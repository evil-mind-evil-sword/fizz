const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../math.zig");
const Vec3 = math.Vec3;

const ecs = @import("../ecs/mod.zig");
const World = ecs.World;
const EntityId = ecs.EntityId;
const TrackState = ecs.TrackState;

const types = @import("../types.zig");
const Camera = types.Camera;

const gmm = @import("../gmm.zig");
const ObservationGrid = gmm.ObservationGrid;

const assignment_mod = @import("assignment.zig");
const Assignment = assignment_mod.Assignment;
const AssignmentHypotheses = assignment_mod.AssignmentHypotheses;
const trackTransitionProb = assignment_mod.trackTransitionProb;

const occlusion_mod = @import("occlusion.zig");
const OcclusionGraph = occlusion_mod.OcclusionGraph;

const likelihood_mod = @import("likelihood.zig");
const singleAssignmentLikelihood = likelihood_mod.singleAssignmentLikelihood;

const config_mod = @import("config.zig");
const AssociationConfig = config_mod.AssociationConfig;

// =============================================================================
// Gibbs Sampler for Data Association
// =============================================================================

/// Gibbs sampler state for data association
pub const GibbsSampler = struct {
    assignment: Assignment,
    occlusion_graph: OcclusionGraph,
    config: AssociationConfig,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        num_observations: usize,
        num_entities: usize,
        config: AssociationConfig,
    ) !GibbsSampler {
        return .{
            .assignment = try Assignment.init(allocator, num_observations, num_entities),
            .occlusion_graph = OcclusionGraph.init(allocator),
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GibbsSampler) void {
        self.assignment.deinit();
        self.occlusion_graph.deinit();
    }

    /// Run Gibbs sampling for data association
    pub fn sample(
        self: *GibbsSampler,
        observation: *const ObservationGrid,
        world: *const World,
        camera: Camera,
        rng: std.Random,
    ) !void {
        // Update occlusion graph
        try self.occlusion_graph.compute(world, camera, self.config.occlusion);

        // Run Gibbs iterations
        for (0..self.config.gibbs_iterations) |_| {
            // Sample observation assignments
            try self.sampleObservationAssignments(observation, world, camera, rng);

            // Sample entity track states
            self.sampleTrackStates(observation, world, rng);
        }

        // Compute final assignment log probability
        self.assignment.log_prob = try likelihood_mod.assignmentLogLikelihood(
            observation,
            &self.assignment,
            world,
            camera,
            self.config,
            self.allocator,
        );
    }

    /// Sample assignment for each observation
    fn sampleObservationAssignments(
        self: *GibbsSampler,
        observation: *const ObservationGrid,
        world: *const World,
        camera: Camera,
        rng: std.Random,
    ) !void {
        const n_obs = self.assignment.obs_to_entity.items.len;
        const n_entities = self.assignment.entity_states.items.len;

        // Random order for observations
        var order = try std.ArrayListUnmanaged(u32).initCapacity(self.allocator, n_obs);
        defer order.deinit(self.allocator);

        for (0..n_obs) |i| {
            order.appendAssumeCapacity(@intCast(i));
        }

        // Shuffle order
        rng.shuffle(u32, order.items);

        // Sample each observation
        for (order.items) |obs_idx| {
            try self.sampleSingleObservation(obs_idx, n_entities, observation, world, camera, rng);
        }
    }

    /// Sample assignment for a single observation
    fn sampleSingleObservation(
        self: *GibbsSampler,
        obs_idx: u32,
        n_entities: usize,
        observation: *const ObservationGrid,
        world: *const World,
        camera: Camera,
        rng: std.Random,
    ) !void {
        // Compute conditional probabilities for each assignment
        var log_probs = try std.ArrayListUnmanaged(f32).initCapacity(self.allocator, n_entities + 1);
        defer log_probs.deinit(self.allocator);

        // Option: clutter
        const clutter_lik = singleAssignmentLikelihood(obs_idx, null, observation, world, camera, self.config);
        log_probs.appendAssumeCapacity(clutter_lik);

        // Option: each entity
        for (0..n_entities) |entity_idx| {
            // Skip if entity already has an observation (one-to-one constraint)
            // unless it's currently assigned to this observation
            const current_obs = self.assignment.entity_to_obs.items[entity_idx];
            if (current_obs != null and current_obs.? != obs_idx) {
                log_probs.appendAssumeCapacity(-std.math.inf(f32));
                continue;
            }

            // Skip if entity is dead
            if (self.assignment.entity_states.items[entity_idx] == .dead) {
                log_probs.appendAssumeCapacity(-std.math.inf(f32));
                continue;
            }

            const entity_lik = singleAssignmentLikelihood(
                obs_idx,
                @intCast(entity_idx),
                observation,
                world,
                camera,
                self.config,
            );
            log_probs.appendAssumeCapacity(entity_lik);
        }

        // Sample from conditional distribution
        const sampled_idx = sampleFromLogProbs(log_probs.items, rng);

        // Update assignment
        if (sampled_idx == 0) {
            self.assignment.markClutter(obs_idx);
        } else {
            self.assignment.assign(obs_idx, @intCast(sampled_idx - 1));
        }
    }

    /// Sample track state for each entity
    fn sampleTrackStates(
        self: *GibbsSampler,
        observation: *const ObservationGrid,
        world: *const World,
        rng: std.Random,
    ) void {
        _ = observation;
        _ = world;

        for (self.assignment.entity_states.items, 0..) |current_state, i| {
            if (current_state == .dead) continue;

            const matched = self.assignment.isEntityMatched(@intCast(i));
            const is_occluded = self.occlusion_graph.isOccluded(@intCast(i));

            // Compute transition probabilities for 4 states
            var log_probs: [4]f32 = undefined;

            // P(detected)
            log_probs[0] = @log(trackTransitionProb(current_state, .detected, matched, self.config.track_transition));
            if (!matched and !is_occluded) {
                // Penalty for being detected but not matched when visible
                log_probs[0] += @log(0.1);
            }

            // P(occluded)
            log_probs[1] = @log(trackTransitionProb(current_state, .occluded, matched, self.config.track_transition));
            if (matched) {
                // Penalty for being occluded but matched
                log_probs[1] += @log(0.1);
            }
            if (!is_occluded) {
                // Penalty for claiming occlusion when visible
                log_probs[1] += @log(0.1);
            }

            // P(tentative)
            log_probs[2] = @log(trackTransitionProb(current_state, .tentative, matched, self.config.track_transition));

            // P(dead)
            log_probs[3] = @log(trackTransitionProb(current_state, .dead, matched, self.config.track_transition));

            // Sample new state
            const sampled = sampleFromLogProbs(&log_probs, rng);
            self.assignment.entity_states.items[i] = switch (sampled) {
                0 => .detected,
                1 => .occluded,
                2 => .tentative,
                else => .dead,
            };
        }
    }
};

/// Sample from a categorical distribution given log probabilities
fn sampleFromLogProbs(log_probs: []const f32, rng: std.Random) usize {
    // Find max for numerical stability
    var max_log: f32 = -std.math.inf(f32);
    for (log_probs) |lp| {
        if (lp > max_log) max_log = lp;
    }

    // Handle all -inf case
    if (max_log == -std.math.inf(f32)) {
        return 0;
    }

    // Compute normalized probabilities
    var probs: [64]f32 = undefined;
    var sum: f32 = 0;

    for (log_probs, 0..) |lp, i| {
        if (i >= 64) break;
        probs[i] = @exp(lp - max_log);
        sum += probs[i];
    }

    // Sample
    const u = rng.float(f32) * sum;
    var cumulative: f32 = 0;

    for (0..log_probs.len) |i| {
        if (i >= 64) break;
        cumulative += probs[i];
        if (u <= cumulative) {
            return i;
        }
    }

    return log_probs.len - 1;
}

// =============================================================================
// Multi-Hypothesis Gibbs (for AssignmentHypotheses)
// =============================================================================

/// Generate multiple assignment hypotheses via Gibbs sampling
pub fn generateHypotheses(
    observation: *const ObservationGrid,
    world: *const World,
    camera: Camera,
    config: AssociationConfig,
    num_hypotheses: usize,
    rng: std.Random,
    allocator: Allocator,
) !AssignmentHypotheses {
    var hypotheses = AssignmentHypotheses.init(allocator);
    errdefer hypotheses.deinit();

    // Count observations and entities
    const num_observations = observation.width * observation.height;

    var entity_count: usize = 0;
    var query = world.queryPhysicsEntities();
    while (query.next()) |_| {
        entity_count += 1;
    }

    if (entity_count == 0) {
        // No entities - single hypothesis with all clutter
        var empty_assignment = try Assignment.init(allocator, num_observations, 0);
        empty_assignment.log_prob = 0;
        try hypotheses.add(empty_assignment, 1.0);
        return hypotheses;
    }

    // Generate hypotheses via multiple Gibbs chains
    for (0..num_hypotheses) |_| {
        var sampler = try GibbsSampler.init(allocator, num_observations, entity_count, config);
        defer sampler.deinit();

        try sampler.sample(observation, world, camera, rng);

        // Clone the assignment into hypotheses
        const cloned = try sampler.assignment.clone(allocator);
        try hypotheses.add(cloned, 0);
    }

    // Normalize and prune
    hypotheses.normalize();
    hypotheses.prune(config.pruning_threshold);
    hypotheses.keepTopK(config.max_hypotheses);

    return hypotheses;
}

// =============================================================================
// Tests
// =============================================================================

test "sampleFromLogProbs" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Simple case: one option much more likely
    const log_probs = [_]f32{ 0, -10, -10 };
    var counts = [_]usize{ 0, 0, 0 };

    for (0..100) |_| {
        const idx = sampleFromLogProbs(&log_probs, rng);
        counts[idx] += 1;
    }

    // First option should be sampled most often
    try std.testing.expect(counts[0] > 80);
}

test "GibbsSampler init" {
    const allocator = std.testing.allocator;

    var sampler = try GibbsSampler.init(allocator, 100, 3, .{});
    defer sampler.deinit();

    try std.testing.expectEqual(@as(usize, 100), sampler.assignment.obs_to_entity.items.len);
    try std.testing.expectEqual(@as(usize, 3), sampler.assignment.entity_states.items.len);
}
