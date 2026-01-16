const std = @import("std");
const Allocator = std.mem.Allocator;

const ecs = @import("../ecs/mod.zig");
const EntityId = ecs.EntityId;
const TrackState = ecs.TrackState;

// =============================================================================
// Assignment Structures
// =============================================================================

/// Single observation-to-entity assignment
/// null = clutter (false alarm)
pub const ObsAssignment = ?u32;

/// Assignment of all observations to entities for a single hypothesis
pub const Assignment = struct {
    /// observation_idx -> entity_idx (null = clutter)
    obs_to_entity: std.ArrayListUnmanaged(ObsAssignment),

    /// entity_idx -> track state
    entity_states: std.ArrayListUnmanaged(TrackState),

    /// entity_idx -> observation_idx that matched (null = unmatched)
    entity_to_obs: std.ArrayListUnmanaged(?u32),

    /// Log probability of this assignment
    log_prob: f32,

    /// Allocator
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_observations: usize, num_entities: usize) !Assignment {
        var self = Assignment{
            .obs_to_entity = .empty,
            .entity_states = .empty,
            .entity_to_obs = .empty,
            .log_prob = 0,
            .allocator = allocator,
        };

        try self.obs_to_entity.ensureTotalCapacity(allocator, num_observations);
        try self.entity_states.ensureTotalCapacity(allocator, num_entities);
        try self.entity_to_obs.ensureTotalCapacity(allocator, num_entities);

        // Initialize all observations as clutter
        for (0..num_observations) |_| {
            self.obs_to_entity.appendAssumeCapacity(null);
        }

        // Initialize all entities as detected
        for (0..num_entities) |_| {
            self.entity_states.appendAssumeCapacity(.detected);
            self.entity_to_obs.appendAssumeCapacity(null);
        }

        return self;
    }

    pub fn deinit(self: *Assignment) void {
        self.obs_to_entity.deinit(self.allocator);
        self.entity_states.deinit(self.allocator);
        self.entity_to_obs.deinit(self.allocator);
    }

    pub fn clone(self: *const Assignment, allocator: Allocator) !Assignment {
        const new = Assignment{
            .obs_to_entity = try self.obs_to_entity.clone(allocator),
            .entity_states = try self.entity_states.clone(allocator),
            .entity_to_obs = try self.entity_to_obs.clone(allocator),
            .log_prob = self.log_prob,
            .allocator = allocator,
        };
        return new;
    }

    /// Assign observation to entity (updates both directions)
    pub fn assign(self: *Assignment, obs_idx: u32, entity_idx: u32) void {
        // Clear previous assignment for this observation
        if (self.obs_to_entity.items[obs_idx]) |prev_entity| {
            self.entity_to_obs.items[prev_entity] = null;
        }

        // Clear previous assignment for this entity
        if (self.entity_to_obs.items[entity_idx]) |prev_obs| {
            self.obs_to_entity.items[prev_obs] = null;
        }

        // Make new assignment
        self.obs_to_entity.items[obs_idx] = entity_idx;
        self.entity_to_obs.items[entity_idx] = obs_idx;
    }

    /// Mark observation as clutter
    pub fn markClutter(self: *Assignment, obs_idx: u32) void {
        if (self.obs_to_entity.items[obs_idx]) |entity_idx| {
            self.entity_to_obs.items[entity_idx] = null;
        }
        self.obs_to_entity.items[obs_idx] = null;
    }

    /// Check if entity has any observation assigned
    pub fn isEntityMatched(self: *const Assignment, entity_idx: u32) bool {
        return self.entity_to_obs.items[entity_idx] != null;
    }

    /// Count matched entities
    pub fn matchedCount(self: *const Assignment) usize {
        var count: usize = 0;
        for (self.entity_to_obs.items) |maybe_obs| {
            if (maybe_obs != null) count += 1;
        }
        return count;
    }

    /// Count clutter observations
    pub fn clutterCount(self: *const Assignment) usize {
        var count: usize = 0;
        for (self.obs_to_entity.items) |maybe_entity| {
            if (maybe_entity == null) count += 1;
        }
        return count;
    }
};

// =============================================================================
// Multiple Assignment Hypotheses (GLMB-style)
// =============================================================================

/// Maintains multiple assignment hypotheses per particle
pub const AssignmentHypotheses = struct {
    hypotheses: std.ArrayListUnmanaged(Assignment),
    weights: std.ArrayListUnmanaged(f32),
    allocator: Allocator,

    pub fn init(allocator: Allocator) AssignmentHypotheses {
        return .{
            .hypotheses = .empty,
            .weights = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AssignmentHypotheses) void {
        for (self.hypotheses.items) |*h| {
            h.deinit();
        }
        self.hypotheses.deinit(self.allocator);
        self.weights.deinit(self.allocator);
    }

    /// Add a new hypothesis
    pub fn add(self: *AssignmentHypotheses, assignment: Assignment, weight: f32) !void {
        try self.hypotheses.append(self.allocator, assignment);
        try self.weights.append(self.allocator, weight);
    }

    /// Normalize weights to sum to 1
    pub fn normalize(self: *AssignmentHypotheses) void {
        if (self.weights.items.len == 0) return;

        // Use log-sum-exp for numerical stability
        var max_log: f32 = -std.math.inf(f32);
        for (self.hypotheses.items) |h| {
            if (h.log_prob > max_log) max_log = h.log_prob;
        }

        var sum: f32 = 0;
        for (self.hypotheses.items) |h| {
            sum += @exp(h.log_prob - max_log);
        }

        const log_sum = max_log + @log(sum);

        for (self.weights.items, self.hypotheses.items) |*w, h| {
            w.* = @exp(h.log_prob - log_sum);
        }
    }

    /// Prune hypotheses below threshold
    pub fn prune(self: *AssignmentHypotheses, threshold: f32) void {
        var i: usize = 0;
        while (i < self.weights.items.len) {
            if (self.weights.items[i] < threshold) {
                self.hypotheses.items[i].deinit();
                _ = self.hypotheses.swapRemove(i);
                _ = self.weights.swapRemove(i);
            } else {
                i += 1;
            }
        }

        // Re-normalize after pruning
        self.normalize();
    }

    /// Keep only top K hypotheses
    pub fn keepTopK(self: *AssignmentHypotheses, k: usize) void {
        if (self.hypotheses.items.len <= k) return;

        // Sort by weight descending (simple bubble sort for small k)
        // TODO: Use proper sorting for larger sets
        for (0..self.hypotheses.items.len) |i| {
            for (i + 1..self.hypotheses.items.len) |j| {
                if (self.weights.items[j] > self.weights.items[i]) {
                    std.mem.swap(Assignment, &self.hypotheses.items[i], &self.hypotheses.items[j]);
                    std.mem.swap(f32, &self.weights.items[i], &self.weights.items[j]);
                }
            }
        }

        // Remove excess hypotheses
        while (self.hypotheses.items.len > k) {
            var h = self.hypotheses.pop().?;
            h.deinit();
            _ = self.weights.pop();
        }

        self.normalize();
    }

    /// Sample a hypothesis according to weights
    pub fn sample(self: *const AssignmentHypotheses, rng: std.Random) ?*const Assignment {
        if (self.hypotheses.items.len == 0) return null;

        const u = rng.float(f32);
        var cumulative: f32 = 0;

        for (self.weights.items, self.hypotheses.items) |w, *h| {
            cumulative += w;
            if (u <= cumulative) {
                return h;
            }
        }

        // Return last if numerical issues
        return &self.hypotheses.items[self.hypotheses.items.len - 1];
    }

    /// Get MAP (maximum a posteriori) hypothesis
    pub fn getMAP(self: *const AssignmentHypotheses) ?*const Assignment {
        if (self.hypotheses.items.len == 0) return null;

        var best_idx: usize = 0;
        var best_weight: f32 = self.weights.items[0];

        for (self.weights.items[1..], 1..) |w, i| {
            if (w > best_weight) {
                best_weight = w;
                best_idx = i;
            }
        }

        return &self.hypotheses.items[best_idx];
    }

    /// Get number of hypotheses
    pub fn count(self: *const AssignmentHypotheses) usize {
        return self.hypotheses.items.len;
    }
};

// =============================================================================
// Track State Transition Model
// =============================================================================

/// Compute track state transition probability
pub fn trackTransitionProb(
    from: TrackState,
    to: TrackState,
    matched: bool,
    config: @import("config.zig").TrackTransitionConfig,
) f32 {
    return switch (from) {
        .detected => switch (to) {
            .detected => if (matched) config.detected_to_detected_matched else config.detected_to_detected_unmatched,
            .occluded => if (!matched) config.detected_to_occluded else 0.04,
            .tentative => 0.0, // detected entities don't become tentative
            .dead => config.detected_to_dead,
        },
        .occluded => switch (to) {
            .detected => if (matched) config.occluded_to_detected_matched else 0.01,
            .occluded => if (!matched) config.occluded_to_occluded else 0.18,
            .tentative => 0.0, // occluded entities don't become tentative
            .dead => config.occluded_to_dead,
        },
        .tentative => switch (to) {
            .detected => if (matched) config.tentative_to_detected else 0.1,
            .occluded => if (!matched) 0.2 else 0.05,
            .tentative => if (!matched) 0.3 else 0.1, // stay tentative if unmatched
            .dead => if (!matched) config.tentative_to_dead else 0.05,
        },
        .dead => switch (to) {
            .dead => 1.0,
            else => 0.0,
        },
    };
}

// =============================================================================
// Tests
// =============================================================================

test "Assignment basic operations" {
    const allocator = std.testing.allocator;

    var assignment = try Assignment.init(allocator, 5, 3);
    defer assignment.deinit();

    // Initially all clutter
    try std.testing.expectEqual(@as(usize, 5), assignment.clutterCount());
    try std.testing.expectEqual(@as(usize, 0), assignment.matchedCount());

    // Assign observation 0 to entity 1
    assignment.assign(0, 1);
    try std.testing.expectEqual(@as(?u32, 1), assignment.obs_to_entity.items[0]);
    try std.testing.expectEqual(@as(?u32, 0), assignment.entity_to_obs.items[1]);
    try std.testing.expect(assignment.isEntityMatched(1));
    try std.testing.expect(!assignment.isEntityMatched(0));
}

test "Assignment reassignment" {
    const allocator = std.testing.allocator;

    var assignment = try Assignment.init(allocator, 3, 2);
    defer assignment.deinit();

    // Assign obs 0 to entity 0
    assignment.assign(0, 0);
    try std.testing.expectEqual(@as(?u32, 0), assignment.obs_to_entity.items[0]);

    // Reassign obs 0 to entity 1 (should clear entity 0)
    assignment.assign(0, 1);
    try std.testing.expectEqual(@as(?u32, 1), assignment.obs_to_entity.items[0]);
    try std.testing.expect(!assignment.isEntityMatched(0));
    try std.testing.expect(assignment.isEntityMatched(1));
}

test "AssignmentHypotheses operations" {
    const allocator = std.testing.allocator;

    var hypotheses = AssignmentHypotheses.init(allocator);
    defer hypotheses.deinit();

    // Add hypotheses
    var h1 = try Assignment.init(allocator, 2, 2);
    h1.log_prob = -1.0;
    try hypotheses.add(h1, 0);

    var h2 = try Assignment.init(allocator, 2, 2);
    h2.log_prob = -2.0;
    try hypotheses.add(h2, 0);

    try std.testing.expectEqual(@as(usize, 2), hypotheses.count());

    // Normalize
    hypotheses.normalize();

    // Weights should sum to ~1
    var sum: f32 = 0;
    for (hypotheses.weights.items) |w| {
        sum += w;
    }
    try std.testing.expect(@abs(sum - 1.0) < 0.01);

    // First hypothesis should have higher weight (higher log prob)
    try std.testing.expect(hypotheses.weights.items[0] > hypotheses.weights.items[1]);
}
