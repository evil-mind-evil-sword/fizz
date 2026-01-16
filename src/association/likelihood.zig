const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../math.zig");
const Vec3 = math.Vec3;
const Vec2 = math.Vec2;

const ecs = @import("../ecs/mod.zig");
const World = ecs.World;
const EntityId = ecs.EntityId;
const TrackState = ecs.TrackState;

const types = @import("../types.zig");
const Camera = types.Camera;

const gmm = @import("../gmm.zig");
const ObservationGrid = gmm.ObservationGrid;
const Observation = gmm.Observation;

const assignment_mod = @import("assignment.zig");
const Assignment = assignment_mod.Assignment;

const occlusion_mod = @import("occlusion.zig");
const OcclusionGraph = occlusion_mod.OcclusionGraph;
const PixelOwnership = occlusion_mod.PixelOwnership;
const renderPixelOwnership = occlusion_mod.renderPixelOwnership;

const config_mod = @import("config.zig");
const AssociationConfig = config_mod.AssociationConfig;

// =============================================================================
// Likelihood Computation
// =============================================================================

/// Compute log likelihood of observation given assignment and world state
pub fn assignmentLogLikelihood(
    observation: *const ObservationGrid,
    assignment: *const Assignment,
    world: *const World,
    camera: Camera,
    config: AssociationConfig,
    allocator: Allocator,
) !f32 {
    var log_lik: f32 = 0;

    // Render expected pixel ownership
    var ownership = try renderPixelOwnership(world, camera, observation.width, observation.height, allocator);
    defer ownership.deinit();

    // Pixel-wise likelihood
    const pixel_log_lik = try computePixelLikelihood(observation, &ownership, world, config, allocator);
    log_lik += pixel_log_lik;

    // Track state transition prior
    const state_log_prior = computeTrackStatePrior(assignment, config);
    log_lik += state_log_prior;

    // Clutter penalty
    const clutter_log_lik = computeClutterLikelihood(assignment, observation, config);
    log_lik += clutter_log_lik;

    return log_lik;
}

/// Compute pixel-wise log likelihood comparing observation to rendered expectation
fn computePixelLikelihood(
    observation: *const ObservationGrid,
    ownership: *const PixelOwnership,
    world: *const World,
    config: AssociationConfig,
    allocator: Allocator,
) !f32 {
    _ = allocator;
    var log_lik: f32 = 0;
    const noise_var = config.observation_noise * config.observation_noise;
    const log_norm = -0.5 * @log(2.0 * std.math.pi * noise_var);

    for (0..observation.height) |y| {
        for (0..observation.width) |x| {
            const obs = observation.get(@intCast(x), @intCast(y));
            const owner = ownership.getOwner(@intCast(x), @intCast(y));

            if (owner) |entity_idx| {
                // Pixel should match entity color
                if (world.appearances.get(entity_idx)) |app| {
                    // Compare RGB
                    const expected_color = app.color;
                    const obs_color = obs.color;
                    const diff = obs_color.sub(expected_color);
                    const sq_err = diff.dot(diff);

                    // 3D Gaussian likelihood (independent RGB channels)
                    log_lik += 3.0 * log_norm - sq_err / (2.0 * noise_var);
                }
            } else {
                // Background pixel - should be dark/neutral
                const bg_color = Vec3.init(0.1, 0.1, 0.1);
                const obs_color = obs.color;
                const diff = obs_color.sub(bg_color);
                const sq_err = diff.dot(diff);

                // Higher variance for background (less certain)
                const bg_var = noise_var * 10.0;
                const bg_log_norm = -0.5 * @log(2.0 * std.math.pi * bg_var);
                log_lik += 3.0 * bg_log_norm - sq_err / (2.0 * bg_var);
            }
        }
    }

    return log_lik;
}

/// Compute log prior for track state configuration
fn computeTrackStatePrior(
    assignment: *const Assignment,
    config: AssociationConfig,
) f32 {
    var log_prior: f32 = 0;

    for (assignment.entity_states.items, 0..) |state, i| {
        const matched = assignment.isEntityMatched(@intCast(i));

        const prob = switch (state) {
            .detected => if (matched) config.detection_prob else (1.0 - config.detection_prob),
            .occluded => config.occlusion.existence_decay,
            .tentative => if (matched) @as(f32, 0.7) else @as(f32, 0.3), // Tentative tracks need confirmation
            .dead => 1.0 - config.track_transition.detected_to_dead,
        };

        log_prior += @log(@max(prob, 1e-10));
    }

    return log_prior;
}

/// Compute log likelihood for clutter observations
fn computeClutterLikelihood(
    assignment: *const Assignment,
    observation: *const ObservationGrid,
    config: AssociationConfig,
) f32 {
    const n_clutter = assignment.clutterCount();
    const total_pixels = observation.width * observation.height;

    // Poisson clutter model: P(n_clutter) = exp(-lambda) * lambda^n / n!
    // log P = -lambda + n * log(lambda) - log(n!)
    const lambda = config.clutter_rate * @as(f32, @floatFromInt(total_pixels));

    if (n_clutter == 0) {
        return -lambda;
    }

    // Stirling's approximation for log(n!)
    const n = @as(f32, @floatFromInt(n_clutter));
    const log_factorial = n * @log(n) - n + 0.5 * @log(2.0 * std.math.pi * n);

    return -lambda + n * @log(lambda + 1e-10) - log_factorial;
}

// =============================================================================
// Single Entity Likelihood (for Gibbs updates)
// =============================================================================

/// Compute likelihood of single observation-entity assignment
pub fn singleAssignmentLikelihood(
    obs_idx: u32,
    entity_idx: ?u32,
    observation: *const ObservationGrid,
    world: *const World,
    camera: Camera,
    config: AssociationConfig,
) f32 {
    // Get observation location (convert from grid index)
    const width = observation.width;
    const x = obs_idx % width;
    const y = obs_idx / width;
    const obs = observation.get(x, y);
    const obs_color = obs.color;

    if (entity_idx) |ent_idx| {
        // Match observation to entity
        const pos = world.positions.get(ent_idx) orelse return -std.math.inf(f32);
        const app = world.appearances.get(ent_idx);

        // Check if observation could plausibly come from entity projection
        if (camera.project(pos.mean)) |proj| {
            const expected_color = if (app) |a| a.color else Vec3.splat(0.5);

            // Spatial gating
            const proj_x = (proj.ndc.x + 1.0) * 0.5 * @as(f32, @floatFromInt(width));
            const proj_y = (1.0 - proj.ndc.y) * 0.5 * @as(f32, @floatFromInt(observation.height));

            const dx = @as(f32, @floatFromInt(x)) - proj_x;
            const dy = @as(f32, @floatFromInt(y)) - proj_y;
            const spatial_dist_sq = dx * dx + dy * dy;

            const projected_radius = camera.projectRadius(if (app) |a| a.radius else 0.5, proj.depth);
            const radius_pixels = projected_radius * @as(f32, @floatFromInt(width)) * 0.5;

            if (spatial_dist_sq > radius_pixels * radius_pixels * 4.0) {
                // Too far from entity center
                return -std.math.inf(f32);
            }

            // Color likelihood
            const color_diff = obs_color.sub(expected_color);
            const color_dist_sq = color_diff.dot(color_diff);
            const noise_var = config.observation_noise * config.observation_noise;

            // Spatial likelihood (Gaussian falloff from center)
            const spatial_var = radius_pixels * radius_pixels;
            const spatial_log_lik = -spatial_dist_sq / (2.0 * spatial_var);

            // Combined likelihood
            const color_log_lik = -color_dist_sq / (2.0 * noise_var);

            return spatial_log_lik + color_log_lik + @log(config.detection_prob);
        }

        return -std.math.inf(f32);
    } else {
        // Clutter observation
        const bg_color = Vec3.init(0.1, 0.1, 0.1);
        const diff = obs_color.sub(bg_color);
        const dist_sq = diff.dot(diff);

        // Clutter has higher variance (uniform-ish)
        const clutter_var = 0.5;
        return -dist_sq / (2.0 * clutter_var) + @log(config.clutter_rate);
    }
}

/// Compute likelihood ratio for assigning observation to entity vs clutter
pub fn assignmentLikelihoodRatio(
    obs_idx: u32,
    entity_idx: u32,
    observation: *const ObservationGrid,
    world: *const World,
    camera: Camera,
    config: AssociationConfig,
) f32 {
    const lik_entity = singleAssignmentLikelihood(obs_idx, entity_idx, observation, world, camera, config);
    const lik_clutter = singleAssignmentLikelihood(obs_idx, null, observation, world, camera, config);

    return lik_entity - lik_clutter;
}

// =============================================================================
// Entity Detection Likelihood
// =============================================================================

/// Compute likelihood of entity being detected vs occluded
pub fn detectionLikelihood(
    entity_idx: u32,
    is_detected: bool,
    observation: *const ObservationGrid,
    world: *const World,
    occlusion_graph: *const OcclusionGraph,
    config: AssociationConfig,
) f32 {
    _ = observation;
    _ = world;

    const is_occluded = occlusion_graph.isOccluded(entity_idx);

    if (is_detected) {
        if (is_occluded) {
            // Detected despite being occluded (unlikely but possible)
            return @log(0.1 * config.detection_prob);
        } else {
            // Detected and visible (expected)
            return @log(config.detection_prob);
        }
    } else {
        if (is_occluded) {
            // Not detected because occluded (expected)
            return @log(config.occlusion.existence_decay);
        } else {
            // Not detected despite being visible (missed detection)
            return @log(1.0 - config.detection_prob);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "computeClutterLikelihood" {
    const allocator = std.testing.allocator;

    var assignment = try Assignment.init(allocator, 10, 2);
    defer assignment.deinit();

    // Create mock observation grid
    var obs = try ObservationGrid.init(10, 10, allocator);
    defer obs.deinit();

    const config = AssociationConfig{};

    // All observations are clutter
    const log_lik = computeClutterLikelihood(&assignment, &obs, config);

    // Should be finite
    try std.testing.expect(std.math.isFinite(log_lik));
}

test "singleAssignmentLikelihood" {
    const allocator = std.testing.allocator;

    var world = World.init(allocator);
    defer world.deinit();

    _ = try world.spawnPhysics(Vec3.init(0, 0, 0), Vec3.zero, ecs.Physics.standard);

    var obs = try ObservationGrid.init(10, 10, allocator);
    defer obs.deinit();

    const camera = Camera.default;
    const config = AssociationConfig{};

    // Clutter likelihood
    const clutter_lik = singleAssignmentLikelihood(55, null, &obs, &world, camera, config);
    try std.testing.expect(std.math.isFinite(clutter_lik));
}
