const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../math.zig");
const Vec3 = math.Vec3;

const types = @import("../types.zig");
const Camera = types.Camera;
const PhysicsParams = types.PhysicsParams;
const Label = types.Label;
const EntityId = types.EntityId;

const gmm = @import("../gmm.zig");
const ObservationGrid = gmm.ObservationGrid;

const association = @import("../association/mod.zig");
const Assignment = association.Assignment;
const PixelOwnership = association.PixelOwnership;
const renderPixelOwnership = association.renderPixelOwnership;

const config_mod = @import("config.zig");
const CRPConfig = config_mod.CRPConfig;

const label_set_mod = @import("label_set.zig");
const LabelSet = label_set_mod.LabelSet;
const crpNewEntityProb = label_set_mod.crpNewEntityProb;

// =============================================================================
// Birth Proposal
// =============================================================================

/// Result of a birth proposal
pub const BirthProposal = struct {
    /// Proposed position for new entity
    position: Vec3,
    /// Proposed velocity for new entity
    velocity: Vec3,
    /// Physics parameters for new entity
    physics_params: PhysicsParams,
    /// Log acceptance probability
    log_accept_prob: f32,
    /// Cluster size that generated this proposal
    cluster_size: usize,
    /// Color from cluster average
    color: Vec3,
};

/// Cluster of unexplained observations
pub const UnexplainedCluster = struct {
    /// Centroid position in world coordinates
    centroid: Vec3,
    /// Number of pixels in cluster
    size: usize,
    /// Average color
    avg_color: Vec3,
    /// Bounding box in pixel space
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
};

// =============================================================================
// Unexplained Observation Detection
// =============================================================================

/// Find unexplained pixels (high residual after entity assignment)
/// world can be any type with getAppearance() method
pub fn findUnexplainedPixels(
    observation: *const ObservationGrid,
    ownership: *const PixelOwnership,
    world: anytype,
    threshold: f32,
    allocator: Allocator,
) !std.ArrayListUnmanaged(u32) {
    var unexplained: std.ArrayListUnmanaged(u32) = .empty;

    for (0..observation.height) |y| {
        for (0..observation.width) |x| {
            const pixel_idx: u32 = @intCast(y * observation.width + x);
            const obs = observation.get(@intCast(x), @intCast(y));
            const owner = ownership.getOwner(@intCast(x), @intCast(y));

            const residual = if (owner) |entity_idx| blk: {
                // Compare to entity color
                const entity_id = EntityId{ .index = entity_idx, .generation = 0 };
                if (world.getAppearance(entity_id)) |app| {
                    const diff = obs.color.sub(app.color);
                    break :blk diff.dot(diff);
                }
                break :blk obs.color.dot(obs.color);
            } else blk: {
                // Background - check if bright enough to be unexplained
                const brightness = (obs.color.x + obs.color.y + obs.color.z) / 3.0;
                break :blk brightness;
            };

            if (residual > threshold) {
                try unexplained.append(allocator, pixel_idx);
            }
        }
    }

    return unexplained;
}

/// Cluster unexplained pixels using simple connected components
pub fn clusterUnexplained(
    unexplained_pixels: []const u32,
    observation: *const ObservationGrid,
    camera: Camera,
    min_cluster_size: u32,
    allocator: Allocator,
) !std.ArrayListUnmanaged(UnexplainedCluster) {
    var clusters: std.ArrayListUnmanaged(UnexplainedCluster) = .empty;

    if (unexplained_pixels.len == 0) return clusters;

    // Simple clustering: group adjacent pixels
    var visited = try allocator.alloc(bool, observation.width * observation.height);
    defer allocator.free(visited);
    @memset(visited, false);

    // Mark unexplained pixels
    var unexplained_set = std.AutoHashMapUnmanaged(u32, void){};
    defer unexplained_set.deinit(allocator);

    for (unexplained_pixels) |idx| {
        try unexplained_set.put(allocator, idx, {});
    }

    // Find connected components
    for (unexplained_pixels) |start_idx| {
        if (visited[start_idx]) continue;

        // BFS to find cluster
        var cluster_pixels: std.ArrayListUnmanaged(u32) = .empty;
        defer cluster_pixels.deinit(allocator);

        var queue: std.ArrayListUnmanaged(u32) = .empty;
        defer queue.deinit(allocator);

        try queue.append(allocator, start_idx);
        visited[start_idx] = true;

        while (queue.items.len > 0) {
            const current = queue.pop().?;
            try cluster_pixels.append(allocator, current);

            const cx = current % observation.width;
            const cy = current / observation.width;

            // Check 4-neighbors
            const neighbors = [_][2]i32{
                .{ -1, 0 },
                .{ 1, 0 },
                .{ 0, -1 },
                .{ 0, 1 },
            };

            for (neighbors) |offset| {
                const nx_i: i32 = @as(i32, @intCast(cx)) + offset[0];
                const ny_i: i32 = @as(i32, @intCast(cy)) + offset[1];

                if (nx_i < 0 or ny_i < 0) continue;
                const nx: u32 = @intCast(nx_i);
                const ny: u32 = @intCast(ny_i);

                if (nx >= observation.width or ny >= observation.height) continue;

                const neighbor_idx: u32 = ny * observation.width + nx;
                if (!visited[neighbor_idx] and unexplained_set.contains(neighbor_idx)) {
                    visited[neighbor_idx] = true;
                    try queue.append(allocator, neighbor_idx);
                }
            }
        }

        // Skip small clusters
        if (cluster_pixels.items.len < min_cluster_size) continue;

        // Compute cluster properties
        var sum_x: f32 = 0;
        var sum_y: f32 = 0;
        var sum_color = Vec3.zero;
        var min_x: u32 = observation.width;
        var max_x: u32 = 0;
        var min_y: u32 = observation.height;
        var max_y: u32 = 0;

        for (cluster_pixels.items) |idx| {
            const px = idx % observation.width;
            const py = idx / observation.width;
            sum_x += @floatFromInt(px);
            sum_y += @floatFromInt(py);

            const obs = observation.get(px, py);
            sum_color = sum_color.add(obs.color);

            min_x = @min(min_x, px);
            max_x = @max(max_x, px);
            min_y = @min(min_y, py);
            max_y = @max(max_y, py);
        }

        const n = @as(f32, @floatFromInt(cluster_pixels.items.len));
        const centroid_x = sum_x / n;
        const centroid_y = sum_y / n;
        const avg_color = sum_color.scale(1.0 / n);

        // Convert pixel centroid to world coordinates
        // Use depth estimate from camera
        const ndc_x = (centroid_x / @as(f32, @floatFromInt(observation.width))) * 2.0 - 1.0;
        const ndc_y = 1.0 - (centroid_y / @as(f32, @floatFromInt(observation.height))) * 2.0;

        const depth_estimate: f32 = 5.0; // Default depth estimate
        const world_pos = ndcToWorld(ndc_x, ndc_y, depth_estimate, camera);

        try clusters.append(allocator, .{
            .centroid = world_pos,
            .size = cluster_pixels.items.len,
            .avg_color = avg_color,
            .min_x = min_x,
            .max_x = max_x,
            .min_y = min_y,
            .max_y = max_y,
        });
    }

    return clusters;
}

/// Convert NDC coordinates to world position at given depth
fn ndcToWorld(ndc_x: f32, ndc_y: f32, depth: f32, camera: Camera) Vec3 {
    const forward = camera.target.sub(camera.position).normalize();
    const right = forward.cross(camera.up).normalize();
    const up = right.cross(forward);

    const tan_half_fov = @tan(camera.fov / 2.0);
    const world_x = ndc_x * depth * tan_half_fov * camera.aspect;
    const world_y = ndc_y * depth * tan_half_fov;

    return camera.position.add(forward.scale(depth)).add(right.scale(world_x)).add(up.scale(world_y));
}

// =============================================================================
// Birth Proposal Generation
// =============================================================================

/// Generate a birth proposal from unexplained observations
/// world can be any type with queryPhysicsEntities(), getPosition(), getAppearance()
pub fn proposeBirth(
    observation: *const ObservationGrid,
    world: anytype,
    label_set: *const LabelSet,
    camera: Camera,
    config: CRPConfig,
    rng: std.Random,
    allocator: Allocator,
) !?BirthProposal {
    // Check entity limit
    if (label_set.count() >= config.max_entities) return null;

    // Render current pixel ownership
    var ownership = try renderPixelOwnership(world, camera, observation.width, observation.height, allocator);
    defer ownership.deinit();

    // Find unexplained pixels
    var unexplained = try findUnexplainedPixels(
        observation,
        &ownership,
        world,
        config.unexplained_threshold,
        allocator,
    );
    defer unexplained.deinit(allocator);

    if (unexplained.items.len == 0) return null;

    // Cluster unexplained pixels
    var clusters = try clusterUnexplained(
        unexplained.items,
        observation,
        camera,
        config.min_cluster_size,
        allocator,
    );
    defer clusters.deinit(allocator);

    if (clusters.items.len == 0) return null;

    // Select cluster (weighted by size)
    var total_size: usize = 0;
    for (clusters.items) |c| {
        total_size += c.size;
    }

    const selected_idx = blk: {
        const u = rng.float(f32) * @as(f32, @floatFromInt(total_size));
        var cumulative: f32 = 0;
        for (clusters.items, 0..) |c, i| {
            cumulative += @floatFromInt(c.size);
            if (u <= cumulative) {
                break :blk i;
            }
        }
        break :blk clusters.items.len - 1;
    };

    const cluster = clusters.items[selected_idx];

    // Compute birth acceptance probability
    const n_existing = label_set.count();
    const prior_prob = crpNewEntityProb(n_existing, config.alpha);

    // Likelihood ratio: compare observation likelihood with vs without new entity
    // With new entity: cluster pixels match entity color
    // Without: cluster pixels are unexplained (high residual)
    const cluster_size_f = @as(f32, @floatFromInt(cluster.size));
    const min_size_f = @as(f32, @floatFromInt(config.min_cluster_size));

    // Log likelihood with entity: pixels match entity color (low residual)
    // Assuming Gaussian observation model with variance sigma^2
    const obs_var: f32 = 0.1 * 0.1;
    const log_lik_with = -cluster_size_f * 0.5 * @log(2.0 * std.math.pi * obs_var);

    // Log likelihood without entity: pixels are unexplained (high residual)
    // Use threshold as typical residual magnitude
    const unexplained_residual = config.unexplained_threshold;
    const log_lik_without = log_lik_with - cluster_size_f * unexplained_residual * unexplained_residual / (2.0 * obs_var);

    // Likelihood ratio in log space
    const log_lik_ratio = log_lik_with - log_lik_without;

    // Proposal ratio for MH correction
    // Birth proposal: proportional to cluster size
    // Death proposal: uniform over entities
    const birth_proposal_density = cluster_size_f / @as(f32, @floatFromInt(total_size));
    const death_proposal_density = 1.0 / @as(f32, @floatFromInt(n_existing + 1));
    const log_proposal_ratio = @log(death_proposal_density) - @log(birth_proposal_density + 1e-10);

    // Total log acceptance probability including size bonus
    const log_accept_prob = @log(prior_prob) + log_lik_ratio + log_proposal_ratio + @log(cluster_size_f / min_size_f);

    // Sample initial velocity (small random)
    const vel_std = @sqrt(config.birth_velocity_variance);
    const velocity = Vec3.init(
        sampleStdNormal(rng) * vel_std,
        sampleStdNormal(rng) * vel_std,
        sampleStdNormal(rng) * vel_std,
    );

    return BirthProposal{
        .position = cluster.centroid,
        .velocity = velocity,
        .physics_params = PhysicsParams.standard,
        .log_accept_prob = log_accept_prob,
        .cluster_size = cluster.size,
        .color = cluster.avg_color,
    };
}

/// Execute a birth proposal (add entity to world)
/// world can be any type with spawnPhysics(), setAppearance(), setOcclusion()
pub fn executeBirth(
    proposal: BirthProposal,
    world: anytype,
    label_set: *LabelSet,
) ?EntityId {
    // Create new entity
    const entity_id = world.spawnPhysics(
        proposal.position,
        proposal.velocity,
        proposal.physics_params,
    ) orelse return null;

    // Register in label set
    const label = label_set.createLabel();
    label_set.register(label, entity_id) catch return null;

    // Set appearance color from cluster average
    const Appearance = @import("../smc/world_view.zig").Appearance;
    world.setAppearance(entity_id, Appearance{
        .color = proposal.color,
        .opacity = 1.0,
        .radius = 0.5,
    });

    // Set initial track state to tentative (needs confirmation)
    const Occlusion = @import("../smc/world_view.zig").Occlusion;
    world.setOcclusion(entity_id, Occlusion{
        .state = .tentative,
        .duration = 0,
        .existence_prob = 0.5, // Start with moderate confidence
    });

    return entity_id;
}

/// Sample from standard normal distribution
fn sampleStdNormal(rng: std.Random) f32 {
    const r1 = rng.float(f32);
    const r2 = rng.float(f32);
    return @sqrt(-2.0 * @log(r1 + 1e-10)) * @cos(2.0 * std.math.pi * r2);
}

// =============================================================================
// Tests
// =============================================================================

const smc = @import("../smc/mod.zig");
const ParticleSwarm = smc.ParticleSwarm;
const particleWorld = smc.particleWorld;

test "findUnexplainedPixels empty" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 1, 8);
    defer swarm.deinit();
    var world = particleWorld(&swarm, 0);

    var obs = try ObservationGrid.init(10, 10, allocator);
    defer obs.deinit();

    var ownership = try PixelOwnership.init(allocator, 10, 10);
    defer ownership.deinit();

    var unexplained = try findUnexplainedPixels(&obs, &ownership, &world, 0.1, allocator);
    defer unexplained.deinit(allocator);

    // All pixels are dark/zero, so none should be unexplained
    try std.testing.expectEqual(@as(usize, 0), unexplained.items.len);
}

test "BirthProposal structure" {
    const proposal = BirthProposal{
        .position = Vec3.init(1, 2, 3),
        .velocity = Vec3.zero,
        .physics_params = PhysicsParams.standard,
        .log_accept_prob = -1.0,
        .cluster_size = 100,
        .color = Vec3.init(0.5, 0.5, 0.5),
    };

    try std.testing.expect(proposal.position.x == 1);
}
