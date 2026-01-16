const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../math.zig");
const Vec3 = math.Vec3;
const Vec2 = math.Vec2;

const types = @import("../types.zig");
const Camera = types.Camera;
const EntityId = types.EntityId;

const config_mod = @import("config.zig");
const OcclusionConfig = config_mod.OcclusionConfig;

// =============================================================================
// Projected Entity for Occlusion Reasoning
// =============================================================================

/// Entity projected to image plane with depth info
pub const ProjectedEntity = struct {
    entity_idx: u32,
    /// Center in NDC [-1, 1]
    center: Vec2,
    /// Depth (distance from camera)
    depth: f32,
    /// Projected radius in NDC
    radius: f32,
    /// Is visible (not fully occluded)
    visible: bool,
};

// =============================================================================
// Occlusion Graph
// =============================================================================

/// Tracks occlusion relationships between entities
pub const OcclusionGraph = struct {
    /// For each entity pair (i,j): P(i occludes j)
    /// Matrix is row-major: occlusion_probs[i * n + j]
    occlusion_probs: std.ArrayListUnmanaged(f32),

    /// Number of entities
    n_entities: usize,

    /// Projected entities (sorted by depth, near to far)
    projections: std.ArrayListUnmanaged(ProjectedEntity),

    /// Depth ordering (indices into projections, front to back)
    depth_order: std.ArrayListUnmanaged(u32),

    allocator: Allocator,

    pub fn init(allocator: Allocator) OcclusionGraph {
        return .{
            .occlusion_probs = .empty,
            .n_entities = 0,
            .projections = .empty,
            .depth_order = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *OcclusionGraph) void {
        self.occlusion_probs.deinit(self.allocator);
        self.projections.deinit(self.allocator);
        self.depth_order.deinit(self.allocator);
    }

    /// Compute occlusion graph from world state and camera
    /// world can be any type with queryPhysicsEntities(), getPosition(), getAppearance()
    pub fn compute(
        self: *OcclusionGraph,
        world: anytype,
        camera: Camera,
        config: OcclusionConfig,
    ) !void {
        // Clear previous state
        self.occlusion_probs.clearRetainingCapacity();
        self.projections.clearRetainingCapacity();
        self.depth_order.clearRetainingCapacity();

        // Project all entities
        var query = world.queryPhysicsEntities();
        while (query.next()) |id| {
            const pos = world.getPosition(id) orelse continue;
            const app = world.getAppearance(id);
            const radius = if (app) |a| a.radius else 0.5;

            if (camera.project(pos.mean)) |proj| {
                const projected_radius = camera.projectRadius(radius, proj.depth);

                try self.projections.append(self.allocator, .{
                    .entity_idx = id.index,
                    .center = proj.ndc,
                    .depth = proj.depth,
                    .radius = projected_radius,
                    .visible = true, // Initially visible
                });
            }
        }

        self.n_entities = self.projections.items.len;
        if (self.n_entities == 0) return;

        // Sort by depth (front to back for occlusion testing)
        std.mem.sort(ProjectedEntity, self.projections.items, {}, struct {
            fn lessThan(_: void, a: ProjectedEntity, b: ProjectedEntity) bool {
                return a.depth < b.depth;
            }
        }.lessThan);

        // Build depth order
        try self.depth_order.ensureTotalCapacity(self.allocator, self.n_entities);
        for (0..self.n_entities) |i| {
            self.depth_order.appendAssumeCapacity(@intCast(i));
        }

        // Compute occlusion probabilities
        const n = self.n_entities;
        try self.occlusion_probs.ensureTotalCapacity(self.allocator, n * n);
        for (0..n * n) |_| {
            self.occlusion_probs.appendAssumeCapacity(0);
        }

        // For each pair, check if one occludes the other
        for (0..n) |i| {
            const proj_i = self.projections.items[i];

            for (i + 1..n) |j| {
                const proj_j = self.projections.items[j];

                // i is closer (sorted by depth), so i can occlude j
                const overlap = computeOverlap(proj_i, proj_j);

                if (overlap > config.overlap_threshold) {
                    // i occludes j with probability based on overlap
                    self.occlusion_probs.items[i * n + j] = overlap;

                    // Mark j as potentially occluded
                    if (overlap > 0.5) {
                        self.projections.items[j].visible = false;
                    }
                }
            }
        }
    }

    /// Check if entity at index is occluded by any other entity
    pub fn isOccluded(self: *const OcclusionGraph, entity_idx: u32) bool {
        // Find projection for this entity
        for (self.projections.items) |proj| {
            if (proj.entity_idx == entity_idx) {
                return !proj.visible;
            }
        }
        // Entity not projected (behind camera)
        return true;
    }

    /// Get list of entities that could occlude the given entity
    pub fn getOccluders(
        self: *const OcclusionGraph,
        entity_idx: u32,
        allocator: Allocator,
    ) !std.ArrayListUnmanaged(u32) {
        var occluders: std.ArrayListUnmanaged(u32) = .empty;

        // Find target entity's projection index
        var target_proj_idx: ?usize = null;
        for (self.projections.items, 0..) |proj, idx| {
            if (proj.entity_idx == entity_idx) {
                target_proj_idx = idx;
                break;
            }
        }

        if (target_proj_idx == null) return occluders;
        const j = target_proj_idx.?;

        // Check all entities in front of target
        for (0..j) |i| {
            const prob = self.occlusion_probs.items[i * self.n_entities + j];
            if (prob > 0) {
                try occluders.append(allocator, self.projections.items[i].entity_idx);
            }
        }

        return occluders;
    }

    /// Get depth-ordered entity indices (front to back)
    pub fn getDepthOrder(self: *const OcclusionGraph) []const u32 {
        return self.depth_order.items;
    }

    /// Get visible entities only
    pub fn getVisibleEntities(
        self: *const OcclusionGraph,
        allocator: Allocator,
    ) !std.ArrayListUnmanaged(u32) {
        var visible: std.ArrayListUnmanaged(u32) = .empty;

        for (self.projections.items) |proj| {
            if (proj.visible) {
                try visible.append(allocator, proj.entity_idx);
            }
        }

        return visible;
    }
};

/// Compute overlap ratio between two projected circles
fn computeOverlap(a: ProjectedEntity, b: ProjectedEntity) f32 {
    const dx = a.center.x - b.center.x;
    const dy = a.center.y - b.center.y;
    const dist = @sqrt(dx * dx + dy * dy);

    const r1 = a.radius;
    const r2 = b.radius;

    // No overlap if centers too far apart
    if (dist >= r1 + r2) return 0;

    // Full overlap if one contains the other
    if (dist + r2 <= r1) return 1.0;
    if (dist + r1 <= r2) return 1.0;

    // Partial overlap - compute circle-circle intersection area
    // Simplified: use ratio of distances
    const overlap_dist = r1 + r2 - dist;
    const max_overlap = @min(r1, r2) * 2;

    return @min(1.0, overlap_dist / max_overlap);
}

// =============================================================================
// Depth-Ordered Rendering
// =============================================================================

/// Pixel ownership buffer for data association
pub const PixelOwnership = struct {
    /// For each pixel, which entity index (null = background)
    owners: std.ArrayListUnmanaged(?u32),
    /// Depth at each pixel
    depths: std.ArrayListUnmanaged(f32),
    width: u32,
    height: u32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: u32, height: u32) !PixelOwnership {
        const size = width * height;
        var self = PixelOwnership{
            .owners = .empty,
            .depths = .empty,
            .width = width,
            .height = height,
            .allocator = allocator,
        };

        try self.owners.ensureTotalCapacity(allocator, size);
        try self.depths.ensureTotalCapacity(allocator, size);

        for (0..size) |_| {
            self.owners.appendAssumeCapacity(null);
            self.depths.appendAssumeCapacity(std.math.inf(f32));
        }

        return self;
    }

    pub fn deinit(self: *PixelOwnership) void {
        self.owners.deinit(self.allocator);
        self.depths.deinit(self.allocator);
    }

    /// Get pixel index from x, y
    pub fn pixelIndex(self: *const PixelOwnership, x: u32, y: u32) usize {
        return y * self.width + x;
    }

    /// Set pixel owner (if closer than current)
    pub fn setIfCloser(self: *PixelOwnership, x: u32, y: u32, entity_idx: u32, depth: f32) void {
        if (x >= self.width or y >= self.height) return;

        const idx = self.pixelIndex(x, y);
        if (depth < self.depths.items[idx]) {
            self.depths.items[idx] = depth;
            self.owners.items[idx] = entity_idx;
        }
    }

    /// Get owner at pixel
    pub fn getOwner(self: *const PixelOwnership, x: u32, y: u32) ?u32 {
        if (x >= self.width or y >= self.height) return null;
        return self.owners.items[self.pixelIndex(x, y)];
    }

    /// Count pixels owned by each entity
    pub fn countOwned(self: *const PixelOwnership, entity_idx: u32) usize {
        var count: usize = 0;
        for (self.owners.items) |owner| {
            if (owner == entity_idx) count += 1;
        }
        return count;
    }
};

/// Render entities to pixel ownership buffer (depth-ordered)
/// world can be any type with queryPhysicsEntities(), getPosition(), getAppearance()
pub fn renderPixelOwnership(
    world: anytype,
    camera: Camera,
    width: u32,
    height: u32,
    allocator: Allocator,
) !PixelOwnership {
    var ownership = try PixelOwnership.init(allocator, width, height);
    errdefer ownership.deinit();

    // Project and render each entity
    var query = world.queryPhysicsEntities();
    while (query.next()) |id| {
        const pos = world.getPosition(id) orelse continue;
        const app = world.getAppearance(id);
        const radius = if (app) |a| a.radius else 0.5;

        if (camera.project(pos.mean)) |proj| {
            const projected_radius = camera.projectRadius(radius, proj.depth);

            // Convert NDC to pixel coordinates
            const px = (proj.ndc.x + 1.0) * 0.5 * @as(f32, @floatFromInt(width));
            const py = (1.0 - proj.ndc.y) * 0.5 * @as(f32, @floatFromInt(height));
            const pr = projected_radius * @as(f32, @floatFromInt(width)) * 0.5;

            // Render disc (simple rasterization)
            const pr_int: i32 = @intFromFloat(@ceil(pr));
            const px_int: i32 = @intFromFloat(px);
            const py_int: i32 = @intFromFloat(py);

            var dy: i32 = -pr_int;
            while (dy <= pr_int) : (dy += 1) {
                var dx: i32 = -pr_int;
                while (dx <= pr_int) : (dx += 1) {
                    const dist_sq = dx * dx + dy * dy;
                    if (dist_sq <= pr_int * pr_int) {
                        const pixel_x = px_int + dx;
                        const pixel_y = py_int + dy;

                        if (pixel_x >= 0 and pixel_x < width and
                            pixel_y >= 0 and pixel_y < height)
                        {
                            ownership.setIfCloser(
                                @intCast(pixel_x),
                                @intCast(pixel_y),
                                id.index,
                                proj.depth,
                            );
                        }
                    }
                }
            }
        }
    }

    return ownership;
}

// =============================================================================
// Tests
// =============================================================================

const smc = @import("../smc/mod.zig");
const ParticleSwarm = smc.ParticleSwarm;
const particleWorld = smc.particleWorld;

test "OcclusionGraph basic" {
    const allocator = std.testing.allocator;

    var graph = OcclusionGraph.init(allocator);
    defer graph.deinit();

    // Test with empty world (use ParticleWorldView)
    var swarm = try ParticleSwarm.init(allocator, 1, 8);
    defer swarm.deinit();
    var world = particleWorld(&swarm, 0);

    const camera = Camera.default;
    try graph.compute(&world, camera, .{});

    try std.testing.expectEqual(@as(usize, 0), graph.n_entities);
}

test "OcclusionGraph with entities" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 1, 8);
    defer swarm.deinit();
    var world = particleWorld(&swarm, 0);

    // Spawn two entities at different depths
    _ = world.spawnPhysics(Vec3.init(0, 0, 0), Vec3.zero, .standard);
    _ = world.spawnPhysics(Vec3.init(0, 0, -2), Vec3.zero, .standard);

    var graph = OcclusionGraph.init(allocator);
    defer graph.deinit();

    const camera = Camera.default;
    try graph.compute(&world, camera, .{});

    try std.testing.expectEqual(@as(usize, 2), graph.n_entities);
}

test "PixelOwnership basic" {
    const allocator = std.testing.allocator;

    var ownership = try PixelOwnership.init(allocator, 10, 10);
    defer ownership.deinit();

    try std.testing.expectEqual(@as(?u32, null), ownership.getOwner(5, 5));

    ownership.setIfCloser(5, 5, 0, 1.0);
    try std.testing.expectEqual(@as(?u32, 0), ownership.getOwner(5, 5));

    // Closer entity should take ownership
    ownership.setIfCloser(5, 5, 1, 0.5);
    try std.testing.expectEqual(@as(?u32, 1), ownership.getOwner(5, 5));

    // Further entity should not take ownership
    ownership.setIfCloser(5, 5, 2, 2.0);
    try std.testing.expectEqual(@as(?u32, 1), ownership.getOwner(5, 5));
}

test "computeOverlap" {
    // No overlap
    const a = ProjectedEntity{
        .entity_idx = 0,
        .center = Vec2.init(0, 0),
        .depth = 1,
        .radius = 0.1,
        .visible = true,
    };
    const b = ProjectedEntity{
        .entity_idx = 1,
        .center = Vec2.init(1, 0),
        .depth = 2,
        .radius = 0.1,
        .visible = true,
    };

    const overlap = computeOverlap(a, b);
    try std.testing.expect(overlap == 0);

    // Full overlap
    const c = ProjectedEntity{
        .entity_idx = 2,
        .center = Vec2.init(0, 0),
        .depth = 1,
        .radius = 0.5,
        .visible = true,
    };
    const d = ProjectedEntity{
        .entity_idx = 3,
        .center = Vec2.init(0, 0),
        .depth = 2,
        .radius = 0.1,
        .visible = true,
    };

    const full_overlap = computeOverlap(c, d);
    try std.testing.expect(full_overlap == 1.0);
}
