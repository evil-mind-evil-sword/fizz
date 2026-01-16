//! ParticleWorldView: World-like facade over SoA ParticleSwarm
//!
//! Provides a World-like API for accessing a single particle's entity state.
//! This enables CRP and other modules to work with the "particle = world hypothesis"
//! mental model while preserving the performance benefits of SoA storage.
//!
//! Usage:
//!     var view = swarm.particleWorld(particle_idx);
//!     const entity = view.spawnPhysics(pos, vel, ptype) orelse return error.Full;
//!     view.setAppearance(entity, appearance);

const std = @import("std");
const swarm_mod = @import("swarm.zig");
const ParticleSwarm = swarm_mod.ParticleSwarm;
const EntityView = swarm_mod.EntityView;
const CovTriangle = swarm_mod.CovTriangle;
const covToTriangle = swarm_mod.covToTriangle;
const triangleToCov = swarm_mod.triangleToCov;

const math = @import("../math.zig");
const Vec3 = math.Vec3;
const Mat3 = math.Mat3;

// Use types.zig for shared type definitions
const types = @import("../types.zig");
const Label = types.Label;
const PhysicsType = types.PhysicsType;
const ContactMode = types.ContactMode;
const TrackState = types.TrackState;
const GoalType = types.GoalType;
const SpatialRelationType = types.SpatialRelationType;
pub const EntityId = types.EntityId;

// Component value types for get/set API
pub const Position = struct {
    mean: Vec3,
    cov: Mat3,
};

pub const Velocity = struct {
    mean: Vec3,
    cov: Mat3,
};

pub const Occlusion = struct {
    state: TrackState,
    duration: u32,
    existence_prob: f32,
};

pub const Appearance = struct {
    color: Vec3,
    opacity: f32,
    radius: f32,
};

/// World-like view into a single particle's entity state
/// Provides spawn/despawn/get/set API operating on underlying SoA storage
pub const ParticleWorldView = struct {
    swarm: *ParticleSwarm,
    particle: usize,

    const Self = @This();

    // =========================================================================
    // Indexing
    // =========================================================================

    /// Get flat SoA index for an entity in this particle
    inline fn idx(self: Self, entity: usize) usize {
        return self.particle * self.swarm.max_entities + entity;
    }

    // =========================================================================
    // Entity Lifecycle
    // =========================================================================

    /// Spawn a new physics entity (finds free slot, initializes components)
    /// Returns EntityId or null if all slots are full
    pub fn spawnPhysics(
        self: Self,
        pos: Vec3,
        vel: Vec3,
        ptype: PhysicsType,
    ) ?EntityId {
        // Find free slot
        for (0..self.swarm.max_entities) |e| {
            const i = self.idx(e);
            if (!self.swarm.alive[i]) {
                // Initialize entity state
                self.swarm.position_mean[i] = pos;
                self.swarm.position_cov[i] = covToTriangle(Mat3.diagonal(Vec3.splat(0.1)));
                self.swarm.velocity_mean[i] = vel;
                self.swarm.velocity_cov[i] = covToTriangle(Mat3.diagonal(Vec3.splat(0.1)));
                self.swarm.physics_type[i] = ptype;
                self.swarm.contact_mode[i] = .free;
                self.swarm.track_state[i] = .detected;
                self.swarm.label[i] = Label{ .birth_time = 0, .birth_index = 0 };
                self.swarm.occlusion_count[i] = 0;
                self.swarm.alive[i] = true;
                self.swarm.color[i] = Vec3.init(0.5, 0.5, 0.5);
                self.swarm.opacity[i] = 1.0;
                self.swarm.radius[i] = 0.5;
                self.swarm.goal_type[i] = .none;
                self.swarm.target_label[i] = null;
                self.swarm.target_position[i] = null;
                self.swarm.spatial_relation_type[i] = .none;
                self.swarm.spatial_reference[i] = null;
                self.swarm.spatial_distance[i] = 0;
                self.swarm.spatial_tolerance[i] = 1.0;

                self.swarm.entity_counts[self.particle] += 1;

                return EntityId{ .index = @intCast(e), .generation = 0 };
            }
        }
        return null; // All slots full
    }

    /// Remove entity from this particle's hypothesis
    pub fn despawn(self: Self, entity: EntityId) void {
        const i = self.idx(entity.index);
        if (self.swarm.alive[i]) {
            self.swarm.alive[i] = false;
            self.swarm.track_state[i] = .dead;
            if (self.swarm.entity_counts[self.particle] > 0) {
                self.swarm.entity_counts[self.particle] -= 1;
            }
        }
    }

    /// Check if entity is alive
    pub fn isAlive(self: Self, entity: EntityId) bool {
        return self.swarm.alive[self.idx(entity.index)];
    }

    // =========================================================================
    // Position Component
    // =========================================================================

    pub fn getPosition(self: Self, entity: EntityId) ?Position {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return Position{
            .mean = self.swarm.position_mean[i],
            .cov = triangleToCov(self.swarm.position_cov[i]),
        };
    }

    pub fn setPosition(self: Self, entity: EntityId, pos: Position) void {
        const i = self.idx(entity.index);
        self.swarm.position_mean[i] = pos.mean;
        self.swarm.position_cov[i] = covToTriangle(pos.cov);
    }

    // =========================================================================
    // Velocity Component
    // =========================================================================

    pub fn getVelocity(self: Self, entity: EntityId) ?Velocity {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return Velocity{
            .mean = self.swarm.velocity_mean[i],
            .cov = triangleToCov(self.swarm.velocity_cov[i]),
        };
    }

    pub fn setVelocity(self: Self, entity: EntityId, vel: Velocity) void {
        const i = self.idx(entity.index);
        self.swarm.velocity_mean[i] = vel.mean;
        self.swarm.velocity_cov[i] = covToTriangle(vel.cov);
    }

    // =========================================================================
    // Physics Type Component
    // =========================================================================

    pub fn getPhysicsType(self: Self, entity: EntityId) ?PhysicsType {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.physics_type[i];
    }

    pub fn setPhysicsType(self: Self, entity: EntityId, ptype: PhysicsType) void {
        const i = self.idx(entity.index);
        self.swarm.physics_type[i] = ptype;
    }

    // =========================================================================
    // Contact Mode Component
    // =========================================================================

    pub fn getContactMode(self: Self, entity: EntityId) ?ContactMode {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.contact_mode[i];
    }

    pub fn setContactMode(self: Self, entity: EntityId, mode: ContactMode) void {
        const i = self.idx(entity.index);
        self.swarm.contact_mode[i] = mode;
    }

    // =========================================================================
    // Track State Component
    // =========================================================================

    pub fn getTrackState(self: Self, entity: EntityId) ?TrackState {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.track_state[i];
    }

    pub fn setTrackState(self: Self, entity: EntityId, state: TrackState) void {
        const i = self.idx(entity.index);
        self.swarm.track_state[i] = state;
    }

    // =========================================================================
    // Label Component
    // =========================================================================

    pub fn getLabel(self: Self, entity: EntityId) ?Label {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.label[i];
    }

    pub fn setLabel(self: Self, entity: EntityId, label: Label) void {
        const i = self.idx(entity.index);
        self.swarm.label[i] = label;
    }

    // =========================================================================
    // Occlusion Component
    // =========================================================================

    pub fn getOcclusion(self: Self, entity: EntityId) ?Occlusion {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return Occlusion{
            .state = self.swarm.track_state[i],
            .duration = self.swarm.occlusion_count[i],
            .existence_prob = 1.0, // TODO: track this if needed
        };
    }

    pub fn setOcclusion(self: Self, entity: EntityId, occ: Occlusion) void {
        const i = self.idx(entity.index);
        self.swarm.track_state[i] = occ.state;
        self.swarm.occlusion_count[i] = occ.duration;
    }

    pub fn getOcclusionCount(self: Self, entity: EntityId) ?u32 {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.occlusion_count[i];
    }

    pub fn setOcclusionCount(self: Self, entity: EntityId, count: u32) void {
        const i = self.idx(entity.index);
        self.swarm.occlusion_count[i] = count;
    }

    // =========================================================================
    // Appearance Component
    // =========================================================================

    pub fn getAppearance(self: Self, entity: EntityId) ?Appearance {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return Appearance{
            .color = self.swarm.color[i],
            .opacity = self.swarm.opacity[i],
            .radius = self.swarm.radius[i],
        };
    }

    pub fn setAppearance(self: Self, entity: EntityId, app: Appearance) void {
        const i = self.idx(entity.index);
        self.swarm.color[i] = app.color;
        self.swarm.opacity[i] = app.opacity;
        self.swarm.radius[i] = app.radius;
    }

    // =========================================================================
    // Goal/Agency Component
    // =========================================================================

    pub fn getGoalType(self: Self, entity: EntityId) ?GoalType {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.goal_type[i];
    }

    pub fn setGoalType(self: Self, entity: EntityId, goal: GoalType) void {
        const i = self.idx(entity.index);
        self.swarm.goal_type[i] = goal;
    }

    pub fn getTargetLabel(self: Self, entity: EntityId) ?Label {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.target_label[i];
    }

    pub fn setTargetLabel(self: Self, entity: EntityId, target: ?Label) void {
        const i = self.idx(entity.index);
        self.swarm.target_label[i] = target;
    }

    pub fn getTargetPosition(self: Self, entity: EntityId) ?Vec3 {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.target_position[i];
    }

    pub fn setTargetPosition(self: Self, entity: EntityId, target: ?Vec3) void {
        const i = self.idx(entity.index);
        self.swarm.target_position[i] = target;
    }

    // =========================================================================
    // Spatial Relation Component
    // =========================================================================

    pub fn getSpatialRelationType(self: Self, entity: EntityId) ?SpatialRelationType {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.spatial_relation_type[i];
    }

    pub fn setSpatialRelationType(self: Self, entity: EntityId, rel: SpatialRelationType) void {
        const i = self.idx(entity.index);
        self.swarm.spatial_relation_type[i] = rel;
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Get count of alive entities in this particle
    pub fn entityCount(self: Self) u32 {
        return self.swarm.entity_counts[self.particle];
    }

    /// Iterate over alive entities in this particle
    pub fn aliveEntities(self: Self) EntityIterator {
        return EntityIterator{
            .view = self,
            .current = 0,
        };
    }

    /// Alias for aliveEntities (compatibility with ecs.World API)
    pub fn queryPhysicsEntities(self: Self) EntityIterator {
        return self.aliveEntities();
    }

    pub const EntityIterator = struct {
        view: ParticleWorldView,
        current: usize,

        pub fn next(self: *EntityIterator) ?EntityId {
            while (self.current < self.view.swarm.max_entities) {
                const e = self.current;
                self.current += 1;
                const i = self.view.idx(e);
                if (self.view.swarm.alive[i]) {
                    return EntityId{ .index = @intCast(e), .generation = 0 };
                }
            }
            return null;
        }

        pub fn reset(self: *EntityIterator) void {
            self.current = 0;
        }
    };

    // =========================================================================
    // Bulk Access (for compatibility with existing code)
    // =========================================================================

    /// Get full entity view (snapshot of all fields)
    pub fn getEntity(self: Self, entity: EntityId) ?EntityView {
        const i = self.idx(entity.index);
        if (!self.swarm.alive[i]) return null;
        return self.swarm.getEntity(self.particle, entity.index);
    }

    /// Set full entity view
    pub fn setEntity(self: Self, entity: EntityId, view: EntityView) void {
        self.swarm.setEntity(self.particle, entity.index, view);
    }
};

// =============================================================================
// ParticleSwarm extension
// =============================================================================

/// Add particleWorld() method to ParticleSwarm
pub fn particleWorld(swarm: *ParticleSwarm, particle: usize) ParticleWorldView {
    return ParticleWorldView{
        .swarm = swarm,
        .particle = particle,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "ParticleWorldView spawn and despawn" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 2, 8);
    defer swarm.deinit();

    var view = particleWorld(&swarm, 0);

    // Initially no entities
    try std.testing.expectEqual(@as(u32, 0), view.entityCount());

    // Spawn entity
    const e1 = view.spawnPhysics(Vec3.init(1, 2, 3), Vec3.zero, .standard);
    try std.testing.expect(e1 != null);
    try std.testing.expectEqual(@as(u32, 1), view.entityCount());

    // Check position
    const pos = view.getPosition(e1.?);
    try std.testing.expect(pos != null);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pos.?.mean.x, 0.001);

    // Spawn another
    const e2 = view.spawnPhysics(Vec3.init(4, 5, 6), Vec3.zero, .bouncy);
    try std.testing.expect(e2 != null);
    try std.testing.expectEqual(@as(u32, 2), view.entityCount());

    // Despawn first
    view.despawn(e1.?);
    try std.testing.expectEqual(@as(u32, 1), view.entityCount());
    try std.testing.expect(!view.isAlive(e1.?));
    try std.testing.expect(view.isAlive(e2.?));
}

test "ParticleWorldView component get/set" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 1, 4);
    defer swarm.deinit();

    var view = particleWorld(&swarm, 0);
    const entity = view.spawnPhysics(Vec3.zero, Vec3.zero, .standard).?;

    // Test appearance
    var app = view.getAppearance(entity).?;
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), app.color.x, 0.001);

    app.color = Vec3.init(1.0, 0.0, 0.0);
    app.radius = 1.5;
    view.setAppearance(entity, app);

    const app2 = view.getAppearance(entity).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), app2.color.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), app2.radius, 0.001);

    // Test label
    view.setLabel(entity, Label{ .birth_time = 42, .birth_index = 7 });
    const label = view.getLabel(entity).?;
    try std.testing.expectEqual(@as(u32, 42), label.birth_time);
    try std.testing.expectEqual(@as(u16, 7), label.birth_index);
}

test "ParticleWorldView iteration" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 1, 8);
    defer swarm.deinit();

    var view = particleWorld(&swarm, 0);

    // Spawn 3 entities
    _ = view.spawnPhysics(Vec3.init(1, 0, 0), Vec3.zero, .standard);
    const e2 = view.spawnPhysics(Vec3.init(2, 0, 0), Vec3.zero, .standard).?;
    _ = view.spawnPhysics(Vec3.init(3, 0, 0), Vec3.zero, .standard);

    // Kill middle one
    view.despawn(e2);

    // Count alive via iteration
    var count: u32 = 0;
    var iter = view.aliveEntities();
    while (iter.next()) |_| {
        count += 1;
    }

    try std.testing.expectEqual(@as(u32, 2), count);
    try std.testing.expectEqual(@as(u32, 2), view.entityCount());
}

test "ParticleWorldView particles are independent" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 3, 4);
    defer swarm.deinit();

    var v0 = particleWorld(&swarm, 0);
    var v1 = particleWorld(&swarm, 1);
    var v2 = particleWorld(&swarm, 2);

    // Spawn different counts per particle
    _ = v0.spawnPhysics(Vec3.zero, Vec3.zero, .standard);

    _ = v1.spawnPhysics(Vec3.zero, Vec3.zero, .standard);
    _ = v1.spawnPhysics(Vec3.zero, Vec3.zero, .standard);

    _ = v2.spawnPhysics(Vec3.zero, Vec3.zero, .standard);
    _ = v2.spawnPhysics(Vec3.zero, Vec3.zero, .standard);
    _ = v2.spawnPhysics(Vec3.zero, Vec3.zero, .standard);

    // Verify independent counts
    try std.testing.expectEqual(@as(u32, 1), v0.entityCount());
    try std.testing.expectEqual(@as(u32, 2), v1.entityCount());
    try std.testing.expectEqual(@as(u32, 3), v2.entityCount());
}
