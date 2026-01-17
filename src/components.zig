//! Spelke Core Knowledge Components for Probabilistic Physics Inference
//!
//! These components encode intuitive physics priors based on developmental psychology:
//! - Objects exist in space (Position)
//! - Objects move continuously (Velocity)
//! - Objects don't pass through each other (ContactMode)
//! - Objects persist through occlusion (TrackState, Occlusion)
//! - Objects maintain identity (Label)
//! - Some objects are agents (Agency)

const std = @import("std");
const math = @import("math.zig");
const Vec3 = math.Vec3;
const Mat3 = math.Mat3;

// =============================================================================
// Entity Identity
// =============================================================================

/// Entity identifier within a particle's hypothesis
/// In ParticleSwarm context, this is an index into the particle's entity slots
pub const EntityId = struct {
    index: u32,
    generation: u16 = 0,

    pub const invalid = EntityId{ .index = std.math.maxInt(u32), .generation = 0 };

    pub fn eql(self: EntityId, other: EntityId) bool {
        return self.index == other.index and self.generation == other.generation;
    }

    pub fn hash(self: EntityId) u64 {
        return @as(u64, self.index) << 16 | @as(u64, self.generation);
    }

    pub fn format(
        self: EntityId,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("Entity({d}:{d})", .{ self.index, self.generation });
    }
};

/// Label for entity identity (GLMB-style tracking)
/// Core knowledge: Objects maintain identity across time
pub const Label = struct {
    birth_time: u32,
    birth_index: u16,

    pub fn eql(self: Label, other: Label) bool {
        return self.birth_time == other.birth_time and self.birth_index == other.birth_index;
    }

    pub fn hash(self: Label) u64 {
        return @as(u64, self.birth_time) << 16 | @as(u64, self.birth_index);
    }
};

/// EntityRef - stable reference to another entity via Label
/// Used for cross-entity relationships (goal targets, spatial references)
/// Survives resampling since it uses Label identity, not array index
pub const EntityRef = struct {
    label: Label,

    pub fn init(label: Label) EntityRef {
        return .{ .label = label };
    }

    pub fn eql(self: EntityRef, other: EntityRef) bool {
        return self.label.eql(other.label);
    }
};

// =============================================================================
// Continuous State Components
// =============================================================================

/// Position component - Gaussian belief over 3D position
/// Core knowledge: Objects exist in space
pub const Position = struct {
    mean: Vec3,
    cov: Mat3,

    pub fn point(pos: Vec3) Position {
        return .{
            .mean = pos,
            .cov = Mat3.diagonal(Vec3.splat(0.001)),
        };
    }

    pub fn isotropic(pos: Vec3, variance: f32) Position {
        return .{
            .mean = pos,
            .cov = Mat3.diagonal(Vec3.splat(variance)),
        };
    }
};

/// Velocity component - Gaussian belief over 3D velocity
/// Core knowledge: Objects move continuously (no teleportation)
pub const Velocity = struct {
    mean: Vec3,
    cov: Mat3,

    pub fn zero() Velocity {
        return .{
            .mean = Vec3.zero,
            .cov = Mat3.diagonal(Vec3.splat(0.001)),
        };
    }

    pub fn point(vel: Vec3) Velocity {
        return .{
            .mean = vel,
            .cov = Mat3.diagonal(Vec3.splat(0.001)),
        };
    }
};

// =============================================================================
// Physics Parameters
// =============================================================================

/// Physics parameters - continuous values for dynamics computation
pub const PhysicsParams = struct {
    friction: f32,
    elasticity: f32,
    process_noise: f32,
    mass: f32,

    pub const standard = PhysicsParams{
        .friction = 0.3,
        .elasticity = 0.5,
        .process_noise = 0.01,
        .mass = 1.0,
    };

    pub const bouncy = PhysicsParams{
        .friction = 0.2,
        .elasticity = 0.9,
        .process_noise = 0.02,
        .mass = 1.0,
    };

    pub const sticky = PhysicsParams{
        .friction = 0.8,
        .elasticity = 0.1,
        .process_noise = 0.005,
        .mass = 1.0,
    };

    pub const slippery = PhysicsParams{
        .friction = 0.05,
        .elasticity = 0.6,
        .process_noise = 0.015,
        .mass = 1.0,
    };
};

// Legacy alias for compatibility
pub const Physics = PhysicsParams;

// =============================================================================
// Discrete Mode Components
// =============================================================================

/// Contact mode - discrete state for SLDS
/// Core knowledge: Solidity (objects don't pass through each other)
pub const ContactMode = enum(u8) {
    /// Free flight under gravity
    free,
    /// Resting on ground plane
    ground,
    /// Resting on another entity (support relationship)
    supported,
    /// Attached to surface (sticky behavior)
    attached,
    /// Self-propelled motion (agency)
    agency,
};

/// Track state - for object permanence
/// Core knowledge: Objects persist through occlusion
pub const TrackState = enum(u8) {
    /// Matched to observation this timestep
    detected,
    /// Not matched but predicted to exist (occluded)
    occluded,
    /// New track, needs confirmation (recently birthed)
    tentative,
    /// Marked for removal
    dead,
};

// =============================================================================
// Composite Components
// =============================================================================

/// Contact state - tracks mode and duration
pub const Contact = struct {
    mode: ContactMode,
    normal: Vec3,
    duration: u32,

    pub fn free() Contact {
        return .{
            .mode = .free,
            .normal = Vec3.unit_y,
            .duration = 0,
        };
    }
};

/// Support relationship
/// Core knowledge: Objects at rest tend to stay at rest (support stability)
pub const Support = struct {
    supporter_index: ?u32,
    contact_normal: Vec3,
    attachment_strength: f32,

    pub fn ground(normal: Vec3) Support {
        return .{
            .supporter_index = null,
            .contact_normal = normal,
            .attachment_strength = 0,
        };
    }

    pub fn onEntity(supporter_index: u32, normal: Vec3) Support {
        return .{
            .supporter_index = supporter_index,
            .contact_normal = normal,
            .attachment_strength = 0,
        };
    }
};

/// Occlusion tracking
pub const Occlusion = struct {
    state: TrackState,
    duration: u32,
    existence_prob: f32,

    pub fn detected() Occlusion {
        return .{
            .state = .detected,
            .duration = 0,
            .existence_prob = 1.0,
        };
    }
};

/// Appearance for rendering
pub const Appearance = struct {
    color: Vec3,
    opacity: f32,
    radius: f32,

    pub const default = Appearance{
        .color = Vec3.init(0.5, 0.5, 0.5),
        .opacity = 1.0,
        .radius = 0.5,
    };
};

// =============================================================================
// Agency Components
// =============================================================================

/// Goal type - discrete variable for agent behavior
/// Core knowledge: Agents have goals that persist and guide behavior
pub const GoalType = enum(u8) {
    none,
    reach,
    track,
    avoid,
    acquire,

    pub fn requiresTarget(self: GoalType) bool {
        return switch (self) {
            .none, .reach => false,
            .track, .avoid, .acquire => true,
        };
    }

    pub fn requiresPosition(self: GoalType) bool {
        return self == .reach;
    }
};

/// Agency component - marks entity as self-propelled
/// Core knowledge: Some objects are agents (different dynamics expectations)
pub const Agency = struct {
    goal_type: GoalType,
    target_entity: ?EntityRef,
    target_position: ?Vec3,
    max_accel: f32,
    noise_scale: f32,

    pub fn passive() Agency {
        return .{
            .goal_type = .none,
            .target_entity = null,
            .target_position = null,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    pub fn reaching(target: Vec3) Agency {
        return .{
            .goal_type = .reach,
            .target_entity = null,
            .target_position = target,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    pub fn tracking(target: EntityRef) Agency {
        return .{
            .goal_type = .track,
            .target_entity = target,
            .target_position = null,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    pub fn avoiding(target: EntityRef) Agency {
        return .{
            .goal_type = .avoid,
            .target_entity = target,
            .target_position = null,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    pub fn basic() Agency {
        return passive();
    }
};

// =============================================================================
// Spatial Relations
// =============================================================================

/// Spatial relation type - discrete variable for geometric relationships
/// Core knowledge: Space/geometry (containment, support, relative positions)
pub const SpatialRelationType = enum(u8) {
    none,
    inside,
    on,
    near,
    left_of,
    right_of,
    above,
    below,

    pub fn requiresReference(self: SpatialRelationType) bool {
        return self != .none;
    }

    pub fn isViewpointDependent(self: SpatialRelationType) bool {
        return self == .left_of or self == .right_of;
    }
};

/// Spatial relation component
pub const SpatialRelation = struct {
    relation_type: SpatialRelationType,
    reference_entity: ?Label,
    expected_distance: f32,
    tolerance: f32,

    pub fn none() SpatialRelation {
        return .{
            .relation_type = .none,
            .reference_entity = null,
            .expected_distance = 0,
            .tolerance = 1.0,
        };
    }

    pub fn inside(container: Label) SpatialRelation {
        return .{
            .relation_type = .inside,
            .reference_entity = container,
            .expected_distance = 0,
            .tolerance = 1.0,
        };
    }

    pub fn on(supporter: Label) SpatialRelation {
        return .{
            .relation_type = .on,
            .reference_entity = supporter,
            .expected_distance = 0,
            .tolerance = 0.5,
        };
    }

    pub fn near(reference: Label, distance: f32) SpatialRelation {
        return .{
            .relation_type = .near,
            .reference_entity = reference,
            .expected_distance = distance,
            .tolerance = distance * 0.3,
        };
    }

    pub fn relative(relation: SpatialRelationType, reference: Label) SpatialRelation {
        return .{
            .relation_type = relation,
            .reference_entity = reference,
            .expected_distance = 0,
            .tolerance = 1.0,
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Position component" {
    const pos = Position.point(Vec3.init(1, 2, 3));
    try std.testing.expect(pos.mean.x == 1);
    try std.testing.expect(pos.cov.get(0, 0) == 0.001);
}

test "PhysicsParams presets" {
    try std.testing.expect(PhysicsParams.bouncy.elasticity > PhysicsParams.sticky.elasticity);
    try std.testing.expect(PhysicsParams.sticky.friction > PhysicsParams.slippery.friction);
}

test "ContactMode values" {
    const mode: ContactMode = .free;
    try std.testing.expect(mode == .free);
}

test "Label equality" {
    const l1 = Label{ .birth_time = 1, .birth_index = 0 };
    const l2 = Label{ .birth_time = 1, .birth_index = 0 };
    const l3 = Label{ .birth_time = 2, .birth_index = 0 };
    try std.testing.expect(l1.eql(l2));
    try std.testing.expect(!l1.eql(l3));
}
