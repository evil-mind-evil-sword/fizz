const std = @import("std");
const math = @import("../math.zig");
const Vec3 = math.Vec3;
const Mat3 = math.Mat3;

// =============================================================================
// Built-in Components for Spelke Core Knowledge Physics
// =============================================================================

/// Position component - Gaussian belief over 3D position
/// Core knowledge: Objects exist in space
pub const Position = struct {
    mean: Vec3,
    cov: Mat3,

    pub const id: u16 = 0;

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

    pub const id: u16 = 1;

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

/// Physics parameters component - controls dynamics behavior
/// Replaces old PhysicsType enum with continuous parameters
pub const Physics = struct {
    friction: f32,
    elasticity: f32,
    process_noise: f32,
    mass: f32,

    pub const id: u16 = 2;

    // Preset configurations (replacing PhysicsType enum)
    pub const standard = Physics{
        .friction = 0.3,
        .elasticity = 0.5,
        .process_noise = 0.01,
        .mass = 1.0,
    };

    pub const bouncy = Physics{
        .friction = 0.2,
        .elasticity = 0.9,
        .process_noise = 0.02,
        .mass = 1.0,
    };

    pub const sticky = Physics{
        .friction = 0.8,
        .elasticity = 0.1,
        .process_noise = 0.005,
        .mass = 1.0,
    };

    pub const slippery = Physics{
        .friction = 0.05,
        .elasticity = 0.6,
        .process_noise = 0.015,
        .mass = 1.0,
    };
};

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

    pub const id: u16 = 3;
};

/// Contact state component - tracks mode and duration
pub const Contact = struct {
    mode: ContactMode,
    /// Surface normal at contact point
    normal: Vec3,
    /// Timesteps in current mode (for stability prior)
    duration: u32,

    pub const id: u16 = 4;

    pub fn free() Contact {
        return .{
            .mode = .free,
            .normal = Vec3.unit_y,
            .duration = 0,
        };
    }
};

/// Support relationship component
/// Core knowledge: Objects at rest tend to stay at rest (support stability)
pub const Support = struct {
    /// Entity providing support (null = ground/self-supported)
    supporter_index: ?u32,
    /// Contact normal pointing toward supported entity
    contact_normal: Vec3,
    /// 0 = just resting, 1 = fully attached
    attachment_strength: f32,

    pub const id: u16 = 5;

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

/// Track state component - for object permanence
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

    pub const id: u16 = 6;
};

/// Occlusion tracking component
pub const Occlusion = struct {
    /// Current track state
    state: TrackState,
    /// Timesteps since last observation
    duration: u32,
    /// Prior belief in continued existence
    existence_prob: f32,

    pub const id: u16 = 7;

    pub fn detected() Occlusion {
        return .{
            .state = .detected,
            .duration = 0,
            .existence_prob = 1.0,
        };
    }
};

/// Appearance component for rendering
pub const Appearance = struct {
    color: Vec3,
    opacity: f32,
    radius: f32,

    pub const id: u16 = 8;

    pub const default = Appearance{
        .color = Vec3.init(0.5, 0.5, 0.5),
        .opacity = 1.0,
        .radius = 0.5,
    };
};

/// Label component for entity identity (GLMB-style)
/// Core knowledge: Objects maintain identity
pub const Label = struct {
    birth_time: u32,
    birth_index: u16,

    pub const id: u16 = 9;

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
    /// Label of referenced entity (stable identity)
    label: Label,

    pub fn init(label: Label) EntityRef {
        return .{ .label = label };
    }

    pub fn eql(self: EntityRef, other: EntityRef) bool {
        return self.label.eql(other.label);
    }
};

/// Goal type - discrete variable for agent behavior
/// Core knowledge: Agents have goals that persist and guide behavior
pub const GoalType = enum(u8) {
    /// No active goal (object-like behavior)
    none,
    /// Move toward target position
    reach,
    /// Follow/pursue target entity
    track,
    /// Move away from target entity
    avoid,
    /// Approach and make contact with target
    acquire,

    /// Returns true if this goal type requires a target entity
    pub fn requiresTarget(self: GoalType) bool {
        return switch (self) {
            .none, .reach => false,
            .track, .avoid, .acquire => true,
        };
    }

    /// Returns true if this goal type requires a target position
    pub fn requiresPosition(self: GoalType) bool {
        return self == .reach;
    }
};

/// Agency component - marks entity as self-propelled
/// Core knowledge: Some objects are agents (different dynamics expectations)
pub const Agency = struct {
    /// Type of goal the agent is pursuing (discrete, Gibbs-sampled)
    goal_type: GoalType,
    /// Target entity for track/avoid/acquire goals
    target_entity: ?EntityRef,
    /// Target position for reach goals
    target_position: ?Vec3,
    /// Maximum acceleration capability
    max_accel: f32,
    /// Process noise multiplier (agents are less predictable)
    noise_scale: f32,

    pub const id: u16 = 10;

    /// Create a passive agent (no active goal)
    pub fn passive() Agency {
        return .{
            .goal_type = .none,
            .target_entity = null,
            .target_position = null,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    /// Create an agent reaching toward a position
    pub fn reaching(target: Vec3) Agency {
        return .{
            .goal_type = .reach,
            .target_entity = null,
            .target_position = target,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    /// Create an agent tracking another entity
    pub fn tracking(target: EntityRef) Agency {
        return .{
            .goal_type = .track,
            .target_entity = target,
            .target_position = null,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    /// Create an agent avoiding another entity
    pub fn avoiding(target: EntityRef) Agency {
        return .{
            .goal_type = .avoid,
            .target_entity = target,
            .target_position = null,
            .max_accel = 5.0,
            .noise_scale = 3.0,
        };
    }

    /// Legacy compatibility - same as passive()
    pub fn basic() Agency {
        return passive();
    }
};

/// Spatial relation type - discrete variable for geometric relationships
/// Core knowledge: Space/geometry (containment, support, relative positions)
pub const SpatialRelationType = enum(u8) {
    /// No spatial relation specified
    none,
    /// Inside a container (containment)
    inside,
    /// On top of / supported by
    on,
    /// Near another entity (proximity)
    near,
    /// Left of reference entity (requires known viewpoint)
    left_of,
    /// Right of reference entity (requires known viewpoint)
    right_of,
    /// Above reference entity
    above,
    /// Below reference entity
    below,

    /// Returns true if this relation requires a reference entity
    pub fn requiresReference(self: SpatialRelationType) bool {
        return self != .none;
    }

    /// Returns true if this relation is viewpoint-dependent
    pub fn isViewpointDependent(self: SpatialRelationType) bool {
        return self == .left_of or self == .right_of;
    }
};

/// Spatial relation component - encodes geometric relationships between entities
/// Core knowledge: Objects have spatial relationships that constrain inference
pub const SpatialRelation = struct {
    /// Type of spatial relation
    relation_type: SpatialRelationType,
    /// Reference entity for the relation (via Label for stability)
    reference_entity: ?Label,
    /// Expected distance for proximity relations (near)
    expected_distance: f32,
    /// Tolerance for relation satisfaction (soft constraint)
    tolerance: f32,

    pub const id: u16 = 11;

    /// Create an empty (no relation) spatial relation
    pub fn none() SpatialRelation {
        return .{
            .relation_type = .none,
            .reference_entity = null,
            .expected_distance = 0,
            .tolerance = 1.0,
        };
    }

    /// Create a containment relation (inside)
    pub fn inside(container: Label) SpatialRelation {
        return .{
            .relation_type = .inside,
            .reference_entity = container,
            .expected_distance = 0,
            .tolerance = 1.0,
        };
    }

    /// Create a support relation (on top of)
    pub fn on(supporter: Label) SpatialRelation {
        return .{
            .relation_type = .on,
            .reference_entity = supporter,
            .expected_distance = 0,
            .tolerance = 0.5,
        };
    }

    /// Create a proximity relation (near)
    pub fn near(reference: Label, distance: f32) SpatialRelation {
        return .{
            .relation_type = .near,
            .reference_entity = reference,
            .expected_distance = distance,
            .tolerance = distance * 0.3, // 30% tolerance by default
        };
    }

    /// Create a relative position relation
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
// Component Bundles (common entity configurations)
// =============================================================================

/// Bundle for spawning a basic physics entity
pub const PhysicsBundle = struct {
    position: Position,
    velocity: Velocity,
    physics: Physics,
    contact: Contact,
    occlusion: Occlusion,
    appearance: Appearance,
    label: Label,
};

/// Bundle for spawning an agent entity
pub const AgentBundle = struct {
    position: Position,
    velocity: Velocity,
    physics: Physics,
    contact: Contact,
    occlusion: Occlusion,
    appearance: Appearance,
    label: Label,
    agency: Agency,
};

// =============================================================================
// Tests
// =============================================================================

test "Position component" {
    const pos = Position.point(Vec3.init(1, 2, 3));
    try std.testing.expect(pos.mean.x == 1);
    try std.testing.expect(pos.cov.get(0, 0) == 0.001);
}

test "Physics presets" {
    try std.testing.expect(Physics.bouncy.elasticity > Physics.sticky.elasticity);
    try std.testing.expect(Physics.sticky.friction > Physics.slippery.friction);
}

test "ContactMode transitions" {
    const mode: ContactMode = .free;
    try std.testing.expect(mode == .free);
}
