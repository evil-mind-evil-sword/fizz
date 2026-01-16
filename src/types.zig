const std = @import("std");
const math = @import("math.zig");
const Vec3 = math.Vec3;
const Mat3 = math.Mat3;

// =============================================================================
// Entity Identification
// =============================================================================

/// Entity identifier - compatible with both ecs.World and ParticleWorldView
/// This is the canonical definition used throughout the codebase
pub const EntityId = struct {
    index: u32,
    generation: u16 = 0,

    pub const invalid = EntityId{ .index = std.math.maxInt(u32), .generation = 0 };

    pub fn eql(self: EntityId, other: EntityId) bool {
        return self.index == other.index and self.generation == other.generation;
    }
};

// =============================================================================
// Entity Labels and Identification
// =============================================================================

/// Unique label for entity identity (persists through occlusion)
/// Format: (birth_time, birth_index) as per GLMB formulation
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

/// Entity tracking state
pub const TrackState = enum(u8) {
    /// Matched to observation this timestep
    detected,
    /// Not matched but predicted to exist (occlusion)
    occluded,
    /// Tentative - newly born, needs confirmation
    tentative,
    /// Marked for removal
    dead,
};

// =============================================================================
// Physics Types (Discrete, Enumerable)
// =============================================================================

/// Discrete physics type for each entity
/// Small enumerable set enables Gibbs enumeration
pub const PhysicsType = enum(u8) {
    /// No special properties, standard dynamics
    standard,
    /// High elasticity (bouncy)
    bouncy,
    /// High friction, low elasticity (sticky)
    sticky,
    /// Low friction (sliding/icy)
    slippery,

    /// Get friction coefficient for this type
    pub fn friction(self: PhysicsType) f32 {
        return switch (self) {
            .standard => 0.3,
            .bouncy => 0.2,
            .sticky => 0.8,
            .slippery => 0.05,
        };
    }

    /// Get elasticity (coefficient of restitution) for this type
    pub fn elasticity(self: PhysicsType) f32 {
        return switch (self) {
            .standard => 0.5,
            .bouncy => 0.9,
            .sticky => 0.1,
            .slippery => 0.6,
        };
    }

    /// Get process noise scale for this type
    pub fn processNoise(self: PhysicsType) f32 {
        return switch (self) {
            .standard => 0.01,
            .bouncy => 0.02,
            .sticky => 0.005,
            .slippery => 0.015,
        };
    }
};

/// Contact mode between entities or with environment
pub const ContactMode = enum(u8) {
    /// No contact, free flight
    free,
    /// Contact with ground plane
    ground,
    /// Contact with wall (environment)
    wall,
    /// Contact with another entity
    entity,
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

// =============================================================================
// Gaussian State Representation (for Rao-Blackwellization)
// =============================================================================

/// Gaussian distribution over Vec3 (mean + covariance)
pub const GaussianVec3 = struct {
    mean: Vec3,
    cov: Mat3,

    /// Create with isotropic covariance
    pub fn isotropic(mean: Vec3, variance: f32) GaussianVec3 {
        return .{
            .mean = mean,
            .cov = Mat3.diagonal(Vec3.splat(variance)),
        };
    }

    /// Create with diagonal covariance
    pub fn diagonal(mean: Vec3, variances: Vec3) GaussianVec3 {
        return .{
            .mean = mean,
            .cov = Mat3.diagonal(variances),
        };
    }

    /// Evaluate log probability density at point x
    pub fn logPdf(self: GaussianVec3, x: Vec3) f32 {
        const diff = x.sub(self.mean);

        // For simplicity, assume diagonal covariance
        // Full implementation would need Cholesky decomposition
        const inv_cov = self.cov.inverse() orelse return -std.math.inf(f32);
        const mahalanobis = diff.dot(inv_cov.mulVec(diff));

        const det = self.cov.determinant();
        if (det <= 0) return -std.math.inf(f32);

        const log_det = @log(det);
        const log_2pi: f32 = 1.8378770664093453; // log(2*pi)

        return -0.5 * (3.0 * log_2pi + log_det + mahalanobis);
    }

    /// Sample from this distribution (requires RNG)
    pub fn sample(self: GaussianVec3, rng: std.Random) Vec3 {
        // Sample standard normal
        const z = Vec3.init(
            sampleStdNormal(rng),
            sampleStdNormal(rng),
            sampleStdNormal(rng),
        );

        // For diagonal covariance, multiply by sqrt of variances
        // Full implementation would use Cholesky decomposition
        const std_dev = Vec3.init(
            @sqrt(self.cov.get(0, 0)),
            @sqrt(self.cov.get(1, 1)),
            @sqrt(self.cov.get(2, 2)),
        );

        return self.mean.add(z.mul(std_dev));
    }
};

/// Sample from standard normal distribution using Box-Muller
fn sampleStdNormal(rng: std.Random) f32 {
    const r1 = rng.float(f32);
    const r2 = rng.float(f32);
    return @sqrt(-2.0 * @log(r1 + 1e-10)) * @cos(2.0 * std.math.pi * r2);
}

// =============================================================================
// Entity State
// =============================================================================

/// Appearance parameters for rendering (GMM component)
pub const Appearance = struct {
    /// Color (RGB, 0-1 range)
    color: Vec3,
    /// Opacity (0-1)
    opacity: f32,
    /// Size/radius of Gaussian blob
    radius: f32,

    pub const default = Appearance{
        .color = Vec3.init(0.5, 0.5, 0.5),
        .opacity = 1.0,
        .radius = 0.5,
    };
};

/// Full entity state
pub const Entity = struct {
    /// Permanent identity label
    label: Label,

    /// Position (as Gaussian for RBPF, or point estimate)
    position: GaussianVec3,

    /// Velocity (as Gaussian for RBPF, or point estimate)
    velocity: GaussianVec3,

    /// Discrete physics type
    physics_type: PhysicsType,

    /// Current contact mode
    contact_mode: ContactMode,

    /// Track management state
    track_state: TrackState,

    /// Timesteps since last observation (for occlusion tracking)
    occlusion_count: u32,

    /// Rendering appearance
    appearance: Appearance,

    /// Create entity with point estimates (converts to Gaussian with small variance)
    pub fn initPoint(
        label: Label,
        position: Vec3,
        velocity: Vec3,
        physics_type: PhysicsType,
    ) Entity {
        const init_variance: f32 = 0.001;
        return .{
            .label = label,
            .position = GaussianVec3.isotropic(position, init_variance),
            .velocity = GaussianVec3.isotropic(velocity, init_variance),
            .physics_type = physics_type,
            .contact_mode = .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = Appearance.default,
        };
    }

    /// Get point estimate of position (mean)
    pub fn positionMean(self: Entity) Vec3 {
        return self.position.mean;
    }

    /// Get point estimate of velocity (mean)
    pub fn velocityMean(self: Entity) Vec3 {
        return self.velocity.mean;
    }

    /// Check if entity is alive (not dead)
    pub fn isAlive(self: Entity) bool {
        return self.track_state != .dead;
    }
};

// =============================================================================
// World Configuration
// =============================================================================

/// Global physics parameters
pub const PhysicsConfig = struct {
    /// Gravity vector (typically (0, -9.81, 0) or (0, 0, -9.81))
    gravity: Vec3 = Vec3.init(0, -9.81, 0),

    /// Simulation timestep
    dt: f32 = 1.0 / 60.0,

    /// Ground plane height (y-coordinate)
    ground_height: f32 = 0.0,

    /// World bounds (AABB min/max)
    bounds_min: Vec3 = Vec3.init(-10, -10, -10),
    bounds_max: Vec3 = Vec3.init(10, 10, 10),

    /// Default process noise for dynamics
    process_noise: f32 = 0.01,

    /// CRP concentration parameter (birth rate)
    crp_alpha: f32 = 1.0,

    /// Survival probability per timestep
    survival_prob: f32 = 0.99,
};

/// Result of projecting a 3D point to screen space
pub const ProjectionResult = struct {
    /// Normalized device coordinates [-1, 1]
    ndc: math.Vec2,
    /// Depth (distance along view axis)
    depth: f32,
};

// =============================================================================
// Camera Pose (for FastSLAM - sampled per particle)
// =============================================================================

/// Camera pose state (sampled in particles)
/// Gravity constrains pitch/roll, so only position + yaw are free
pub const CameraPose = struct {
    /// Camera position in world coordinates
    position: Vec3,
    /// Yaw angle (rotation around Y/up axis, in radians)
    /// 0 = looking along -Z, positive = counter-clockwise from above
    yaw: f32,

    pub const default = CameraPose{
        .position = Vec3.init(0, 5, 10),
        .yaw = 0,
    };

    /// Create camera pose from position and yaw
    pub fn init(position: Vec3, yaw: f32) CameraPose {
        return .{ .position = position, .yaw = yaw };
    }

    /// Get forward direction (where camera looks)
    pub fn forward(self: CameraPose) Vec3 {
        // Yaw rotates around Y axis
        // At yaw=0, looking along -Z
        const cos_yaw = @cos(self.yaw);
        const sin_yaw = @sin(self.yaw);
        return Vec3.init(-sin_yaw, 0, -cos_yaw);
    }

    /// Get right direction
    pub fn right(self: CameraPose) Vec3 {
        const cos_yaw = @cos(self.yaw);
        const sin_yaw = @sin(self.yaw);
        return Vec3.init(cos_yaw, 0, -sin_yaw);
    }

    /// Get up direction (always world up due to gravity constraint)
    pub fn up(self: CameraPose) Vec3 {
        _ = self;
        return Vec3.unit_y;
    }

    /// Convert to full Camera with given intrinsics
    pub fn toCamera(self: CameraPose, intrinsics: CameraIntrinsics) Camera {
        const fwd = self.forward();
        return .{
            .position = self.position,
            .target = self.position.add(fwd),
            .up = Vec3.unit_y,
            .fov = intrinsics.fov,
            .aspect = intrinsics.aspect,
            .near = intrinsics.near,
            .far = intrinsics.far,
        };
    }

    /// Sample from prior (uniform in bounds)
    pub fn samplePrior(
        position_min: Vec3,
        position_max: Vec3,
        yaw_min: f32,
        yaw_max: f32,
        rng: std.Random,
    ) CameraPose {
        return .{
            .position = Vec3.init(
                position_min.x + rng.float(f32) * (position_max.x - position_min.x),
                position_min.y + rng.float(f32) * (position_max.y - position_min.y),
                position_min.z + rng.float(f32) * (position_max.z - position_min.z),
            ),
            .yaw = yaw_min + rng.float(f32) * (yaw_max - yaw_min),
        };
    }

    /// Apply random walk dynamics
    pub fn step(self: CameraPose, position_noise: f32, yaw_noise: f32, rng: std.Random) CameraPose {
        return .{
            .position = Vec3.init(
                self.position.x + sampleStdNormal(rng) * position_noise,
                self.position.y + sampleStdNormal(rng) * position_noise,
                self.position.z + sampleStdNormal(rng) * position_noise,
            ),
            .yaw = self.yaw + sampleStdNormal(rng) * yaw_noise,
        };
    }
};

/// Camera intrinsic parameters (fixed, not inferred)
pub const CameraIntrinsics = struct {
    /// Field of view (radians)
    fov: f32 = std.math.pi / 4.0,
    /// Aspect ratio (width/height)
    aspect: f32 = 1.0,
    /// Near clip plane
    near: f32 = 0.1,
    /// Far clip plane
    far: f32 = 100.0,

    pub const default = CameraIntrinsics{};
};

/// Camera parameters for projection
pub const Camera = struct {
    /// Camera position
    position: Vec3,
    /// Look-at target
    target: Vec3,
    /// Up vector
    up: Vec3,
    /// Field of view (radians)
    fov: f32,
    /// Aspect ratio (width/height)
    aspect: f32,
    /// Near clip plane
    near: f32,
    /// Far clip plane
    far: f32,

    pub const default = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.zero,
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 100.0,
    };

    /// Project 3D point to normalized device coordinates [-1, 1] with depth
    pub fn project(self: Camera, point: Vec3) ?ProjectionResult {
        // View direction
        const forward = self.target.sub(self.position).normalize();
        const right = forward.cross(self.up).normalize();
        const up = right.cross(forward);

        // Transform to camera space
        const rel = point.sub(self.position);
        const cam_x = rel.dot(right);
        const cam_y = rel.dot(up);
        const cam_z = rel.dot(forward); // Positive when point is in front of camera

        // Behind camera check
        if (cam_z <= self.near) return null;

        // Perspective projection
        const tan_half_fov = @tan(self.fov / 2.0);
        const ndc_x = cam_x / (cam_z * tan_half_fov * self.aspect);
        const ndc_y = cam_y / (cam_z * tan_half_fov);

        // Clip to NDC bounds
        if (ndc_x < -1 or ndc_x > 1 or ndc_y < -1 or ndc_y > 1) return null;

        return .{
            .ndc = math.Vec2.init(ndc_x, ndc_y),
            .depth = cam_z,
        };
    }

    /// Compute the projected radius in NDC space for a world-space radius at given depth
    pub fn projectRadius(self: Camera, world_radius: f32, depth: f32) f32 {
        const tan_half_fov = @tan(self.fov / 2.0);
        // Radius in NDC = world_radius / (depth * tan_half_fov)
        return world_radius / (depth * tan_half_fov);
    }
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "Label equality and hash" {
    const l1 = Label{ .birth_time = 10, .birth_index = 5 };
    const l2 = Label{ .birth_time = 10, .birth_index = 5 };
    const l3 = Label{ .birth_time = 10, .birth_index = 6 };

    try testing.expect(l1.eql(l2));
    try testing.expect(!l1.eql(l3));
    try testing.expect(l1.hash() == l2.hash());
    try testing.expect(l1.hash() != l3.hash());
}

test "PhysicsType parameters" {
    try testing.expect(PhysicsType.bouncy.elasticity() > PhysicsType.sticky.elasticity());
    try testing.expect(PhysicsType.sticky.friction() > PhysicsType.slippery.friction());
}

test "GaussianVec3 sample and logPdf" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const g = GaussianVec3.isotropic(Vec3.init(1, 2, 3), 0.1);

    // Sample should be near mean
    var sum = Vec3.zero;
    const n_samples = 100;
    for (0..n_samples) |_| {
        sum = sum.add(g.sample(rng));
    }
    const avg = sum.scale(1.0 / @as(f32, n_samples));
    try testing.expect(avg.distance(g.mean) < 0.5);

    // Log PDF at mean should be highest
    const log_p_mean = g.logPdf(g.mean);
    const log_p_far = g.logPdf(g.mean.add(Vec3.splat(10)));
    try testing.expect(log_p_mean > log_p_far);
}

test "Entity creation" {
    const label = Label{ .birth_time = 0, .birth_index = 0 };
    const e = Entity.initPoint(label, Vec3.init(1, 2, 3), Vec3.zero, .standard);

    try testing.expect(e.isAlive());
    try testing.expect(e.positionMean().approxEql(Vec3.init(1, 2, 3), 1e-6));
    try testing.expect(e.track_state == .detected);
}

test "Camera projection" {
    const cam = Camera.default;

    // Point in front of camera should project
    const p1 = Vec3.init(0, 0, 0);
    const proj1 = cam.project(p1);
    try testing.expect(proj1 != null);
    // Origin is at distance ~11.18 from camera at (0,5,10)
    try testing.expect(proj1.?.depth > 10.0);

    // Point behind camera should not project
    const p2 = Vec3.init(0, 0, 20); // Behind default camera at z=10
    const proj2 = cam.project(p2);
    try testing.expect(proj2 == null);
}

test "Camera projectRadius" {
    const cam = Camera.default;

    // At depth 10, radius 1 should project to some reasonable NDC value
    const projected_r = cam.projectRadius(1.0, 10.0);
    try testing.expect(projected_r > 0.01);
    try testing.expect(projected_r < 1.0);

    // Closer objects should project larger
    const closer_r = cam.projectRadius(1.0, 5.0);
    try testing.expect(closer_r > projected_r);
}
