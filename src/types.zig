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
// Continuous Physics Parameters (Spelke-aligned)
// =============================================================================

/// Continuous physical parameters per entity
/// These are INFERRED from observed motion, not set a priori
/// Aligned with Spelke's core knowledge: infants infer physical properties
/// from observing how objects move and interact
pub const PhysicsParams = struct {
    /// Coefficient of restitution [0,1] - inferred from bounces
    /// 0 = perfectly inelastic (no bounce), 1 = perfectly elastic
    elasticity: f32 = 0.5,

    /// Friction coefficient [0,1] - inferred from sliding
    /// 0 = frictionless (ice), 1 = maximum friction (sticky)
    friction: f32 = 0.3,

    // Preset configurations (convenience constants)
    pub const standard = PhysicsParams{ .elasticity = 0.5, .friction = 0.5 };
    pub const bouncy = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 };
    pub const sticky = PhysicsParams{ .elasticity = 0.2, .friction = 0.8 };
    pub const slippery = PhysicsParams{ .elasticity = 0.7, .friction = 0.1 };

    /// Default prior: moderate uncertainty centered at neutral values
    pub const prior = PhysicsParams{
        .elasticity = 0.5,
        .friction = 0.3,
    };

    /// Prior variances (Beta distribution pseudo-counts)
    /// Lower = more uncertain, higher = more confident
    pub const prior_pseudo_counts: f32 = 2.0; // Weak prior

    /// Get elasticity (for compatibility during transition)
    pub fn getElasticity(self: PhysicsParams) f32 {
        return self.elasticity;
    }

    /// Get friction (for compatibility during transition)
    pub fn getFriction(self: PhysicsParams) f32 {
        return self.friction;
    }
};

/// Uncertainty in physical parameters (for Bayesian updates)
/// Uses Beta distribution parameterization: mean = α/(α+β), variance = αβ/((α+β)²(α+β+1))
/// We store pseudo-counts α and β for conjugate updates
pub const PhysicsParamsUncertainty = struct {
    /// Elasticity Beta distribution: α (successes)
    elasticity_alpha: f32 = 2.0,
    /// Elasticity Beta distribution: β (failures)
    elasticity_beta: f32 = 2.0,

    /// Friction Beta distribution: α
    friction_alpha: f32 = 2.0,
    /// Friction Beta distribution: β
    friction_beta: f32 = 2.0,

    /// Weak prior (high variance)
    pub const weak_prior = PhysicsParamsUncertainty{
        .elasticity_alpha = 1.0,
        .elasticity_beta = 1.0,
        .friction_alpha = 1.0,
        .friction_beta = 1.0,
    };

    /// Vague prior - alias for weak_prior (maximum uncertainty)
    pub const vague = weak_prior;

    /// Get elasticity mean from Beta distribution
    pub fn elasticityMean(self: PhysicsParamsUncertainty) f32 {
        return self.elasticity_alpha / (self.elasticity_alpha + self.elasticity_beta);
    }

    /// Get elasticity variance from Beta distribution
    pub fn elasticityVariance(self: PhysicsParamsUncertainty) f32 {
        const a = self.elasticity_alpha;
        const b = self.elasticity_beta;
        const n = a + b;
        return (a * b) / (n * n * (n + 1));
    }

    /// Get friction mean from Beta distribution
    pub fn frictionMean(self: PhysicsParamsUncertainty) f32 {
        return self.friction_alpha / (self.friction_alpha + self.friction_beta);
    }

    /// Get friction variance from Beta distribution
    pub fn frictionVariance(self: PhysicsParamsUncertainty) f32 {
        const a = self.friction_alpha;
        const b = self.friction_beta;
        const n = a + b;
        return (a * b) / (n * n * (n + 1));
    }

    /// Update elasticity after observing a bounce
    /// observed_elasticity = |v_after| / |v_before| (coefficient of restitution)
    pub fn updateElasticity(self: *PhysicsParamsUncertainty, observed: f32, confidence: f32) void {
        // Approximate Beta update: treat observation as pseudo-count contribution
        // Higher confidence = more pseudo-counts
        const obs_clamped = std.math.clamp(observed, 0.01, 0.99);
        const weight = confidence * 2.0; // Scale confidence to pseudo-count contribution
        self.elasticity_alpha += weight * obs_clamped;
        self.elasticity_beta += weight * (1.0 - obs_clamped);
    }

    /// Update friction after observing sliding deceleration
    /// observed_friction = a_decel / g (deceleration relative to gravity)
    pub fn updateFriction(self: *PhysicsParamsUncertainty, observed: f32, confidence: f32) void {
        const obs_clamped = std.math.clamp(observed, 0.01, 0.99);
        const weight = confidence * 2.0;
        self.friction_alpha += weight * obs_clamped;
        self.friction_beta += weight * (1.0 - obs_clamped);
    }

    /// Get PhysicsParams from current posterior means
    pub fn toParams(self: PhysicsParamsUncertainty) PhysicsParams {
        return .{
            .elasticity = self.elasticityMean(),
            .friction = self.frictionMean(),
        };
    }
};

// =============================================================================
/// Contact mode between entities or with environment
/// Spelke core knowledge: Support emerges from contact with static geometry or entities
pub const ContactMode = enum(u8) {
    /// No contact, free flight under gravity
    free,
    /// Contact with static geometry (ground plane, walls)
    /// Renamed from 'ground' for domain-generality
    environment,
    /// Contact with another dynamic entity (support relationship)
    entity,
    /// Self-propelled motion (agency) - SLDS only
    agency,
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

/// Upper triangle of 6x6 symmetric covariance matrix (21 elements)
/// Layout: row-major upper triangle
/// [ 0  1  2  3  4  5 ]
/// [    6  7  8  9 10 ]
/// [      11 12 13 14 ]
/// [         15 16 17 ]
/// [            18 19 ]
/// [               20 ]
///
/// Blocks:
/// - P_pp (position-position): indices 0,1,2,6,7,11 (upper triangle of 3x3)
/// - P_vv (velocity-velocity): indices 15,16,17,18,19,20 (upper triangle of 3x3)
/// - P_pv (position-velocity cross): indices 3,4,5,8,9,10,12,13,14 (full 3x3)
pub const CovTriangle21 = struct {
    data: [21]f32,

    /// Identity covariance (uncorrelated, unit variance)
    pub const identity: CovTriangle21 = .{
        .data = .{
            1, 0, 0, 0, 0, 0, // row 0
            1, 0, 0, 0, 0, // row 1 (starting from col 1)
            1, 0, 0, 0, // row 2 (starting from col 2)
            1, 0, 0, // row 3 (starting from col 3)
            1, 0, // row 4 (starting from col 4)
            1, // row 5 (starting from col 5)
        },
    };

    /// Zero covariance
    pub const zero: CovTriangle21 = .{ .data = .{0} ** 21 };

    /// Get linear index for upper triangle (i <= j)
    fn triangleIndex(i: u3, j: u3) usize {
        std.debug.assert(i <= j);
        // Index = sum(6-k for k in 0..i) + (j - i)
        // = 6*i - i*(i-1)/2 + (j - i)
        // = 6*i - i*(i+1)/2 + j
        const ii: usize = @intCast(i);
        const jj: usize = @intCast(j);
        return ii * 6 - (ii * (ii + 1)) / 2 + jj;
    }

    /// Get element at (i, j), handling symmetry
    pub fn get(self: CovTriangle21, i: u3, j: u3) f32 {
        if (i <= j) {
            return self.data[triangleIndex(i, j)];
        } else {
            return self.data[triangleIndex(j, i)];
        }
    }

    /// Set element at (i, j), handling symmetry
    pub fn set(self: *CovTriangle21, i: u3, j: u3, val: f32) void {
        if (i <= j) {
            self.data[triangleIndex(i, j)] = val;
        } else {
            self.data[triangleIndex(j, i)] = val;
        }
    }

    /// Extract position-position block (upper-left 3x3)
    pub fn positionBlock(self: CovTriangle21) [6]f32 {
        return .{
            self.data[0], self.data[1], self.data[2], // row 0: (0,0), (0,1), (0,2)
            self.data[6], self.data[7], // row 1: (1,1), (1,2)
            self.data[11], // row 2: (2,2)
        };
    }

    /// Extract velocity-velocity block (lower-right 3x3)
    pub fn velocityBlock(self: CovTriangle21) [6]f32 {
        return .{
            self.data[15], self.data[16], self.data[17], // row 3: (3,3), (3,4), (3,5)
            self.data[18], self.data[19], // row 4: (4,4), (4,5)
            self.data[20], // row 5: (5,5)
        };
    }

    /// Extract position-velocity cross block (upper-right 3x3, stored as full matrix)
    /// Returns as column-major Mat3 for compatibility
    pub fn crossBlock(self: CovTriangle21) Mat3 {
        // P_pv occupies positions (0,3)-(2,5) in the 6x6 matrix
        // Row 0: (0,3), (0,4), (0,5) = indices 3, 4, 5
        // Row 1: (1,3), (1,4), (1,5) = indices 8, 9, 10
        // Row 2: (2,3), (2,4), (2,5) = indices 12, 13, 14
        return Mat3.fromRows(
            Vec3.init(self.data[3], self.data[4], self.data[5]),
            Vec3.init(self.data[8], self.data[9], self.data[10]),
            Vec3.init(self.data[12], self.data[13], self.data[14]),
        );
    }

    /// Set position-position block from upper triangle array
    pub fn setPositionBlock(self: *CovTriangle21, block: [6]f32) void {
        self.data[0] = block[0];
        self.data[1] = block[1];
        self.data[2] = block[2];
        self.data[6] = block[3];
        self.data[7] = block[4];
        self.data[11] = block[5];
    }

    /// Set velocity-velocity block from upper triangle array
    pub fn setVelocityBlock(self: *CovTriangle21, block: [6]f32) void {
        self.data[15] = block[0];
        self.data[16] = block[1];
        self.data[17] = block[2];
        self.data[18] = block[3];
        self.data[19] = block[4];
        self.data[20] = block[5];
    }

    /// Set cross block from Mat3 (row-major interpretation)
    pub fn setCrossBlock(self: *CovTriangle21, block: Mat3) void {
        // Store as row-major
        self.data[3] = block.get(0, 0);
        self.data[4] = block.get(0, 1);
        self.data[5] = block.get(0, 2);
        self.data[8] = block.get(1, 0);
        self.data[9] = block.get(1, 1);
        self.data[10] = block.get(1, 2);
        self.data[12] = block.get(2, 0);
        self.data[13] = block.get(2, 1);
        self.data[14] = block.get(2, 2);
    }

    /// Create from separate position, velocity, and cross covariances
    pub fn fromBlocks(pos_cov: Mat3, vel_cov: Mat3, cross_cov: Mat3) CovTriangle21 {
        var result = CovTriangle21.zero;

        // Position block (upper triangle of pos_cov)
        result.data[0] = pos_cov.get(0, 0);
        result.data[1] = pos_cov.get(0, 1);
        result.data[2] = pos_cov.get(0, 2);
        result.data[6] = pos_cov.get(1, 1);
        result.data[7] = pos_cov.get(1, 2);
        result.data[11] = pos_cov.get(2, 2);

        // Cross block
        result.data[3] = cross_cov.get(0, 0);
        result.data[4] = cross_cov.get(0, 1);
        result.data[5] = cross_cov.get(0, 2);
        result.data[8] = cross_cov.get(1, 0);
        result.data[9] = cross_cov.get(1, 1);
        result.data[10] = cross_cov.get(1, 2);
        result.data[12] = cross_cov.get(2, 0);
        result.data[13] = cross_cov.get(2, 1);
        result.data[14] = cross_cov.get(2, 2);

        // Velocity block (upper triangle of vel_cov)
        result.data[15] = vel_cov.get(0, 0);
        result.data[16] = vel_cov.get(0, 1);
        result.data[17] = vel_cov.get(0, 2);
        result.data[18] = vel_cov.get(1, 1);
        result.data[19] = vel_cov.get(1, 2);
        result.data[20] = vel_cov.get(2, 2);

        return result;
    }

    /// Create diagonal covariance (no cross-correlation)
    pub fn diagonal(pos_var: Vec3, vel_var: Vec3) CovTriangle21 {
        var result = CovTriangle21.zero;
        result.data[0] = pos_var.x;
        result.data[6] = pos_var.y;
        result.data[11] = pos_var.z;
        result.data[15] = vel_var.x;
        result.data[18] = vel_var.y;
        result.data[20] = vel_var.z;
        return result;
    }

    /// Scale entire covariance by scalar
    pub fn scale(self: CovTriangle21, s: f32) CovTriangle21 {
        var result: CovTriangle21 = undefined;
        for (0..21) |i| {
            result.data[i] = self.data[i] * s;
        }
        return result;
    }

    /// Add two covariances
    pub fn add(self: CovTriangle21, other: CovTriangle21) CovTriangle21 {
        var result: CovTriangle21 = undefined;
        for (0..21) |i| {
            result.data[i] = self.data[i] + other.data[i];
        }
        return result;
    }
};

/// 6D Gaussian distribution (position + velocity coupled)
/// Used for Rao-Blackwellized Kalman filtering with cross-covariance
pub const Gaussian6D = struct {
    mean: [6]f32, // [px, py, pz, vx, vy, vz]
    cov: CovTriangle21,

    /// Create with diagonal covariance (no position-velocity correlation)
    pub fn diagonal(pos_mean: Vec3, pos_var: Vec3, vel_mean: Vec3, vel_var: Vec3) Gaussian6D {
        return .{
            .mean = .{ pos_mean.x, pos_mean.y, pos_mean.z, vel_mean.x, vel_mean.y, vel_mean.z },
            .cov = CovTriangle21.diagonal(pos_var, vel_var),
        };
    }

    /// Create with isotropic position and velocity variances
    pub fn isotropic(pos_mean: Vec3, pos_var: f32, vel_mean: Vec3, vel_var: f32) Gaussian6D {
        return diagonal(pos_mean, Vec3.splat(pos_var), vel_mean, Vec3.splat(vel_var));
    }

    /// Get position components as Vec3
    pub fn position(self: Gaussian6D) Vec3 {
        return Vec3.init(self.mean[0], self.mean[1], self.mean[2]);
    }

    /// Get velocity components as Vec3
    pub fn velocity(self: Gaussian6D) Vec3 {
        return Vec3.init(self.mean[3], self.mean[4], self.mean[5]);
    }

    /// Set position components from Vec3
    pub fn setPosition(self: *Gaussian6D, pos: Vec3) void {
        self.mean[0] = pos.x;
        self.mean[1] = pos.y;
        self.mean[2] = pos.z;
    }

    /// Set velocity components from Vec3
    pub fn setVelocity(self: *Gaussian6D, vel: Vec3) void {
        self.mean[3] = vel.x;
        self.mean[4] = vel.y;
        self.mean[5] = vel.z;
    }

    /// Get position covariance as Mat3
    pub fn positionCov(self: Gaussian6D) Mat3 {
        const tri = self.cov.positionBlock();
        // Convert upper triangle to full symmetric matrix
        return Mat3{
            .data = .{
                tri[0], tri[1], tri[2], // col 0
                tri[1], tri[3], tri[4], // col 1
                tri[2], tri[4], tri[5], // col 2
            },
        };
    }

    /// Get velocity covariance as Mat3
    pub fn velocityCov(self: Gaussian6D) Mat3 {
        const tri = self.cov.velocityBlock();
        return Mat3{
            .data = .{
                tri[0], tri[1], tri[2], // col 0
                tri[1], tri[3], tri[4], // col 1
                tri[2], tri[4], tri[5], // col 2
            },
        };
    }

    /// Get position-velocity cross-covariance (P_pv)
    pub fn crossCov(self: Gaussian6D) Mat3 {
        return self.cov.crossBlock();
    }

    /// Convert from separate position and velocity Gaussians (no cross-covariance)
    pub fn fromSeparate(pos: GaussianVec3, vel: GaussianVec3) Gaussian6D {
        return .{
            .mean = .{ pos.mean.x, pos.mean.y, pos.mean.z, vel.mean.x, vel.mean.y, vel.mean.z },
            .cov = CovTriangle21.fromBlocks(pos.cov, vel.cov, Mat3.zero),
        };
    }

    /// Extract as separate position and velocity Gaussians (loses cross-covariance!)
    pub fn toSeparate(self: Gaussian6D) struct { pos: GaussianVec3, vel: GaussianVec3 } {
        return .{
            .pos = .{ .mean = self.position(), .cov = self.positionCov() },
            .vel = .{ .mean = self.velocity(), .cov = self.velocityCov() },
        };
    }
};

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

    /// Continuous physics parameters (Spelke-aligned inference)
    physics_params: PhysicsParams,

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
        physics_params: PhysicsParams,
    ) Entity {
        const init_variance: f32 = 0.001;
        return .{
            .label = label,
            .position = GaussianVec3.isotropic(position, init_variance),
            .velocity = GaussianVec3.isotropic(velocity, init_variance),
            .physics_params = physics_params,
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
// Environment Entities (Static Geometry)
// =============================================================================

/// Static environment geometry (ground, walls)
/// Conceptually an entity, but with O(1) collision optimization
/// Spelke core knowledge: Support emerges from contact with any entity below
pub const EnvironmentEntity = struct {
    /// Plane equation: point · normal = height
    height: f32,
    normal: Vec3,

    /// Physics properties (for collision response)
    friction: f32,
    elasticity: f32,

    /// Default ground plane at y=0
    pub const ground = EnvironmentEntity{
        .height = 0.0,
        .normal = Vec3.unit_y,
        .friction = 0.5,
        .elasticity = 0.5,
    };

    /// Check if a point is below/inside this plane
    pub fn isBelow(self: EnvironmentEntity, pos: Vec3, radius: f32) bool {
        const dist_to_plane = pos.dot(self.normal) - self.height;
        return dist_to_plane < radius;
    }

    /// Get signed distance from point to plane
    pub fn signedDistance(self: EnvironmentEntity, pos: Vec3) f32 {
        return pos.dot(self.normal) - self.height;
    }
};

/// Environment configuration (ground plane, walls)
pub const EnvironmentConfig = struct {
    /// Ground plane (null = no ground, freefall world)
    ground: ?EnvironmentEntity = EnvironmentEntity.ground,

    /// Get ground height (for backwards compatibility)
    /// Returns 0.0 if no ground is configured
    pub fn groundHeight(self: EnvironmentConfig) f32 {
        if (self.ground) |g| {
            return g.height;
        }
        return 0.0;
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

    /// Environment entities (ground plane, walls)
    /// Configure ground via: config.environment.ground = .{ .height = 0.0, ... }
    /// Set to null for freefall world: config.environment.ground = null
    environment: EnvironmentConfig = .{},

    /// World bounds (AABB min/max)
    bounds_min: Vec3 = Vec3.init(-10, -10, -10),
    bounds_max: Vec3 = Vec3.init(10, 10, 10),

    /// Default process noise for dynamics
    process_noise: f32 = 0.01,

    /// CRP concentration parameter (birth rate)
    crp_alpha: f32 = 1.0,

    /// Survival probability per timestep
    survival_prob: f32 = 0.99,

    /// Get ground height from environment config
    /// Returns -inf if no ground is configured (freefall world)
    pub fn groundHeight(self: PhysicsConfig) f32 {
        if (self.environment.ground) |g| {
            return g.height;
        }
        return -std.math.inf(f32);
    }
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

test "PhysicsParams Bayesian update" {
    var unc = PhysicsParamsUncertainty.weak_prior;
    const initial_elasticity = unc.elasticityMean();

    // Observing high elasticity should increase mean
    unc.updateElasticity(0.9, 1.0);
    try testing.expect(unc.elasticityMean() > initial_elasticity);

    // Observing low elasticity should decrease mean
    var unc2 = PhysicsParamsUncertainty.weak_prior;
    unc2.updateElasticity(0.1, 1.0);
    try testing.expect(unc2.elasticityMean() < initial_elasticity);
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

test "EnvironmentEntity ground plane" {
    const ground = EnvironmentEntity.ground;

    // Ground plane should be at y=0 with normal pointing up
    try testing.expect(ground.height == 0.0);
    try testing.expect(ground.normal.y == 1.0);

    // Point above ground should not be below
    try testing.expect(!ground.isBelow(Vec3.init(0, 1, 0), 0.5));

    // Point at ground should be below (within radius)
    try testing.expect(ground.isBelow(Vec3.init(0, 0.3, 0), 0.5));

    // Signed distance should be positive above, negative below
    try testing.expect(ground.signedDistance(Vec3.init(0, 2, 0)) > 0);
    try testing.expect(ground.signedDistance(Vec3.init(0, -1, 0)) < 0);
}

test "EnvironmentConfig ground height" {
    // Default config should have ground at y=0
    const default_config = EnvironmentConfig{};
    try testing.expect(default_config.groundHeight() == 0.0);

    // Config with no ground should return 0
    const no_ground = EnvironmentConfig{ .ground = null };
    try testing.expect(no_ground.groundHeight() == 0.0);

    // Config with custom ground height
    const custom_ground = EnvironmentConfig{
        .ground = .{
            .height = 5.0,
            .normal = Vec3.unit_y,
            .friction = 0.3,
            .elasticity = 0.7,
        },
    };
    try testing.expect(custom_ground.groundHeight() == 5.0);
}

test "PhysicsConfig ground height" {
    // Default config should use environment ground at y=0
    const default_config = PhysicsConfig{};
    try testing.expect(default_config.groundHeight() == 0.0);

    // Config with custom environment ground
    var custom_config = PhysicsConfig{};
    custom_config.environment.ground = .{
        .height = 3.0,
        .normal = Vec3.unit_y,
        .friction = 0.5,
        .elasticity = 0.5,
    };
    try testing.expect(custom_config.groundHeight() == 3.0);

    // Config with no ground should return -inf
    var no_ground_config = PhysicsConfig{};
    no_ground_config.environment.ground = null;
    try testing.expect(no_ground_config.groundHeight() == -std.math.inf(f32));
}

test "CovTriangle21 indexing" {
    // Test that triangleIndex works correctly
    var cov = CovTriangle21.identity;

    // Diagonal elements should be 1
    try testing.expectApproxEqAbs(@as(f32, 1.0), cov.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), cov.get(1, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), cov.get(2, 2), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), cov.get(3, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), cov.get(4, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), cov.get(5, 5), 1e-6);

    // Off-diagonal elements should be 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), cov.get(0, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), cov.get(0, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), cov.get(2, 5), 1e-6);

    // Test symmetry
    cov.set(0, 3, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), cov.get(0, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), cov.get(3, 0), 1e-6);
}

test "CovTriangle21 block extraction" {
    // Create covariance with known structure
    var cov = CovTriangle21.zero;

    // Set position block diagonal
    cov.set(0, 0, 1.0);
    cov.set(1, 1, 2.0);
    cov.set(2, 2, 3.0);

    // Set velocity block diagonal
    cov.set(3, 3, 4.0);
    cov.set(4, 4, 5.0);
    cov.set(5, 5, 6.0);

    // Set cross block
    cov.set(0, 3, 0.1);
    cov.set(1, 4, 0.2);
    cov.set(2, 5, 0.3);

    // Extract and verify position block
    const pos_block = cov.positionBlock();
    try testing.expectApproxEqAbs(@as(f32, 1.0), pos_block[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), pos_block[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3.0), pos_block[5], 1e-6);

    // Extract and verify velocity block
    const vel_block = cov.velocityBlock();
    try testing.expectApproxEqAbs(@as(f32, 4.0), vel_block[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), vel_block[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), vel_block[5], 1e-6);

    // Extract and verify cross block
    const cross = cov.crossBlock();
    try testing.expectApproxEqAbs(@as(f32, 0.1), cross.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), cross.get(1, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.3), cross.get(2, 2), 1e-6);
}

test "Gaussian6D creation and accessors" {
    const pos = Vec3.init(1, 2, 3);
    const vel = Vec3.init(0.1, 0.2, 0.3);

    const g = Gaussian6D.isotropic(pos, 0.5, vel, 0.01);

    // Check mean extraction
    try testing.expect(g.position().approxEql(pos, 1e-6));
    try testing.expect(g.velocity().approxEql(vel, 1e-6));

    // Check position covariance is diagonal with value 0.5
    const pos_cov = g.positionCov();
    try testing.expectApproxEqAbs(@as(f32, 0.5), pos_cov.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), pos_cov.get(1, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), pos_cov.get(2, 2), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), pos_cov.get(0, 1), 1e-6);

    // Check velocity covariance is diagonal with value 0.01
    const vel_cov = g.velocityCov();
    try testing.expectApproxEqAbs(@as(f32, 0.01), vel_cov.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.01), vel_cov.get(1, 1), 1e-6);

    // Cross covariance should be zero (isotropic has no correlation)
    const cross = g.crossCov();
    try testing.expectApproxEqAbs(@as(f32, 0.0), cross.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), cross.get(1, 1), 1e-6);
}

test "Gaussian6D round-trip from separate" {
    const pos = GaussianVec3.isotropic(Vec3.init(1, 2, 3), 0.5);
    const vel = GaussianVec3.isotropic(Vec3.init(0.1, 0.2, 0.3), 0.01);

    const g6d = Gaussian6D.fromSeparate(pos, vel);
    const separated = g6d.toSeparate();

    try testing.expect(separated.pos.mean.approxEql(pos.mean, 1e-6));
    try testing.expect(separated.vel.mean.approxEql(vel.mean, 1e-6));
    try testing.expectApproxEqAbs(pos.cov.get(0, 0), separated.pos.cov.get(0, 0), 1e-6);
    try testing.expectApproxEqAbs(vel.cov.get(0, 0), separated.vel.cov.get(0, 0), 1e-6);
}
