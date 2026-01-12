const std = @import("std");
const math = @import("math.zig");
const types = @import("types.zig");

const Vec3 = math.Vec3;
const Mat3 = math.Mat3;
const Entity = types.Entity;
const GaussianVec3 = types.GaussianVec3;
const PhysicsType = types.PhysicsType;
const ContactMode = types.ContactMode;
const PhysicsConfig = types.PhysicsConfig;

// =============================================================================
// Linear-Gaussian Dynamics
// =============================================================================

/// State transition matrices for linear-Gaussian dynamics
/// x_{t+1} = A * x_t + B * u_t + w, where w ~ N(0, Q)
pub const DynamicsMatrices = struct {
    /// State transition matrix (6x6 for pos+vel, but we use 3x3 blocks)
    A_pos: Mat3, // Position update from position
    A_vel: Mat3, // Position update from velocity
    B_pos: Mat3, // Velocity update from position (usually zero)
    B_vel: Mat3, // Velocity update from velocity

    /// Process noise covariance
    Q_pos: Mat3,
    Q_vel: Mat3,

    /// Control input (gravity, forces)
    gravity: Vec3,
    dt: f32,

    /// Create dynamics matrices for given physics type and config
    pub fn forPhysicsType(physics_type: PhysicsType, config: PhysicsConfig) DynamicsMatrices {
        const dt = config.dt;
        const friction = physics_type.friction();
        const noise_scale = physics_type.processNoise();

        // Position update: p_{t+1} = p_t + v_t * dt
        const A_pos = Mat3.identity;
        const A_vel = Mat3.scale(dt);

        // Velocity update: v_{t+1} = (1 - friction * dt) * v_t + g * dt
        const vel_decay = 1.0 - friction * dt;
        const B_pos = Mat3.zero;
        const B_vel = Mat3.scale(vel_decay);

        // Process noise (scaled by dt and physics type)
        const pos_noise = noise_scale * dt * dt;
        const vel_noise = noise_scale * dt;
        const Q_pos = Mat3.scale(pos_noise);
        const Q_vel = Mat3.scale(vel_noise);

        return .{
            .A_pos = A_pos,
            .A_vel = A_vel,
            .B_pos = B_pos,
            .B_vel = B_vel,
            .Q_pos = Q_pos,
            .Q_vel = Q_vel,
            .gravity = config.gravity,
            .dt = dt,
        };
    }

    /// Create dynamics matrices for contact mode
    pub fn forContactMode(
        contact_mode: ContactMode,
        physics_type: PhysicsType,
        config: PhysicsConfig,
        contact_normal: Vec3,
    ) DynamicsMatrices {
        var matrices = forPhysicsType(physics_type, config);

        switch (contact_mode) {
            .free => {
                // Standard free-flight dynamics
            },
            .ground => {
                // Constrain to ground plane, apply friction
                // Zero out vertical velocity component in update
                matrices.B_vel = applyConstraint(matrices.B_vel, Vec3.unit_y);
                // Increase friction effect
                matrices.Q_vel = matrices.Q_vel.scaleMat(0.5);
            },
            .wall => {
                // Constrain perpendicular to wall normal
                matrices.B_vel = applyConstraint(matrices.B_vel, contact_normal);
            },
            .entity => {
                // Entity-entity contact: handled separately in collision resolution
            },
        }

        return matrices;
    }
};

/// Apply constraint by zeroing out component along normal
fn applyConstraint(M: Mat3, normal: Vec3) Mat3 {
    // Project out normal component: M' = M - (n * n^T) * M
    const projection = Mat3.outer(normal, normal);
    return M.sub(projection.mulMat(M));
}

// =============================================================================
// Kalman Filter for Rao-Blackwellization
// =============================================================================

/// Kalman filter predict step for position
pub fn kalmanPredictPosition(
    pos: GaussianVec3,
    vel: GaussianVec3,
    matrices: DynamicsMatrices,
) GaussianVec3 {
    // mu_p' = A_pos * mu_p + A_vel * mu_v + gravity * dt^2 / 2 (for position)
    const new_mean = matrices.A_pos.mulVec(pos.mean)
        .add(matrices.A_vel.mulVec(vel.mean))
        .add(matrices.gravity.scale(0.5 * matrices.dt * matrices.dt));

    // cov_p' = A_pos * cov_p * A_pos^T + A_vel * cov_v * A_vel^T + Q_pos
    const cov_from_pos = matrices.A_pos.mulMat(pos.cov).mulMat(matrices.A_pos.transpose());
    const cov_from_vel = matrices.A_vel.mulMat(vel.cov).mulMat(matrices.A_vel.transpose());
    const new_cov = cov_from_pos.add(cov_from_vel).add(matrices.Q_pos);

    return .{ .mean = new_mean, .cov = new_cov };
}

/// Kalman filter predict step for velocity
pub fn kalmanPredictVelocity(
    vel: GaussianVec3,
    matrices: DynamicsMatrices,
) GaussianVec3 {
    // mu_v' = B_vel * mu_v + gravity * dt
    const new_mean = matrices.B_vel.mulVec(vel.mean)
        .add(matrices.gravity.scale(matrices.dt));

    // cov_v' = B_vel * cov_v * B_vel^T + Q_vel
    const new_cov = matrices.B_vel.mulMat(vel.cov)
        .mulMat(matrices.B_vel.transpose())
        .add(matrices.Q_vel);

    return .{ .mean = new_mean, .cov = new_cov };
}

/// Kalman filter update step given observation
/// Assumes observation is of position with measurement noise R
pub fn kalmanUpdate(
    predicted: GaussianVec3,
    observation: Vec3,
    R: Mat3, // Measurement noise covariance
) GaussianVec3 {
    // Innovation: y = z - H * mu (H = I for position observation)
    const innovation = observation.sub(predicted.mean);

    // Innovation covariance: S = H * P * H^T + R = P + R
    const S = predicted.cov.add(R);

    // Kalman gain: K = P * H^T * S^{-1} = P * S^{-1}
    const S_inv = S.inverse() orelse return predicted; // Return unchanged if singular
    const K = predicted.cov.mulMat(S_inv);

    // Updated mean: mu' = mu + K * y
    const new_mean = predicted.mean.add(K.mulVec(innovation));

    // Updated covariance: P' = (I - K * H) * P = (I - K) * P
    const I_minus_K = Mat3.identity.sub(K);
    const new_cov = I_minus_K.mulMat(predicted.cov);

    return .{ .mean = new_mean, .cov = new_cov };
}

/// Compute marginal log-likelihood for Kalman update
/// Used for particle weight update
pub fn kalmanLogLikelihood(
    predicted: GaussianVec3,
    observation: Vec3,
    R: Mat3,
) f32 {
    // Innovation covariance: S = P + R
    const S = predicted.cov.add(R);

    // Innovation
    const innovation = observation.sub(predicted.mean);

    // Log likelihood = log N(observation | predicted.mean, S)
    const S_inv = S.inverse() orelse return -std.math.inf(f32);
    const mahalanobis = innovation.dot(S_inv.mulVec(innovation));

    const det = S.determinant();
    if (det <= 0) return -std.math.inf(f32);

    const log_det = @log(det);
    const log_2pi: f32 = 1.8378770664093453;

    return -0.5 * (3.0 * log_2pi + log_det + mahalanobis);
}

// =============================================================================
// Physics Step (Point Estimate Version)
// =============================================================================

/// Simple physics step for point estimates (non-Rao-Blackwellized)
pub fn physicsStepPoint(
    position: Vec3,
    velocity: Vec3,
    physics_type: PhysicsType,
    config: PhysicsConfig,
) struct { position: Vec3, velocity: Vec3 } {
    const dt = config.dt;
    const friction = physics_type.friction();

    // Velocity update: v' = (1 - friction * dt) * v + g * dt
    const vel_decay = 1.0 - friction * dt;
    var new_vel = velocity.scale(vel_decay).add(config.gravity.scale(dt));

    // Position update: p' = p + v' * dt
    var new_pos = position.add(new_vel.scale(dt));

    // Ground collision
    if (new_pos.y < config.ground_height) {
        new_pos.y = config.ground_height;
        // Reflect and apply elasticity
        if (new_vel.y < 0) {
            new_vel.y = -new_vel.y * physics_type.elasticity();
        }
    }

    // Bounds clamping
    new_pos = Vec3.init(
        @max(config.bounds_min.x, @min(config.bounds_max.x, new_pos.x)),
        @max(config.bounds_min.y, @min(config.bounds_max.y, new_pos.y)),
        @max(config.bounds_min.z, @min(config.bounds_max.z, new_pos.z)),
    );

    return .{ .position = new_pos, .velocity = new_vel };
}

/// Full entity physics step with Gaussian state
pub fn entityPhysicsStep(
    entity: *Entity,
    config: PhysicsConfig,
    rng: ?std.Random,
) void {
    const matrices = DynamicsMatrices.forContactMode(
        entity.contact_mode,
        entity.physics_type,
        config,
        Vec3.unit_y, // Default contact normal
    );

    // Kalman predict
    entity.velocity = kalmanPredictVelocity(entity.velocity, matrices);
    entity.position = kalmanPredictPosition(entity.position, entity.velocity, matrices);

    // Add process noise if RNG provided (for generative sampling)
    if (rng) |r| {
        entity.position.mean = entity.position.mean.add(sampleNoise(r, matrices.Q_pos));
        entity.velocity.mean = entity.velocity.mean.add(sampleNoise(r, matrices.Q_vel));
    }

    // Ground collision detection and response
    if (entity.position.mean.y < config.ground_height) {
        entity.position.mean.y = config.ground_height;
        entity.contact_mode = .ground;

        if (entity.velocity.mean.y < 0) {
            entity.velocity.mean.y = -entity.velocity.mean.y * entity.physics_type.elasticity();
        }
    } else if (entity.position.mean.y > config.ground_height + 0.1) {
        // Clear ground contact if sufficiently above
        if (entity.contact_mode == .ground) {
            entity.contact_mode = .free;
        }
    }
}

/// Sample noise from covariance matrix (diagonal approximation)
fn sampleNoise(rng: std.Random, cov: Mat3) Vec3 {
    const std_x = @sqrt(@max(0.0, cov.get(0, 0)));
    const std_y = @sqrt(@max(0.0, cov.get(1, 1)));
    const std_z = @sqrt(@max(0.0, cov.get(2, 2)));

    return Vec3.init(
        sampleStdNormal(rng) * std_x,
        sampleStdNormal(rng) * std_y,
        sampleStdNormal(rng) * std_z,
    );
}

fn sampleStdNormal(rng: std.Random) f32 {
    const r1 = rng.float(f32);
    const r2 = rng.float(f32);
    return @sqrt(-2.0 * @log(r1 + 1e-10)) * @cos(2.0 * std.math.pi * r2);
}

// =============================================================================
// Contact Detection
// =============================================================================

/// Check for contact between two entities (sphere collision)
pub fn checkEntityContact(e1: Entity, e2: Entity) bool {
    const dist = e1.positionMean().distance(e2.positionMean());
    const combined_radius = e1.appearance.radius + e2.appearance.radius;
    return dist < combined_radius;
}

/// Resolve collision between two entities (impulse-based)
pub fn resolveEntityCollision(
    e1: *Entity,
    e2: *Entity,
) void {
    const p1 = e1.positionMean();
    const p2 = e2.positionMean();
    const v1 = e1.velocityMean();
    const v2 = e2.velocityMean();

    // Collision normal
    const normal = p2.sub(p1).normalize();
    if (normal.eql(Vec3.zero)) return;

    // Relative velocity
    const rel_vel = v1.sub(v2);
    const vel_along_normal = rel_vel.dot(normal);

    // Only resolve if approaching
    if (vel_along_normal > 0) return;

    // Coefficient of restitution (use minimum of the two)
    const e = @min(e1.physics_type.elasticity(), e2.physics_type.elasticity());

    // Impulse magnitude (assuming equal mass)
    const j = -(1.0 + e) * vel_along_normal / 2.0;

    // Apply impulse
    const impulse = normal.scale(j);
    e1.velocity.mean = v1.add(impulse);
    e2.velocity.mean = v2.sub(impulse);

    // Separate entities to prevent overlap
    const dist = p1.distance(p2);
    const combined_radius = e1.appearance.radius + e2.appearance.radius;
    const overlap = combined_radius - dist;

    if (overlap > 0) {
        const separation = normal.scale(overlap / 2.0);
        e1.position.mean = p1.sub(separation);
        e2.position.mean = p2.add(separation);
    }

    // Update contact modes
    e1.contact_mode = .entity;
    e2.contact_mode = .entity;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "DynamicsMatrices creation" {
    const config = PhysicsConfig{};
    const matrices = DynamicsMatrices.forPhysicsType(.standard, config);

    // A_pos should be identity
    try testing.expect(matrices.A_pos.approxEql(Mat3.identity, 1e-6));

    // A_vel should scale by dt
    const expected_A_vel = Mat3.scale(config.dt);
    try testing.expect(matrices.A_vel.approxEql(expected_A_vel, 1e-6));
}

test "Kalman predict position" {
    const pos = GaussianVec3.isotropic(Vec3.init(0, 10, 0), 0.1);
    const vel = GaussianVec3.isotropic(Vec3.init(1, 0, 0), 0.01);

    const config = PhysicsConfig{ .gravity = Vec3.init(0, -10, 0), .dt = 0.1 };
    const matrices = DynamicsMatrices.forPhysicsType(.standard, config);

    const new_pos = kalmanPredictPosition(pos, vel, matrices);

    // Position should move in x direction
    try testing.expect(new_pos.mean.x > pos.mean.x);
    // Position should drop due to gravity
    try testing.expect(new_pos.mean.y < pos.mean.y);
}

test "Kalman predict velocity" {
    const vel = GaussianVec3.isotropic(Vec3.init(0, 0, 0), 0.01);

    const config = PhysicsConfig{ .gravity = Vec3.init(0, -10, 0), .dt = 0.1 };
    const matrices = DynamicsMatrices.forPhysicsType(.standard, config);

    const new_vel = kalmanPredictVelocity(vel, matrices);

    // Velocity should increase downward due to gravity
    try testing.expect(new_vel.mean.y < 0);
}

test "Kalman update" {
    const predicted = GaussianVec3.isotropic(Vec3.init(0, 0, 0), 1.0);
    const observation = Vec3.init(1, 1, 1);
    const R = Mat3.scale(0.1); // Low measurement noise

    const updated = kalmanUpdate(predicted, observation, R);

    // Updated mean should be closer to observation
    try testing.expect(updated.mean.distance(observation) < predicted.mean.distance(observation));

    // Updated covariance should be smaller
    try testing.expect(updated.cov.trace() < predicted.cov.trace());
}

test "Physics step point estimate" {
    const config = PhysicsConfig{ .gravity = Vec3.init(0, -10, 0), .dt = 0.1 };

    const result = physicsStepPoint(
        Vec3.init(0, 5, 0),
        Vec3.zero,
        .standard,
        config,
    );

    // Should fall due to gravity
    try testing.expect(result.position.y < 5);
    try testing.expect(result.velocity.y < 0);
}

test "Physics step with ground collision" {
    const config = PhysicsConfig{
        .gravity = Vec3.init(0, -10, 0),
        .dt = 0.1,
        .ground_height = 0,
    };

    // Start at ground with downward velocity
    const result = physicsStepPoint(
        Vec3.init(0, 0.05, 0),
        Vec3.init(0, -5, 0),
        .bouncy,
        config,
    );

    // Should bounce (velocity should be positive after hitting ground)
    try testing.expect(result.velocity.y > 0);
    try testing.expect(result.position.y >= 0);
}

test "Entity contact detection" {
    const label1 = types.Label{ .birth_time = 0, .birth_index = 0 };
    const label2 = types.Label{ .birth_time = 0, .birth_index = 1 };

    var e1 = Entity.initPoint(label1, Vec3.init(0, 0, 0), Vec3.zero, .standard);
    var e2 = Entity.initPoint(label2, Vec3.init(0.5, 0, 0), Vec3.zero, .standard);

    e1.appearance.radius = 0.5;
    e2.appearance.radius = 0.5;

    // Should be in contact (distance 0.5, combined radius 1.0)
    try testing.expect(checkEntityContact(e1, e2));

    // Move apart
    e2.position.mean = Vec3.init(2, 0, 0);
    try testing.expect(!checkEntityContact(e1, e2));
}
