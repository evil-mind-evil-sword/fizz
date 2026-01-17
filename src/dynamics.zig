const std = @import("std");
const math = @import("math.zig");
const types = @import("types.zig");

const Vec3 = math.Vec3;
const Mat3 = math.Mat3;
const Entity = types.Entity;
const GaussianVec3 = types.GaussianVec3;
const Gaussian6D = types.Gaussian6D;
const CovTriangle21 = types.CovTriangle21;
const PhysicsParams = types.PhysicsParams;
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

    /// Create dynamics matrices for given physics params and config
    pub fn forPhysicsParams(physics_params: PhysicsParams, config: PhysicsConfig) DynamicsMatrices {
        const dt = config.dt;
        const friction = physics_params.friction;
        // Use config.process_noise instead of hardcoded physics_type value
        // This allows proper Kalman filter responsiveness to observations
        const noise_scale = config.process_noise;

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

    /// Create dynamics matrices for contact mode with continuous physics params
    pub fn forContactModeWithParams(
        contact_mode: ContactMode,
        physics_params: PhysicsParams,
        config: PhysicsConfig,
        contact_normal: Vec3,
    ) DynamicsMatrices {
        var matrices = forPhysicsParams(physics_params, config);

        switch (contact_mode) {
            .free => {
                // Standard free-flight dynamics
            },
            .environment => {
                // Contact with static geometry (ground, walls)
                // Constrain perpendicular to contact normal, apply friction
                matrices.B_vel = applyConstraint(matrices.B_vel, contact_normal);
                // Increase friction effect for stability
                matrices.Q_vel = matrices.Q_vel.scaleMat(0.5);
            },
            .entity => {
                // Entity-entity contact: handled separately in collision resolution
            },
            .agency => {
                // Self-propelled motion: higher noise, less predictable
                matrices.Q_vel = matrices.Q_vel.scaleMat(3.0);
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
// 6D Coupled Kalman Filter
// =============================================================================
//
// This implements a coupled 6D Kalman filter where position and velocity are
// tracked jointly with cross-covariance. The key benefit is that position
// observations can update velocity via the cross-covariance term:
//
//     K_v = P_vp · S⁻¹
//
// This allows bounces (detected via position observations) to correctly
// update velocity estimates, fixing the factored filter's inability to
// track velocity through discontinuities.
//
// State: x = [p; v] (6D)
// Dynamics: x' = F·x + Bu + w, w ~ N(0, Q)
// Observation: z = H·x + r, r ~ N(0, R), H = [I_3, 0_3] (position only)
// =============================================================================

/// 6x6 state transition matrix stored as 4 blocks of 3x3
/// [ F_pp  F_pv ]   [ I     dt·I ]
/// [ F_vp  F_vv ] = [ 0  (1-f·dt)·I ]
pub const StateTransition6 = struct {
    F_pp: Mat3, // position-position: I
    F_pv: Mat3, // position-velocity: dt·I
    F_vp: Mat3, // velocity-position: 0
    F_vv: Mat3, // velocity-velocity: (1-friction·dt)·I
};

/// 6D dynamics matrices for coupled Kalman filter
pub const DynamicsMatrices6D = struct {
    /// State transition (4 blocks)
    F: StateTransition6,
    /// Process noise covariance (21 elements, upper triangle of 6x6)
    Q: CovTriangle21,
    /// Gravity vector
    gravity: Vec3,
    /// Timestep
    dt: f32,

    /// Create dynamics matrices for given physics params and config
    pub fn forPhysicsParams(physics_params: PhysicsParams, config: PhysicsConfig) DynamicsMatrices6D {
        const dt = config.dt;
        const friction = physics_params.friction;
        const noise_scale = config.process_noise;

        // State transition blocks
        // Position: p' = p + v·dt
        // Velocity: v' = (1-friction·dt)·v + g·dt
        const vel_decay = 1.0 - friction * dt;

        const F = StateTransition6{
            .F_pp = Mat3.identity,
            .F_pv = Mat3.scale(dt),
            .F_vp = Mat3.zero,
            .F_vv = Mat3.scale(vel_decay),
        };

        // Process noise covariance
        // For coupled dynamics, there's correlation between position and velocity noise
        // Q = [Q_pp, Q_pv; Q_vp, Q_vv]
        // Using dt² for position (more responsive) and standard cross-covariance
        const pos_noise = noise_scale * dt * dt;
        const vel_noise = noise_scale * dt;
        const cross_noise = noise_scale * dt * dt * 0.5; // dt²/2 factor for cross-covariance

        const Q_pp = Mat3.scale(pos_noise);
        const Q_vv = Mat3.scale(vel_noise);
        const Q_pv = Mat3.scale(cross_noise);

        const Q = CovTriangle21.fromBlocks(Q_pp, Q_vv, Q_pv);

        return .{
            .F = F,
            .Q = Q,
            .gravity = config.gravity,
            .dt = dt,
        };
    }

    /// Create dynamics matrices for contact mode with continuous physics params
    pub fn forContactModeWithParams(
        contact_mode: ContactMode,
        physics_params: PhysicsParams,
        config: PhysicsConfig,
        contact_normal: Vec3,
    ) DynamicsMatrices6D {
        var matrices = forPhysicsParams(physics_params, config);

        switch (contact_mode) {
            .free => {
                // Standard free-flight dynamics
            },
            .environment => {
                // Contact with static geometry (ground, walls)
                // Constrain velocity perpendicular to contact normal
                matrices.F.F_vv = applyConstraint(matrices.F.F_vv, contact_normal);
                // Reduce process noise for stability
                matrices.Q = matrices.Q.scale(0.5);
            },
            .entity => {
                // Entity-entity contact: handled separately
            },
            .agency => {
                // Self-propelled motion: higher noise
                matrices.Q = matrices.Q.scale(3.0);
            },
        }

        return matrices;
    }
};

/// Kalman filter predict step for 6D coupled state
/// Updates both mean and covariance using full state transition
pub fn kalmanPredict6D(state: Gaussian6D, m: DynamicsMatrices6D) Gaussian6D {
    const pos = state.position();
    const vel = state.velocity();

    // Mean prediction:
    // p' = F_pp·p + F_pv·v + 0.5·g·dt²
    // v' = F_vp·p + F_vv·v + g·dt
    const new_pos = m.F.F_pp.mulVec(pos)
        .add(m.F.F_pv.mulVec(vel))
        .add(m.gravity.scale(0.5 * m.dt * m.dt));

    const new_vel = m.F.F_vp.mulVec(pos)
        .add(m.F.F_vv.mulVec(vel))
        .add(m.gravity.scale(m.dt));

    // Covariance prediction:
    // P' = F·P·F' + Q
    // Using block structure:
    // P'_pp = F_pp·P_pp·F_pp' + F_pv·P_vp·F_pp' + F_pp·P_pv·F_pv' + F_pv·P_vv·F_pv' + Q_pp
    // P'_pv = F_pp·P_pp·F_vp' + F_pv·P_vp·F_vp' + F_pp·P_pv·F_vv' + F_pv·P_vv·F_vv' + Q_pv
    // P'_vv = F_vp·P_pp·F_vp' + F_vv·P_vp·F_vp' + F_vp·P_pv·F_vv' + F_vv·P_vv·F_vv' + Q_vv

    const P_pp = state.positionCov();
    const P_vv = state.velocityCov();
    const P_pv = state.crossCov();
    const P_vp = P_pv.transpose();

    // P'_pp (simplified since F_vp = 0)
    // = F_pp·P_pp·F_pp' + F_pv·P_vp·F_pp' + F_pp·P_pv·F_pv' + F_pv·P_vv·F_pv' + Q_pp
    const term1_pp = m.F.F_pp.mulMat(P_pp).mulMat(m.F.F_pp.transpose());
    const term2_pp = m.F.F_pv.mulMat(P_vp).mulMat(m.F.F_pp.transpose());
    const term3_pp = m.F.F_pp.mulMat(P_pv).mulMat(m.F.F_pv.transpose());
    const term4_pp = m.F.F_pv.mulMat(P_vv).mulMat(m.F.F_pv.transpose());

    // P'_pv (simplified since F_vp = 0)
    // = F_pp·P_pv·F_vv' + F_pv·P_vv·F_vv' + Q_pv
    const term1_pv = m.F.F_pp.mulMat(P_pv).mulMat(m.F.F_vv.transpose());
    const term2_pv = m.F.F_pv.mulMat(P_vv).mulMat(m.F.F_vv.transpose());

    // P'_vv (simplified since F_vp = 0)
    // = F_vv·P_vv·F_vv' + Q_vv
    const term1_vv = m.F.F_vv.mulMat(P_vv).mulMat(m.F.F_vv.transpose());

    // Extract Q blocks
    const Q_pp_tri = m.Q.positionBlock();
    const Q_vv_tri = m.Q.velocityBlock();
    const Q_pv = m.Q.crossBlock();

    // Convert Q blocks to Mat3
    const Q_pp = Mat3{
        .data = .{
            Q_pp_tri[0], Q_pp_tri[1], Q_pp_tri[2],
            Q_pp_tri[1], Q_pp_tri[3], Q_pp_tri[4],
            Q_pp_tri[2], Q_pp_tri[4], Q_pp_tri[5],
        },
    };
    const Q_vv = Mat3{
        .data = .{
            Q_vv_tri[0], Q_vv_tri[1], Q_vv_tri[2],
            Q_vv_tri[1], Q_vv_tri[3], Q_vv_tri[4],
            Q_vv_tri[2], Q_vv_tri[4], Q_vv_tri[5],
        },
    };

    const new_P_pp = term1_pp.add(term2_pp).add(term3_pp).add(term4_pp).add(Q_pp);
    const new_P_pv = term1_pv.add(term2_pv).add(Q_pv);
    const new_P_vv = term1_vv.add(Q_vv);

    var result: Gaussian6D = undefined;
    result.mean = .{ new_pos.x, new_pos.y, new_pos.z, new_vel.x, new_vel.y, new_vel.z };
    result.cov = CovTriangle21.fromBlocks(new_P_pp, new_P_vv, new_P_pv);

    return result;
}

/// Kalman filter update step for 6D state with position observation
/// Position observation updates BOTH position AND velocity via cross-covariance
/// Returns updated state and marginal log-likelihood for particle weighting
pub fn kalmanUpdate6D(
    predicted: Gaussian6D,
    observation: Vec3,
    R: Mat3, // Measurement noise covariance (3x3)
) struct { state: Gaussian6D, log_lik: f32 } {
    const pred_pos = predicted.position();
    const pred_vel = predicted.velocity();
    const P_pp = predicted.positionCov();
    const P_vv = predicted.velocityCov();
    const P_pv = predicted.crossCov();
    const P_vp = P_pv.transpose();

    // Innovation: y = z - H·x̂ = z - p̂
    const innovation = observation.sub(pred_pos);

    // Innovation covariance: S = H·P·H' + R = P_pp + R
    const S = P_pp.add(R);

    // Invert S
    const S_inv = S.inverse() orelse {
        // Singular matrix, return predicted state unchanged
        return .{ .state = predicted, .log_lik = -std.math.inf(f32) };
    };

    // Kalman gains:
    // K_p = P_pp · S⁻¹  (position gain)
    // K_v = P_vp · S⁻¹  (velocity gain - KEY for bounce tracking!)
    const K_p = P_pp.mulMat(S_inv);
    const K_v = P_vp.mulMat(S_inv);

    // Updated means:
    // p' = p̂ + K_p · y
    // v' = v̂ + K_v · y  ← This is the critical update!
    const new_pos = pred_pos.add(K_p.mulVec(innovation));
    const new_vel = pred_vel.add(K_v.mulVec(innovation));

    // Updated covariances (Joseph form for numerical stability):
    // P' = (I - K·H) · P · (I - K·H)' + K·R·K'
    // Simplified for position-only observation:
    // P'_pp = (I - K_p)·P_pp·(I - K_p)' + K_p·R·K_p' = (I - K_p)·P_pp (when using standard form)
    // P'_pv = (I - K_p)·P_pv - K_p·P_pp·K_v' (approximately)
    // P'_vv = P_vv - K_v·P_pp·K_v' (approximately)

    // Using standard Kalman update (simpler, equivalent for well-conditioned matrices):
    // P'_pp = (I - K_p)·P_pp
    // P'_pv = P_pv - K_p·P_pv (since H = [I, 0])
    // P'_vv = P_vv - K_v·S·K_v'
    const I_minus_Kp = Mat3.identity.sub(K_p);
    const new_P_pp = I_minus_Kp.mulMat(P_pp);

    // Cross-covariance update: P'_pv = (I - K_p·H)·P_pv = (I - K_p)·P_pv
    // But also need to account for velocity update: P'_pv = P_pv - K_p·P_pv = (I - K_p)·P_pv
    const new_P_pv = I_minus_Kp.mulMat(P_pv);

    // Velocity covariance update: P'_vv = P_vv - K_v·S·K_v'
    const K_v_S = K_v.mulMat(S);
    const new_P_vv = P_vv.sub(K_v_S.mulMat(K_v.transpose()));

    // Compute marginal log-likelihood: log p(z | prior) = log N(z; p̂, S)
    const mahalanobis = innovation.dot(S_inv.mulVec(innovation));
    const det = S.determinant();
    const log_lik = if (det > 0)
        -0.5 * (3.0 * 1.8378770664093453 + @log(det) + mahalanobis)
    else
        -std.math.inf(f32);

    var result: Gaussian6D = undefined;
    result.mean = .{ new_pos.x, new_pos.y, new_pos.z, new_vel.x, new_vel.y, new_vel.z };
    result.cov = CovTriangle21.fromBlocks(new_P_pp, new_P_vv, new_P_pv);

    return .{ .state = result, .log_lik = log_lik };
}

/// Compute marginal log-likelihood for 6D state position observation
/// Used for particle weight update without modifying state
pub fn kalmanLogLikelihood6D(
    predicted: Gaussian6D,
    observation: Vec3,
    R: Mat3,
) f32 {
    const pred_pos = predicted.position();
    const P_pp = predicted.positionCov();

    // Innovation: y = z - p̂
    const innovation = observation.sub(pred_pos);

    // Innovation covariance: S = P_pp + R
    const S = P_pp.add(R);

    const S_inv = S.inverse() orelse return -std.math.inf(f32);
    const mahalanobis = innovation.dot(S_inv.mulVec(innovation));

    const det = S.determinant();
    if (det <= 0) return -std.math.inf(f32);

    const log_det = @log(det);
    const log_2pi: f32 = 1.8378770664093453;

    return -0.5 * (3.0 * log_2pi + log_det + mahalanobis);
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
    physics_params: PhysicsParams,
    config: PhysicsConfig,
) struct { position: Vec3, velocity: Vec3 } {
    const dt = config.dt;
    const friction = physics_params.friction;

    // Velocity update: v' = (1 - friction * dt) * v + g * dt
    const vel_decay = 1.0 - friction * dt;
    var new_vel = velocity.scale(vel_decay).add(config.gravity.scale(dt));

    // Position update: p' = p + v' * dt
    var new_pos = position.add(new_vel.scale(dt));

    // Environment collision (unified ground/wall handling)
    const ground_height = config.groundHeight();
    if (new_pos.y < ground_height) {
        new_pos.y = ground_height;
        // Reflect and apply elasticity
        if (new_vel.y < 0) {
            new_vel.y = -new_vel.y * physics_params.elasticity;
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
    // Get contact normal (default to up for ground contact)
    const contact_normal = if (config.environment.ground) |g| g.normal else Vec3.unit_y;

    const matrices = DynamicsMatrices.forContactModeWithParams(
        entity.contact_mode,
        entity.physics_params,
        config,
        contact_normal,
    );

    // Kalman predict
    entity.velocity = kalmanPredictVelocity(entity.velocity, matrices);
    entity.position = kalmanPredictPosition(entity.position, entity.velocity, matrices);

    // Add process noise if RNG provided (for generative sampling)
    if (rng) |r| {
        entity.position.mean = entity.position.mean.add(sampleNoise(r, matrices.Q_pos));
        entity.velocity.mean = entity.velocity.mean.add(sampleNoise(r, matrices.Q_vel));
    }

    // Environment collision detection and response (unified ground/wall handling)
    const ground_height = config.groundHeight();
    if (entity.position.mean.y < ground_height) {
        entity.position.mean.y = ground_height;
        entity.contact_mode = .environment;

        if (entity.velocity.mean.y < 0) {
            entity.velocity.mean.y = -entity.velocity.mean.y * entity.physics_params.elasticity;
        }
    } else if (entity.position.mean.y > ground_height + 0.1) {
        // Clear environment contact if sufficiently above
        if (entity.contact_mode == .environment) {
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
// Environment Contact Detection
// =============================================================================

/// Result of environment contact check
pub const ContactResult = struct {
    /// Contact mode (always .environment for environment contacts)
    mode: ContactMode,
    /// Surface normal at contact point
    normal: Vec3,
    /// Penetration depth (positive = inside environment)
    penetration: f32,
    /// Friction coefficient of contacted surface
    friction: f32,
    /// Elasticity (coefficient of restitution) of contacted surface
    elasticity: f32,
};

/// Check collision with environment (O(1) plane test)
/// Returns contact result if collision detected, null otherwise
pub fn checkEnvironmentContact(
    pos: Vec3,
    radius: f32,
    env: types.EnvironmentConfig,
) ?ContactResult {
    if (env.ground) |ground| {
        const dist_to_plane = ground.signedDistance(pos);
        if (dist_to_plane < radius) {
            return ContactResult{
                .mode = .environment,
                .normal = ground.normal,
                .penetration = radius - dist_to_plane,
                .friction = ground.friction,
                .elasticity = ground.elasticity,
            };
        }
    }
    return null;
}

/// Resolve environment contact by correcting position and reflecting velocity
pub fn resolveEnvironmentContact(
    pos: *Vec3,
    vel: *Vec3,
    contact: ContactResult,
) void {
    // 1. Correct position (remove penetration)
    pos.* = pos.add(contact.normal.scale(contact.penetration));

    // 2. Reflect velocity with elasticity if moving into surface
    const vel_along_normal = vel.dot(contact.normal);
    if (vel_along_normal < 0) {
        // Reflect normal component with elasticity
        const impulse = contact.normal.scale(-vel_along_normal * (1.0 + contact.elasticity));
        vel.* = vel.add(impulse);

        // Apply friction to tangential component
        const tangent_vel = vel.sub(contact.normal.scale(vel.dot(contact.normal)));
        const friction_decay = 1.0 - contact.friction * 0.1; // Scale friction effect
        vel.* = contact.normal.scale(vel.dot(contact.normal)).add(tangent_vel.scale(friction_decay));
    }
}

// =============================================================================
// Entity Contact Detection
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
    const e = @min(e1.physics_params.elasticity, e2.physics_params.elasticity);

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
    const matrices = DynamicsMatrices.forPhysicsParams(PhysicsParams.prior, config);

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
    const matrices = DynamicsMatrices.forPhysicsParams(PhysicsParams.prior, config);

    const new_pos = kalmanPredictPosition(pos, vel, matrices);

    // Position should move in x direction
    try testing.expect(new_pos.mean.x > pos.mean.x);
    // Position should drop due to gravity
    try testing.expect(new_pos.mean.y < pos.mean.y);
}

test "Kalman predict velocity" {
    const vel = GaussianVec3.isotropic(Vec3.init(0, 0, 0), 0.01);

    const config = PhysicsConfig{ .gravity = Vec3.init(0, -10, 0), .dt = 0.1 };
    const matrices = DynamicsMatrices.forPhysicsParams(PhysicsParams.prior, config);

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
        PhysicsParams.prior,
        config,
    );

    // Should fall due to gravity
    try testing.expect(result.position.y < 5);
    try testing.expect(result.velocity.y < 0);
}

test "Physics step with environment collision" {
    // Default config has ground at y=0 via environment config
    const config = PhysicsConfig{
        .gravity = Vec3.init(0, -10, 0),
        .dt = 0.1,
    };

    // Start at ground with downward velocity
    // Use high elasticity params for bouncy behavior
    const bouncy_params = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 };
    const result = physicsStepPoint(
        Vec3.init(0, 0.05, 0),
        Vec3.init(0, -5, 0),
        bouncy_params,
        config,
    );

    // Should bounce (velocity should be positive after hitting ground)
    try testing.expect(result.velocity.y > 0);
    try testing.expect(result.position.y >= 0);
}

test "Entity contact detection" {
    const label1 = types.Label{ .birth_time = 0, .birth_index = 0 };
    const label2 = types.Label{ .birth_time = 0, .birth_index = 1 };

    var e1 = Entity.initPoint(label1, Vec3.init(0, 0, 0), Vec3.zero, PhysicsParams.prior);
    var e2 = Entity.initPoint(label2, Vec3.init(0.5, 0, 0), Vec3.zero, PhysicsParams.prior);

    e1.appearance.radius = 0.5;
    e2.appearance.radius = 0.5;

    // Should be in contact (distance 0.5, combined radius 1.0)
    try testing.expect(checkEntityContact(e1, e2));

    // Move apart
    e2.position.mean = Vec3.init(2, 0, 0);
    try testing.expect(!checkEntityContact(e1, e2));
}

test "Environment contact detection" {
    const env = types.EnvironmentConfig{};

    // Point above ground should not have contact
    const no_contact = checkEnvironmentContact(Vec3.init(0, 1, 0), 0.5, env);
    try testing.expect(no_contact == null);

    // Point at ground level should have contact (within radius)
    const contact = checkEnvironmentContact(Vec3.init(0, 0.3, 0), 0.5, env);
    try testing.expect(contact != null);
    try testing.expect(contact.?.mode == .environment);
    try testing.expect(contact.?.normal.y == 1.0);
    try testing.expect(contact.?.penetration > 0);

    // Point below ground should have contact
    const below_contact = checkEnvironmentContact(Vec3.init(0, -0.5, 0), 0.5, env);
    try testing.expect(below_contact != null);
    try testing.expect(below_contact.?.penetration > contact.?.penetration);
}

test "Environment contact with no ground" {
    // No ground configured = freefall world
    const env = types.EnvironmentConfig{ .ground = null };

    // Even point below y=0 should not have contact
    const no_contact = checkEnvironmentContact(Vec3.init(0, -10, 0), 0.5, env);
    try testing.expect(no_contact == null);
}

test "DynamicsMatrices6D creation" {
    const config = PhysicsConfig{};
    const matrices = DynamicsMatrices6D.forPhysicsParams(PhysicsParams.prior, config);

    // F_pp should be identity
    try testing.expect(matrices.F.F_pp.approxEql(Mat3.identity, 1e-6));

    // F_pv should scale by dt
    const expected_F_pv = Mat3.scale(config.dt);
    try testing.expect(matrices.F.F_pv.approxEql(expected_F_pv, 1e-6));

    // F_vp should be zero
    try testing.expect(matrices.F.F_vp.approxEql(Mat3.zero, 1e-6));

    // Q should have non-zero diagonal
    try testing.expect(matrices.Q.get(0, 0) > 0);
    try testing.expect(matrices.Q.get(3, 3) > 0);
}

test "6D Kalman predict updates cross-covariance" {
    const config = PhysicsConfig{ .gravity = Vec3.init(0, -10, 0), .dt = 0.1 };
    const matrices = DynamicsMatrices6D.forPhysicsParams(PhysicsParams.prior, config);

    // Start with diagonal (no cross-covariance)
    const initial = Gaussian6D.isotropic(
        Vec3.init(0, 10, 0), // position
        0.1, // pos variance
        Vec3.init(1, 0, 0), // velocity
        0.01, // vel variance
    );

    const predicted = kalmanPredict6D(initial, matrices);

    // Position should move in x direction (from velocity)
    try testing.expect(predicted.position().x > initial.position().x);

    // Position should drop due to gravity
    try testing.expect(predicted.position().y < initial.position().y);

    // Velocity should increase downward due to gravity
    try testing.expect(predicted.velocity().y < 0);

    // Cross-covariance should be non-zero after prediction (from Q_pv)
    const cross = predicted.crossCov();
    // The diagonal of cross should be non-zero
    try testing.expect(cross.get(0, 0) > 0);
}

test "6D Kalman update propagates to velocity" {
    // Create state with non-zero cross-covariance
    // This simulates a state where position and velocity are correlated
    var initial: Gaussian6D = undefined;
    initial.mean = .{ 0, 0, 0, 0, 0, 0 }; // Both at origin

    // Set up covariance with correlation between position and velocity
    // When P_pv is non-zero, position observations will update velocity
    const P_pp = Mat3.scale(1.0);
    const P_vv = Mat3.scale(0.1);
    // Strong positive correlation between position and velocity
    // (moving rightward and being observed to the right are correlated)
    const P_pv = Mat3.scale(0.3);
    initial.cov = CovTriangle21.fromBlocks(P_pp, P_vv, P_pv);

    // Observation is to the right
    const observation = Vec3.init(1, 0, 0);
    const R = Mat3.scale(0.1); // Low measurement noise

    const result = kalmanUpdate6D(initial, observation, R);

    // Position should move toward observation
    try testing.expect(result.state.position().x > 0);

    // Velocity should ALSO update (the key feature!)
    // Because position was observed to the right, and there's positive
    // cross-covariance, velocity should be inferred to be rightward
    try testing.expect(result.state.velocity().x > 0);

    // Log-likelihood should be finite
    try testing.expect(!std.math.isInf(result.log_lik));
}

test "6D Kalman log-likelihood" {
    const state = Gaussian6D.isotropic(
        Vec3.init(0, 0, 0),
        1.0,
        Vec3.init(0, 0, 0),
        0.1,
    );

    // Observation at mean should have highest likelihood
    const ll_at_mean = kalmanLogLikelihood6D(state, Vec3.zero, Mat3.scale(0.1));

    // Observation far from mean should have lower likelihood
    const ll_far = kalmanLogLikelihood6D(state, Vec3.init(10, 0, 0), Mat3.scale(0.1));

    try testing.expect(ll_at_mean > ll_far);
    try testing.expect(!std.math.isInf(ll_at_mean));
}
