const std = @import("std");
const math = @import("../math.zig");
const Vec3 = math.Vec3;
const Mat3 = math.Mat3;

const ecs = @import("../ecs/mod.zig");
const ContactMode = ecs.ContactMode;
const Physics = ecs.Physics;

// =============================================================================
// Physics Configuration
// =============================================================================

/// Global physics parameters
pub const PhysicsConfig = struct {
    /// Gravity vector
    gravity: Vec3 = Vec3.init(0, -9.81, 0),
    /// Simulation timestep
    dt: f32 = 1.0 / 60.0,
    /// Ground plane height
    ground_height: f32 = 0.0,
    /// World bounds
    bounds_min: Vec3 = Vec3.init(-10, -10, -10),
    bounds_max: Vec3 = Vec3.init(10, 10, 10),
    /// Default process noise
    process_noise: f32 = 0.01,
    /// CRP concentration parameter
    crp_alpha: f32 = 1.0,
    /// Survival probability per timestep
    survival_prob: f32 = 0.99,
    /// Contact detection threshold
    contact_threshold: f32 = 0.1,
    /// Stability snap probability (for islands of stability)
    stability_snap_prob: f32 = 0.1,
    /// Speed threshold for stability
    stability_speed_threshold: f32 = 0.1,
};

// =============================================================================
// SLDS Dynamics Matrices
// =============================================================================

/// 6x6 matrix for (position, velocity) state space
pub const Mat6 = struct {
    data: [6][6]f32,

    pub fn zero() Mat6 {
        return .{ .data = std.mem.zeroes([6][6]f32) };
    }

    pub fn identity() Mat6 {
        var m = zero();
        for (0..6) |i| {
            m.data[i][i] = 1.0;
        }
        return m;
    }

    pub fn get(self: Mat6, row: usize, col: usize) f32 {
        return self.data[row][col];
    }

    pub fn set(self: *Mat6, row: usize, col: usize, val: f32) void {
        self.data[row][col] = val;
    }

    /// Multiply 6x6 matrix by 6-vector
    pub fn mulVec(self: Mat6, v: [6]f32) [6]f32 {
        var result: [6]f32 = undefined;
        for (0..6) |i| {
            var sum: f32 = 0;
            for (0..6) |j| {
                sum += self.data[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /// Standard position-velocity transition matrix
    /// x_{t+1} = x_t + dt * v_t
    /// v_{t+1} = damping * v_t
    pub fn posVelTransition(dt: f32, damping: f32) Mat6 {
        var m = identity();
        // Position rows: x += dt * v
        m.data[0][3] = dt; // x += dt * vx
        m.data[1][4] = dt; // y += dt * vy
        m.data[2][5] = dt; // z += dt * vz
        // Velocity rows: v *= damping
        m.data[3][3] = damping;
        m.data[4][4] = damping;
        m.data[5][5] = damping;
        return m;
    }

    /// Constrained transition (e.g., on ground plane)
    /// Velocity in normal direction is zeroed
    pub fn constrainedTransition(dt: f32, damping: f32, normal: Vec3) Mat6 {
        var m = posVelTransition(dt, damping);

        // Zero out velocity component along normal
        // For ground (y-up), zero vy
        if (@abs(normal.y) > 0.9) {
            m.data[4][4] = 0; // vy stays zero
            m.data[1][4] = 0; // y doesn't change from vy
        }

        return m;
    }

    /// Process noise covariance
    pub fn processNoise(base_noise: f32, dt: f32) Mat6 {
        var m = zero();
        const pos_noise = base_noise * dt * dt;
        const vel_noise = base_noise * dt;

        // Position noise
        m.data[0][0] = pos_noise;
        m.data[1][1] = pos_noise;
        m.data[2][2] = pos_noise;

        // Velocity noise
        m.data[3][3] = vel_noise;
        m.data[4][4] = vel_noise;
        m.data[5][5] = vel_noise;

        return m;
    }

    /// Anisotropic process noise (different in each direction)
    pub fn anisotropicNoise(pos_noise: Vec3, vel_noise: Vec3) Mat6 {
        var m = zero();
        m.data[0][0] = pos_noise.x;
        m.data[1][1] = pos_noise.y;
        m.data[2][2] = pos_noise.z;
        m.data[3][3] = vel_noise.x;
        m.data[4][4] = vel_noise.y;
        m.data[5][5] = vel_noise.z;
        return m;
    }
};

/// SLDS matrices for a specific mode
pub const SLDSMatrices = struct {
    /// State transition: x_{t+1} = A * x_t + b
    A: Mat6,
    /// Constant term (gravity, etc.)
    b: [6]f32,
    /// Process noise covariance
    Q: Mat6,

    /// Get matrices for a specific contact mode
    pub fn forMode(
        mode: ContactMode,
        physics: Physics,
        config: PhysicsConfig,
    ) SLDSMatrices {
        return switch (mode) {
            .free => freeDynamics(physics, config),
            .ground => groundDynamics(physics, config),
            .supported => supportedDynamics(physics, config),
            .attached => attachedDynamics(physics, config),
            .agency => agencyDynamics(physics, config),
        };
    }
};

/// Free flight dynamics - standard Newtonian with gravity
fn freeDynamics(physics: Physics, config: PhysicsConfig) SLDSMatrices {
    const dt = config.dt;
    const damping = 1.0 - physics.friction * dt;

    return .{
        .A = Mat6.posVelTransition(dt, damping),
        .b = .{
            0, 0, 0, // position constant term
            config.gravity.x * dt,
            config.gravity.y * dt,
            config.gravity.z * dt, // velocity += gravity * dt
        },
        .Q = Mat6.processNoise(physics.process_noise, dt),
    };
}

/// Ground contact dynamics - constrained to ground plane
/// Key Spelke principle: "Islands of stability" - low noise when supported
fn groundDynamics(physics: Physics, config: PhysicsConfig) SLDSMatrices {
    const dt = config.dt;
    const damping = 1.0 - physics.friction * dt * 5.0; // Higher friction on ground

    return .{
        .A = Mat6.constrainedTransition(dt, damping, Vec3.unit_y),
        .b = .{ 0, 0, 0, 0, 0, 0 }, // No gravity effect (supported)
        // Very low noise = stable rest state (10x lower than free mode)
        .Q = Mat6.processNoise(0.001 * dt, dt), // Much lower than free mode's 0.01
    };
}

/// Supported dynamics - resting on another entity
/// Even more stable than ground (inherited motion)
fn supportedDynamics(physics: Physics, config: PhysicsConfig) SLDSMatrices {
    _ = physics;
    const dt = config.dt;

    return .{
        .A = Mat6.constrainedTransition(dt, 0.9, Vec3.unit_y),
        .b = .{ 0, 0, 0, 0, 0, 0 },
        // Minimal noise - stable stacking
        .Q = Mat6.anisotropicNoise(
            Vec3.init(0.00001, 0.00001, 0.00001),
            Vec3.init(0.0001, 0.0001, 0.0001),
        ),
    };
}

/// Attached dynamics - stuck to surface
/// Near-zero motion relative to attachment
fn attachedDynamics(physics: Physics, config: PhysicsConfig) SLDSMatrices {
    _ = physics;
    _ = config;

    var A = Mat6.identity();
    // Almost no motion
    for (3..6) |i| {
        A.data[i][i] = 0.01; // Velocity decays rapidly
    }

    return .{
        .A = A,
        .b = .{ 0, 0, 0, 0, 0, 0 },
        // Nearly zero noise
        .Q = Mat6.anisotropicNoise(
            Vec3.splat(0.000001),
            Vec3.splat(0.00001),
        ),
    };
}

/// Agency dynamics - self-propelled motion
/// Spelke principle: Agents are less predictable
fn agencyDynamics(physics: Physics, config: PhysicsConfig) SLDSMatrices {
    const dt = config.dt;
    const damping = 1.0 - physics.friction * dt * 0.5; // Less friction for agents

    return .{
        .A = Mat6.posVelTransition(dt, damping),
        .b = .{ 0, 0, 0, 0, 0, 0 }, // No external forces (self-propelled)
        // High noise - unpredictable motion
        .Q = Mat6.processNoise(physics.process_noise * 3.0, dt),
    };
}

// =============================================================================
// Tests
// =============================================================================

test "Mat6 basic operations" {
    const m = Mat6.identity();
    try std.testing.expectEqual(@as(f32, 1), m.get(0, 0));
    try std.testing.expectEqual(@as(f32, 0), m.get(0, 1));

    const v = m.mulVec(.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expectEqual(@as(f32, 1), v[0]);
    try std.testing.expectEqual(@as(f32, 6), v[5]);
}

test "SLDS matrices per mode" {
    const config = PhysicsConfig{};
    const physics = Physics.standard;

    const free_m = SLDSMatrices.forMode(.free, physics, config);
    const ground_m = SLDSMatrices.forMode(.ground, physics, config);

    // Free mode should have gravity in b
    try std.testing.expect(free_m.b[4] < 0); // Negative y gravity

    // Ground mode should have lower noise
    try std.testing.expect(ground_m.Q.get(0, 0) < free_m.Q.get(0, 0));
}
