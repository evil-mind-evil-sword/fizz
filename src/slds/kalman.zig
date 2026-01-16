const std = @import("std");
const math = @import("../math.zig");
const Vec3 = math.Vec3;
const Mat3 = math.Mat3;

const dynamics = @import("dynamics.zig");
const Mat6 = dynamics.Mat6;
const SLDSMatrices = dynamics.SLDSMatrices;

const ecs = @import("../ecs/mod.zig");
const Position = ecs.Position;
const Velocity = ecs.Velocity;

// =============================================================================
// State Vector Operations
// =============================================================================

/// Combined state: [position, velocity]
pub const State6 = struct {
    data: [6]f32,

    pub fn fromPosVel(pos: Vec3, vel: Vec3) State6 {
        return .{ .data = .{ pos.x, pos.y, pos.z, vel.x, vel.y, vel.z } };
    }

    pub fn position(self: State6) Vec3 {
        return Vec3.init(self.data[0], self.data[1], self.data[2]);
    }

    pub fn velocity(self: State6) Vec3 {
        return Vec3.init(self.data[3], self.data[4], self.data[5]);
    }

    pub fn add(self: State6, other: State6) State6 {
        var result: State6 = undefined;
        for (0..6) |i| {
            result.data[i] = self.data[i] + other.data[i];
        }
        return result;
    }

    pub fn addArray(self: State6, arr: [6]f32) State6 {
        var result: State6 = undefined;
        for (0..6) |i| {
            result.data[i] = self.data[i] + arr[i];
        }
        return result;
    }
};

/// 6x6 covariance matrix
pub const Cov6 = struct {
    data: [6][6]f32,

    pub fn fromMat6(m: Mat6) Cov6 {
        return .{ .data = m.data };
    }

    pub fn toMat6(self: Cov6) Mat6 {
        return .{ .data = self.data };
    }

    pub fn zero() Cov6 {
        return .{ .data = std.mem.zeroes([6][6]f32) };
    }

    pub fn diagonal(pos_var: Vec3, vel_var: Vec3) Cov6 {
        var c = zero();
        c.data[0][0] = pos_var.x;
        c.data[1][1] = pos_var.y;
        c.data[2][2] = pos_var.z;
        c.data[3][3] = vel_var.x;
        c.data[4][4] = vel_var.y;
        c.data[5][5] = vel_var.z;
        return c;
    }

    /// Extract position covariance (3x3 upper-left)
    pub fn positionCov(self: Cov6) Mat3 {
        var m = Mat3.zero;
        for (0..3) |i| {
            for (0..3) |j| {
                m.set(i, j, self.data[i][j]);
            }
        }
        return m;
    }

    /// Extract velocity covariance (3x3 lower-right)
    pub fn velocityCov(self: Cov6) Mat3 {
        var m = Mat3.zero;
        for (0..3) |i| {
            for (0..3) |j| {
                m.set(i, j, self.data[i + 3][j + 3]);
            }
        }
        return m;
    }

    pub fn get(self: Cov6, row: usize, col: usize) f32 {
        return self.data[row][col];
    }
};

// =============================================================================
// Kalman Filter Operations
// =============================================================================

/// Kalman filter state
pub const KalmanState = struct {
    mean: State6,
    cov: Cov6,

    pub fn fromComponents(pos: Position, vel: Velocity) KalmanState {
        return .{
            .mean = State6.fromPosVel(pos.mean, vel.mean),
            .cov = Cov6.diagonal(
                Vec3.init(pos.cov.get(0, 0), pos.cov.get(1, 1), pos.cov.get(2, 2)),
                Vec3.init(vel.cov.get(0, 0), vel.cov.get(1, 1), vel.cov.get(2, 2)),
            ),
        };
    }

    pub fn toComponents(self: KalmanState) struct { pos: Position, vel: Velocity } {
        return .{
            .pos = .{
                .mean = self.mean.position(),
                .cov = self.cov.positionCov(),
            },
            .vel = .{
                .mean = self.mean.velocity(),
                .cov = self.cov.velocityCov(),
            },
        };
    }
};

/// Kalman predict step
/// x_{t+1|t} = A * x_{t|t} + b
/// P_{t+1|t} = A * P_{t|t} * A^T + Q
pub fn kalmanPredict(state: KalmanState, matrices: SLDSMatrices) KalmanState {
    // Mean prediction
    const ax = State6{ .data = matrices.A.mulVec(state.mean.data) };
    const predicted_mean = ax.addArray(matrices.b);

    // Covariance prediction: P = A * P * A^T + Q
    // For simplicity, use diagonal approximation
    var predicted_cov = Cov6.zero();
    for (0..6) |i| {
        for (0..6) |j| {
            var sum: f32 = 0;
            for (0..6) |k| {
                for (0..6) |l| {
                    sum += matrices.A.data[i][k] * state.cov.data[k][l] * matrices.A.data[j][l];
                }
            }
            predicted_cov.data[i][j] = sum + matrices.Q.data[i][j];
        }
    }

    return .{
        .mean = predicted_mean,
        .cov = predicted_cov,
    };
}

/// Kalman update step with position observation
/// Uses simple position-only observation model: z = H * x + v
/// where H selects position components
pub fn kalmanUpdate(state: KalmanState, observed_pos: Vec3, obs_noise: f32) KalmanState {
    const z = [3]f32{ observed_pos.x, observed_pos.y, observed_pos.z };

    // Innovation: y = z - H * x (H selects first 3 components)
    const y = [3]f32{
        z[0] - state.mean.data[0],
        z[1] - state.mean.data[1],
        z[2] - state.mean.data[2],
    };

    // Innovation covariance: S = H * P * H^T + R
    // For diagonal R and position-only H, S is 3x3
    var S: [3][3]f32 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            S[i][j] = state.cov.data[i][j];
            if (i == j) S[i][j] += obs_noise;
        }
    }

    // Invert S (3x3)
    const S_inv = invert3x3(S) orelse return state; // Return unchanged if singular

    // Kalman gain: K = P * H^T * S^-1
    // K is 6x3
    var K: [6][3]f32 = undefined;
    for (0..6) |i| {
        for (0..3) |j| {
            var sum: f32 = 0;
            for (0..3) |k| {
                sum += state.cov.data[i][k] * S_inv[k][j];
            }
            K[i][j] = sum;
        }
    }

    // Update mean: x = x + K * y
    var updated_mean = state.mean;
    for (0..6) |i| {
        for (0..3) |j| {
            updated_mean.data[i] += K[i][j] * y[j];
        }
    }

    // Update covariance: P = (I - K*H) * P
    // K*H is 6x6 with non-zero only in first 3 columns
    var updated_cov = state.cov;
    for (0..6) |i| {
        for (0..6) |j| {
            var correction: f32 = 0;
            for (0..3) |k| {
                correction += K[i][k] * state.cov.data[k][j];
            }
            updated_cov.data[i][j] = state.cov.data[i][j] - correction;
        }
    }

    return .{
        .mean = updated_mean,
        .cov = updated_cov,
    };
}

/// Compute log likelihood of observation
pub fn kalmanLogLikelihood(state: KalmanState, observed_pos: Vec3, obs_noise: f32) f32 {
    // Innovation
    const y = Vec3.init(
        observed_pos.x - state.mean.data[0],
        observed_pos.y - state.mean.data[1],
        observed_pos.z - state.mean.data[2],
    );

    // Innovation covariance (position block + observation noise)
    var S = state.cov.positionCov();
    S.set(0, 0, S.get(0, 0) + obs_noise);
    S.set(1, 1, S.get(1, 1) + obs_noise);
    S.set(2, 2, S.get(2, 2) + obs_noise);

    // Mahalanobis distance
    const S_inv = S.inverse() orelse return -std.math.inf(f32);
    const mahal = y.dot(S_inv.mulVec(y));

    // Log determinant
    const det = S.determinant();
    if (det <= 0) return -std.math.inf(f32);

    const log_det = @log(det);
    const log_2pi: f32 = 1.8378770664093453;

    return -0.5 * (3.0 * log_2pi + log_det + mahal);
}

/// Invert 3x3 matrix
fn invert3x3(m: [3][3]f32) ?[3][3]f32 {
    // Compute determinant
    const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (@abs(det) < 1e-10) return null;

    const inv_det = 1.0 / det;

    return [3][3]f32{
        .{
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        },
        .{
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        },
        .{
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        },
    };
}

// =============================================================================
// Tests
// =============================================================================

test "State6 conversions" {
    const pos = Vec3.init(1, 2, 3);
    const vel = Vec3.init(4, 5, 6);

    const state = State6.fromPosVel(pos, vel);
    try std.testing.expect(state.position().approxEql(pos, 1e-6));
    try std.testing.expect(state.velocity().approxEql(vel, 1e-6));
}

test "Kalman predict" {
    const pos = Position.point(Vec3.init(0, 5, 0));
    const vel = Velocity.point(Vec3.init(0, 0, 0));
    const state = KalmanState.fromComponents(pos, vel);

    const config = dynamics.PhysicsConfig{};
    const matrices = SLDSMatrices.forMode(.free, ecs.Physics.standard, config);

    const predicted = kalmanPredict(state, matrices);

    // Position should change due to gravity
    // Velocity should become negative due to gravity
    try std.testing.expect(predicted.mean.data[4] < 0); // vy < 0
}

test "Kalman update" {
    const pos = Position.isotropic(Vec3.init(0, 5, 0), 1.0);
    const vel = Velocity.point(Vec3.zero);
    const state = KalmanState.fromComponents(pos, vel);

    // Observe at slightly different position
    const obs = Vec3.init(0.1, 5.1, 0.1);
    const updated = kalmanUpdate(state, obs, 0.1);

    // Updated position should be pulled toward observation
    try std.testing.expect(updated.mean.data[0] > 0); // x pulled toward 0.1
    try std.testing.expect(updated.mean.data[1] > 5); // y pulled toward 5.1
}
