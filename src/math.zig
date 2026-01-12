const std = @import("std");
const testing = std.testing;

/// 3D vector with f32 components
pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub const zero = Vec3{ .x = 0, .y = 0, .z = 0 };
    pub const one = Vec3{ .x = 1, .y = 1, .z = 1 };
    pub const unit_x = Vec3{ .x = 1, .y = 0, .z = 0 };
    pub const unit_y = Vec3{ .x = 0, .y = 1, .z = 0 };
    pub const unit_z = Vec3{ .x = 0, .y = 0, .z = 1 };

    pub fn init(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn splat(v: f32) Vec3 {
        return .{ .x = v, .y = v, .z = v };
    }

    pub fn add(self: Vec3, other: Vec3) Vec3 {
        return .{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }

    pub fn sub(self: Vec3, other: Vec3) Vec3 {
        return .{
            .x = self.x - other.x,
            .y = self.y - other.y,
            .z = self.z - other.z,
        };
    }

    pub fn mul(self: Vec3, other: Vec3) Vec3 {
        return .{
            .x = self.x * other.x,
            .y = self.y * other.y,
            .z = self.z * other.z,
        };
    }

    pub fn scale(self: Vec3, s: f32) Vec3 {
        return .{
            .x = self.x * s,
            .y = self.y * s,
            .z = self.z * s,
        };
    }

    pub fn div(self: Vec3, s: f32) Vec3 {
        const inv = 1.0 / s;
        return self.scale(inv);
    }

    pub fn negate(self: Vec3) Vec3 {
        return .{
            .x = -self.x,
            .y = -self.y,
            .z = -self.z,
        };
    }

    pub fn dot(self: Vec3, other: Vec3) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    pub fn cross(self: Vec3, other: Vec3) Vec3 {
        return .{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    pub fn lengthSquared(self: Vec3) f32 {
        return self.dot(self);
    }

    pub fn length(self: Vec3) f32 {
        return @sqrt(self.lengthSquared());
    }

    pub fn normalize(self: Vec3) Vec3 {
        const len = self.length();
        if (len < 1e-10) return Vec3.zero;
        return self.scale(1.0 / len);
    }

    pub fn distance(self: Vec3, other: Vec3) f32 {
        return self.sub(other).length();
    }

    pub fn distanceSquared(self: Vec3, other: Vec3) f32 {
        return self.sub(other).lengthSquared();
    }

    pub fn lerp(self: Vec3, other: Vec3, t: f32) Vec3 {
        return self.add(other.sub(self).scale(t));
    }

    pub fn reflect(self: Vec3, normal: Vec3) Vec3 {
        // v - 2 * (v . n) * n
        const d = 2.0 * self.dot(normal);
        return self.sub(normal.scale(d));
    }

    pub fn clamp(self: Vec3, min_val: f32, max_val: f32) Vec3 {
        return .{
            .x = @max(min_val, @min(max_val, self.x)),
            .y = @max(min_val, @min(max_val, self.y)),
            .z = @max(min_val, @min(max_val, self.z)),
        };
    }

    pub fn abs(self: Vec3) Vec3 {
        return .{
            .x = @abs(self.x),
            .y = @abs(self.y),
            .z = @abs(self.z),
        };
    }

    pub fn eql(self: Vec3, other: Vec3) bool {
        return self.x == other.x and self.y == other.y and self.z == other.z;
    }

    pub fn approxEql(self: Vec3, other: Vec3, epsilon: f32) bool {
        return @abs(self.x - other.x) < epsilon and
            @abs(self.y - other.y) < epsilon and
            @abs(self.z - other.z) < epsilon;
    }

    /// Convert to array for interop
    pub fn toArray(self: Vec3) [3]f32 {
        return .{ self.x, self.y, self.z };
    }

    pub fn fromArray(arr: [3]f32) Vec3 {
        return .{ .x = arr[0], .y = arr[1], .z = arr[2] };
    }
};

/// 3x3 matrix stored in column-major order (for OpenGL compatibility)
/// Columns are: [col0, col1, col2]
pub const Mat3 = struct {
    data: [9]f32,

    pub const identity = Mat3{
        .data = .{
            1, 0, 0, // col 0
            0, 1, 0, // col 1
            0, 0, 1, // col 2
        },
    };

    pub const zero = Mat3{ .data = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

    /// Create from column vectors
    pub fn fromColumns(c0: Vec3, c1: Vec3, c2: Vec3) Mat3 {
        return .{
            .data = .{
                c0.x, c0.y, c0.z,
                c1.x, c1.y, c1.z,
                c2.x, c2.y, c2.z,
            },
        };
    }

    /// Create from row vectors
    pub fn fromRows(r0: Vec3, r1: Vec3, r2: Vec3) Mat3 {
        return .{
            .data = .{
                r0.x, r1.x, r2.x,
                r0.y, r1.y, r2.y,
                r0.z, r1.z, r2.z,
            },
        };
    }

    /// Create diagonal matrix
    pub fn diagonal(d: Vec3) Mat3 {
        return .{
            .data = .{
                d.x, 0,   0,
                0,   d.y, 0,
                0,   0,   d.z,
            },
        };
    }

    /// Create uniform scale matrix
    pub fn scale(s: f32) Mat3 {
        return diagonal(Vec3.splat(s));
    }

    /// Get element at row r, column c
    pub fn get(self: Mat3, r: usize, c: usize) f32 {
        return self.data[c * 3 + r];
    }

    /// Set element at row r, column c
    pub fn set(self: *Mat3, r: usize, c: usize, val: f32) void {
        self.data[c * 3 + r] = val;
    }

    /// Get column as Vec3
    pub fn col(self: Mat3, c: usize) Vec3 {
        const base = c * 3;
        return .{
            .x = self.data[base],
            .y = self.data[base + 1],
            .z = self.data[base + 2],
        };
    }

    /// Get row as Vec3
    pub fn row(self: Mat3, r: usize) Vec3 {
        return .{
            .x = self.data[r],
            .y = self.data[r + 3],
            .z = self.data[r + 6],
        };
    }

    /// Matrix-vector multiplication: M * v
    pub fn mulVec(self: Mat3, v: Vec3) Vec3 {
        return .{
            .x = self.row(0).dot(v),
            .y = self.row(1).dot(v),
            .z = self.row(2).dot(v),
        };
    }

    /// Matrix-matrix multiplication: A * B
    pub fn mulMat(self: Mat3, other: Mat3) Mat3 {
        var result: Mat3 = undefined;
        inline for (0..3) |c| {
            const col_vec = other.col(c);
            inline for (0..3) |r| {
                result.data[c * 3 + r] = self.row(r).dot(col_vec);
            }
        }
        return result;
    }

    /// Scalar multiplication
    pub fn scaleMat(self: Mat3, s: f32) Mat3 {
        var result: Mat3 = undefined;
        inline for (0..9) |i| {
            result.data[i] = self.data[i] * s;
        }
        return result;
    }

    /// Matrix addition
    pub fn add(self: Mat3, other: Mat3) Mat3 {
        var result: Mat3 = undefined;
        inline for (0..9) |i| {
            result.data[i] = self.data[i] + other.data[i];
        }
        return result;
    }

    /// Matrix subtraction
    pub fn sub(self: Mat3, other: Mat3) Mat3 {
        var result: Mat3 = undefined;
        inline for (0..9) |i| {
            result.data[i] = self.data[i] - other.data[i];
        }
        return result;
    }

    /// Transpose
    pub fn transpose(self: Mat3) Mat3 {
        return .{
            .data = .{
                self.data[0], self.data[3], self.data[6],
                self.data[1], self.data[4], self.data[7],
                self.data[2], self.data[5], self.data[8],
            },
        };
    }

    /// Determinant
    pub fn determinant(self: Mat3) f32 {
        const a = self.data[0];
        const b = self.data[3];
        const c = self.data[6];
        const d = self.data[1];
        const e = self.data[4];
        const f = self.data[7];
        const g = self.data[2];
        const h = self.data[5];
        const i = self.data[8];

        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    /// Inverse (returns null if singular)
    pub fn inverse(self: Mat3) ?Mat3 {
        const det = self.determinant();
        if (@abs(det) < 1e-10) return null;

        const inv_det = 1.0 / det;

        const a = self.data[0];
        const b = self.data[3];
        const c = self.data[6];
        const d = self.data[1];
        const e = self.data[4];
        const f = self.data[7];
        const g = self.data[2];
        const h = self.data[5];
        const i = self.data[8];

        return Mat3{
            .data = .{
                (e * i - f * h) * inv_det,
                (f * g - d * i) * inv_det,
                (d * h - e * g) * inv_det,
                (c * h - b * i) * inv_det,
                (a * i - c * g) * inv_det,
                (b * g - a * h) * inv_det,
                (b * f - c * e) * inv_det,
                (c * d - a * f) * inv_det,
                (a * e - b * d) * inv_det,
            },
        };
    }

    /// Trace (sum of diagonal elements)
    pub fn trace(self: Mat3) f32 {
        return self.data[0] + self.data[4] + self.data[8];
    }

    /// Outer product: v1 * v2^T
    pub fn outer(v1: Vec3, v2: Vec3) Mat3 {
        return .{
            .data = .{
                v1.x * v2.x, v1.y * v2.x, v1.z * v2.x,
                v1.x * v2.y, v1.y * v2.y, v1.z * v2.y,
                v1.x * v2.z, v1.y * v2.z, v1.z * v2.z,
            },
        };
    }

    /// Check if approximately equal
    pub fn approxEql(self: Mat3, other: Mat3, epsilon: f32) bool {
        inline for (0..9) |i| {
            if (@abs(self.data[i] - other.data[i]) >= epsilon) return false;
        }
        return true;
    }
};

/// 2D vector for screen-space operations
pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub const zero = Vec2{ .x = 0, .y = 0 };

    pub fn init(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }

    pub fn add(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x + other.x, .y = self.y + other.y };
    }

    pub fn sub(self: Vec2, other: Vec2) Vec2 {
        return .{ .x = self.x - other.x, .y = self.y - other.y };
    }

    pub fn scale(self: Vec2, s: f32) Vec2 {
        return .{ .x = self.x * s, .y = self.y * s };
    }

    pub fn dot(self: Vec2, other: Vec2) f32 {
        return self.x * other.x + self.y * other.y;
    }

    pub fn length(self: Vec2) f32 {
        return @sqrt(self.dot(self));
    }

    pub fn normalize(self: Vec2) Vec2 {
        const len = self.length();
        if (len < 1e-10) return Vec2.zero;
        return self.scale(1.0 / len);
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Vec3 basic operations" {
    const a = Vec3.init(1, 2, 3);
    const b = Vec3.init(4, 5, 6);

    // Addition
    const sum = a.add(b);
    try testing.expectApproxEqAbs(sum.x, 5.0, 1e-6);
    try testing.expectApproxEqAbs(sum.y, 7.0, 1e-6);
    try testing.expectApproxEqAbs(sum.z, 9.0, 1e-6);

    // Subtraction
    const diff = b.sub(a);
    try testing.expectApproxEqAbs(diff.x, 3.0, 1e-6);
    try testing.expectApproxEqAbs(diff.y, 3.0, 1e-6);
    try testing.expectApproxEqAbs(diff.z, 3.0, 1e-6);

    // Dot product
    const d = a.dot(b);
    try testing.expectApproxEqAbs(d, 32.0, 1e-6); // 1*4 + 2*5 + 3*6 = 32

    // Scale
    const scaled = a.scale(2.0);
    try testing.expectApproxEqAbs(scaled.x, 2.0, 1e-6);
    try testing.expectApproxEqAbs(scaled.y, 4.0, 1e-6);
    try testing.expectApproxEqAbs(scaled.z, 6.0, 1e-6);
}

test "Vec3 cross product" {
    const x = Vec3.unit_x;
    const y = Vec3.unit_y;
    const z = x.cross(y);

    try testing.expectApproxEqAbs(z.x, 0.0, 1e-6);
    try testing.expectApproxEqAbs(z.y, 0.0, 1e-6);
    try testing.expectApproxEqAbs(z.z, 1.0, 1e-6);
}

test "Vec3 normalize" {
    const v = Vec3.init(3, 4, 0);
    const n = v.normalize();

    try testing.expectApproxEqAbs(n.length(), 1.0, 1e-6);
    try testing.expectApproxEqAbs(n.x, 0.6, 1e-6);
    try testing.expectApproxEqAbs(n.y, 0.8, 1e-6);
}

test "Vec3 reflect" {
    const v = Vec3.init(1, -1, 0).normalize();
    const n = Vec3.unit_y;
    const r = v.reflect(n);

    // Reflection of (1,-1,0) normalized over y-axis should give (1,1,0) normalized
    try testing.expectApproxEqAbs(r.x, v.x, 1e-6);
    try testing.expectApproxEqAbs(r.y, -v.y, 1e-6);
}

test "Mat3 identity" {
    const I = Mat3.identity;
    const v = Vec3.init(1, 2, 3);
    const result = I.mulVec(v);

    try testing.expect(result.approxEql(v, 1e-6));
}

test "Mat3 multiplication" {
    const A = Mat3.fromRows(
        Vec3.init(1, 2, 3),
        Vec3.init(4, 5, 6),
        Vec3.init(7, 8, 9),
    );
    const I = Mat3.identity;

    const result = A.mulMat(I);
    try testing.expect(result.approxEql(A, 1e-6));
}

test "Mat3 transpose" {
    const A = Mat3.fromRows(
        Vec3.init(1, 2, 3),
        Vec3.init(4, 5, 6),
        Vec3.init(7, 8, 9),
    );
    const At = A.transpose();
    const Att = At.transpose();

    try testing.expect(Att.approxEql(A, 1e-6));
}

test "Mat3 determinant and inverse" {
    const A = Mat3.fromRows(
        Vec3.init(1, 2, 3),
        Vec3.init(0, 1, 4),
        Vec3.init(5, 6, 0),
    );

    const det = A.determinant();
    try testing.expectApproxEqAbs(det, 1.0, 1e-6);

    if (A.inverse()) |Ainv| {
        const product = A.mulMat(Ainv);
        try testing.expect(product.approxEql(Mat3.identity, 1e-5));
    } else {
        try testing.expect(false); // Should have inverse
    }
}

test "Mat3 outer product" {
    const u = Vec3.init(1, 2, 3);
    const v = Vec3.init(4, 5, 6);
    const M = Mat3.outer(u, v);

    // M[i,j] = u[i] * v[j]
    try testing.expectApproxEqAbs(M.get(0, 0), 4.0, 1e-6); // 1*4
    try testing.expectApproxEqAbs(M.get(1, 0), 8.0, 1e-6); // 2*4
    try testing.expectApproxEqAbs(M.get(0, 1), 5.0, 1e-6); // 1*5
    try testing.expectApproxEqAbs(M.get(2, 2), 18.0, 1e-6); // 3*6
}
