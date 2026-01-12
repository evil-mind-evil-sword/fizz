const std = @import("std");
const math = @import("math.zig");
const types = @import("types.zig");

const Vec3 = math.Vec3;
const Vec2 = math.Vec2;
const Mat3 = math.Mat3;
const Entity = types.Entity;
const GaussianVec3 = types.GaussianVec3;
const Camera = types.Camera;
const Appearance = types.Appearance;

// =============================================================================
// 3D Gaussian Mixture Model for Observation
// =============================================================================

/// A single Gaussian component in the mixture (corresponds to one entity)
pub const GaussianComponent = struct {
    /// 3D position (center of Gaussian)
    position: Vec3,
    /// 3D covariance (shape of Gaussian blob)
    covariance: Mat3,
    /// Mixture weight (proportional to opacity * size)
    weight: f32,
    /// RGB color
    color: Vec3,

    /// Create component from entity
    pub fn fromEntity(entity: Entity) GaussianComponent {
        const radius = entity.appearance.radius;
        const variance = radius * radius;

        return .{
            .position = entity.positionMean(),
            .covariance = Mat3.diagonal(Vec3.splat(variance)),
            .weight = entity.appearance.opacity,
            .color = entity.appearance.color,
        };
    }

    /// Evaluate 3D Gaussian density at point x
    pub fn evaluate(self: GaussianComponent, x: Vec3) f32 {
        const diff = x.sub(self.position);
        const inv_cov = self.covariance.inverse() orelse return 0;
        const mahalanobis = diff.dot(inv_cov.mulVec(diff));
        return @exp(-0.5 * mahalanobis);
    }

    /// Log density at point x (unnormalized)
    pub fn logDensity(self: GaussianComponent, x: Vec3) f32 {
        const diff = x.sub(self.position);
        const inv_cov = self.covariance.inverse() orelse return -std.math.inf(f32);
        return -0.5 * diff.dot(inv_cov.mulVec(diff));
    }
};

/// Gaussian Mixture Model for observation likelihood
pub const GaussianMixture = struct {
    /// Components (one per entity)
    components: []GaussianComponent,
    /// Allocator for dynamic allocation
    allocator: std.mem.Allocator,

    /// Create GMM from entity slice
    pub fn fromEntities(entities: []const Entity, allocator: std.mem.Allocator) !GaussianMixture {
        // Count alive entities
        var n_alive: usize = 0;
        for (entities) |e| {
            if (e.isAlive()) n_alive += 1;
        }

        const components = try allocator.alloc(GaussianComponent, n_alive);
        errdefer allocator.free(components);

        var i: usize = 0;
        for (entities) |e| {
            if (e.isAlive()) {
                components[i] = GaussianComponent.fromEntity(e);
                i += 1;
            }
        }

        return .{
            .components = components,
            .allocator = allocator,
        };
    }

    /// Free allocated memory
    pub fn deinit(self: *GaussianMixture) void {
        self.allocator.free(self.components);
    }

    /// Evaluate mixture density at 3D point
    pub fn evaluate(self: GaussianMixture, x: Vec3) f32 {
        var total: f32 = 0;
        var weight_sum: f32 = 0;

        for (self.components) |comp| {
            total += comp.weight * comp.evaluate(x);
            weight_sum += comp.weight;
        }

        if (weight_sum > 0) {
            return total / weight_sum;
        }
        return 0;
    }

    /// Evaluate log mixture density at 3D point (properly normalized)
    pub fn logDensity(self: GaussianMixture, x: Vec3) f32 {
        if (self.components.len == 0) return -std.math.inf(f32);

        // Compute weight sum for normalization
        var weight_sum: f32 = 0;
        for (self.components) |comp| {
            weight_sum += comp.weight;
        }
        const log_weight_sum = @log(weight_sum + 1e-10);

        // Log-sum-exp for numerical stability
        var max_log: f32 = -std.math.inf(f32);
        for (self.components) |comp| {
            const log_w = @log(comp.weight + 1e-10);
            const log_p = comp.logDensity(x);
            max_log = @max(max_log, log_w + log_p);
        }

        var sum_exp: f32 = 0;
        for (self.components) |comp| {
            const log_w = @log(comp.weight + 1e-10);
            const log_p = comp.logDensity(x);
            sum_exp += @exp(log_w + log_p - max_log);
        }

        // Subtract log_weight_sum to normalize the mixture weights
        return max_log + @log(sum_exp + 1e-10) - log_weight_sum;
    }

    /// Get color at 3D point (weighted by density)
    pub fn colorAt(self: GaussianMixture, x: Vec3) Vec3 {
        var color_sum = Vec3.zero;
        var weight_sum: f32 = 0;

        for (self.components) |comp| {
            const w = comp.weight * comp.evaluate(x);
            color_sum = color_sum.add(comp.color.scale(w));
            weight_sum += w;
        }

        if (weight_sum > 1e-10) {
            return color_sum.scale(1.0 / weight_sum);
        }
        return Vec3.zero; // Background color
    }
};

// =============================================================================
// Observation Model (Image Likelihood)
// =============================================================================

/// Pixel observation (what we actually see)
pub const Observation = struct {
    /// RGB color
    color: Vec3,
    /// Depth (distance from camera)
    depth: f32,
    /// Whether pixel is occupied or background
    occupied: bool,
};

/// Grid of observations (rendered image)
pub const ObservationGrid = struct {
    /// Width in pixels
    width: u32,
    /// Height in pixels
    height: u32,
    /// Pixel observations (row-major)
    pixels: []Observation,
    /// Allocator
    allocator: std.mem.Allocator,

    /// Create empty observation grid
    pub fn init(width: u32, height: u32, allocator: std.mem.Allocator) !ObservationGrid {
        const pixels = try allocator.alloc(Observation, @as(usize, width) * height);
        @memset(pixels, Observation{
            .color = Vec3.zero,
            .depth = std.math.inf(f32),
            .occupied = false,
        });

        return .{
            .width = width,
            .height = height,
            .pixels = pixels,
            .allocator = allocator,
        };
    }

    /// Free allocated memory
    pub fn deinit(self: *ObservationGrid) void {
        self.allocator.free(self.pixels);
    }

    /// Get pixel at (x, y)
    pub fn get(self: ObservationGrid, x: u32, y: u32) Observation {
        const idx = @as(usize, y) * self.width + x;
        return self.pixels[idx];
    }

    /// Set pixel at (x, y)
    pub fn set(self: *ObservationGrid, x: u32, y: u32, obs: Observation) void {
        const idx = @as(usize, y) * self.width + x;
        self.pixels[idx] = obs;
    }

    /// Render GMM to observation grid using camera
    pub fn renderGMM(
        self: *ObservationGrid,
        gmm: GaussianMixture,
        camera: Camera,
        samples_per_ray: u32,
    ) void {
        const half_w: f32 = @floatFromInt(self.width / 2);
        const half_h: f32 = @floatFromInt(self.height / 2);

        for (0..self.height) |yi| {
            for (0..self.width) |xi| {
                const x: u32 = @intCast(xi);
                const y: u32 = @intCast(yi);

                // Convert pixel to NDC
                const ndc_x = (@as(f32, @floatFromInt(x)) - half_w) / half_w;
                const ndc_y = (@as(f32, @floatFromInt(y)) - half_h) / half_h;

                // Ray march through scene
                const ray_dir = computeRayDirection(camera, ndc_x, ndc_y);
                const obs = rayMarchGMM(camera.position, ray_dir, gmm, camera.near, camera.far, samples_per_ray);

                self.set(x, y, obs);
            }
        }
    }
};

/// Compute ray direction for pixel in NDC coordinates
fn computeRayDirection(camera: Camera, ndc_x: f32, ndc_y: f32) Vec3 {
    const forward = camera.target.sub(camera.position).normalize();
    const right = forward.cross(camera.up).normalize();
    const up = right.cross(forward);

    const tan_half_fov = @tan(camera.fov / 2.0);
    const dir_x = right.scale(ndc_x * tan_half_fov * camera.aspect);
    const dir_y = up.scale(ndc_y * tan_half_fov);

    return forward.add(dir_x).add(dir_y).normalize();
}

/// Ray march through GMM to compute observation
/// Uses adaptive step size based on maximum entity radius for proper sampling
fn rayMarchGMM(
    origin: Vec3,
    direction: Vec3,
    gmm: GaussianMixture,
    near: f32,
    far: f32,
    n_samples: u32,
) Observation {
    var accumulated_color = Vec3.zero;
    var accumulated_alpha: f32 = 0;
    var first_hit_depth: f32 = std.math.inf(f32);

    // Use actual depth range for sampling, but limit to reasonable bounds
    const effective_far = @min(far, 30.0); // Limit far plane for denser sampling
    const step_size = (effective_far - near) / @as(f32, @floatFromInt(n_samples));

    // Volumetric integration: alpha = 1 - exp(-density * step_size * extinction_coeff)
    // For small values, alpha ≈ density * step_size * extinction_coeff
    // extinction_coeff scales how quickly density converts to opacity
    const extinction_coeff: f32 = 10.0;

    for (0..n_samples) |i| {
        const t = near + @as(f32, @floatFromInt(i)) * step_size;
        const sample_pos = origin.add(direction.scale(t));

        // Evaluate GMM at sample point
        const density = gmm.evaluate(sample_pos);

        if (density > 1e-8) { // Lower threshold for detection
            const color = gmm.colorAt(sample_pos);
            // Correct volumetric integration: alpha depends on density * step_size
            const alpha = @min(1.0, density * step_size * extinction_coeff);

            // Front-to-back compositing
            const weight = alpha * (1.0 - accumulated_alpha);
            accumulated_color = accumulated_color.add(color.scale(weight));
            accumulated_alpha += weight;

            if (first_hit_depth == std.math.inf(f32)) {
                first_hit_depth = t;
            }

            // Early termination if fully opaque
            if (accumulated_alpha > 0.99) break;
        }
    }

    return .{
        .color = accumulated_color,
        .depth = first_hit_depth,
        .occupied = accumulated_alpha > 0.05, // Lower threshold for occupancy
    };
}

// =============================================================================
// Image Likelihood
// =============================================================================

/// Compute log-likelihood of observed image given GMM
pub fn imageLogLikelihood(
    observed: ObservationGrid,
    gmm: GaussianMixture,
    camera: Camera,
    color_noise: f32,
) f32 {
    var log_lik: f32 = 0;

    // Render expected image
    var expected = ObservationGrid.init(observed.width, observed.height, observed.allocator) catch return -std.math.inf(f32);
    defer expected.deinit();

    expected.renderGMM(gmm, camera, 32);

    // Compare each pixel
    for (0..observed.height) |yi| {
        for (0..observed.width) |xi| {
            const x: u32 = @intCast(xi);
            const y: u32 = @intCast(yi);

            const obs = observed.get(x, y);
            const exp = expected.get(x, y);

            // Color likelihood (Gaussian)
            const color_diff = obs.color.sub(exp.color);
            const color_sq_dist = color_diff.dot(color_diff);
            log_lik += -color_sq_dist / (2.0 * color_noise * color_noise);

            // Occupancy likelihood (Bernoulli)
            if (obs.occupied == exp.occupied) {
                log_lik += @log(0.9);
            } else {
                log_lik += @log(0.1);
            }
        }
    }

    return log_lik;
}

/// Compute per-entity log-likelihood contribution
/// Used for data association in particle filter
pub fn entityLogLikelihood(
    observed: ObservationGrid,
    entity: Entity,
    camera: Camera,
    color_noise: f32,
) f32 {
    // Project entity to image space (now returns ndc + depth)
    const pos = entity.positionMean();
    const proj = camera.project(pos) orelse return -std.math.inf(f32);

    // Convert NDC to pixel coordinates
    const half_w: f32 = @floatFromInt(observed.width / 2);
    const half_h: f32 = @floatFromInt(observed.height / 2);
    const px = @as(u32, @intFromFloat(@max(0, @min(@as(f32, @floatFromInt(observed.width - 1)), (proj.ndc.x + 1) * half_w))));
    const py = @as(u32, @intFromFloat(@max(0, @min(@as(f32, @floatFromInt(observed.height - 1)), (proj.ndc.y + 1) * half_h))));

    // Compute perspective-correct projected radius in pixel space
    // projectRadius returns vertical NDC units, so multiply by half_h
    const ndc_radius = camera.projectRadius(entity.appearance.radius, proj.depth);
    const pixel_radius = ndc_radius * half_h; // Vertical NDC to pixel radius
    const radius: i32 = @max(1, @as(i32, @intFromFloat(pixel_radius)));

    var log_lik: f32 = 0;
    var n_pixels: f32 = 0;

    const px_i: i32 = @intCast(px);
    const py_i: i32 = @intCast(py);

    var dy: i32 = -radius;
    while (dy <= radius) : (dy += 1) {
        var dx: i32 = -radius;
        while (dx <= radius) : (dx += 1) {
            const x = px_i + dx;
            const y = py_i + dy;

            if (x >= 0 and x < @as(i32, @intCast(observed.width)) and
                y >= 0 and y < @as(i32, @intCast(observed.height)))
            {
                const obs = observed.get(@intCast(x), @intCast(y));

                // Compare observed color to entity color
                const expected_color = entity.appearance.color;
                const color_diff = obs.color.sub(expected_color);
                const color_sq_dist = color_diff.dot(color_diff);

                log_lik += -color_sq_dist / (2.0 * color_noise * color_noise);
                n_pixels += 1;
            }
        }
    }

    if (n_pixels > 0) {
        return log_lik / n_pixels;
    }
    return 0;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "GaussianComponent from entity" {
    const label = types.Label{ .birth_time = 0, .birth_index = 0 };
    const entity = Entity.initPoint(label, Vec3.init(1, 2, 3), Vec3.zero, .standard);

    const comp = GaussianComponent.fromEntity(entity);

    try testing.expect(comp.position.approxEql(Vec3.init(1, 2, 3), 1e-6));
    try testing.expect(comp.weight == 1.0);
}

test "GaussianComponent evaluate" {
    const comp = GaussianComponent{
        .position = Vec3.zero,
        .covariance = Mat3.diagonal(Vec3.splat(1.0)),
        .weight = 1.0,
        .color = Vec3.init(1, 0, 0),
    };

    // Density at center should be 1.0
    const density_center = comp.evaluate(Vec3.zero);
    try testing.expect(density_center > 0.99);

    // Density should decrease with distance
    const density_far = comp.evaluate(Vec3.init(3, 0, 0));
    try testing.expect(density_far < density_center);
}

test "GaussianMixture fromEntities" {
    const allocator = testing.allocator;

    var entities: [2]Entity = undefined;
    entities[0] = Entity.initPoint(.{ .birth_time = 0, .birth_index = 0 }, Vec3.init(0, 0, 0), Vec3.zero, .standard);
    entities[1] = Entity.initPoint(.{ .birth_time = 0, .birth_index = 1 }, Vec3.init(2, 0, 0), Vec3.zero, .standard);

    var gmm = try GaussianMixture.fromEntities(&entities, allocator);
    defer gmm.deinit();

    try testing.expect(gmm.components.len == 2);
}

test "GaussianMixture evaluate" {
    const allocator = testing.allocator;

    var entities: [1]Entity = undefined;
    entities[0] = Entity.initPoint(.{ .birth_time = 0, .birth_index = 0 }, Vec3.zero, Vec3.zero, .standard);

    var gmm = try GaussianMixture.fromEntities(&entities, allocator);
    defer gmm.deinit();

    // Should have high density at entity center
    const density = gmm.evaluate(Vec3.zero);
    try testing.expect(density > 0);
}

test "ObservationGrid init and access" {
    const allocator = testing.allocator;

    var grid = try ObservationGrid.init(4, 4, allocator);
    defer grid.deinit();

    const obs = Observation{
        .color = Vec3.init(1, 0, 0),
        .depth = 5.0,
        .occupied = true,
    };

    grid.set(1, 2, obs);
    const retrieved = grid.get(1, 2);

    try testing.expect(retrieved.color.approxEql(Vec3.init(1, 0, 0), 1e-6));
    try testing.expect(retrieved.depth == 5.0);
    try testing.expect(retrieved.occupied == true);
}

test "Ray direction computation" {
    const camera = Camera.default;

    // Center ray should point toward target
    const center_dir = computeRayDirection(camera, 0, 0);
    const expected_dir = camera.target.sub(camera.position).normalize();

    try testing.expect(center_dir.dot(expected_dir) > 0.9);
}
