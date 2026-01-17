const std = @import("std");
const math = @import("math.zig");
const types = @import("types.zig");

const Vec3 = math.Vec3;
const Vec2 = math.Vec2;
const Mat3 = math.Mat3;
const Mat2 = math.Mat2;
const Entity = types.Entity;
const GaussianVec3 = types.GaussianVec3;
const Camera = types.Camera;
const Appearance = types.Appearance;

// =============================================================================
// SURROGATE POSTERIOR OBSERVATION MODEL
// =============================================================================
//
// This module implements a SURROGATE (detection-based) observation model for
// Rao-Blackwellized SMC inference. Instead of computing p(RGB | entities),
// we compute p(detections | entities) where detections are extracted features.
//
// ## Mathematical Foundation
//
// The true posterior we want is:
//     p(entities | RGB) ∝ p(RGB | entities) × p(entities)
//
// But p(RGB | entities) requires pixel-by-pixel rendering comparison, which is:
//     - O(W × H) per particle per timestep
//     - Non-conjugate with Gaussian entity beliefs
//     - Computationally expensive for real-time inference
//
// Instead, we define a SURROGATE posterior:
//     p(entities | detections) ∝ p(detections | entities) × p(entities)
//
// Where detections = f(RGB) are extracted features (blob centroids, optical flow).
//
// ## Soundness Justification
//
// This approach is valid under several interpretations:
//
// 1. **Cut Posterior / Semi-Modular Inference** (Plummer 2015, Jacob et al. 2017)
//    We "cut" the Bayesian network at the detection level, treating detections
//    as primary observations rather than derived quantities. The resulting
//    posterior is CONSISTENT (proper probability) but targets a different
//    distribution than the pixel-based posterior.
//
// 2. **ABC (Approximate Bayesian Computation)** (Marin et al. 2012)
//    Detection extraction can be viewed as computing summary statistics S(RGB).
//    The surrogate likelihood p(S_obs | entities) ∝ K(S_obs, S_simulated)
//    where K is a kernel. Our Gaussian likelihood is a soft kernel.
//
// 3. **Multi-Target Tracking Convention** (Bar-Shalom & Li 1995)
//    Standard MTT algorithms operate on detections, not raw sensor data.
//    The sensor model p(detection | target state) is well-established.
//
// ## Observation Modalities
//
// ### Blob Detection (IMPLEMENTED)
// - Input: RGB image
// - Extraction: Connected component analysis with brightness threshold
// - Output: 2D centroids + radii
// - Likelihood: Back-project to 3D ray, Gaussian in position space
// - Conjugacy: YES (Gaussian × Gaussian → Gaussian)
//
// ### Optical Flow (PLANNED)
// - Input: RGB sequence (frames t-1, t)
// - Extraction: Lucas-Kanade or Horn-Schunck flow estimation
// - Output: 2D velocity field at detection locations
// - Likelihood: Project entity velocity to 2D, Gaussian comparison
// - Conjugacy: YES (velocity is linear function of state)
//
// Optical flow provides DIRECT velocity observations, complementing blob
// detection's position observations. Together they constrain the full
// (position, velocity) state.
//
// ## RBPF Integration
//
// For Rao-Blackwellization to work, the observation model must support:
//
// 1. Kalman update: p(position | detection) = N(μ', Σ')
//    - IMPLEMENTED in smc.zig:rbpfEntityObservationUpdate()
//
// 2. Marginal likelihood: p(detection | prior) = ∫ p(det | x) p(x) dx
//    - IMPLEMENTED in dynamics.zig:kalmanLogLikelihood()
//    - Used for particle weighting
//
// 3. Per-particle evaluation: Each particle has different camera pose
//    - IMPLEMENTED in smc.zig:rbpfParticleObservationUpdate()
//
// ## Limitations and Future Work
//
// Current limitations:
// - Single blob = single entity (no hierarchical parts)
// - Hard assignment (MAP) for data association
// - Fixed depth prior for back-projection
// - No occlusion reasoning in likelihood
//
// See tissue issues:
// - workshop-b9wk02o: Hierarchical entity representations
// - workshop-4uh0sbp: Soft assignment / marginalized association
// - workshop-8e4ssfd: Adaptive depth prior
//
// References:
// - Doucet et al. (2000): Rao-Blackwellized Particle Filtering
// - Bar-Shalom & Li (1995): Multitarget-Multisensor Tracking
// - Lew et al. (2023): SMCP3 - Sequential Monte Carlo with Probabilistic Programs
// - Jacob et al. (2017): Better together? Statistical learning in models made of modules
// =============================================================================

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
                // Image coords: (0,0) top-left, Y increases downward
                // NDC coords: (-1,-1) bottom-left, Y increases upward
                const ndc_x = (@as(f32, @floatFromInt(x)) - half_w) / half_w;
                const ndc_y = (half_h - @as(f32, @floatFromInt(y))) / half_h; // Flip Y

                // Ray march through scene
                const ray_dir = computeRayDirection(camera, ndc_x, ndc_y);
                const obs = rayMarchGMM(camera.position, ray_dir, gmm, camera.near, camera.far, samples_per_ray);

                self.set(x, y, obs);
            }
        }
    }

    /// Write observation grid to binary PPM file (P6 format)
    pub fn writePPM(self: ObservationGrid, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // PPM header: P6 (binary RGB)
        var header_buf: [64]u8 = undefined;
        const header = std.fmt.bufPrint(&header_buf, "P6\n{d} {d}\n255\n", .{ self.width, self.height }) catch unreachable;
        try file.writeAll(header);

        // Write pixel data (RGB bytes) - collect into buffer for efficiency
        const pixel_count = @as(usize, self.width) * self.height * 3;
        var pixel_data = self.allocator.alloc(u8, pixel_count) catch {
            // Fallback to unbuffered writes if allocation fails
            for (0..self.height) |yi| {
                for (0..self.width) |xi| {
                    const obs = self.get(@intCast(xi), @intCast(yi));
                    const r: u8 = @intFromFloat(@min(255.0, @max(0.0, obs.color.x * 255.0)));
                    const g: u8 = @intFromFloat(@min(255.0, @max(0.0, obs.color.y * 255.0)));
                    const b: u8 = @intFromFloat(@min(255.0, @max(0.0, obs.color.z * 255.0)));
                    try file.writeAll(&[_]u8{ r, g, b });
                }
            }
            return;
        };
        defer self.allocator.free(pixel_data);

        var idx: usize = 0;
        for (0..self.height) |yi| {
            for (0..self.width) |xi| {
                const obs = self.get(@intCast(xi), @intCast(yi));
                pixel_data[idx] = @intFromFloat(@min(255.0, @max(0.0, obs.color.x * 255.0)));
                pixel_data[idx + 1] = @intFromFloat(@min(255.0, @max(0.0, obs.color.y * 255.0)));
                pixel_data[idx + 2] = @intFromFloat(@min(255.0, @max(0.0, obs.color.z * 255.0)));
                idx += 3;
            }
        }

        try file.writeAll(pixel_data);
    }
};

/// Compute ray direction for pixel in NDC coordinates
pub fn computeRayDirection(camera: Camera, ndc_x: f32, ndc_y: f32) Vec3 {
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
// Back-Projection Observation Model
// =============================================================================
//
// THEORETICAL NOTES ON DETECTION EXTRACTION:
//
// Detection extraction (blob detection) is an approximation that converts
// the pixel-level observation into a sparse set of 2D detections. This is
// NOT a sufficient statistic for the full pixel grid, meaning information
// is lost.
//
// Key approximations:
// 1. Hard thresholding: Pixels below brightness threshold are ignored
// 2. Connected components: Nearby objects may be merged
// 3. Centroid summary: Each blob is summarized by its centroid + radius
//
// Soundness analysis:
// - The detection-based observation model is CONSISTENT: if we generate
//   observations by rendering GMM -> pixels -> detections, and interpret
//   them via the same detection extraction, the inference is internally
//   consistent (though not pixel-optimal).
// - For RBPF: The marginal likelihood p(detections | entity positions)
//   is well-defined, even if it's not the same as p(pixels | entities).
// - Trade-off: O(K) detections vs O(W×H) pixels → massive speedup
//
// Alternative approaches (not implemented):
// - Soft weighting: Weight pixels by brightness instead of hard threshold
// - Hierarchical: Multi-scale detection pyramid
// - Pixel-wise: Full rendering comparison (legacy mode)
//
// References:
// - Bar-Shalom & Li (1995): Multi-target tracking with detection extraction
// - SMCP3 (Lew et al. 2023): Structured observations in probabilistic programs
// =============================================================================

/// 2D detection extracted from observation grid
pub const Detection2D = struct {
    /// Pixel coordinates (center of detection)
    pixel_x: f32,
    pixel_y: f32,
    /// Detected color (RGB)
    color: Vec3,
    /// Detection radius in pixels
    radius: f32,
    /// Detection weight (brightness/confidence)
    weight: f32,

    /// Extract detections from observation grid using connected component analysis
    ///
    /// Approximation properties:
    /// - Sound within detection-based observation model
    /// - Not a sufficient statistic for pixel-level observations
    /// - May merge nearby objects (depends on threshold)
    /// - O(W×H) time complexity, O(K) output where K = number of detections
    pub fn extractFromGrid(grid: ObservationGrid, allocator: std.mem.Allocator) ![]Detection2D {
        const brightness_threshold: f32 = 0.1;
        const min_blob_pixels: usize = 4;

        // Blob detection via connected components (flood-fill)
        // Approximation: hard threshold loses sub-threshold information
        const n_pixels = @as(usize, grid.width) * grid.height;
        const visited = try allocator.alloc(bool, n_pixels);
        defer allocator.free(visited);
        @memset(visited, false);

        var detections: std.ArrayList(Detection2D) = .empty;
        errdefer detections.deinit(allocator);

        for (0..grid.height) |yi| {
            for (0..grid.width) |xi| {
                const idx = yi * grid.width + xi;
                if (visited[idx]) continue;

                const px = grid.get(@intCast(xi), @intCast(yi));
                const brightness = (px.color.x + px.color.y + px.color.z) / 3.0;

                if (brightness > brightness_threshold) {
                    // Found start of a blob - flood fill to find extent
                    var blob = BlobAccumulator{};
                    try floodFill(grid, @intCast(xi), @intCast(yi), visited, brightness_threshold, &blob, allocator);

                    if (blob.count >= min_blob_pixels) {
                        // Convert accumulated blob to detection
                        const cx = blob.sum_x / @as(f32, @floatFromInt(blob.count));
                        const cy = blob.sum_y / @as(f32, @floatFromInt(blob.count));
                        const avg_color = blob.sum_color.scale(1.0 / @as(f32, @floatFromInt(blob.count)));

                        // Estimate radius from blob size (assuming roughly circular)
                        const area = @as(f32, @floatFromInt(blob.count));
                        const radius = @sqrt(area / std.math.pi);

                        try detections.append(allocator, .{
                            .pixel_x = cx,
                            .pixel_y = cy,
                            .color = avg_color,
                            .radius = radius,
                            .weight = blob.sum_brightness / @as(f32, @floatFromInt(blob.count)),
                        });
                    }
                }
            }
        }

        return detections.toOwnedSlice(allocator);
    }
};

/// Helper for blob accumulation during flood fill
const BlobAccumulator = struct {
    sum_x: f32 = 0,
    sum_y: f32 = 0,
    sum_color: Vec3 = Vec3.zero,
    sum_brightness: f32 = 0,
    count: usize = 0,
};

/// Flood fill helper for blob detection
fn floodFill(
    grid: ObservationGrid,
    start_x: u32,
    start_y: u32,
    visited: []bool,
    threshold: f32,
    blob: *BlobAccumulator,
    allocator: std.mem.Allocator,
) !void {
    const StackItem = struct { x: u32, y: u32 };
    var stack: std.ArrayList(StackItem) = .empty;
    defer stack.deinit(allocator);

    try stack.append(allocator, .{ .x = start_x, .y = start_y });

    while (stack.items.len > 0) {
        const pos = stack.pop().?;
        const idx = @as(usize, pos.y) * grid.width + pos.x;

        if (visited[idx]) continue;
        visited[idx] = true;

        const px = grid.get(pos.x, pos.y);
        const brightness = (px.color.x + px.color.y + px.color.z) / 3.0;

        if (brightness > threshold) {
            // Add to blob
            blob.sum_x += @floatFromInt(pos.x);
            blob.sum_y += @floatFromInt(pos.y);
            blob.sum_color = blob.sum_color.add(px.color);
            blob.sum_brightness += brightness;
            blob.count += 1;

            // Add neighbors to stack
            if (pos.x > 0) try stack.append(allocator, .{ .x = pos.x - 1, .y = pos.y });
            if (pos.x < grid.width - 1) try stack.append(allocator, .{ .x = pos.x + 1, .y = pos.y });
            if (pos.y > 0) try stack.append(allocator, .{ .x = pos.x, .y = pos.y - 1 });
            if (pos.y < grid.height - 1) try stack.append(allocator, .{ .x = pos.x, .y = pos.y + 1 });
        }
    }
}

/// Ray-Gaussian: a Gaussian distribution representing uncertainty along a viewing ray
/// Used for back-projection from 2D detections to 3D
pub const RayGaussian = struct {
    /// Ray origin (camera position)
    origin: Vec3,
    /// Ray direction (normalized)
    direction: Vec3,
    /// Mean depth along ray
    depth_mean: f32,
    /// Depth variance (uncertainty along ray)
    depth_var: f32,
    /// Lateral variance (uncertainty perpendicular to ray)
    lateral_var: f32,
    /// Color from detection
    color: Vec3,
    /// Weight (detection confidence)
    weight: f32,

    /// Construct ray-Gaussian from camera and detection
    pub fn fromDetection(
        detection: Detection2D,
        camera: Camera,
        image_width: u32,
        image_height: u32,
        depth_prior_mean: f32,
        depth_prior_var: f32,
    ) RayGaussian {
        // Convert pixel to NDC
        // Image: Y increases downward, NDC: Y increases upward
        const half_w: f32 = @floatFromInt(image_width / 2);
        const half_h: f32 = @floatFromInt(image_height / 2);
        const ndc_x = (detection.pixel_x - half_w) / half_w;
        const ndc_y = (half_h - detection.pixel_y) / half_h; // Flip Y

        // Compute ray direction
        const ray_dir = computeRayDirection(camera, ndc_x, ndc_y);

        // Lateral variance based on detection size in image
        // Larger detections in image → closer objects → smaller lateral variance
        // Convert pixel radius to angular uncertainty
        const angular_radius = detection.radius / half_h * @tan(camera.fov / 2.0);
        const lateral_var = angular_radius * angular_radius * depth_prior_mean * depth_prior_mean;

        return .{
            .origin = camera.position,
            .direction = ray_dir,
            .depth_mean = depth_prior_mean,
            .depth_var = depth_prior_var,
            .lateral_var = lateral_var,
            .color = detection.color,
            .weight = detection.weight,
        };
    }

    /// Get the 3D position mean (point on ray at depth_mean)
    pub fn positionMean(self: RayGaussian) Vec3 {
        return self.origin.add(self.direction.scale(self.depth_mean));
    }

    /// Get the 3D covariance matrix
    /// Diagonal in ray coordinates: large variance along ray, small perpendicular
    pub fn covariance3D(self: RayGaussian) Mat3 {
        // Build coordinate frame: ray direction + two perpendicular directions
        const d = self.direction;

        // Find perpendicular vectors using Gram-Schmidt
        var perp1: Vec3 = undefined;
        if (@abs(d.x) < 0.9) {
            perp1 = Vec3.unit_x.sub(d.scale(d.x)).normalize();
        } else {
            perp1 = Vec3.unit_y.sub(d.scale(d.y)).normalize();
        }
        const perp2 = d.cross(perp1);

        // Covariance in ray frame: Σ_ray = diag(σ²_depth, σ²_lateral, σ²_lateral)
        // Transform to world: Σ_world = R * Σ_ray * R^T
        // Where R = [d | perp1 | perp2] as columns

        // For efficiency, compute directly:
        // Σ = σ²_d * d*d^T + σ²_l * (perp1*perp1^T + perp2*perp2^T)
        const outer_d = outerProduct(d, d);
        const outer_p1 = outerProduct(perp1, perp1);
        const outer_p2 = outerProduct(perp2, perp2);

        var result = outer_d.scaleMat(self.depth_var);
        result = result.add(outer_p1.scaleMat(self.lateral_var));
        result = result.add(outer_p2.scaleMat(self.lateral_var));

        return result;
    }

    /// Compute log-likelihood of overlap with a 3D Gaussian component
    /// Uses the Gaussian product formula: ∫ N(x;μ₁,Σ₁)·N(x;μ₂,Σ₂)dx = N(μ₁;μ₂,Σ₁+Σ₂)
    pub fn overlapLogLikelihood(self: RayGaussian, component: GaussianComponent) f32 {
        const mu1 = self.positionMean();
        const mu2 = component.position;
        const sigma1 = self.covariance3D();
        const sigma2 = component.covariance;

        // Combined covariance: Σ₁ + Σ₂
        const sigma_sum = sigma1.add(sigma2);

        // Inverse of combined covariance
        const sigma_inv = sigma_sum.inverse() orelse return -std.math.inf(f32);

        // Mahalanobis distance: (μ₁-μ₂)^T (Σ₁+Σ₂)^{-1} (μ₁-μ₂)
        const diff = mu1.sub(mu2);
        const mahal = diff.dot(sigma_inv.mulVec(diff));

        // Log-likelihood: -0.5 * mahal - 0.5 * log|Σ₁+Σ₂| - (d/2)*log(2π)
        // We omit the constant term and determinant for now (just comparing)
        const log_det = @log(sigma_sum.determinant() + 1e-10);

        return -0.5 * mahal - 0.5 * log_det;
    }
};

/// Compute outer product of two vectors: v * w^T
/// Result is a 3x3 matrix stored column-major in flat array
fn outerProduct(v: Vec3, w: Vec3) Mat3 {
    // Mat3.data is column-major: [col0, col1, col2]
    // Outer product v * w^T has element (i,j) = v[i] * w[j]
    // Column j = v * w[j]
    return .{
        .data = .{
            // Column 0: v * w.x
            v.x * w.x, v.y * w.x, v.z * w.x,
            // Column 1: v * w.y
            v.x * w.y, v.y * w.y, v.z * w.y,
            // Column 2: v * w.z
            v.x * w.z, v.y * w.z, v.z * w.z,
        },
    };
}

// =============================================================================
// OPTICAL FLOW OBSERVATION MODEL
// =============================================================================
//
// Sparse optical flow between consecutive detection centroids provides
// VELOCITY observations that complement the POSITION observations from
// blob detection. Together they constrain the full (position, velocity) state.
//
// ## Mathematical Foundation
//
// Lucas-Kanade optical flow assumes brightness constancy:
//     I(x, y, t) = I(x + dx, y + dy, t + dt)
//
// Linearizing gives the optical flow constraint:
//     Ix * u + Iy * v + It = 0
//
// where (Ix, Iy) are spatial gradients, It is temporal gradient,
// and (u, v) is the flow velocity.
//
// ## Heteroscedastic Covariance (arXiv:2010.06871)
//
// The uncertainty of Lucas-Kanade flow depends on local image texture.
// The structure tensor M = [[Ix², IxIy], [IxIy, Iy²]] captures this:
// - High eigenvalues → strong texture → low uncertainty
// - Low eigenvalues → weak texture (aperture problem) → high uncertainty
//
// Covariance is inversely proportional to the structure tensor:
//     Σ_flow ∝ M⁻¹
//
// This provides heteroscedastic (texture-dependent) noise estimation.
//
// ## Conjugacy
//
// Flow observation is Gaussian: p(observed_flow | true_velocity) = N(H*v, Σ)
// where H projects 3D velocity to 2D image plane.
// Combined with Gaussian velocity prior → Gaussian posterior (conjugate).
// =============================================================================

/// Sparse optical flow observation between two consecutive frames
/// Used for velocity estimation in SMC inference
pub const FlowObservation = struct {
    /// Index of detection in previous frame
    prev_detection_idx: usize,
    /// Index of matched detection in current frame
    curr_detection_idx: usize,
    /// Optical flow (dx, dy) in pixels
    flow: Vec2,
    /// Heteroscedastic covariance (texture-dependent uncertainty)
    covariance: Mat2,
    /// Confidence weight (based on structure tensor eigenvalues)
    weight: f32,

    /// Compute log-likelihood of this flow observation given projected velocity
    pub fn logLikelihood(self: FlowObservation, projected_velocity: Vec2) f32 {
        const residual = self.flow.sub(projected_velocity);
        const inv_cov = self.covariance.inverse() orelse return -std.math.inf(f32);
        const mahal = residual.dot(inv_cov.mulVec(residual));
        const log_det = @log(self.covariance.determinant() + 1e-10);
        return -0.5 * mahal - 0.5 * log_det;
    }
};

/// Compute structure tensor at a pixel location
/// The structure tensor captures local image gradient statistics:
///     M = Σ_window [[Ix², IxIy], [IxIy, Iy²]]
///
/// This determines the reliability of optical flow estimation.
fn structureTensor(
    grid: ObservationGrid,
    x: u32,
    y: u32,
    window_size: u32,
) Mat2 {
    var sum_ix_ix: f32 = 0;
    var sum_ix_iy: f32 = 0;
    var sum_iy_iy: f32 = 0;

    const half_w = window_size / 2;
    const width = grid.width;
    const height = grid.height;

    // Compute structure tensor over local window
    var wy: u32 = 0;
    while (wy < window_size) : (wy += 1) {
        const py = @as(i32, @intCast(y)) - @as(i32, @intCast(half_w)) + @as(i32, @intCast(wy));
        if (py < 1 or py >= @as(i32, @intCast(height)) - 1) continue;

        var wx: u32 = 0;
        while (wx < window_size) : (wx += 1) {
            const px = @as(i32, @intCast(x)) - @as(i32, @intCast(half_w)) + @as(i32, @intCast(wx));
            if (px < 1 or px >= @as(i32, @intCast(width)) - 1) continue;

            const ux = @as(u32, @intCast(px));
            const uy = @as(u32, @intCast(py));

            // Compute spatial gradients using Sobel-like kernel
            const left = grid.get(ux - 1, uy);
            const right = grid.get(ux + 1, uy);
            const up = grid.get(ux, uy - 1);
            const down = grid.get(ux, uy + 1);

            // Gradient on brightness (average of RGB)
            const b_left = (left.color.x + left.color.y + left.color.z) / 3.0;
            const b_right = (right.color.x + right.color.y + right.color.z) / 3.0;
            const b_up = (up.color.x + up.color.y + up.color.z) / 3.0;
            const b_down = (down.color.x + down.color.y + down.color.z) / 3.0;

            const ix = (b_right - b_left) / 2.0;
            const iy = (b_down - b_up) / 2.0;

            sum_ix_ix += ix * ix;
            sum_ix_iy += ix * iy;
            sum_iy_iy += iy * iy;
        }
    }

    return Mat2.init(sum_ix_ix, sum_ix_iy, sum_ix_iy, sum_iy_iy);
}

/// Compute Lucas-Kanade optical flow at a detection centroid
/// Returns the flow vector and its heteroscedastic covariance
fn lucasKanadeAtPoint(
    prev_grid: ObservationGrid,
    curr_grid: ObservationGrid,
    x: u32,
    y: u32,
    window_size: u32,
    noise_floor: f32,
) struct { flow: Vec2, covariance: Mat2, weight: f32 } {
    // Compute structure tensor from previous frame
    const M = structureTensor(prev_grid, x, y, window_size);

    // Check if structure tensor is invertible (sufficient texture)
    const det = M.determinant();
    const trace = M.get(0, 0) + M.get(1, 1);
    const min_eigenvalue = (trace - @sqrt(@max(0, trace * trace - 4 * det))) / 2.0;

    // If insufficient texture, return zero flow with high uncertainty
    if (min_eigenvalue < 0.01) {
        return .{
            .flow = Vec2.zero,
            .covariance = Mat2.diagonal(Vec2.init(100.0, 100.0)),
            .weight = 0.01,
        };
    }

    // Compute temporal gradient (brightness difference)
    var sum_ix_it: f32 = 0;
    var sum_iy_it: f32 = 0;

    const half_w = window_size / 2;
    const width = prev_grid.width;
    const height = prev_grid.height;

    var wy: u32 = 0;
    while (wy < window_size) : (wy += 1) {
        const py = @as(i32, @intCast(y)) - @as(i32, @intCast(half_w)) + @as(i32, @intCast(wy));
        if (py < 1 or py >= @as(i32, @intCast(height)) - 1) continue;

        var wx: u32 = 0;
        while (wx < window_size) : (wx += 1) {
            const px = @as(i32, @intCast(x)) - @as(i32, @intCast(half_w)) + @as(i32, @intCast(wx));
            if (px < 1 or px >= @as(i32, @intCast(width)) - 1) continue;

            const ux = @as(u32, @intCast(px));
            const uy = @as(u32, @intCast(py));

            // Spatial gradients from previous frame
            const left = prev_grid.get(ux - 1, uy);
            const right = prev_grid.get(ux + 1, uy);
            const up = prev_grid.get(ux, uy - 1);
            const down = prev_grid.get(ux, uy + 1);

            const b_left = (left.color.x + left.color.y + left.color.z) / 3.0;
            const b_right = (right.color.x + right.color.y + right.color.z) / 3.0;
            const b_up = (up.color.x + up.color.y + up.color.z) / 3.0;
            const b_down = (down.color.x + down.color.y + down.color.z) / 3.0;

            const ix = (b_right - b_left) / 2.0;
            const iy = (b_down - b_up) / 2.0;

            // Temporal gradient
            const prev_px = prev_grid.get(ux, uy);
            const curr_px = curr_grid.get(ux, uy);
            const b_prev = (prev_px.color.x + prev_px.color.y + prev_px.color.z) / 3.0;
            const b_curr = (curr_px.color.x + curr_px.color.y + curr_px.color.z) / 3.0;
            const it = b_curr - b_prev;

            sum_ix_it += ix * it;
            sum_iy_it += iy * it;
        }
    }

    // Solve M * flow = -b where b = [sum_ix_it, sum_iy_it]
    const M_inv = M.inverse() orelse {
        return .{
            .flow = Vec2.zero,
            .covariance = Mat2.diagonal(Vec2.init(100.0, 100.0)),
            .weight = 0.01,
        };
    };

    const b = Vec2.init(-sum_ix_it, -sum_iy_it);
    const flow = M_inv.mulVec(b);

    // Heteroscedastic covariance: Σ = σ² * M⁻¹
    // where σ² is the noise floor (residual variance)
    const covariance = M_inv.scale(noise_floor * noise_floor);

    // Weight based on smallest eigenvalue (texture strength)
    const weight = @min(1.0, min_eigenvalue / 1.0);

    return .{ .flow = flow, .covariance = covariance, .weight = weight };
}

/// Compute sparse optical flow between consecutive detection sets
/// Uses Lucas-Kanade at each detection centroid with heteroscedastic covariance
pub fn computeSparseFlow(
    prev_detections: []const Detection2D,
    curr_detections: []const Detection2D,
    prev_grid: ObservationGrid,
    curr_grid: ObservationGrid,
    config: SparseFlowConfig,
    allocator: std.mem.Allocator,
) ![]FlowObservation {
    var observations: std.ArrayList(FlowObservation) = .empty;
    errdefer observations.deinit(allocator);

    // For each previous detection, find best match in current frame
    // and compute optical flow
    for (prev_detections, 0..) |prev_det, prev_idx| {
        // Find nearest detection in current frame (simple greedy matching)
        var best_curr_idx: ?usize = null;
        var best_dist: f32 = config.max_match_distance;

        for (curr_detections, 0..) |curr_det, curr_idx| {
            const dx = curr_det.pixel_x - prev_det.pixel_x;
            const dy = curr_det.pixel_y - prev_det.pixel_y;
            const dist = @sqrt(dx * dx + dy * dy);

            // Also check color similarity
            const color_diff = curr_det.color.sub(prev_det.color);
            const color_dist = @sqrt(color_diff.dot(color_diff));

            if (dist < best_dist and color_dist < config.max_color_distance) {
                best_dist = dist;
                best_curr_idx = curr_idx;
            }
        }

        if (best_curr_idx) |curr_idx| {
            // Compute Lucas-Kanade flow at the detection centroid
            const px = @as(u32, @intFromFloat(@max(0, prev_det.pixel_x)));
            const py = @as(u32, @intFromFloat(@max(0, prev_det.pixel_y)));

            // Clamp to valid range
            const clamped_x = @min(px, prev_grid.width - 1);
            const clamped_y = @min(py, prev_grid.height - 1);

            const lk_result = lucasKanadeAtPoint(
                prev_grid,
                curr_grid,
                clamped_x,
                clamped_y,
                config.window_size,
                config.noise_floor,
            );

            // Only add if confidence is above threshold
            if (lk_result.weight >= config.min_weight) {
                try observations.append(allocator, FlowObservation{
                    .prev_detection_idx = prev_idx,
                    .curr_detection_idx = curr_idx,
                    .flow = lk_result.flow,
                    .covariance = lk_result.covariance,
                    .weight = lk_result.weight,
                });
            }
        }
    }

    return observations.toOwnedSlice(allocator);
}

/// Configuration for sparse optical flow computation
pub const SparseFlowConfig = struct {
    /// Lucas-Kanade window size (pixels)
    window_size: u32 = 7,
    /// Noise floor for covariance estimation
    noise_floor: f32 = 0.1,
    /// Maximum pixel distance for detection matching
    max_match_distance: f32 = 50.0,
    /// Maximum color distance for detection matching
    max_color_distance: f32 = 0.5,
    /// Minimum weight (texture confidence) to include observation
    min_weight: f32 = 0.1,
};

/// Resolution-adaptive flow configuration
/// Scales optical flow parameters based on image resolution to maintain
/// consistent tracking behavior across different image sizes.
///
/// ## Resolution Scaling Rationale
///
/// At low resolutions (e.g., 32px), fixed parameters designed for higher
/// resolutions cause problems:
/// - 7px window = 22% of image (too large, captures multiple entities)
/// - 50px match distance = 156% of image (matches anything)
/// - Texture gradients are weak (few pixels per object)
///
/// At high resolutions, the same parameters may be too conservative.
///
/// ## Scaling Strategy
///
/// Window size: ~1.4% of min dimension (e.g., 9px at 640px, 5px at 360px)
/// Match distance: ~6.9% of image diagonal
/// Noise floor: Inversely proportional to resolution (more noise at low res)
///
/// Below 48px minimum dimension, flow is disabled entirely as reliable
/// gradient computation becomes impossible.
pub const AdaptedFlowConfig = struct {
    /// Base configuration (scaled for resolution)
    base: SparseFlowConfig,
    /// Whether flow is enabled for this resolution
    enabled: bool,

    /// Default disabled configuration
    pub const disabled: AdaptedFlowConfig = .{
        .base = .{},
        .enabled = false,
    };

    /// Create resolution-adapted flow configuration
    ///
    /// Parameters:
    /// - width: Image width in pixels
    /// - height: Image height in pixels
    ///
    /// Returns configuration with:
    /// - enabled = false if min(width, height) < 48
    /// - Scaled window_size, match_distance, noise_floor otherwise
    pub fn forResolution(width: u32, height: u32) AdaptedFlowConfig {
        const min_dim = @min(width, height);

        // Disable flow below 48px (texture gradients unreliable)
        if (min_dim < 48) {
            return disabled;
        }

        const dim_f: f32 = @floatFromInt(min_dim);
        const width_f: f32 = @floatFromInt(width);
        const height_f: f32 = @floatFromInt(height);
        const diag = @sqrt(width_f * width_f + height_f * height_f);

        // Window size: ~1.4% of min dimension, odd, clamped to [3, 15]
        // At 640px: 9px window
        // At 64px: 3px window (minimum)
        var window_raw = dim_f * 0.014;
        window_raw = @max(3.0, @min(15.0, window_raw));
        // Make odd
        const window_int: u32 = @intFromFloat(window_raw);
        const window_size = window_int | 1; // Ensure odd

        // Match distance: ~6.9% of diagonal
        // At 640x480 (diag ~800): 55px
        // At 64x64 (diag ~90): 6px
        const max_match_distance = diag * 0.069;

        // Noise floor: inversely proportional to resolution
        // Higher noise at lower resolution (fewer pixels, more uncertainty)
        // At 512px: 0.1 (base)
        // At 64px: 0.8 (8x higher noise)
        const noise_floor = 0.1 * (512.0 / dim_f);

        return .{
            .base = .{
                .window_size = window_size,
                .max_match_distance = max_match_distance,
                .noise_floor = noise_floor,
                .max_color_distance = 0.5, // Keep fixed
                .min_weight = 0.1, // Keep fixed
            },
            .enabled = true,
        };
    }

    /// Get the base config (for use with computeSparseFlow)
    pub fn getConfig(self: AdaptedFlowConfig) SparseFlowConfig {
        return self.base;
    }
};

/// Compute back-projection observation log-likelihood
/// This is the conjugate alternative to pixel-wise rendering comparison
pub fn backProjectionLogLikelihood(
    observed: ObservationGrid,
    gmm: GaussianMixture,
    camera: Camera,
    depth_prior_mean: f32,
    depth_prior_var: f32,
    allocator: std.mem.Allocator,
) f32 {
    // Step 1: Extract detections from observed image
    const detections = Detection2D.extractFromGrid(observed, allocator) catch return -std.math.inf(f32);
    defer allocator.free(detections);

    return backProjectionLogLikelihoodWithDetections(
        detections,
        observed.width,
        observed.height,
        gmm,
        camera,
        depth_prior_mean,
        depth_prior_var,
    );
}

/// Compute back-projection log-likelihood with pre-extracted detections
/// Use this when computing likelihood for multiple camera hypotheses to avoid
/// redundant detection extraction (O(n) per call -> O(1) per call)
pub fn backProjectionLogLikelihoodWithDetections(
    detections: []const Detection2D,
    image_width: u32,
    image_height: u32,
    gmm: GaussianMixture,
    camera: Camera,
    depth_prior_mean: f32,
    depth_prior_var: f32,
) f32 {
    if (detections.len == 0) {
        // No detections - penalize if GMM has components
        if (gmm.components.len > 0) {
            return -10.0 * @as(f32, @floatFromInt(gmm.components.len));
        }
        return 0;
    }

    // Convert detections to ray-Gaussians and compute overlap
    var log_lik: f32 = 0;

    for (detections) |detection| {
        const ray = RayGaussian.fromDetection(
            detection,
            camera,
            image_width,
            image_height,
            depth_prior_mean,
            depth_prior_var,
        );

        // Compute overlap with GMM (max over components)
        var max_overlap: f32 = -std.math.inf(f32);
        for (gmm.components) |component| {
            const overlap = ray.overlapLogLikelihood(component);
            max_overlap = @max(max_overlap, overlap);
        }

        // Add best-matching overlap to log-likelihood
        if (max_overlap > -std.math.inf(f32)) {
            log_lik += max_overlap * detection.weight;
        } else {
            // Detection with no matching component - penalty
            log_lik -= 5.0;
        }
    }

    // Penalize unmatched GMM components (entities not detected)
    // This encourages the GMM to explain all detections
    const expected_detections = @as(f32, @floatFromInt(gmm.components.len));
    const actual_detections = @as(f32, @floatFromInt(detections.len));
    const count_diff = @abs(expected_detections - actual_detections);
    log_lik -= count_diff * 2.0;

    return log_lik;
}

// =============================================================================
// Image Likelihood (Legacy - Forward Rendering)
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
    // NDC: Y increases upward, Image: Y increases downward
    const half_w: f32 = @floatFromInt(observed.width / 2);
    const half_h: f32 = @floatFromInt(observed.height / 2);
    const px = @as(u32, @intFromFloat(@max(0, @min(@as(f32, @floatFromInt(observed.width - 1)), (proj.ndc.x + 1) * half_w))));
    const py = @as(u32, @intFromFloat(@max(0, @min(@as(f32, @floatFromInt(observed.height - 1)), (1 - proj.ndc.y) * half_h)))); // Flip Y

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

test "RayGaussian construction and position" {
    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };

    const detection = Detection2D{
        .pixel_x = 16, // Center of 32x32 image
        .pixel_y = 16,
        .color = Vec3.init(1, 0, 0),
        .radius = 5.0,
        .weight = 1.0,
    };

    const ray = RayGaussian.fromDetection(
        detection,
        camera,
        32,
        32,
        10.0, // depth prior mean
        4.0, // depth prior var
    );

    // Ray should originate from camera
    try testing.expect(ray.origin.approxEql(camera.position, 1e-6));

    // Position mean should be along ray at depth 10
    const pos = ray.positionMean();
    const expected_dir = camera.target.sub(camera.position).normalize();
    const actual_dir = pos.sub(camera.position).normalize();
    try testing.expect(actual_dir.dot(expected_dir) > 0.9);
}

test "RayGaussian overlap with GaussianComponent" {
    const ray = RayGaussian{
        .origin = Vec3.init(0, 5, 10),
        .direction = Vec3.init(0, -0.287, -0.958).normalize(),
        .depth_mean = 10.0,
        .depth_var = 4.0,
        .lateral_var = 0.1,
        .color = Vec3.init(1, 0, 0),
        .weight = 1.0,
    };

    // Component near ray should have high overlap
    const near_component = GaussianComponent{
        .position = ray.positionMean(),
        .covariance = Mat3.diagonal(Vec3.splat(0.5)),
        .weight = 1.0,
        .color = Vec3.init(1, 0, 0),
    };

    // Component far from ray should have low overlap
    const far_component = GaussianComponent{
        .position = Vec3.init(10, 10, 10),
        .covariance = Mat3.diagonal(Vec3.splat(0.5)),
        .weight = 1.0,
        .color = Vec3.init(1, 0, 0),
    };

    const near_overlap = ray.overlapLogLikelihood(near_component);
    const far_overlap = ray.overlapLogLikelihood(far_component);

    // Near should have much higher overlap
    try testing.expect(near_overlap > far_overlap + 10.0);
}

test "outerProduct symmetry" {
    const v = Vec3.init(1, 2, 3);
    const w = Vec3.init(4, 5, 6);

    const vw = outerProduct(v, w);
    const wv = outerProduct(w, v);

    // vw^T should equal wv (by definition of outer product)
    const vw_t = vw.transpose();
    for (0..9) |i| {
        try testing.expect(@abs(vw_t.data[i] - wv.data[i]) < 1e-6);
    }
}

test "FlowObservation log-likelihood" {
    // Flow observation with known covariance
    const flow_obs = FlowObservation{
        .prev_detection_idx = 0,
        .curr_detection_idx = 0,
        .flow = Vec2.init(2.0, 1.0),
        .covariance = Mat2.diagonal(Vec2.init(1.0, 1.0)),
        .weight = 1.0,
    };

    // Perfect match should have highest likelihood
    const perfect_ll = flow_obs.logLikelihood(Vec2.init(2.0, 1.0));
    const offset_ll = flow_obs.logLikelihood(Vec2.init(4.0, 1.0));

    // Perfect match should be higher
    try testing.expect(perfect_ll > offset_ll);

    // Log-likelihood should be negative (unnormalized Gaussian)
    try testing.expect(perfect_ll <= 0);
}

test "Structure tensor computation" {
    const allocator = testing.allocator;

    // Create a grid with a gradient (textured region)
    var grid = try ObservationGrid.init(16, 16, allocator);
    defer grid.deinit();

    // Create horizontal gradient
    for (0..16) |yi| {
        for (0..16) |xi| {
            const brightness = @as(f32, @floatFromInt(xi)) / 16.0;
            grid.set(@intCast(xi), @intCast(yi), .{
                .color = Vec3.init(brightness, brightness, brightness),
                .depth = 1.0,
                .occupied = true,
            });
        }
    }

    // Structure tensor at center should have strong horizontal gradient
    const M = structureTensor(grid, 8, 8, 5);

    // Ix² should be non-zero (horizontal gradient exists)
    try testing.expect(M.get(0, 0) > 0.001);

    // Iy² should be small (no vertical gradient)
    try testing.expect(M.get(1, 1) < M.get(0, 0));
}

test "Sparse flow config defaults" {
    const config = SparseFlowConfig{};

    try testing.expect(config.window_size == 7);
    try testing.expect(config.noise_floor > 0);
    try testing.expect(config.max_match_distance > 0);
}

test "AdaptedFlowConfig disabled at low resolution" {
    // Below 48px minimum dimension, flow should be disabled
    const config_32 = AdaptedFlowConfig.forResolution(32, 32);
    try testing.expect(!config_32.enabled);

    const config_40 = AdaptedFlowConfig.forResolution(40, 40);
    try testing.expect(!config_40.enabled);

    const config_47 = AdaptedFlowConfig.forResolution(47, 100);
    try testing.expect(!config_47.enabled);
}

test "AdaptedFlowConfig enabled at sufficient resolution" {
    // At 48px and above, flow should be enabled
    const config_48 = AdaptedFlowConfig.forResolution(48, 48);
    try testing.expect(config_48.enabled);

    const config_64 = AdaptedFlowConfig.forResolution(64, 64);
    try testing.expect(config_64.enabled);

    const config_640 = AdaptedFlowConfig.forResolution(640, 480);
    try testing.expect(config_640.enabled);
}

test "AdaptedFlowConfig scales window size" {
    // At 64px, window should be small (3-5px)
    const config_64 = AdaptedFlowConfig.forResolution(64, 64);
    try testing.expect(config_64.base.window_size >= 3);
    try testing.expect(config_64.base.window_size <= 5);
    // Window should be odd
    try testing.expect(config_64.base.window_size % 2 == 1);

    // At 640px, window should be larger
    const config_640 = AdaptedFlowConfig.forResolution(640, 480);
    try testing.expect(config_640.base.window_size > config_64.base.window_size);
    try testing.expect(config_640.base.window_size % 2 == 1);
}

test "AdaptedFlowConfig scales match distance" {
    // At low resolution, match distance should be small
    const config_64 = AdaptedFlowConfig.forResolution(64, 64);
    // Diagonal of 64x64 is ~90, so match distance should be ~6px
    try testing.expect(config_64.base.max_match_distance < 15.0);
    try testing.expect(config_64.base.max_match_distance > 3.0);

    // At high resolution, match distance should be larger
    const config_640 = AdaptedFlowConfig.forResolution(640, 480);
    // Diagonal of 640x480 is ~800, so match distance should be ~55px
    try testing.expect(config_640.base.max_match_distance > 40.0);
    try testing.expect(config_640.base.max_match_distance < 70.0);
}

test "AdaptedFlowConfig increases noise at low resolution" {
    const config_64 = AdaptedFlowConfig.forResolution(64, 64);
    const config_512 = AdaptedFlowConfig.forResolution(512, 512);

    // Lower resolution should have higher noise floor
    try testing.expect(config_64.base.noise_floor > config_512.base.noise_floor);

    // At 512px, noise should be close to base (0.1)
    try testing.expectApproxEqAbs(@as(f32, 0.1), config_512.base.noise_floor, 0.05);
}
