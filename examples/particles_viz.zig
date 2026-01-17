//! Particle Cloud Visualization Demo
//!
//! Shows the full particle filter belief as a cloud of hypotheses:
//! - Left: Ground truth observation
//! - Right: All particles rendered (transparency based on weight)
//!
//! Run with: zig build particles-viz

const std = @import("std");
const fizz = @import("fizz");

const Vec3 = fizz.Vec3;
const Mat3 = fizz.Mat3;
const Entity = fizz.Entity;
const Label = fizz.Label;
const PhysicsParams = fizz.PhysicsParams;
const GaussianVec3 = fizz.GaussianVec3;
const Camera = fizz.Camera;
const GaussianMixture = fizz.GaussianMixture;
const ObservationGrid = fizz.ObservationGrid;
const SMCState = fizz.SMCState;
const SMCConfig = fizz.SMCConfig;

const print = std.debug.print;

/// Fast rasterization: draw filled circles instead of ray marching
fn renderParticlesFast(
    positions: []const Vec3,
    weights: []const f32,
    base_color: Vec3,
    radius: f32,
    camera: Camera,
    width: u32,
    height: u32,
    allocator: std.mem.Allocator,
) !ObservationGrid {
    var grid = try ObservationGrid.init(width, height, allocator);
    errdefer grid.deinit();

    // Find max weight for normalization
    var max_weight: f32 = 0;
    for (weights) |w| {
        if (w > max_weight) max_weight = w;
    }
    if (max_weight == 0) max_weight = 1;

    const half_w: f32 = @floatFromInt(width / 2);
    const half_h: f32 = @floatFromInt(height / 2);

    // Project and draw each particle as a circle
    for (positions, weights) |pos, w| {
        const proj = camera.project(pos) orelse continue;

        // NDC to pixel (Y flipped for image coords)
        const px = (proj.ndc.x + 1) * half_w;
        const py = (1 - proj.ndc.y) * half_h;

        // Pixel radius based on depth
        const pixel_radius = radius * half_h / (proj.depth * @tan(camera.fov / 2));
        const r_int: i32 = @intFromFloat(@max(2, pixel_radius));

        // Opacity from weight
        const alpha = @max(0.2, w / max_weight);
        const color = base_color.scale(alpha);

        // Draw filled circle
        const cx: i32 = @intFromFloat(px);
        const cy: i32 = @intFromFloat(py);
        const w_i: i32 = @intCast(width);
        const h_i: i32 = @intCast(height);

        var dy: i32 = -r_int;
        while (dy <= r_int) : (dy += 1) {
            var dx: i32 = -r_int;
            while (dx <= r_int) : (dx += 1) {
                if (dx * dx + dy * dy <= r_int * r_int) {
                    const x = cx + dx;
                    const y = cy + dy;
                    if (x >= 0 and x < w_i and y >= 0 and y < h_i) {
                        const existing = grid.get(@intCast(x), @intCast(y));
                        // Additive blend
                        const blended = existing.color.add(color);
                        grid.set(@intCast(x), @intCast(y), .{
                            .color = Vec3.init(
                                @min(1.0, blended.x),
                                @min(1.0, blended.y),
                                @min(1.0, blended.z),
                            ),
                            .depth = @min(existing.depth, proj.depth),
                            .occupied = true,
                        });
                    }
                }
            }
        }
    }

    return grid;
}

/// Fast single entity render
fn renderEntityFast(
    pos: Vec3,
    color: Vec3,
    radius: f32,
    camera: Camera,
    width: u32,
    height: u32,
    allocator: std.mem.Allocator,
) !ObservationGrid {
    var grid = try ObservationGrid.init(width, height, allocator);
    errdefer grid.deinit();

    const proj = camera.project(pos) orelse return grid;

    const half_w: f32 = @floatFromInt(width / 2);
    const half_h: f32 = @floatFromInt(height / 2);

    const px = (proj.ndc.x + 1) * half_w;
    const py = (1 - proj.ndc.y) * half_h;

    const pixel_radius = radius * half_h / (proj.depth * @tan(camera.fov / 2));
    const r_int: i32 = @intFromFloat(@max(3, pixel_radius));

    const cx: i32 = @intFromFloat(px);
    const cy: i32 = @intFromFloat(py);
    const w_i: i32 = @intCast(width);
    const h_i: i32 = @intCast(height);

    var dy: i32 = -r_int;
    while (dy <= r_int) : (dy += 1) {
        var dx: i32 = -r_int;
        while (dx <= r_int) : (dx += 1) {
            const dist_sq = dx * dx + dy * dy;
            if (dist_sq <= r_int * r_int) {
                const x = cx + dx;
                const y = cy + dy;
                if (x >= 0 and x < w_i and y >= 0 and y < h_i) {
                    // Soft edge falloff
                    const dist: f32 = @sqrt(@as(f32, @floatFromInt(dist_sq)));
                    const edge = 1.0 - @min(1.0, dist / @as(f32, @floatFromInt(r_int)));
                    grid.set(@intCast(x), @intCast(y), .{
                        .color = color.scale(edge),
                        .depth = proj.depth,
                        .occupied = true,
                    });
                }
            }
        }
    }

    return grid;
}

/// Accurate GMM render for SMC observations
fn renderEntityGMM(
    pos: Vec3,
    color: Vec3,
    radius: f32,
    camera: Camera,
    width: u32,
    height: u32,
    allocator: std.mem.Allocator,
) !ObservationGrid {
    var grid = try ObservationGrid.init(width, height, allocator);
    errdefer grid.deinit();

    const entity = Entity{
        .label = Label{ .birth_time = 0, .birth_index = 0 },
        .position = GaussianVec3{ .mean = pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
        .velocity = GaussianVec3{ .mean = Vec3.zero, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
        .physics_params = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }, // Bouncy
        .contact_mode = .free,
        .track_state = .detected,
        .occlusion_count = 0,
        .appearance = .{ .color = color, .opacity = 1.0, .radius = radius },
    };

    const entities = [_]Entity{entity};
    var gmm = try GaussianMixture.fromEntities(&entities, allocator);
    defer gmm.deinit();

    grid.renderGMM(gmm, camera, 16); // Fewer samples for speed

    return grid;
}

/// Composite two grids side-by-side
fn compositeSideBySide(
    left: ObservationGrid,
    right: ObservationGrid,
    allocator: std.mem.Allocator,
) !ObservationGrid {
    const gap: u32 = 4;
    const total_width = left.width + gap + right.width;
    const height = @max(left.height, right.height);

    var composite = try ObservationGrid.init(total_width, height, allocator);

    for (0..left.height) |yi| {
        for (0..left.width) |xi| {
            composite.set(@intCast(xi), @intCast(yi), left.get(@intCast(xi), @intCast(yi)));
        }
    }

    for (0..height) |yi| {
        for (0..gap) |gi| {
            const x: u32 = left.width + @as(u32, @intCast(gi));
            composite.set(x, @intCast(yi), .{
                .color = Vec3.init(0.3, 0.3, 0.3),
                .depth = std.math.inf(f32),
                .occupied = false,
            });
        }
    }

    for (0..right.height) |yi| {
        for (0..right.width) |xi| {
            const x: u32 = left.width + gap + @as(u32, @intCast(xi));
            composite.set(x, @intCast(yi), right.get(@intCast(xi), @intCast(yi)));
        }
    }

    return composite;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\n=== Particle Cloud Visualization ===\n\n", .{});

    std.fs.cwd().makeDir("viz_particles") catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    // SMC config - reduced for speed
    var config = SMCConfig{};
    config.num_particles = 30; // Fewer particles
    config.max_entities = 2;
    config.physics.gravity = Vec3.init(0, -9.81, 0);
    config.physics.dt = 1.0 / 30.0;
    config.physics.process_noise = 1.0; // Higher process noise = Kalman filter stays responsive
    // Set ground height to ball radius (0.5) so SMC bounce matches GT
    config.physics.environment.ground = .{
        .height = 0.5,
        .normal = Vec3.unit_y,
        .friction = 0.5,
        .elasticity = 0.9, // Match bouncy physics type
    };
    config.observation_noise = 0.1; // Lower noise = trust observations more
    config.ess_threshold = 0.3;

    // DISABLED: Flow observations corrupt velocity at low resolution
    config.use_flow_observations = false;
    // config.sparse_flow_config = .{
    //     .window_size = 5,
    //     .noise_floor = 0.1,
    //     .min_weight = 0.1,
    // };

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    // Camera prior must match horizontal look direction (CameraPose has no pitch)
    config.camera_prior_min = Vec3.init(-0.1, 3.9, 11.9);
    config.camera_prior_max = Vec3.init(0.1, 4.1, 12.1);
    config.camera_yaw_min = -0.05;
    config.camera_yaw_max = 0.05;
    config.back_projection_depth_mean = 12.0;
    config.back_projection_depth_var = 0.1; // Much tighter depth prior for better lateral tracking

    var prng = std.Random.DefaultPrng.init(12345);
    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Ground truth: ball with horizontal motion - starts high for multiple bounces
    const gt_physics = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }; // Bouncy
    var gt_pos = Vec3.init(-3, 5, 0);
    var gt_vel = Vec3.init(2.0, 0.0, 0); // Slower horizontal for longer trajectory

    // Camera must be HORIZONTAL (CameraPose only models yaw, not pitch)
    const camera = Camera{
        .position = Vec3.init(0, 4, 12),
        .target = Vec3.init(0, 4, 0), // Same Y = horizontal look
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };

    const panel_width: u32 = 64; // Smaller for speed
    const panel_height: u32 = 64;

    // Initialize with uncertain prior - use GT velocity to test position tracking
    const init_pos = [_]Vec3{Vec3.init(-3, 5, 0)};
    const init_vel = [_]Vec3{Vec3.init(2.0, 0, 0)}; // Match GT velocity
    smc.initializeWithPrior(&init_pos, &init_vel);

    const n_frames: u32 = 120; // More frames for multiple bounces

    print("Rendering {d} frames with {d} particles...\n", .{ n_frames, config.num_particles });

    // Allocate arrays for particle positions
    var particle_positions = try allocator.alloc(Vec3, config.num_particles);
    defer allocator.free(particle_positions);
    var particle_weights = try allocator.alloc(f32, config.num_particles);
    defer allocator.free(particle_weights);

    for (0..n_frames) |frame| {
        // Step ground truth
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        // Ground collision
        if (gt_pos.y < 0.5) {
            gt_pos.y = 0.5;
            gt_vel.y = -gt_vel.y * gt_physics.elasticity;
        }

        // Wall collisions
        if (gt_pos.x > 4.0) {
            gt_pos.x = 4.0;
            gt_vel.x = -gt_vel.x * 0.9;
        }
        if (gt_pos.x < -4.0) {
            gt_pos.x = -4.0;
            gt_vel.x = -gt_vel.x * 0.9;
        }

        // Fast GT visualization (for display only)
        var gt_grid = try renderEntityFast(gt_pos, Vec3.init(0.3, 1.0, 0.3), 0.5, camera, panel_width, panel_height, allocator);
        defer gt_grid.deinit();

        // SMC needs accurate observation
        var smc_obs = try renderEntityGMM(gt_pos, Vec3.init(0.3, 1.0, 0.3), 0.5, camera, 64, 64, allocator);
        defer smc_obs.deinit();

        // SMC step
        try smc.step(smc_obs);

        // Collect particle positions and weights
        var n_active: usize = 0;
        for (0..config.num_particles) |p| {
            const entity_idx = p * config.max_entities;
            if (smc.swarm.track_state[entity_idx] != .dead) {
                particle_positions[n_active] = smc.swarm.position_mean[entity_idx];
                particle_weights[n_active] = smc.weights[p];
                n_active += 1;
            }
        }

        // Render particle cloud
        var particle_grid: ObservationGrid = undefined;
        if (n_active > 0) {
            particle_grid = try renderParticlesFast(
                particle_positions[0..n_active],
                particle_weights[0..n_active],
                Vec3.init(1.0, 0.5, 0.2), // Orange particles
                0.5,
                camera,
                panel_width,
                panel_height,
                allocator,
            );
        } else {
            particle_grid = try ObservationGrid.init(panel_width, panel_height, allocator);
        }
        defer particle_grid.deinit();

        // Composite
        var composite = try compositeSideBySide(gt_grid, particle_grid, allocator);
        defer composite.deinit();

        // Save frame
        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "viz_particles/frame_{d:0>3}.ppm", .{frame});
        try composite.writePPM(path);

        // Compute weighted mean of particles
        var mean_pos = Vec3.zero;
        var total_w: f32 = 0;
        for (0..n_active) |idx| {
            mean_pos = mean_pos.add(particle_positions[idx].scale(particle_weights[idx]));
            total_w += particle_weights[idx];
        }
        if (total_w > 0) mean_pos = mean_pos.scale(1.0 / total_w);

        if (frame % 10 == 0) {
            const err = gt_pos.sub(mean_pos).length();
            print("  Frame {d:3}: GT=({d:5.2}, {d:5.2}) Est=({d:5.2}, {d:5.2}) Err={d:.2}\n", .{
                frame,
                gt_pos.x,
                gt_pos.y,
                mean_pos.x,
                mean_pos.y,
                err,
            });
        }
    }

    print("\nFrames saved to viz_particles/\n", .{});
}
