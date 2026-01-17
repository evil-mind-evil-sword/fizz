//! Inference Visualization Demo
//!
//! Renders an animation showing SMC inference in action:
//! - Left: Ground truth RGB observation
//! - Right: Particle filter's belief state (weighted mean)
//! - Markers show detected blobs
//!
//! Run with: zig build inference-viz

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
const Detection2D = fizz.Detection2D;

const print = std.debug.print;

/// Render a single entity to observation grid
fn renderEntity(
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
        .appearance = .{
            .color = color,
            .opacity = 1.0,
            .radius = radius,
        },
    };

    const entities = [_]Entity{entity};
    var gmm = try GaussianMixture.fromEntities(&entities, allocator);
    defer gmm.deinit();

    grid.renderGMM(gmm, camera, 32);

    return grid;
}

/// Composite two grids side-by-side with a separator
fn compositeSideBySide(
    left: ObservationGrid,
    right: ObservationGrid,
    allocator: std.mem.Allocator,
) !ObservationGrid {
    const gap: u32 = 4; // Separator width
    const total_width = left.width + gap + right.width;
    const height = @max(left.height, right.height);

    var composite = try ObservationGrid.init(total_width, height, allocator);

    // Copy left panel
    for (0..left.height) |yi| {
        for (0..left.width) |xi| {
            composite.set(@intCast(xi), @intCast(yi), left.get(@intCast(xi), @intCast(yi)));
        }
    }

    // Draw separator (white line)
    for (0..height) |yi| {
        for (0..gap) |gi| {
            const x: u32 = left.width + @as(u32, @intCast(gi));
            composite.set(x, @intCast(yi), .{
                .color = Vec3.init(0.3, 0.3, 0.3), // Gray separator
                .depth = std.math.inf(f32),
                .occupied = false,
            });
        }
    }

    // Copy right panel
    for (0..right.height) |yi| {
        for (0..right.width) |xi| {
            const x: u32 = left.width + gap + @as(u32, @intCast(xi));
            composite.set(x, @intCast(yi), right.get(@intCast(xi), @intCast(yi)));
        }
    }

    return composite;
}

/// Draw a crosshair marker at a 2D position
fn drawMarker(grid: *ObservationGrid, px: f32, py: f32, color: Vec3) void {
    const ix: i32 = @intFromFloat(px);
    const iy: i32 = @intFromFloat(py);
    const w: i32 = @intCast(grid.width);
    const h: i32 = @intCast(grid.height);

    // Draw a small cross
    const offsets = [_][2]i32{
        .{ 0, 0 },
        .{ -1, 0 },
        .{ 1, 0 },
        .{ -2, 0 },
        .{ 2, 0 },
        .{ 0, -1 },
        .{ 0, 1 },
        .{ 0, -2 },
        .{ 0, 2 },
    };

    for (offsets) |off| {
        const x = ix + off[0];
        const y = iy + off[1];
        if (x >= 0 and x < w and y >= 0 and y < h) {
            grid.set(@intCast(x), @intCast(y), .{
                .color = color,
                .depth = 0,
                .occupied = true,
            });
        }
    }
}

/// Project 3D position to 2D pixel coordinates
fn projectToPixel(pos: Vec3, camera: Camera, width: u32, height: u32) ?[2]f32 {
    // View direction
    const forward = camera.target.sub(camera.position).normalize();
    const right = forward.cross(camera.up).normalize();
    const up = right.cross(forward);

    // Vector from camera to point
    const to_point = pos.sub(camera.position);
    const depth = to_point.dot(forward);

    if (depth <= camera.near) return null;

    // Project onto image plane
    const x_cam = to_point.dot(right);
    const y_cam = to_point.dot(up);

    const tan_half_fov = @tan(camera.fov / 2.0);
    const ndc_x = x_cam / (depth * tan_half_fov * camera.aspect);
    const ndc_y = y_cam / (depth * tan_half_fov);

    // NDC to pixel
    const half_w: f32 = @floatFromInt(width / 2);
    const half_h: f32 = @floatFromInt(height / 2);
    const px = ndc_x * half_w + half_w;
    const py = half_h - ndc_y * half_h; // Flip Y

    if (px < 0 or px >= @as(f32, @floatFromInt(width)) or
        py < 0 or py >= @as(f32, @floatFromInt(height)))
    {
        return null;
    }

    return .{ px, py };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\n=== Inference Visualization Demo ===\n\n", .{});

    // Create output directory
    std.fs.cwd().makeDir("viz_inference") catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    // SMC configuration
    var config = SMCConfig{};
    config.num_particles = 50;
    config.max_entities = 2;
    config.physics.gravity = Vec3.init(0, -9.81, 0);
    config.physics.dt = 1.0 / 30.0;
    config.observation_noise = 0.2;
    config.ess_threshold = 0.5;

    // Camera setup
    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.1, 4.9, 11.9);
    config.camera_prior_max = Vec3.init(0.1, 5.1, 12.1);
    config.back_projection_depth_mean = 12.0;
    config.back_projection_depth_var = 2.0;

    var prng = std.Random.DefaultPrng.init(42);
    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Ground truth: bouncy ball
    const gt_physics = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }; // Bouncy
    var gt_pos = Vec3.init(0, 5, 0);
    var gt_vel = Vec3.init(2.0, 0, 0); // Moving right

    // Fixed camera for rendering
    const camera = Camera{
        .position = Vec3.init(0, 5, 12),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };

    const panel_width: u32 = 128;
    const panel_height: u32 = 128;

    // Initialize SMC with prior
    const init_pos = [_]Vec3{Vec3.init(0, 5, 0)};
    const init_vel = [_]Vec3{Vec3.init(0, 0, 0)}; // Unknown velocity
    smc.initializeWithPrior(&init_pos, &init_vel);

    const n_frames: u32 = 60;

    print("Rendering {d} frames...\n", .{n_frames});

    for (0..n_frames) |frame| {
        // Step ground truth physics
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        // Ground collision
        if (gt_pos.y < 0.5) {
            gt_pos.y = 0.5;
            gt_vel.y = -gt_vel.y * gt_physics.elasticity;
        }

        // Wall collision (keep in view)
        if (gt_pos.x > 4.0) {
            gt_pos.x = 4.0;
            gt_vel.x = -gt_vel.x * 0.8;
        }
        if (gt_pos.x < -4.0) {
            gt_pos.x = -4.0;
            gt_vel.x = -gt_vel.x * 0.8;
        }

        // Render ground truth observation
        var gt_grid = try renderEntity(gt_pos, Vec3.init(0.3, 1.0, 0.3), 0.5, camera, panel_width, panel_height, allocator);
        defer gt_grid.deinit();

        // Run SMC inference step
        try smc.step(gt_grid);

        // Get SMC's belief (weighted mean position)
        var belief_pos = Vec3.zero;
        var total_weight: f32 = 0;
        for (0..config.num_particles) |p| {
            const w = smc.weights[p];
            const entity_idx = p * config.max_entities;
            if (smc.swarm.track_state[entity_idx] != .dead) {
                belief_pos = belief_pos.add(smc.swarm.position_mean[entity_idx].scale(w));
                total_weight += w;
            }
        }
        if (total_weight > 0) {
            belief_pos = belief_pos.scale(1.0 / total_weight);
        }

        // Render belief state (show as orange ball)
        var belief_grid = try renderEntity(belief_pos, Vec3.init(1.0, 0.6, 0.2), 0.5, camera, panel_width, panel_height, allocator);
        defer belief_grid.deinit();

        // Draw markers for some particles (show uncertainty)
        for (0..@min(10, config.num_particles)) |p| {
            const entity_idx = p * config.max_entities;
            if (smc.swarm.track_state[entity_idx] != .dead) {
                const particle_pos = smc.swarm.position_mean[entity_idx];
                if (projectToPixel(particle_pos, camera, panel_width, panel_height)) |px| {
                    // Draw faint particle markers
                    drawMarker(&belief_grid, px[0], px[1], Vec3.init(0.5, 0.3, 0.1));
                }
            }
        }

        // Draw ground truth marker on left panel
        if (projectToPixel(gt_pos, camera, panel_width, panel_height)) |px| {
            drawMarker(&gt_grid, px[0], px[1], Vec3.init(1.0, 1.0, 1.0));
        }

        // Draw belief marker on right panel
        if (projectToPixel(belief_pos, camera, panel_width, panel_height)) |px| {
            drawMarker(&belief_grid, px[0], px[1], Vec3.init(1.0, 1.0, 1.0));
        }

        // Composite side-by-side
        var composite = try compositeSideBySide(gt_grid, belief_grid, allocator);
        defer composite.deinit();

        // Save frame
        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "viz_inference/frame_{d:0>3}.ppm", .{frame});
        try composite.writePPM(path);

        // Progress indicator
        if (frame % 10 == 0) {
            const err = @abs(gt_pos.sub(belief_pos).length());
            print("  Frame {d:3}: GT=({d:5.2}, {d:5.2}) Belief=({d:5.2}, {d:5.2}) Err={d:.3}\n", .{
                frame,
                gt_pos.x,
                gt_pos.y,
                belief_pos.x,
                belief_pos.y,
                err,
            });
        }
    }

    print("\nFrames saved to viz_inference/\n", .{});
    print("Creating GIF...\n", .{});
}
