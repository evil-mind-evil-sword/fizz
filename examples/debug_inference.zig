//! Debug Inference - diagnose why tracking fails
//!
//! Run with: zig build debug-inference

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
const gmm_module = fizz.gmm;

const print = std.debug.print;

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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\n=== Debug Inference ===\n\n", .{});

    var config = SMCConfig{};
    config.num_particles = 20;
    config.max_entities = 2;
    config.physics.gravity = Vec3.init(0, -9.81, 0);
    config.physics.dt = 1.0 / 30.0;
    config.observation_noise = 0.3;
    config.ess_threshold = 0.3;

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.1, 3.9, 11.9);
    config.camera_prior_max = Vec3.init(0.1, 4.1, 12.1);
    // CRITICAL: Constrain yaw to match rendering camera (looking at -Z)
    config.camera_yaw_min = -0.05;
    config.camera_yaw_max = 0.05;
    config.back_projection_depth_mean = 12.0;
    config.back_projection_depth_var = 2.0;

    var prng = std.Random.DefaultPrng.init(42);
    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    var gt_pos = Vec3.init(0, 4, 0);
    var gt_vel = Vec3.init(2.0, 0, 0);

    // IMPORTANT: Camera must be HORIZONTAL (same Y for position and target)
    // because CameraPose only models yaw, not pitch
    const camera = Camera{
        .position = Vec3.init(0, 4, 12),
        .target = Vec3.init(0, 4, 0), // Same Y as position = horizontal look
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };

    const width: u32 = 64;
    const height: u32 = 64;

    // Initialize
    const init_pos = [_]Vec3{Vec3.init(0, 4, 0)};
    const init_vel = [_]Vec3{Vec3.zero};
    smc.initializeWithPrior(&init_pos, &init_vel);

    print("Initial particle positions:\n", .{});
    for (0..@min(5, config.num_particles)) |p| {
        const idx = p * config.max_entities;
        const pos = smc.swarm.position_mean[idx];
        print("  P{d}: ({d:.2}, {d:.2}, {d:.2})\n", .{ p, pos.x, pos.y, pos.z });
    }

    for (0..10) |frame| {
        // Step GT
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));
        if (gt_pos.y < 0.5) {
            gt_pos.y = 0.5;
            gt_vel.y = -gt_vel.y * 0.8;
        }

        print("\n--- Frame {d} ---\n", .{frame});
        print("GT position: ({d:.2}, {d:.2}, {d:.2})\n", .{ gt_pos.x, gt_pos.y, gt_pos.z });

        // Render observation
        var obs_grid = try renderEntity(gt_pos, Vec3.init(0.3, 1.0, 0.3), 0.5, camera, width, height, allocator);
        defer obs_grid.deinit();

        // Check what's in the observation grid
        var total_brightness: f32 = 0;
        var bright_pixels: u32 = 0;
        for (0..height) |yi| {
            for (0..width) |xi| {
                const px = obs_grid.get(@intCast(xi), @intCast(yi));
                const b = (px.color.x + px.color.y + px.color.z) / 3.0;
                total_brightness += b;
                if (b > 0.1) bright_pixels += 1;
            }
        }
        print("Observation: {d} bright pixels, total brightness={d:.1}\n", .{ bright_pixels, total_brightness });

        // Extract detections manually to debug
        const detections = try Detection2D.extractFromGrid(obs_grid, allocator);
        defer allocator.free(detections);

        print("Detections found: {d}\n", .{detections.len});
        for (detections, 0..) |det, i| {
            print("  Det{d}: pixel=({d:.1}, {d:.1}) radius={d:.1} weight={d:.2}\n", .{
                i,
                det.pixel_x,
                det.pixel_y,
                det.radius,
                det.weight,
            });
        }

        // Project GT to image to see where it should be
        if (camera.project(gt_pos)) |proj| {
            const half_w: f32 = @floatFromInt(width / 2);
            const half_h: f32 = @floatFromInt(height / 2);
            const px = (proj.ndc.x + 1) * half_w;
            const py = (1 - proj.ndc.y) * half_h; // Flip Y for image coords
            print("GT projects to pixel: ({d:.1}, {d:.1})\n", .{ px, py });
        } else {
            print("GT not visible in camera\n", .{});
        }

        // Manually compute what the back-projection would give for particle 0
        const camera_pose = smc.swarm.camera_poses[0];
        const test_camera = camera_pose.toCamera(smc.config.camera_intrinsics);
        if (detections.len > 0) {
            const det = detections[0];
            const half_w2: f32 = @floatFromInt(width / 2);
            const half_h2: f32 = @floatFromInt(height / 2);
            const ndc_x = (det.pixel_x - half_w2) / half_w2;
            const ndc_y = (half_h2 - det.pixel_y) / half_h2;
            const ray_dir = gmm_module.computeRayDirection(test_camera, ndc_x, ndc_y);
            const depth_mean = smc.config.back_projection_depth_mean;
            const back_proj = test_camera.position.add(ray_dir.scale(depth_mean));
            print("Back-projection: cam=({d:.2},{d:.2},{d:.2}) ray=({d:.3},{d:.3},{d:.3}) -> 3D=({d:.2},{d:.2},{d:.2})\n", .{
                test_camera.position.x,
                test_camera.position.y,
                test_camera.position.z,
                ray_dir.x,
                ray_dir.y,
                ray_dir.z,
                back_proj.x,
                back_proj.y,
                back_proj.z,
            });
        }

        // Run SMC step
        try smc.step(obs_grid);

        // Check particle positions after step
        print("After SMC step:\n", .{});
        var sum_pos = Vec3.zero;
        var sum_weight: f32 = 0;
        for (0..@min(5, config.num_particles)) |p| {
            const idx = p * config.max_entities;
            const pos = smc.swarm.position_mean[idx];
            const w = smc.weights[p];
            sum_pos = sum_pos.add(pos.scale(w));
            sum_weight += w;
            if (p < 3) {
                print("  P{d}: ({d:.2}, {d:.2}, {d:.2}) w={d:.4}\n", .{ p, pos.x, pos.y, pos.z, w });
            }
        }
        if (sum_weight > 0) {
            const mean_pos = sum_pos.scale(1.0 / sum_weight);
            const err = gt_pos.sub(mean_pos).length();
            print("Weighted mean: ({d:.2}, {d:.2}, {d:.2}) error={d:.2}\n", .{
                mean_pos.x,
                mean_pos.y,
                mean_pos.z,
                err,
            });
        }
        print("ESS: {d:.1}\n", .{smc.effectiveSampleSize()});
    }
}
