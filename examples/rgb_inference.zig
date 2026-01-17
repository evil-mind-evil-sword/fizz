//! RGB Inference Demo
//!
//! Demonstrates end-to-end inference on synthetic RGB observations.
//! Run with: zig build rgb
//!
//! Ground truth: bouncy ball dropped from height
//! Inference: SMC particle filter estimates physics type from observations

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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(12345);

    std.debug.print("\n=== RGB Inference Demo ===\n\n", .{});

    // SMC configuration
    var config = SMCConfig{};
    config.num_particles = 100;
    config.max_entities = 4;
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    // Ground is at y=0 by default via environment config
    config.observation_noise = 0.3;
    config.ess_threshold = 0.5;
    config.use_tempering = true;
    config.initial_temperature = 0.3;
    config.temperature_increment = 0.05;

    // Fixed camera
    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.5, 4.5, 9.5);
    config.camera_prior_max = Vec3.init(0.5, 5.5, 10.5);

    // Back-projection config: camera is ~10 units from ball at z=0
    // Distance along viewing ray from camera at (0,5,10) to origin is ~11.2
    config.back_projection_depth_mean = 11.0;
    config.back_projection_depth_var = 4.0; // Tighter variance for better constraint

    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Ground truth: BOUNCY ball
    const gt_physics = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }; // Bouncy
    const gt_initial_pos = Vec3.init(0, 5, 0);
    const gt_initial_vel = Vec3.zero;

    std.debug.print("Ground truth: bouncy ball (elasticity={d:.2})\n", .{
        gt_physics.elasticity,
    });
    std.debug.print("Initial position: ({d:.1}, {d:.1}, {d:.1})\n\n", .{
        gt_initial_pos.x,
        gt_initial_pos.y,
        gt_initial_pos.z,
    });

    // Initialize SMC
    const init_positions = [_]Vec3{gt_initial_pos};
    const init_velocities = [_]Vec3{gt_initial_vel};
    smc.initializeWithPrior(&init_positions, &init_velocities);

    // Ground truth state
    var gt_pos = gt_initial_pos;
    var gt_vel = gt_initial_vel;

    // Camera for rendering
    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = config.camera_intrinsics.fov,
        .aspect = config.camera_intrinsics.aspect,
        .near = config.camera_intrinsics.near,
        .far = config.camera_intrinsics.far,
    };

    const n_steps = 50;
    const obs_width: u32 = 32;
    const obs_height: u32 = 32;

    std.debug.print("Step | GT_Y  | ESS   | Temp  | Standard | Bouncy  | Sticky  | Slippery\n", .{});
    std.debug.print("-----|-------|-------|-------|----------|---------|---------|----------\n", .{});

    var bounce_count: u32 = 0;

    for (0..n_steps) |step| {
        // Step ground truth
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        // Ground collision
        if (gt_pos.y < config.physics.groundHeight()) {
            gt_pos.y = config.physics.groundHeight();
            const old_vel_y = gt_vel.y;
            gt_vel.y = -gt_vel.y * gt_physics.elasticity;
            if (old_vel_y < -1.0) bounce_count += 1;
        }

        // Render observation
        var observation = try ObservationGrid.init(obs_width, obs_height, allocator);
        defer observation.deinit();

        const gt_entity = Entity{
            .label = Label{ .birth_time = 0, .birth_index = 0 },
            .position = GaussianVec3{ .mean = gt_pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = gt_vel, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = gt_physics,
            .contact_mode = if (gt_pos.y <= config.physics.groundHeight() + 0.1) .environment else .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = Vec3.init(0.3, 1.0, 0.3),
                .opacity = 1.0,
                .radius = 0.5,
            },
        };

        const entities = [_]Entity{gt_entity};
        var gt_gmm = try GaussianMixture.fromEntities(&entities, allocator);
        defer gt_gmm.deinit();

        observation.renderGMM(gt_gmm, camera, 32);

        // Debug: Check observation stats
        if (step == 10) {
            var obs_brightness: f32 = 0;
            var obs_nonzero: u32 = 0;
            for (0..observation.height) |yi| {
                for (0..observation.width) |xi| {
                    const px = observation.get(@intCast(xi), @intCast(yi));
                    const b = px.color.x + px.color.y + px.color.z;
                    obs_brightness += b;
                    if (b > 0.01) obs_nonzero += 1;
                }
            }
            std.debug.print("\n[OBS DEBUG] GT observation: brightness={d:.1}, non_zero_pixels={d}\n", .{
                obs_brightness,
                obs_nonzero,
            });
        }

        // SMC step
        try smc.step(observation);

        // Diagnostic: Print particle position spread at bounce frames
        if (step == 10) {
            std.debug.print("\n--- Particle state at step {d} (GT_Y={d:.2}, GT_VY={d:.2}) ---\n", .{ step, gt_pos.y, gt_vel.y });
            // Compute incremental log-likelihoods for first 5 particles
            for (0..@min(5, smc.config.num_particles)) |p| {
                const entity_idx = p * smc.config.max_entities;
                const pos_y = smc.swarm.position_mean[entity_idx].y;
                const params = smc.swarm.physics_params[entity_idx];
                const weight = smc.weights[p];
                const cam = smc.swarm.camera_poses[p];
                // Compute this step's observation likelihood
                const obs_ll = smc.observationLogLikelihood(p, observation);
                std.debug.print("  P{d}: y={d:.3}, e={d:.2}, cam=({d:.2},{d:.2},{d:.2}), yaw={d:.2}, w={d:.4}, obs_ll={d:.1}\n", .{
                    p,
                    pos_y,
                    params.elasticity,
                    cam.position.x,
                    cam.position.y,
                    cam.position.z,
                    cam.yaw,
                    weight,
                    obs_ll,
                });
            }
            std.debug.print("---\n\n", .{});
        }

        // Get physics beliefs
        const beliefs = try smc.getPhysicsBelief();
        defer allocator.free(beliefs);

        if (beliefs.len > 0) {
            std.debug.print("{d:4} | {d:5.2} | {d:5.1} | {d:5.2} | {d:8.3} | {d:7.3} | {d:7.3} | {d:8.3}\n", .{
                step,
                gt_pos.y,
                smc.effectiveSampleSize(),
                smc.temperature,
                beliefs[0].type_probabilities[0], // standard
                beliefs[0].type_probabilities[1], // bouncy
                beliefs[0].type_probabilities[2], // sticky
                beliefs[0].type_probabilities[3], // slippery
            });
        }
    }

    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("Bounces observed: {d}\n", .{bounce_count});

    const final_beliefs = try smc.getPhysicsBelief();
    defer allocator.free(final_beliefs);

    if (final_beliefs.len > 0) {
        const probs = final_beliefs[0].type_probabilities;
        var max_prob: f32 = 0;
        var max_idx: usize = 0;
        for (probs, 0..) |p, i| {
            if (p > max_prob) {
                max_prob = p;
                max_idx = i;
            }
        }

        const types = [_][]const u8{ "standard", "bouncy", "sticky", "slippery" };
        std.debug.print("MAP estimate: {s} (p={d:.3})\n", .{ types[max_idx], max_prob });
        std.debug.print("Ground truth: bouncy (elasticity={d:.2})\n", .{gt_physics.elasticity});

        // Bouncy = index 1
        if (max_idx == 1) {
            std.debug.print("\n✓ Inference CORRECT!\n", .{});
        } else {
            std.debug.print("\n✗ Inference incorrect (expected bouncy)\n", .{});
        }
    }
}
