//! Fizz Inference Demo: Recovering Physics Types via SMC
//!
//! This demo:
//! 1. Generates "ground truth" observations from entities with KNOWN physics types
//! 2. Runs SMC inference with physics types UNKNOWN
//! 3. Compares inferred physics types to ground truth
//!
//! The key insight: By observing how entities bounce/slide/stick,
//! we can infer their underlying physics properties.

const std = @import("std");
const fizz = @import("root.zig");

const Vec3 = fizz.Vec3;
const Entity = fizz.Entity;
const Label = fizz.Label;
const PhysicsType = fizz.PhysicsType;
const PhysicsConfig = fizz.PhysicsConfig;
const Camera = fizz.Camera;
const ObservationGrid = fizz.ObservationGrid;
const SMCConfig = fizz.SMCConfig;
const SMCState = fizz.SMCState;

const print = std.debug.print;

/// Generate ground truth observations by simulating with known physics types
fn generateGroundTruth(
    allocator: std.mem.Allocator,
    positions: []const Vec3,
    velocities: []const Vec3,
    true_physics: []const PhysicsType,
    config: PhysicsConfig,
    camera: Camera,
    num_steps: u32,
    rng: std.Random,
) ![]ObservationGrid {
    var observations = try allocator.alloc(ObservationGrid, num_steps);
    errdefer {
        for (observations) |*obs| {
            obs.deinit();
        }
        allocator.free(observations);
    }

    // Create entities with known physics types
    var entities: std.ArrayList(Entity) = .empty;
    defer entities.deinit(allocator);

    for (positions, velocities, true_physics, 0..) |pos, vel, ptype, idx| {
        const label = Label{ .birth_time = 0, .birth_index = @intCast(idx) };
        var entity = Entity.initPoint(label, pos, vel, ptype);
        entity.appearance.radius = 0.5;
        entity.appearance.color = switch (ptype) {
            .standard => Vec3.init(1.0, 0.3, 0.3),
            .bouncy => Vec3.init(0.3, 1.0, 0.3),
            .sticky => Vec3.init(0.3, 0.3, 1.0),
            .slippery => Vec3.init(1.0, 1.0, 0.3),
        };
        try entities.append(allocator, entity);
    }

    // Simulate and render
    for (0..num_steps) |step| {
        // Render current state
        var gmm_model = try fizz.GaussianMixture.fromEntities(entities.items, allocator);
        defer gmm_model.deinit();

        var grid = try ObservationGrid.init(8, 8, allocator);
        grid.renderGMM(gmm_model, camera, 64);
        observations[step] = grid;

        // Step physics
        for (entities.items) |*entity| {
            if (entity.isAlive()) {
                fizz.entityPhysicsStep(entity, config, rng);
            }
        }

        // Entity-entity collisions
        for (0..entities.items.len) |i| {
            for (i + 1..entities.items.len) |j| {
                var e1 = &entities.items[i];
                var e2 = &entities.items[j];
                if (e1.isAlive() and e2.isAlive()) {
                    if (fizz.checkEntityContact(e1.*, e2.*)) {
                        fizz.resolveEntityCollision(e1, e2);
                    }
                }
            }
        }
    }

    return observations;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Fizz: Physics Type Inference via SMC ===\n\n", .{});

    // Setup
    const physics_config = PhysicsConfig{
        .gravity = Vec3.init(0, -9.81, 0),
        .dt = 1.0 / 30.0, // 30 FPS for faster simulation
        .ground_height = 0.0,
    };

    const camera = Camera{
        .position = Vec3.init(0, 3, 10),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 30.0,
    };

    // Ground truth: 2 entities with DIFFERENT physics types
    const positions = [_]Vec3{ Vec3.init(-1, 5, 0), Vec3.init(1, 5, 0) };
    const velocities = [_]Vec3{ Vec3.zero, Vec3.zero };
    const true_physics = [_]PhysicsType{ .bouncy, .sticky };

    print("Ground Truth:\n", .{});
    for (true_physics, 0..) |ptype, i| {
        print("  Entity {d}: {s}\n", .{ i, @tagName(ptype) });
    }

    // Initialize RNG
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const rng = prng.random();

    // Generate observations
    const num_steps: u32 = 60; // 2 seconds of simulation
    print("\nGenerating {d} ground truth observations...\n", .{num_steps});

    const observations = try generateGroundTruth(
        allocator,
        &positions,
        &velocities,
        &true_physics,
        physics_config,
        camera,
        num_steps,
        rng,
    );
    defer {
        for (observations) |*obs| {
            obs.deinit();
        }
        allocator.free(observations);
    }

    // Setup SMC
    const smc_config = SMCConfig{
        .num_particles = 50,
        .ess_threshold = 0.5,
        .observation_noise = 0.3, // Soft likelihood for stable inference
        .use_tempering = true,
        .initial_temperature = 0.1,
        .temperature_increment = 0.05,
        .gibbs_sweeps = 2,
        .physics = physics_config,
    };

    print("\nSMC Configuration:\n", .{});
    print("  Particles: {d}\n", .{smc_config.num_particles});
    print("  Observation noise: {d:.2}\n", .{smc_config.observation_noise});
    print("  Tempering: {s} (initial={d:.2}, increment={d:.2})\n", .{
        if (smc_config.use_tempering) "yes" else "no",
        smc_config.initial_temperature,
        smc_config.temperature_increment,
    });

    // Initialize SMC
    var smc = try SMCState.init(allocator, smc_config, rng);
    defer smc.deinit();

    // Initialize particles with prior (unknown physics types)
    try smc.initializeWithPrior(&positions, &velocities);

    print("\nRunning SMC inference...\n", .{});

    // Run inference
    for (observations, 0..) |observation, step| {
        try smc.step(observation, camera);

        // Print progress every 10 steps
        if (step % 10 == 0 or step == observations.len - 1) {
            const ess = smc.effectiveSampleSize();
            const posteriors = try smc.getPhysicsTypePosterior();
            defer allocator.free(posteriors);

            print("\n  Step {d}/{d} (temp={d:.2}, ESS={d:.1}):\n", .{
                step,
                observations.len,
                smc.temperature,
                ess,
            });

            for (posteriors, 0..) |post, i| {
                print("    Entity {d}: std={d:.2} bouncy={d:.2} sticky={d:.2} slip={d:.2}\n", .{
                    i,
                    post[0],
                    post[1],
                    post[2],
                    post[3],
                });
            }
        }
    }

    // Final results
    print("\n=== Inference Results ===\n", .{});

    const estimates = try smc.getPhysicsTypeEstimate();
    defer allocator.free(estimates);

    const posteriors = try smc.getPhysicsTypePosterior();
    defer allocator.free(posteriors);

    var correct: u32 = 0;
    for (estimates, true_physics, posteriors, 0..) |est, truth, post, i| {
        const match = est == truth;
        if (match) correct += 1;

        print("\nEntity {d}:\n", .{i});
        print("  True:      {s}\n", .{@tagName(truth)});
        print("  Inferred:  {s} {s}\n", .{ @tagName(est), if (match) "(CORRECT)" else "(WRONG)" });
        print("  Posterior: std={d:.2} bouncy={d:.2} sticky={d:.2} slip={d:.2}\n", .{
            post[0],
            post[1],
            post[2],
            post[3],
        });
    }

    print("\n=== Summary ===\n", .{});
    print("Accuracy: {d}/{d} entities correctly identified\n", .{ correct, estimates.len });
    print("Final temperature: {d:.2}\n", .{smc.temperature});
    print("Final ESS: {d:.1}/{d}\n", .{ smc.effectiveSampleSize(), smc_config.num_particles });

    if (correct == estimates.len) {
        print("\nSUCCESS: All physics types correctly inferred!\n", .{});
    } else {
        print("\nNote: Inference is probabilistic - results may vary.\n", .{});
        print("Try adjusting observation_noise or num_particles.\n", .{});
    }

    // ==========================================================================
    // Test 2: Uniform colors (physics-only inference, no color cues)
    // ==========================================================================
    print("\n\n=== Test 2: Uniform Colors (Physics-Only Inference) ===\n", .{});
    print("This tests inference without color cues - must rely on dynamics behavior.\n\n", .{});

    // Generate observations with uniform gray color
    var uniform_observations = try allocator.alloc(ObservationGrid, num_steps);
    errdefer {
        for (uniform_observations) |*obs| {
            obs.deinit();
        }
        allocator.free(uniform_observations);
    }

    // Create entities with uniform color but different physics
    {
        var entities: std.ArrayList(Entity) = .empty;
        defer entities.deinit(allocator);

        for (positions, velocities, true_physics, 0..) |pos, vel, ptype, idx| {
            const label = Label{ .birth_time = 0, .birth_index = @intCast(idx) };
            var entity = Entity.initPoint(label, pos, vel, ptype);
            entity.appearance.radius = 0.5;
            entity.appearance.color = Vec3.init(0.7, 0.7, 0.7); // Uniform gray
            try entities.append(allocator, entity);
        }

        // Simulate and render with uniform colors
        for (0..num_steps) |step| {
            var gmm_model = try fizz.GaussianMixture.fromEntities(entities.items, allocator);
            defer gmm_model.deinit();

            var grid = try ObservationGrid.init(8, 8, allocator);
            grid.renderGMM(gmm_model, camera, 64);
            uniform_observations[step] = grid;

            for (entities.items) |*entity| {
                if (entity.isAlive()) {
                    fizz.entityPhysicsStep(entity, physics_config, rng);
                }
            }

            for (0..entities.items.len) |i| {
                for (i + 1..entities.items.len) |j| {
                    var e1 = &entities.items[i];
                    var e2 = &entities.items[j];
                    if (e1.isAlive() and e2.isAlive()) {
                        if (fizz.checkEntityContact(e1.*, e2.*)) {
                            fizz.resolveEntityCollision(e1, e2);
                        }
                    }
                }
            }
        }
    }
    defer {
        for (uniform_observations) |*obs| {
            obs.deinit();
        }
        allocator.free(uniform_observations);
    }

    // Run SMC with uniform colors
    const uniform_config = SMCConfig{
        .num_particles = 100, // More particles needed without color cues
        .ess_threshold = 0.5,
        .observation_noise = 0.3,
        .use_tempering = true,
        .initial_temperature = 0.1,
        .temperature_increment = 0.03, // Slower tempering
        .gibbs_sweeps = 3,
        .physics = physics_config,
        .use_uniform_colors = true, // Critical: disable color-physics coupling
        .uniform_color = Vec3.init(0.7, 0.7, 0.7),
    };

    var smc2 = try SMCState.init(allocator, uniform_config, rng);
    defer smc2.deinit();

    try smc2.initializeWithPrior(&positions, &velocities);

    print("Running SMC inference (uniform colors)...\n", .{});

    for (uniform_observations, 0..) |observation, step| {
        try smc2.step(observation, camera);

        if (step % 20 == 0 or step == uniform_observations.len - 1) {
            const posteriors2 = try smc2.getPhysicsTypePosterior();
            defer allocator.free(posteriors2);

            print("\n  Step {d}/{d} (temp={d:.2}, ESS={d:.1}):\n", .{
                step,
                uniform_observations.len,
                smc2.temperature,
                smc2.effectiveSampleSize(),
            });

            for (posteriors2, 0..) |post, i| {
                print("    Entity {d}: std={d:.2} bouncy={d:.2} sticky={d:.2} slip={d:.2}\n", .{
                    i,
                    post[0],
                    post[1],
                    post[2],
                    post[3],
                });
            }
        }
    }

    // Final results for uniform colors test
    print("\n=== Uniform Colors Results ===\n", .{});

    const estimates2 = try smc2.getPhysicsTypeEstimate();
    defer allocator.free(estimates2);

    const posteriors2 = try smc2.getPhysicsTypePosterior();
    defer allocator.free(posteriors2);

    var correct2: u32 = 0;
    for (estimates2, true_physics, posteriors2, 0..) |est, truth, post, i| {
        const match = est == truth;
        if (match) correct2 += 1;

        print("\nEntity {d}:\n", .{i});
        print("  True:      {s}\n", .{@tagName(truth)});
        print("  Inferred:  {s} {s}\n", .{ @tagName(est), if (match) "(CORRECT)" else "(WRONG)" });
        print("  Posterior: std={d:.2} bouncy={d:.2} sticky={d:.2} slip={d:.2}\n", .{
            post[0],
            post[1],
            post[2],
            post[3],
        });
    }

    print("\n=== Uniform Colors Summary ===\n", .{});
    print("Accuracy: {d}/{d} entities correctly identified\n", .{ correct2, estimates2.len });

    if (correct2 == estimates2.len) {
        print("\nSUCCESS: Physics types inferred from dynamics alone!\n", .{});
    } else {
        print("\nNote: Physics-only inference is harder - requires observing distinctive behavior.\n", .{});
        print("Bouncy entities bounce high; sticky entities stop at ground.\n", .{});
    }
}
