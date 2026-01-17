//! Multi-Entity Visualization Demo
//!
//! Shows SMC tracking of TWO bouncing balls with entity-entity collision:
//! - Left: Ground truth (green + red balls)
//! - Right: Particle filter estimates (orange clouds for each entity)
//!
//! Run with: zig build multi-entity-viz

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

/// Fast rasterization: draw filled circles
fn renderParticlesFast(
    positions: []const Vec3,
    weights: []const f32,
    base_color: Vec3,
    radius: f32,
    camera: Camera,
    grid: *ObservationGrid,
) void {
    var max_weight: f32 = 0;
    for (weights) |w| {
        if (w > max_weight) max_weight = w;
    }
    if (max_weight == 0) max_weight = 1;

    const half_w: f32 = @floatFromInt(grid.width / 2);
    const half_h: f32 = @floatFromInt(grid.height / 2);

    for (positions, weights) |pos, w| {
        const proj = camera.project(pos) orelse continue;

        const px = (proj.ndc.x + 1) * half_w;
        const py = (1 - proj.ndc.y) * half_h;

        const pixel_radius = radius * half_h / (proj.depth * @tan(camera.fov / 2));
        const r_int: i32 = @intFromFloat(@max(2, pixel_radius));

        const alpha = @max(0.2, w / max_weight);
        const color = base_color.scale(alpha);

        const cx: i32 = @intFromFloat(px);
        const cy: i32 = @intFromFloat(py);
        const w_i: i32 = @intCast(grid.width);
        const h_i: i32 = @intCast(grid.height);

        var dy: i32 = -r_int;
        while (dy <= r_int) : (dy += 1) {
            var dx: i32 = -r_int;
            while (dx <= r_int) : (dx += 1) {
                if (dx * dx + dy * dy <= r_int * r_int) {
                    const x = cx + dx;
                    const y = cy + dy;
                    if (x >= 0 and x < w_i and y >= 0 and y < h_i) {
                        const existing = grid.get(@intCast(x), @intCast(y));
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
}

/// Fast single entity render
fn renderEntityFast(
    pos: Vec3,
    color: Vec3,
    radius: f32,
    camera: Camera,
    grid: *ObservationGrid,
) void {
    const proj = camera.project(pos) orelse return;

    const half_w: f32 = @floatFromInt(grid.width / 2);
    const half_h: f32 = @floatFromInt(grid.height / 2);

    const px = (proj.ndc.x + 1) * half_w;
    const py = (1 - proj.ndc.y) * half_h;

    const pixel_radius = radius * half_h / (proj.depth * @tan(camera.fov / 2));
    const r_int: i32 = @intFromFloat(@max(3, pixel_radius));

    const cx: i32 = @intFromFloat(px);
    const cy: i32 = @intFromFloat(py);
    const w_i: i32 = @intCast(grid.width);
    const h_i: i32 = @intCast(grid.height);

    var dy: i32 = -r_int;
    while (dy <= r_int) : (dy += 1) {
        var dx: i32 = -r_int;
        while (dx <= r_int) : (dx += 1) {
            const dist_sq = dx * dx + dy * dy;
            if (dist_sq <= r_int * r_int) {
                const x = cx + dx;
                const y = cy + dy;
                if (x >= 0 and x < w_i and y >= 0 and y < h_i) {
                    const dist: f32 = @sqrt(@as(f32, @floatFromInt(dist_sq)));
                    const edge = 1.0 - @min(1.0, dist / @as(f32, @floatFromInt(r_int)));
                    const existing = grid.get(@intCast(x), @intCast(y));
                    const blended = existing.color.add(color.scale(edge));
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

/// Render multiple entities for SMC observation
fn renderMultiEntityGMM(
    positions: []const Vec3,
    colors: []const Vec3,
    radius: f32,
    camera: Camera,
    width: u32,
    height: u32,
    allocator: std.mem.Allocator,
) !ObservationGrid {
    var grid = try ObservationGrid.init(width, height, allocator);
    errdefer grid.deinit();

    var entities: [8]Entity = undefined;
    const n = @min(positions.len, 8);

    for (0..n) |i| {
        entities[i] = Entity{
            .label = Label{ .birth_time = 0, .birth_index = @intCast(i) },
            .position = GaussianVec3{ .mean = positions[i], .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = Vec3.zero, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }, // Bouncy
            .contact_mode = .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{ .color = colors[i], .opacity = 1.0, .radius = radius },
        };
    }

    var gmm = try GaussianMixture.fromEntities(entities[0..n], allocator);
    defer gmm.deinit();

    grid.renderGMM(gmm, camera, 16);

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

    print("\n=== Multi-Entity Visualization ===\n\n", .{});

    std.fs.cwd().makeDir("viz_multi") catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    // SMC config
    var config = SMCConfig{};
    config.num_particles = 30;
    config.max_entities = 4; // Support up to 4 entities
    config.physics.gravity = Vec3.init(0, -9.81, 0);
    config.physics.dt = 1.0 / 30.0;
    config.physics.process_noise = 1.0;
    config.physics.environment.ground = .{
        .height = 0.5,
        .normal = Vec3.unit_y,
        .friction = 0.5,
        .elasticity = 0.9,
    };
    config.observation_noise = 0.1;
    config.ess_threshold = 0.3;
    config.use_flow_observations = false;

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.1, 2.9, 11.9);
    config.camera_prior_max = Vec3.init(0.1, 3.1, 12.1);
    config.camera_yaw_min = -0.05;
    config.camera_yaw_max = 0.05;
    config.back_projection_depth_mean = 12.0;
    config.back_projection_depth_var = 0.1;

    var prng = std.Random.DefaultPrng.init(42);
    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Ground truth: TWO balls
    const gt_physics = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }; // Bouncy

    // Ball 1: Green, starts left, moves right
    var gt_pos1 = Vec3.init(-2.5, 4, 0);
    var gt_vel1 = Vec3.init(2.5, 0, 0);
    const color1 = Vec3.init(0.3, 1.0, 0.3); // Green

    // Ball 2: Red, starts right, moves left (will collide!)
    var gt_pos2 = Vec3.init(2.5, 3, 0);
    var gt_vel2 = Vec3.init(-2.0, 0, 0);
    const color2 = Vec3.init(1.0, 0.3, 0.3); // Red

    const camera = Camera{
        .position = Vec3.init(0, 3, 12),
        .target = Vec3.init(0, 3, 0),
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };

    const panel_width: u32 = 80;
    const panel_height: u32 = 64;

    // Initialize SMC with two entities
    const init_pos = [_]Vec3{ gt_pos1, gt_pos2 };
    const init_vel = [_]Vec3{ gt_vel1, gt_vel2 };
    smc.initializeWithPrior(&init_pos, &init_vel);

    const n_frames: u32 = 120;

    print("Rendering {d} frames with {d} particles, 2 entities...\n", .{ n_frames, config.num_particles });

    // Allocate arrays for particle positions (for both entities)
    var particle_positions1 = try allocator.alloc(Vec3, config.num_particles);
    defer allocator.free(particle_positions1);
    var particle_positions2 = try allocator.alloc(Vec3, config.num_particles);
    defer allocator.free(particle_positions2);
    var particle_weights = try allocator.alloc(f32, config.num_particles);
    defer allocator.free(particle_weights);

    for (0..n_frames) |frame| {
        // Step ground truth for both balls
        gt_vel1 = gt_vel1.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos1 = gt_pos1.add(gt_vel1.scale(config.physics.dt));

        gt_vel2 = gt_vel2.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos2 = gt_pos2.add(gt_vel2.scale(config.physics.dt));

        // Ground collisions
        if (gt_pos1.y < 0.5) {
            gt_pos1.y = 0.5;
            gt_vel1.y = -gt_vel1.y * gt_physics.elasticity;
        }
        if (gt_pos2.y < 0.5) {
            gt_pos2.y = 0.5;
            gt_vel2.y = -gt_vel2.y * gt_physics.elasticity;
        }

        // Wall collisions
        inline for ([_]*Vec3{ &gt_pos1, &gt_pos2 }, [_]*Vec3{ &gt_vel1, &gt_vel2 }) |pos, vel| {
            if (pos.x > 4.0) {
                pos.x = 4.0;
                vel.x = -vel.x * 0.9;
            }
            if (pos.x < -4.0) {
                pos.x = -4.0;
                vel.x = -vel.x * 0.9;
            }
        }

        // Entity-entity collision (sphere-sphere)
        const dist = gt_pos1.sub(gt_pos2).length();
        const min_dist: f32 = 1.0; // Sum of radii
        if (dist < min_dist and dist > 0.001) {
            const normal = gt_pos1.sub(gt_pos2).normalize();
            const overlap = min_dist - dist;

            // Push apart
            gt_pos1 = gt_pos1.add(normal.scale(overlap * 0.5));
            gt_pos2 = gt_pos2.sub(normal.scale(overlap * 0.5));

            // Elastic collision (swap normal components)
            const v1n = normal.scale(gt_vel1.dot(normal));
            const v2n = normal.scale(gt_vel2.dot(normal));
            gt_vel1 = gt_vel1.sub(v1n).add(v2n);
            gt_vel2 = gt_vel2.sub(v2n).add(v1n);
        }

        // Render ground truth
        var gt_grid = try ObservationGrid.init(panel_width, panel_height, allocator);
        defer gt_grid.deinit();
        renderEntityFast(gt_pos1, color1, 0.5, camera, &gt_grid);
        renderEntityFast(gt_pos2, color2, 0.5, camera, &gt_grid);

        // SMC observation (render both balls)
        const obs_pos = [_]Vec3{ gt_pos1, gt_pos2 };
        const obs_colors = [_]Vec3{ color1, color2 };
        var smc_obs = try renderMultiEntityGMM(&obs_pos, &obs_colors, 0.5, camera, 48, 48, allocator);
        defer smc_obs.deinit();

        // SMC step
        try smc.step(smc_obs);

        // Collect particle positions for both entities
        var n_active: usize = 0;
        for (0..config.num_particles) |p| {
            const e0_idx = p * config.max_entities;
            const e1_idx = p * config.max_entities + 1;
            if (smc.swarm.track_state[e0_idx] != .dead and smc.swarm.track_state[e1_idx] != .dead) {
                particle_positions1[n_active] = smc.swarm.position_mean[e0_idx];
                particle_positions2[n_active] = smc.swarm.position_mean[e1_idx];
                particle_weights[n_active] = smc.weights[p];
                n_active += 1;
            }
        }

        // Render particle clouds
        var particle_grid = try ObservationGrid.init(panel_width, panel_height, allocator);
        defer particle_grid.deinit();
        if (n_active > 0) {
            // Entity 1: Orange-green particles
            renderParticlesFast(
                particle_positions1[0..n_active],
                particle_weights[0..n_active],
                Vec3.init(0.5, 1.0, 0.3),
                0.5,
                camera,
                &particle_grid,
            );
            // Entity 2: Orange-red particles
            renderParticlesFast(
                particle_positions2[0..n_active],
                particle_weights[0..n_active],
                Vec3.init(1.0, 0.5, 0.3),
                0.5,
                camera,
                &particle_grid,
            );
        }

        // Composite
        var composite = try compositeSideBySide(gt_grid, particle_grid, allocator);
        defer composite.deinit();

        // Save frame
        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "viz_multi/frame_{d:0>3}.ppm", .{frame});
        try composite.writePPM(path);

        // Compute weighted mean of particles
        var mean_pos1 = Vec3.zero;
        var mean_pos2 = Vec3.zero;
        var total_w: f32 = 0;
        for (0..n_active) |idx| {
            mean_pos1 = mean_pos1.add(particle_positions1[idx].scale(particle_weights[idx]));
            mean_pos2 = mean_pos2.add(particle_positions2[idx].scale(particle_weights[idx]));
            total_w += particle_weights[idx];
        }
        if (total_w > 0) {
            mean_pos1 = mean_pos1.scale(1.0 / total_w);
            mean_pos2 = mean_pos2.scale(1.0 / total_w);
        }

        if (frame % 10 == 0) {
            const err1 = gt_pos1.sub(mean_pos1).length();
            const err2 = gt_pos2.sub(mean_pos2).length();
            print("  Frame {d:3}: Ball1 err={d:.2}  Ball2 err={d:.2}\n", .{ frame, err1, err2 });
        }
    }

    print("\nFrames saved to viz_multi/\n", .{});
}
