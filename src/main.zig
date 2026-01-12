//! Fizz Demo: Generative Physics Simulation
//!
//! This demo runs the generative model forward:
//! - Creates entities with random physics types
//! - Steps physics with ground collisions
//! - Renders each frame as GMM observation
//!
//! In inference mode (Phase 2+), we would run this backward
//! using SMC to infer entity properties from observations.

const std = @import("std");
const fizz = @import("fizz");

const Vec3 = fizz.Vec3;
const World = fizz.World;
const PhysicsConfig = fizz.PhysicsConfig;
const PhysicsType = fizz.PhysicsType;
const Camera = fizz.Camera;

const print = std.debug.print;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Fizz: Probabilistic Physics Engine ===\n\n", .{});

    // Initialize world
    const config = PhysicsConfig{
        .gravity = Vec3.init(0, -9.81, 0),
        .dt = 1.0 / 60.0,
        .ground_height = 0.0,
    };

    var world = World.init(allocator, config);
    defer world.deinit();

    // Initialize RNG
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const rng = prng.random();

    // Create entities with different physics types
    const physics_types = [_]PhysicsType{ .standard, .bouncy, .sticky, .slippery };
    const colors = [_]Vec3{
        Vec3.init(1.0, 0.3, 0.3), // Red - standard
        Vec3.init(0.3, 1.0, 0.3), // Green - bouncy
        Vec3.init(0.3, 0.3, 1.0), // Blue - sticky
        Vec3.init(1.0, 1.0, 0.3), // Yellow - slippery
    };

    print("Creating entities:\n", .{});

    for (physics_types, colors, 0..) |ptype, color, i| {
        const x = @as(f32, @floatFromInt(i)) * 2.0 - 3.0;
        const entity = try world.addEntity(
            Vec3.init(x, 5.0, 0),
            Vec3.init(0, 0, 0),
            ptype,
        );
        entity.appearance.color = color;
        entity.appearance.radius = 0.5; // Larger radius for better visibility

        print("  Entity {d}: {s} at ({d:.2}, {d:.2}, {d:.2})\n", .{
            i,
            @tagName(ptype),
            x,
            @as(f32, 5.0),
            @as(f32, 0.0),
        });
    }

    print("\nRunning simulation ({d} steps):\n", .{180});

    // Camera for rendering
    const camera = Camera{
        .position = Vec3.init(0, 3, 10),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 100.0,
    };

    // Simulation loop
    var frame: u32 = 0;
    const total_frames: u32 = 180; // 3 seconds at 60 FPS

    while (frame < total_frames) : (frame += 1) {
        // Step physics (generative, with process noise)
        world.step(rng);

        // Print status every 30 frames (0.5 seconds)
        if (frame % 30 == 0) {
            print("\n  Frame {d}/{d}:\n", .{ frame, total_frames });

            for (world.entities.items, 0..) |entity, i| {
                const pos = entity.positionMean();
                const vel = entity.velocityMean();
                print("    Entity {d}: pos=({d:.2}, {d:.2}, {d:.2}) vel=({d:.2}, {d:.2}, {d:.2}) contact={s}\n", .{
                    i,
                    pos.x,
                    pos.y,
                    pos.z,
                    vel.x,
                    vel.y,
                    vel.z,
                    @tagName(entity.contact_mode),
                });
            }

            // Render to GMM and compute observation (64 samples for proper coverage)
            var obs_grid = try world.render(camera, 8, 8, 64);
            defer obs_grid.deinit();

            // Count occupied pixels
            var occupied_count: u32 = 0;
            for (obs_grid.pixels) |pixel| {
                if (pixel.occupied) occupied_count += 1;
            }

            print("    Observation: {d}/64 pixels occupied\n", .{occupied_count});
        }
    }

    print("\n=== Simulation Complete ===\n", .{});
    print("\nFinal entity states:\n", .{});

    for (world.entities.items, 0..) |entity, i| {
        const pos = entity.positionMean();
        print("  Entity {d} ({s}): final_y = {d:.3}, contact = {s}\n", .{
            i,
            @tagName(entity.physics_type),
            pos.y,
            @tagName(entity.contact_mode),
        });
    }

    // Summary of physics behavior
    print("\nExpected behaviors:\n", .{});
    print("  - standard: moderate bounce, settles quickly\n", .{});
    print("  - bouncy: high bounce, takes longer to settle\n", .{});
    print("  - sticky: minimal bounce, settles immediately\n", .{});
    print("  - slippery: moderate bounce, may slide on ground\n", .{});

    print("\n=== Phase 1 Complete ===\n", .{});
    print("Next: Phase 2 - SMC inference to infer physics types from observations\n", .{});
}

test "main runs without error" {
    // Just verify it compiles - actual test would need to capture output
}
