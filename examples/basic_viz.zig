//! Visualization Demo: Render test scenarios to PPM images
//!
//! Run with: zig build viz
//!
//! Outputs to viz_scenarios/ directory

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

const print = std.debug.print;

/// Render entities to observation grid
fn renderEntities(
    positions: []const Vec3,
    colors: []const Vec3,
    radii: []const f32,
    camera: Camera,
    width: u32,
    height: u32,
    allocator: std.mem.Allocator,
) !ObservationGrid {
    var grid = try ObservationGrid.init(width, height, allocator);
    errdefer grid.deinit();

    var entities = try allocator.alloc(Entity, positions.len);
    defer allocator.free(entities);

    for (positions, colors, radii, 0..) |pos, color, radius, i| {
        entities[i] = Entity{
            .label = Label{ .birth_time = 0, .birth_index = @intCast(i) },
            .position = GaussianVec3{ .mean = pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = Vec3.zero, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = PhysicsParams.standard,
            .contact_mode = .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = color,
                .opacity = 1.0,
                .radius = radius,
            },
        };
    }

    var gmm = try GaussianMixture.fromEntities(entities, allocator);
    defer gmm.deinit();

    grid.renderGMM(gmm, camera, 32);

    return grid;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\n=== Fizz Visualization Demo ===\n\n", .{});

    // Create output directory
    std.fs.cwd().makeDir("viz_scenarios") catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    const width: u32 = 256;
    const height: u32 = 256;

    // Camera setup
    const camera = Camera{
        .position = Vec3.init(0, 5, 15),
        .target = Vec3.init(0, 3, 0),
        .up = Vec3.unit_y,
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 100.0,
    };

    // =========================================================================
    // Scenario 1: Single bouncing ball (different heights for bounce sequence)
    // =========================================================================
    print("Rendering: Bouncing ball sequence...\n", .{});

    const bounce_heights = [_]f32{ 5.0, 4.0, 2.5, 0.5, 1.5, 3.0, 4.5 };
    for (bounce_heights, 0..) |h, i| {
        const positions = [_]Vec3{Vec3.init(0, h, 0)};
        const colors = [_]Vec3{Vec3.init(0.3, 1.0, 0.3)}; // Green = bouncy
        const radii = [_]f32{0.5};

        var grid = try renderEntities(&positions, &colors, &radii, camera, width, height, allocator);
        defer grid.deinit();

        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "viz_scenarios/bounce_{d:0>2}.ppm", .{i});
        try grid.writePPM(path);
    }

    // =========================================================================
    // Scenario 2: Multiple entities with different physics types
    // =========================================================================
    print("Rendering: Multiple entities...\n", .{});
    {
        const positions = [_]Vec3{
            Vec3.init(-3.0, 4.0, 0.0), // Left - Standard (red)
            Vec3.init(-1.0, 5.0, 0.0), // Left-center - Bouncy (green)
            Vec3.init(1.0, 3.5, 0.0), // Right-center - Sticky (blue)
            Vec3.init(3.0, 4.5, 0.0), // Right - Slippery (yellow)
        };
        const colors = [_]Vec3{
            Vec3.init(1.0, 0.3, 0.3), // Red = standard
            Vec3.init(0.3, 1.0, 0.3), // Green = bouncy
            Vec3.init(0.3, 0.3, 1.0), // Blue = sticky
            Vec3.init(1.0, 1.0, 0.3), // Yellow = slippery
        };
        const radii = [_]f32{ 0.5, 0.4, 0.6, 0.45 };

        var grid = try renderEntities(&positions, &colors, &radii, camera, width, height, allocator);
        defer grid.deinit();

        try grid.writePPM("viz_scenarios/multiple_entities.ppm");
    }

    // =========================================================================
    // Scenario 3: Occlusion (two entities overlapping)
    // =========================================================================
    print("Rendering: Occlusion scenario...\n", .{});
    {
        // Front entity partially occludes back entity
        const positions = [_]Vec3{
            Vec3.init(0.0, 3.0, 2.0), // Front (closer to camera)
            Vec3.init(0.5, 3.5, -1.0), // Back (further from camera)
        };
        const colors = [_]Vec3{
            Vec3.init(1.0, 0.3, 0.3), // Red
            Vec3.init(0.3, 0.3, 1.0), // Blue
        };
        const radii = [_]f32{ 0.6, 0.6 };

        var grid = try renderEntities(&positions, &colors, &radii, camera, width, height, allocator);
        defer grid.deinit();

        try grid.writePPM("viz_scenarios/occlusion.ppm");
    }

    // =========================================================================
    // Scenario 4: Close-up view (fast motion simulation)
    // =========================================================================
    print("Rendering: Close-up view...\n", .{});
    {
        const close_camera = Camera{
            .position = Vec3.init(0, 3, 5),
            .target = Vec3.init(0, 3, 0),
            .up = Vec3.unit_y,
            .fov = std.math.pi / 3.0, // Wider FOV
            .aspect = 1.0,
            .near = 0.1,
            .far = 50.0,
        };

        const positions = [_]Vec3{Vec3.init(0, 3, 0)};
        const colors = [_]Vec3{Vec3.init(0.3, 1.0, 0.3)};
        const radii = [_]f32{0.5};

        var grid = try renderEntities(&positions, &colors, &radii, close_camera, width, height, allocator);
        defer grid.deinit();

        try grid.writePPM("viz_scenarios/closeup.ppm");
    }

    // =========================================================================
    // Scenario 5: Ground contact
    // =========================================================================
    print("Rendering: Ground contact...\n", .{});
    {
        // Entity resting on ground (y = radius)
        const positions = [_]Vec3{Vec3.init(0, 0.5, 0)};
        const colors = [_]Vec3{Vec3.init(0.3, 0.3, 1.0)}; // Blue = sticky
        const radii = [_]f32{0.5};

        const low_camera = Camera{
            .position = Vec3.init(0, 2, 8),
            .target = Vec3.init(0, 1, 0),
            .up = Vec3.unit_y,
            .fov = std.math.pi / 4.0,
            .aspect = 1.0,
            .near = 0.1,
            .far = 50.0,
        };

        var grid = try renderEntities(&positions, &colors, &radii, low_camera, width, height, allocator);
        defer grid.deinit();

        try grid.writePPM("viz_scenarios/ground_contact.ppm");
    }

    // =========================================================================
    // Scenario 6: Size variation
    // =========================================================================
    print("Rendering: Size variation...\n", .{});
    {
        const positions = [_]Vec3{
            Vec3.init(-2.5, 3.0, 0.0),
            Vec3.init(0.0, 3.5, 0.0),
            Vec3.init(2.5, 4.0, 0.0),
        };
        const colors = [_]Vec3{
            Vec3.init(1.0, 0.5, 0.3),
            Vec3.init(0.5, 1.0, 0.3),
            Vec3.init(0.3, 0.5, 1.0),
        };
        const radii = [_]f32{ 0.3, 0.6, 0.9 };

        var grid = try renderEntities(&positions, &colors, &radii, camera, width, height, allocator);
        defer grid.deinit();

        try grid.writePPM("viz_scenarios/size_variation.ppm");
    }

    print("\nImages saved to viz_scenarios/\n", .{});
    print("Convert to PNG: magick viz_scenarios/*.ppm viz_scenarios/output.png\n", .{});
}
