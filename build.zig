const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the fizz module
    const fizz_mod = b.addModule("fizz", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Static library
    const lib = b.addLibrary(.{
        .name = "fizz",
        .root_module = fizz_mod,
        .linkage = .static,
    });
    b.installArtifact(lib);

    // Shared library (for C ABI consumers)
    const shared_lib = b.addLibrary(.{
        .name = "fizz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cabi.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
        .linkage = .dynamic,
    });
    shared_lib.root_module.addCMacro("FIZZ_SHARED", "1");
    b.installArtifact(shared_lib);

    // Unit tests
    const lib_unit_tests = b.addTest(.{
        .root_module = fizz_mod,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // Example executable
    const exe = b.addExecutable(.{
        .name = "fizz-demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the demo");
    run_step.dependOn(&run_cmd.step);

    // RGB inference demo
    const rgb_demo = b.addExecutable(.{
        .name = "rgb-inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/rgb_inference.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
    });
    b.installArtifact(rgb_demo);

    const run_rgb = b.addRunArtifact(rgb_demo);
    run_rgb.step.dependOn(b.getInstallStep());

    const rgb_step = b.step("rgb", "Run RGB inference demo");
    rgb_step.dependOn(&run_rgb.step);

    // Visualization demo
    const viz_demo = b.addExecutable(.{
        .name = "viz-demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic_viz.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
    });
    b.installArtifact(viz_demo);

    const run_viz = b.addRunArtifact(viz_demo);
    run_viz.step.dependOn(b.getInstallStep());

    const viz_step = b.step("viz", "Run visualization demo");
    viz_step.dependOn(&run_viz.step);

    // Inference visualization demo
    const inference_viz = b.addExecutable(.{
        .name = "inference-viz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/inference_viz.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
    });
    b.installArtifact(inference_viz);

    const run_inference_viz = b.addRunArtifact(inference_viz);
    run_inference_viz.step.dependOn(b.getInstallStep());

    const inference_viz_step = b.step("inference-viz", "Run inference visualization demo");
    inference_viz_step.dependOn(&run_inference_viz.step);

    // Particle cloud visualization demo
    const particles_viz = b.addExecutable(.{
        .name = "particles-viz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/particles_viz.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
    });
    b.installArtifact(particles_viz);

    const run_particles_viz = b.addRunArtifact(particles_viz);
    run_particles_viz.step.dependOn(b.getInstallStep());

    const particles_viz_step = b.step("particles-viz", "Run particle cloud visualization");
    particles_viz_step.dependOn(&run_particles_viz.step);

    // Debug inference
    const debug_inference = b.addExecutable(.{
        .name = "debug-inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/debug_inference.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
    });
    b.installArtifact(debug_inference);

    const run_debug_inference = b.addRunArtifact(debug_inference);
    run_debug_inference.step.dependOn(b.getInstallStep());

    const debug_inference_step = b.step("debug-inference", "Debug inference issues");
    debug_inference_step.dependOn(&run_debug_inference.step);

    // Multi-entity visualization demo
    const multi_entity_viz = b.addExecutable(.{
        .name = "multi-entity-viz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/multi_entity_viz.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fizz", .module = fizz_mod },
            },
        }),
    });
    b.installArtifact(multi_entity_viz);

    const run_multi_entity_viz = b.addRunArtifact(multi_entity_viz);
    run_multi_entity_viz.step.dependOn(b.getInstallStep());

    const multi_entity_viz_step = b.step("multi-entity-viz", "Run multi-entity visualization");
    multi_entity_viz_step.dependOn(&run_multi_entity_viz.step);
}
