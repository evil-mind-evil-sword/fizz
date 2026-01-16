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
            .root_source_file = b.path("src/demo_rgb_inference.zig"),
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
}
