const std = @import("std");
const World = @import("world.zig").World;
const EntityId = @import("entity.zig").EntityId;

// =============================================================================
// System Interface
// =============================================================================

/// Query specification for systems
pub const Query = struct {
    /// Components that must be present
    required: []const u16,
    /// Components that must NOT be present
    excluded: []const u16,
    /// Components that may be present
    optional: []const u16,
};

/// System function signature
pub const SystemFn = *const fn (*World, f32, std.Random) void;

/// System definition
pub const System = struct {
    name: []const u8,
    query: Query,
    run: SystemFn,
};

/// System scheduler - runs systems in order
pub const Scheduler = struct {
    systems: std.ArrayListUnmanaged(System),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Scheduler {
        return .{
            .systems = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Scheduler) void {
        self.systems.deinit(self.allocator);
    }

    pub fn addSystem(self: *Scheduler, system: System) !void {
        try self.systems.append(self.allocator, system);
    }

    pub fn run(self: *Scheduler, world: *World, dt: f32, rng: std.Random) void {
        for (self.systems.items) |system| {
            system.run(world, dt, rng);
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Scheduler basic" {
    const allocator = std.testing.allocator;

    var scheduler = Scheduler.init(allocator);
    defer scheduler.deinit();

    // Empty scheduler should run without error
    var world = World.init(allocator);
    defer world.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    scheduler.run(&world, 1.0 / 60.0, prng.random());
}
