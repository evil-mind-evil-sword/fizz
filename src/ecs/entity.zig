const std = @import("std");
const Allocator = std.mem.Allocator;

/// Entity identifier with generational index for safe reuse
/// Format: 32-bit index + 16-bit generation
pub const EntityId = struct {
    index: u32,
    generation: u16,

    pub const invalid = EntityId{ .index = std.math.maxInt(u32), .generation = 0 };

    pub fn eql(self: EntityId, other: EntityId) bool {
        return self.index == other.index and self.generation == other.generation;
    }

    pub fn hash(self: EntityId) u64 {
        return @as(u64, self.index) << 16 | @as(u64, self.generation);
    }

    pub fn format(
        self: EntityId,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("Entity({d}:{d})", .{ self.index, self.generation });
    }
};

/// Entry in the entity allocator
const EntityEntry = struct {
    generation: u16,
    alive: bool,
};

/// Entity allocator with generational indices
/// Supports efficient allocation, deallocation, and validity checking
pub const EntityAllocator = struct {
    entries: std.ArrayListUnmanaged(EntityEntry),
    free_list: std.ArrayListUnmanaged(u32),
    alive_count: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) EntityAllocator {
        return .{
            .entries = .empty,
            .free_list = .empty,
            .alive_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *EntityAllocator) void {
        self.entries.deinit(self.allocator);
        self.free_list.deinit(self.allocator);
    }

    /// Allocate a new entity
    pub fn alloc(self: *EntityAllocator) !EntityId {
        if (self.free_list.pop()) |index| {
            // Reuse freed slot
            self.entries.items[index].alive = true;
            self.alive_count += 1;
            return .{
                .index = index,
                .generation = self.entries.items[index].generation,
            };
        } else {
            // Allocate new slot
            const index: u32 = @intCast(self.entries.items.len);
            try self.entries.append(self.allocator, .{ .generation = 0, .alive = true });
            self.alive_count += 1;
            return .{ .index = index, .generation = 0 };
        }
    }

    /// Free an entity (marks as dead, increments generation)
    pub fn free(self: *EntityAllocator, id: EntityId) void {
        if (!self.isValid(id)) return;

        self.entries.items[id.index].alive = false;
        self.entries.items[id.index].generation +%= 1;
        self.free_list.append(self.allocator, id.index) catch {};
        self.alive_count -= 1;
    }

    /// Check if entity ID is still valid
    pub fn isValid(self: *const EntityAllocator, id: EntityId) bool {
        if (id.index >= self.entries.items.len) return false;
        const entry = self.entries.items[id.index];
        return entry.alive and entry.generation == id.generation;
    }

    /// Get count of alive entities
    pub fn count(self: *const EntityAllocator) usize {
        return self.alive_count;
    }

    /// Iterator over all alive entity IDs
    pub fn aliveIterator(self: *const EntityAllocator) AliveIterator {
        return .{
            .allocator = self,
            .current_index = 0,
        };
    }

    pub const AliveIterator = struct {
        allocator: *const EntityAllocator,
        current_index: u32,

        pub fn next(self: *AliveIterator) ?EntityId {
            while (self.current_index < self.allocator.entries.items.len) {
                const idx = self.current_index;
                self.current_index += 1;
                const entry = self.allocator.entries.items[idx];
                if (entry.alive) {
                    return .{ .index = idx, .generation = entry.generation };
                }
            }
            return null;
        }
    };
};

// =============================================================================
// Tests
// =============================================================================

test "EntityAllocator basic operations" {
    const allocator = std.testing.allocator;

    var entities = EntityAllocator.init(allocator);
    defer entities.deinit();

    // Allocate entities
    const e0 = try entities.alloc();
    const e1 = try entities.alloc();
    const e2 = try entities.alloc();

    try std.testing.expectEqual(@as(u32, 0), e0.index);
    try std.testing.expectEqual(@as(u32, 1), e1.index);
    try std.testing.expectEqual(@as(u32, 2), e2.index);
    try std.testing.expectEqual(@as(usize, 3), entities.count());

    // All should be valid
    try std.testing.expect(entities.isValid(e0));
    try std.testing.expect(entities.isValid(e1));
    try std.testing.expect(entities.isValid(e2));

    // Free middle entity
    entities.free(e1);
    try std.testing.expect(!entities.isValid(e1));
    try std.testing.expectEqual(@as(usize, 2), entities.count());

    // Allocate new - should reuse slot 1
    const e3 = try entities.alloc();
    try std.testing.expectEqual(@as(u32, 1), e3.index);
    try std.testing.expectEqual(@as(u16, 1), e3.generation); // Generation incremented

    // Old e1 should still be invalid
    try std.testing.expect(!entities.isValid(e1));
    // New e3 should be valid
    try std.testing.expect(entities.isValid(e3));
}

test "EntityAllocator iteration" {
    const allocator = std.testing.allocator;

    var entities = EntityAllocator.init(allocator);
    defer entities.deinit();

    const e0 = try entities.alloc();
    _ = try entities.alloc();
    const e2 = try entities.alloc();

    entities.free(e0);

    var iter = entities.aliveIterator();
    var count: usize = 0;
    while (iter.next()) |id| {
        try std.testing.expect(id.index == 1 or id.index == 2);
        try std.testing.expect(!id.eql(e0));
        _ = e2;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), count);
}
