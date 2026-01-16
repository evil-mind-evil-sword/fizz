const std = @import("std");
const Allocator = std.mem.Allocator;

/// Component type identifier
pub const ComponentId = u16;

/// Maximum number of component types (static for simplicity)
pub const MAX_COMPONENTS: usize = 256;

/// Component storage for a single component type
/// Uses sparse set for O(1) lookup and cache-friendly iteration
pub fn ComponentStorage(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Dense array of component data
        dense: std.ArrayListUnmanaged(T),
        /// Dense array of entity indices (parallel to dense)
        dense_to_entity: std.ArrayListUnmanaged(u32),
        /// Sparse array: entity index -> dense index (or null)
        sparse: std.ArrayListUnmanaged(?u32),
        /// Allocator
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .dense = .empty,
                .dense_to_entity = .empty,
                .sparse = .empty,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.dense.deinit(self.allocator);
            self.dense_to_entity.deinit(self.allocator);
            self.sparse.deinit(self.allocator);
        }

        /// Ensure sparse array can hold entity_index
        fn ensureSparse(self: *Self, entity_index: u32) !void {
            const current_len = self.sparse.items.len;
            if (entity_index >= current_len) {
                const new_len = entity_index + 1;
                try self.sparse.resize(self.allocator, new_len);
                // Fill new slots with null
                for (self.sparse.items[current_len..]) |*slot| {
                    slot.* = null;
                }
            }
        }

        /// Set component for entity
        pub fn set(self: *Self, entity_index: u32, value: T) !void {
            try self.ensureSparse(entity_index);

            if (self.sparse.items[entity_index]) |dense_idx| {
                // Update existing
                self.dense.items[dense_idx] = value;
            } else {
                // Insert new
                const dense_idx: u32 = @intCast(self.dense.items.len);
                try self.dense.append(self.allocator, value);
                try self.dense_to_entity.append(self.allocator, entity_index);
                self.sparse.items[entity_index] = dense_idx;
            }
        }

        /// Get component for entity (optional)
        pub fn get(self: *const Self, entity_index: u32) ?*const T {
            if (entity_index >= self.sparse.items.len) return null;
            const dense_idx = self.sparse.items[entity_index] orelse return null;
            return &self.dense.items[dense_idx];
        }

        /// Get mutable component for entity
        pub fn getMut(self: *Self, entity_index: u32) ?*T {
            if (entity_index >= self.sparse.items.len) return null;
            const dense_idx = self.sparse.items[entity_index] orelse return null;
            return &self.dense.items[dense_idx];
        }

        /// Check if entity has this component
        pub fn has(self: *const Self, entity_index: u32) bool {
            if (entity_index >= self.sparse.items.len) return false;
            return self.sparse.items[entity_index] != null;
        }

        /// Remove component from entity
        pub fn remove(self: *Self, entity_index: u32) void {
            if (entity_index >= self.sparse.items.len) return;
            const dense_idx = self.sparse.items[entity_index] orelse return;

            // Swap with last element in dense array
            const last_dense = self.dense.items.len - 1;
            if (dense_idx != last_dense) {
                self.dense.items[dense_idx] = self.dense.items[last_dense];
                const swapped_entity = self.dense_to_entity.items[last_dense];
                self.dense_to_entity.items[dense_idx] = swapped_entity;
                self.sparse.items[swapped_entity] = dense_idx;
            }

            _ = self.dense.pop();
            _ = self.dense_to_entity.pop();
            self.sparse.items[entity_index] = null;
        }

        /// Iterate over all components
        pub fn iter(self: *const Self) []const T {
            return self.dense.items;
        }

        /// Iterate over all entity indices that have this component
        pub fn entities(self: *const Self) []const u32 {
            return self.dense_to_entity.items;
        }

        /// Get count of entities with this component
        pub fn count(self: *const Self) usize {
            return self.dense.items.len;
        }
    };
}

/// Type-erased component storage interface
pub const ErasedStorage = struct {
    ptr: *anyopaque,
    deinit_fn: *const fn (*anyopaque) void,
    has_fn: *const fn (*anyopaque, u32) bool,
    remove_fn: *const fn (*anyopaque, u32) void,
    count_fn: *const fn (*anyopaque) usize,

    pub fn deinit(self: *ErasedStorage) void {
        self.deinit_fn(self.ptr);
    }

    pub fn has(self: *const ErasedStorage, entity_index: u32) bool {
        return self.has_fn(@constCast(self.ptr), entity_index);
    }

    pub fn remove(self: *ErasedStorage, entity_index: u32) void {
        self.remove_fn(self.ptr, entity_index);
    }

    pub fn count(self: *const ErasedStorage) usize {
        return self.count_fn(@constCast(self.ptr));
    }
};

/// Create type-erased storage wrapper
pub fn eraseStorage(comptime T: type, storage: *ComponentStorage(T)) ErasedStorage {
    const Wrapper = struct {
        fn deinitFn(ptr: *anyopaque) void {
            const s: *ComponentStorage(T) = @ptrCast(@alignCast(ptr));
            s.deinit();
        }

        fn hasFn(ptr: *anyopaque, entity_index: u32) bool {
            const s: *const ComponentStorage(T) = @ptrCast(@alignCast(ptr));
            return s.has(entity_index);
        }

        fn removeFn(ptr: *anyopaque, entity_index: u32) void {
            const s: *ComponentStorage(T) = @ptrCast(@alignCast(ptr));
            s.remove(entity_index);
        }

        fn countFn(ptr: *anyopaque) usize {
            const s: *const ComponentStorage(T) = @ptrCast(@alignCast(ptr));
            return s.count();
        }
    };

    return .{
        .ptr = storage,
        .deinit_fn = Wrapper.deinitFn,
        .has_fn = Wrapper.hasFn,
        .remove_fn = Wrapper.removeFn,
        .count_fn = Wrapper.countFn,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "ComponentStorage basic operations" {
    const allocator = std.testing.allocator;

    const Vec3 = struct { x: f32, y: f32, z: f32 };
    var storage = ComponentStorage(Vec3).init(allocator);
    defer storage.deinit();

    // Set component for entity 0
    try storage.set(0, .{ .x = 1, .y = 2, .z = 3 });
    try std.testing.expect(storage.has(0));
    try std.testing.expect(!storage.has(1));

    // Get component
    const comp = storage.get(0).?;
    try std.testing.expectEqual(@as(f32, 1), comp.x);

    // Set for another entity
    try storage.set(5, .{ .x = 5, .y = 6, .z = 7 });
    try std.testing.expect(storage.has(5));
    try std.testing.expectEqual(@as(usize, 2), storage.count());

    // Remove
    storage.remove(0);
    try std.testing.expect(!storage.has(0));
    try std.testing.expect(storage.has(5));
    try std.testing.expectEqual(@as(usize, 1), storage.count());
}
