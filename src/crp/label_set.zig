const std = @import("std");
const Allocator = std.mem.Allocator;

const types = @import("../types.zig");
const Label = types.Label;
const EntityId = types.EntityId;

// =============================================================================
// Per-Particle Label Set
// =============================================================================

/// Manages the set of entity labels for a single particle
/// Different particles can have different numbers and configurations of entities
pub const LabelSet = struct {
    /// Active labels -> EntityId mapping
    labels: std.AutoHashMapUnmanaged(u64, EntityId),

    /// Current timestep
    timestep: u32,

    /// Next birth index for this particle
    next_birth_index: u16,

    /// Allocator
    allocator: Allocator,

    pub fn init(allocator: Allocator) LabelSet {
        return .{
            .labels = .{},
            .timestep = 0,
            .next_birth_index = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LabelSet) void {
        self.labels.deinit(self.allocator);
    }

    /// Clone this label set
    pub fn clone(self: *const LabelSet, allocator: Allocator) !LabelSet {
        return .{
            .labels = try self.labels.clone(allocator),
            .timestep = self.timestep,
            .next_birth_index = self.next_birth_index,
            .allocator = allocator,
        };
    }

    /// Create a new label for entity birth
    pub fn createLabel(self: *LabelSet) Label {
        const label = Label{
            .birth_time = self.timestep,
            .birth_index = self.next_birth_index,
        };
        self.next_birth_index += 1;
        return label;
    }

    /// Register an entity with its label
    pub fn register(self: *LabelSet, label: Label, entity_id: EntityId) !void {
        try self.labels.put(self.allocator, label.hash(), entity_id);
    }

    /// Unregister a label (entity death)
    pub fn unregister(self: *LabelSet, label: Label) void {
        _ = self.labels.remove(label.hash());
    }

    /// Get entity ID for a label
    pub fn get(self: *const LabelSet, label: Label) ?EntityId {
        return self.labels.get(label.hash());
    }

    /// Check if label exists
    pub fn contains(self: *const LabelSet, label: Label) bool {
        return self.labels.contains(label.hash());
    }

    /// Number of active labels
    pub fn count(self: *const LabelSet) usize {
        return self.labels.count();
    }

    /// Advance timestep
    pub fn tick(self: *LabelSet) void {
        self.timestep += 1;
    }

    /// Iterator over active labels
    pub fn iterator(self: *const LabelSet) LabelIterator {
        return .{
            .inner = self.labels.iterator(),
        };
    }

    pub const LabelIterator = struct {
        inner: std.AutoHashMapUnmanaged(u64, EntityId).Iterator,

        pub fn next(self: *LabelIterator) ?EntityId {
            if (self.inner.next()) |entry| {
                return entry.value_ptr.*;
            }
            return null;
        }
    };
};

// =============================================================================
// CRP Prior Computations
// =============================================================================

/// Compute CRP prior probability for entity counts
pub fn crpPrior(n_entities: usize, alpha: f32) f32 {
    // CRP prior: P(n entities) ∝ α^n * Gamma(n) / Gamma(n + α)
    // Simplified: log P ≈ n * log(α) - log_gamma_ratio
    const n = @as(f32, @floatFromInt(n_entities));

    if (n_entities == 0) {
        return @log(alpha) - alpha; // Empty partition probability
    }

    // Stirling approximation for log ratio
    const log_n = @log(n);
    const log_alpha = @log(alpha);

    return n * log_alpha - n * log_n + n;
}

/// Compute probability of new entity vs existing under CRP
pub fn crpNewEntityProb(n_existing: usize, alpha: f32) f32 {
    const n = @as(f32, @floatFromInt(n_existing));
    return alpha / (n + alpha);
}

/// Compute probability of joining existing entity k under CRP
pub fn crpExistingEntityProb(n_existing: usize, count_k: usize, alpha: f32) f32 {
    const n = @as(f32, @floatFromInt(n_existing));
    const c_k = @as(f32, @floatFromInt(count_k));
    return c_k / (n + alpha);
}

// =============================================================================
// Tests
// =============================================================================

test "LabelSet basic operations" {
    const allocator = std.testing.allocator;

    var label_set = LabelSet.init(allocator);
    defer label_set.deinit();

    // Create and register a label
    const label = label_set.createLabel();
    const entity_id = EntityId{ .index = 0, .generation = 0 };
    try label_set.register(label, entity_id);

    try std.testing.expectEqual(@as(usize, 1), label_set.count());
    try std.testing.expect(label_set.contains(label));

    const retrieved = label_set.get(label);
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqual(entity_id.index, retrieved.?.index);

    // Unregister
    label_set.unregister(label);
    try std.testing.expectEqual(@as(usize, 0), label_set.count());
}

test "LabelSet clone" {
    const allocator = std.testing.allocator;

    var label_set = LabelSet.init(allocator);
    defer label_set.deinit();

    const label = label_set.createLabel();
    try label_set.register(label, EntityId{ .index = 5, .generation = 1 });

    var cloned = try label_set.clone(allocator);
    defer cloned.deinit();

    try std.testing.expectEqual(label_set.count(), cloned.count());
    try std.testing.expect(cloned.contains(label));
}

test "CRP probabilities" {
    // New entity probability decreases with more existing entities
    const prob_0 = crpNewEntityProb(0, 1.0);
    const prob_5 = crpNewEntityProb(5, 1.0);
    const prob_10 = crpNewEntityProb(10, 1.0);

    try std.testing.expect(prob_0 > prob_5);
    try std.testing.expect(prob_5 > prob_10);

    // With alpha=1 and n=0, prob of new entity is 1
    try std.testing.expect(@abs(prob_0 - 1.0) < 0.01);
}
