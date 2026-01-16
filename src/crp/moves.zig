const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../math.zig");
const Vec3 = math.Vec3;

const types = @import("../types.zig");
const Camera = types.Camera;
const Label = types.Label;
const EntityId = types.EntityId;

const gmm = @import("../gmm.zig");
const ObservationGrid = gmm.ObservationGrid;

const config_mod = @import("config.zig");
const CRPConfig = config_mod.CRPConfig;
const CRPStats = config_mod.CRPStats;

const label_set_mod = @import("label_set.zig");
const LabelSet = label_set_mod.LabelSet;

const birth_mod = @import("birth.zig");
const BirthProposal = birth_mod.BirthProposal;
const proposeBirth = birth_mod.proposeBirth;
const executeBirth = birth_mod.executeBirth;

const death_mod = @import("death.zig");
const DeathProposal = death_mod.DeathProposal;
const proposeDeath = death_mod.proposeDeath;
const executeDeath = death_mod.executeDeath;
const updateOcclusionStates = death_mod.updateOcclusionStates;
const shouldAutoDeath = death_mod.shouldAutoDeath;

// =============================================================================
// Trans-Dimensional Move Result
// =============================================================================

pub const MoveResult = union(enum) {
    none: void,
    birth: struct {
        entity_id: EntityId,
        accepted: bool,
    },
    death: struct {
        entity_id: EntityId,
        accepted: bool,
    },
};

// =============================================================================
// Trans-Dimensional Move Step
// =============================================================================

/// Perform a single trans-dimensional move (birth or death)
/// world can be any type with the full CRP interface (spawn, despawn, getters, setters, iterators)
pub fn transDimensionalStep(
    observation: *const ObservationGrid,
    world: anytype,
    label_set: *LabelSet,
    camera: Camera,
    config: CRPConfig,
    stats: ?*CRPStats,
    rng: std.Random,
    allocator: Allocator,
) !MoveResult {
    // Update occlusion states first
    try updateOcclusionStates(observation, world, camera, allocator);

    // Randomly choose move type
    const move_type = rng.float(f32);
    const birth_threshold = config.birth_proposal_prob;
    const death_threshold = config.birth_proposal_prob + config.death_proposal_prob;

    if (move_type < birth_threshold) {
        // Attempt birth
        return try attemptBirth(observation, world, label_set, camera, config, stats, rng, allocator);
    } else if (move_type < death_threshold) {
        // Attempt death
        return try attemptDeath(observation, world, label_set, camera, config, stats, rng, allocator);
    } else {
        // No move (standard update)
        return .{ .none = {} };
    }
}

/// Attempt a birth move with Metropolis-Hastings acceptance
fn attemptBirth(
    observation: *const ObservationGrid,
    world: anytype,
    label_set: *LabelSet,
    camera: Camera,
    config: CRPConfig,
    stats: ?*CRPStats,
    rng: std.Random,
    allocator: Allocator,
) !MoveResult {
    if (stats) |s| s.birth_attempts += 1;

    const maybe_proposal = try proposeBirth(
        observation,
        world,
        label_set,
        camera,
        config,
        rng,
        allocator,
    );

    if (maybe_proposal) |proposal| {
        // Metropolis-Hastings acceptance
        const accept_prob = @min(1.0, @exp(proposal.log_accept_prob));
        const accepted = rng.float(f32) < accept_prob;

        if (accepted) {
            if (executeBirth(proposal, world, label_set)) |entity_id| {
                if (stats) |s| s.birth_accepts += 1;
                return .{ .birth = .{ .entity_id = entity_id, .accepted = true } };
            }
        }
        return .{ .birth = .{ .entity_id = EntityId{ .index = 0, .generation = 0 }, .accepted = false } };
    }

    return .{ .none = {} };
}

/// Attempt a death move with Metropolis-Hastings acceptance
fn attemptDeath(
    observation: *const ObservationGrid,
    world: anytype,
    label_set: *LabelSet,
    camera: Camera,
    config: CRPConfig,
    stats: ?*CRPStats,
    rng: std.Random,
    allocator: Allocator,
) !MoveResult {
    if (stats) |s| s.death_attempts += 1;

    // First check for auto-death (long occlusion)
    var auto_death_entity: ?EntityId = null;
    var iter = label_set.iterator();
    while (iter.next()) |entity_id| {
        if (shouldAutoDeath(entity_id, world, config)) {
            auto_death_entity = entity_id;
            break;
        }
    }

    if (auto_death_entity) |entity_id| {
        // Get label and execute death
        if (world.getLabel(entity_id)) |label| {
            executeDeath(.{
                .entity_id = entity_id,
                .label = label,
                .log_accept_prob = 0,
                .occlusion_duration = config.max_occlusion_before_death,
            }, world, label_set);

            if (stats) |s| s.death_accepts += 1;
            return .{ .death = .{ .entity_id = entity_id, .accepted = true } };
        }
    }

    // Standard MH death proposal
    const maybe_proposal = try proposeDeath(
        observation,
        world,
        label_set,
        camera,
        config,
        rng,
        allocator,
    );

    if (maybe_proposal) |proposal| {
        // Metropolis-Hastings acceptance
        const accept_prob = @min(1.0, @exp(proposal.log_accept_prob));
        const accepted = rng.float(f32) < accept_prob;

        if (accepted) {
            const entity_id = proposal.entity_id;
            executeDeath(proposal, world, label_set);
            if (stats) |s| s.death_accepts += 1;
            return .{ .death = .{ .entity_id = entity_id, .accepted = true } };
        } else {
            return .{ .death = .{ .entity_id = proposal.entity_id, .accepted = false } };
        }
    }

    return .{ .none = {} };
}

// =============================================================================
// Entity Count Inference
// =============================================================================

/// Posterior distribution over entity count (from multiple particles)
pub const EntityCountPosterior = struct {
    counts: std.AutoHashMapUnmanaged(usize, f32),
    allocator: Allocator,

    pub fn init(allocator: Allocator) EntityCountPosterior {
        return .{
            .counts = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *EntityCountPosterior) void {
        self.counts.deinit(self.allocator);
    }

    /// Add a particle's entity count with given weight
    pub fn addSample(self: *EntityCountPosterior, count: usize, weight: f32) !void {
        const current = self.counts.get(count) orelse 0;
        try self.counts.put(self.allocator, count, current + weight);
    }

    /// Normalize to proper distribution
    pub fn normalize(self: *EntityCountPosterior) void {
        var total: f32 = 0;
        var iter = self.counts.valueIterator();
        while (iter.next()) |v| {
            total += v.*;
        }

        if (total > 0) {
            var value_iter = self.counts.valueIterator();
            while (value_iter.next()) |v| {
                v.* /= total;
            }
        }
    }

    /// Get most likely entity count (MAP estimate)
    pub fn getMAP(self: *const EntityCountPosterior) usize {
        var best_count: usize = 0;
        var best_prob: f32 = 0;

        var iter = self.counts.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* > best_prob) {
                best_prob = entry.value_ptr.*;
                best_count = entry.key_ptr.*;
            }
        }

        return best_count;
    }

    /// Get mean entity count
    pub fn getMean(self: *const EntityCountPosterior) f32 {
        var sum: f32 = 0;
        var total_weight: f32 = 0;

        var iter = self.counts.iterator();
        while (iter.next()) |entry| {
            sum += @as(f32, @floatFromInt(entry.key_ptr.*)) * entry.value_ptr.*;
            total_weight += entry.value_ptr.*;
        }

        if (total_weight > 0) {
            return sum / total_weight;
        }
        return 0;
    }

    /// Get probability of specific count
    pub fn prob(self: *const EntityCountPosterior, count: usize) f32 {
        return self.counts.get(count) orelse 0;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "MoveResult union" {
    const birth_result = MoveResult{ .birth = .{
        .entity_id = EntityId{ .index = 1, .generation = 0 },
        .accepted = true,
    } };

    switch (birth_result) {
        .birth => |b| {
            try std.testing.expect(b.accepted);
            try std.testing.expectEqual(@as(u32, 1), b.entity_id.index);
        },
        else => unreachable,
    }
}

test "EntityCountPosterior" {
    const allocator = std.testing.allocator;

    var posterior = EntityCountPosterior.init(allocator);
    defer posterior.deinit();

    try posterior.addSample(3, 0.5);
    try posterior.addSample(3, 0.3);
    try posterior.addSample(4, 0.2);

    posterior.normalize();

    // MAP should be 3 (highest weight)
    try std.testing.expectEqual(@as(usize, 3), posterior.getMAP());

    // Mean should be between 3 and 4
    const mean = posterior.getMean();
    try std.testing.expect(mean >= 3.0 and mean <= 4.0);
}
