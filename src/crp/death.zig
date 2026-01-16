const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../math.zig");
const Vec3 = math.Vec3;

const types = @import("../types.zig");
const Camera = types.Camera;
const Label = types.Label;
const TrackState = types.TrackState;
const EntityId = types.EntityId;

const gmm = @import("../gmm.zig");
const ObservationGrid = gmm.ObservationGrid;

const association = @import("../association/mod.zig");
const PixelOwnership = association.PixelOwnership;
const renderPixelOwnership = association.renderPixelOwnership;

const config_mod = @import("config.zig");
const CRPConfig = config_mod.CRPConfig;

const label_set_mod = @import("label_set.zig");
const LabelSet = label_set_mod.LabelSet;
const crpNewEntityProb = label_set_mod.crpNewEntityProb;

// =============================================================================
// Death Proposal
// =============================================================================

/// Result of a death proposal
pub const DeathProposal = struct {
    /// Entity to remove
    entity_id: EntityId,
    /// Label of entity
    label: Label,
    /// Log acceptance probability
    log_accept_prob: f32,
    /// Occlusion duration (frames since last observation)
    occlusion_duration: u32,
};

// =============================================================================
// Death Proposal Generation
// =============================================================================

/// Generate a death proposal for entities with no recent observations
/// world can be any type with queryPhysicsEntities(), getPosition(), getAppearance(), getOcclusion(), getLabel()
pub fn proposeDeath(
    observation: *const ObservationGrid,
    world: anytype,
    label_set: *const LabelSet,
    camera: Camera,
    config: CRPConfig,
    rng: std.Random,
    allocator: Allocator,
) !?DeathProposal {
    // No entities to kill
    if (label_set.count() == 0) return null;

    // Render pixel ownership to check which entities are visible
    var ownership = try renderPixelOwnership(world, camera, observation.width, observation.height, allocator);
    defer ownership.deinit();

    // Collect candidate entities for death
    var candidates: std.ArrayListUnmanaged(DeathCandidate) = .empty;
    defer candidates.deinit(allocator);

    var iter = label_set.iterator();
    while (iter.next()) |entity_id| {
        // Get occlusion info
        const occlusion = world.getOcclusion(entity_id);
        const duration = if (occlusion) |o| o.duration else 0;

        // Only consider entities that have been occluded/unmatched
        if (duration == 0) continue;

        // Compute survival probability based on occlusion duration
        const survival = std.math.pow(f32, config.survival_prob, @floatFromInt(duration));

        // Death probability increases with occlusion duration
        const death_prob = 1.0 - survival;

        if (death_prob > 0.01) {
            try candidates.append(allocator, .{
                .entity_id = entity_id,
                .occlusion_duration = duration,
                .death_prob = death_prob,
            });
        }
    }

    if (candidates.items.len == 0) return null;

    // Sample candidate weighted by death probability
    var total_prob: f32 = 0;
    for (candidates.items) |c| {
        total_prob += c.death_prob;
    }

    const selected_idx = blk: {
        const u = rng.float(f32) * total_prob;
        var cumulative: f32 = 0;
        for (candidates.items, 0..) |c, i| {
            cumulative += c.death_prob;
            if (u <= cumulative) {
                break :blk i;
            }
        }
        break :blk candidates.items.len - 1;
    };

    const candidate = candidates.items[selected_idx];

    // Get label for this entity
    const label = blk: {
        if (world.getLabel(candidate.entity_id)) |l| {
            break :blk l;
        }
        return null;
    };

    // Compute acceptance probability (MH ratio for death move)
    // CRP prior ratio: P(n-1 entities) / P(n entities) = (n-1+alpha) / alpha
    // This favors death when we have many entities
    const n_current = label_set.count();
    const n_after_death = @as(f32, @floatFromInt(n_current - 1));
    const crp_death_prior_ratio = (n_after_death + config.alpha) / config.alpha;

    // Likelihood ratio for removing entity
    // If entity is occluded, removing it doesn't hurt likelihood much
    const lik_ratio = 1.0 / (1.0 + candidate.death_prob);

    // Proposal ratio (birth/death symmetry)
    const proposal_ratio = config.birth_proposal_prob / config.death_proposal_prob;

    const log_accept_prob = @log(crp_death_prior_ratio) + @log(lik_ratio) + @log(proposal_ratio);

    return DeathProposal{
        .entity_id = candidate.entity_id,
        .label = label,
        .log_accept_prob = log_accept_prob,
        .occlusion_duration = candidate.occlusion_duration,
    };
}

const DeathCandidate = struct {
    entity_id: EntityId,
    occlusion_duration: u32,
    death_prob: f32,
};

/// Execute a death proposal (remove entity from world)
/// world can be any type with despawn() method
pub fn executeDeath(
    proposal: DeathProposal,
    world: anytype,
    label_set: *LabelSet,
) void {
    // Unregister from label set
    label_set.unregister(proposal.label);

    // Despawn entity
    world.despawn(proposal.entity_id);
}

/// Update occlusion state for all entities based on current observations
/// world can be any type with queryPhysicsEntities(), getPosition(), getAppearance(), getOcclusion(), setOcclusion()
pub fn updateOcclusionStates(
    observation: *const ObservationGrid,
    world: anytype,
    camera: Camera,
    allocator: Allocator,
) !void {
    // Render pixel ownership
    var ownership = try renderPixelOwnership(world, camera, observation.width, observation.height, allocator);
    defer ownership.deinit();

    const Occlusion = @import("../smc/world_view.zig").Occlusion;

    // For each entity, check if it has any visible pixels
    var query = world.queryPhysicsEntities();
    while (query.next()) |entity_id| {
        const visible_pixels = ownership.countOwned(entity_id.index);

        if (world.getOcclusion(entity_id)) |occlusion| {
            if (visible_pixels > 0) {
                // Entity is visible - reset occlusion
                world.setOcclusion(entity_id, Occlusion{
                    .state = .detected,
                    .duration = 0,
                    .existence_prob = 1.0,
                });
            } else {
                // Entity not visible - increment occlusion
                const new_state: TrackState = if (occlusion.state == .detected) .occluded else occlusion.state;
                world.setOcclusion(entity_id, Occlusion{
                    .state = new_state,
                    .duration = occlusion.duration + 1,
                    .existence_prob = occlusion.existence_prob * 0.99, // Slow decay
                });
            }
        }
    }
}

/// Check if entity should be automatically killed (long occlusion)
/// world can be any type with getOcclusion() method
pub fn shouldAutoDeath(
    entity_id: EntityId,
    world: anytype,
    config: CRPConfig,
) bool {
    if (world.getOcclusion(entity_id)) |occlusion| {
        return occlusion.duration > config.max_occlusion_before_death and
            occlusion.existence_prob < 0.1;
    }
    return false;
}

// =============================================================================
// Tests
// =============================================================================

const smc = @import("../smc/mod.zig");
const ParticleSwarm = smc.ParticleSwarm;
const particleWorld = smc.particleWorld;

test "DeathProposal structure" {
    const proposal = DeathProposal{
        .entity_id = EntityId{ .index = 5, .generation = 0 },
        .label = Label{ .birth_time = 0, .birth_index = 5 },
        .log_accept_prob = -0.5,
        .occlusion_duration = 10,
    };

    try std.testing.expect(proposal.occlusion_duration == 10);
}

test "proposeDeath with no entities" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 1, 8);
    defer swarm.deinit();
    var world = particleWorld(&swarm, 0);

    const label_set = LabelSet.init(allocator);
    // Note: don't defer deinit because we pass by value

    var obs = try ObservationGrid.init(10, 10, allocator);
    defer obs.deinit();

    const camera = Camera.default;
    const config = CRPConfig{};

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const proposal = try proposeDeath(&obs, &world, &label_set, camera, config, rng, allocator);

    // Should be null with no entities
    try std.testing.expect(proposal == null);
}
