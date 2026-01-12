//! C ABI Interface for libfizz
//!
//! This module exports a C-compatible API for the fizz physics engine,
//! enabling integration with Swift/AppKit, GTK, and other GUI frameworks.
//!
//! Architecture follows the Ghostty pattern:
//! - Opaque handles for Zig types (FizzWorld, FizzSMC)
//! - C-compatible structs for data exchange
//! - Explicit memory management (create/destroy pairs)
//!
//! Thread safety: All functions are thread-safe for distinct handles.
//! Do not call functions on the same handle from multiple threads.

const std = @import("std");
const fizz = @import("fizz");

const Vec3 = fizz.Vec3;
const Entity = fizz.Entity;
const Label = fizz.Label;
const PhysicsType = fizz.PhysicsType;
const PhysicsConfig = fizz.PhysicsConfig;
const Camera = fizz.Camera;
const World = fizz.World;
const SMCConfig = fizz.SMCConfig;
const SMCState = fizz.SMCState;
const ObservationGrid = fizz.ObservationGrid;
const GaussianMixture = fizz.GaussianMixture;

// =============================================================================
// C-Compatible Types
// =============================================================================

/// Opaque handle to a Fizz world.
pub const FizzWorld = opaque {};

/// Opaque handle to an SMC inference state.
pub const FizzSMC = opaque {};

/// Entity identifier (birth_time << 16 | birth_index).
/// Uses u64 to preserve full u32 birth_time without truncation.
/// Value 0 is reserved as invalid/error indicator.
pub const FizzEntityId = u64;

/// Invalid entity ID (returned on error).
pub const FIZZ_INVALID_ENTITY_ID: FizzEntityId = 0;

/// 3D vector for C interop.
pub const FizzVec3 = extern struct {
    x: f32,
    y: f32,
    z: f32,
};

/// Physics type enumeration.
/// Uses c_int to match C enum ABI (typically 4 bytes).
pub const FizzPhysicsType = enum(c_int) {
    standard = 0,
    bouncy = 1,
    sticky = 2,
    slippery = 3,
};

/// Physics configuration.
pub const FizzPhysicsConfig = extern struct {
    gravity_x: f32 = 0,
    gravity_y: f32 = -9.81,
    gravity_z: f32 = 0,
    dt: f32 = 1.0 / 60.0,
    ground_height: f32 = 0.0,
    bounds_min_x: f32 = -10,
    bounds_min_y: f32 = -10,
    bounds_min_z: f32 = -10,
    bounds_max_x: f32 = 10,
    bounds_max_y: f32 = 10,
    bounds_max_z: f32 = 10,
    process_noise: f32 = 0.01,
    crp_alpha: f32 = 1.0,
    survival_prob: f32 = 0.99,
};

/// SMC configuration.
pub const FizzSMCConfig = extern struct {
    num_particles: u32 = 100,
    ess_threshold: f32 = 0.5,
    observation_noise: f32 = 0.1,
    use_tempering: bool = true,
    initial_temperature: f32 = 0.1,
    temperature_increment: f32 = 0.1,
    gibbs_sweeps: u32 = 1,
};

/// Camera configuration for rendering observations.
pub const FizzCamera = extern struct {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    target_x: f32,
    target_y: f32,
    target_z: f32,
    up_x: f32 = 0,
    up_y: f32 = 1,
    up_z: f32 = 0,
    fov: f32 = 0.785398, // pi/4
    aspect: f32 = 1.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
};

/// Entity state for queries.
pub const FizzEntityState = extern struct {
    id: FizzEntityId,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    vel_x: f32,
    vel_y: f32,
    vel_z: f32,
    physics_type: FizzPhysicsType,
    is_alive: bool,
};

/// Physics type posterior for a single entity.
pub const FizzPosterior = extern struct {
    prob_standard: f32,
    prob_bouncy: f32,
    prob_sticky: f32,
    prob_slippery: f32,
};

/// Error codes.
pub const FizzError = enum(i32) {
    ok = 0,
    out_of_memory = -1,
    invalid_handle = -2,
    invalid_entity = -3,
    invalid_argument = -4,
};

// =============================================================================
// Internal State
// =============================================================================

/// Internal world state with allocator.
const WorldState = struct {
    world: World,
    allocator: std.mem.Allocator,
    rng: std.Random.DefaultPrng,
};

/// Internal SMC state with allocator.
const SMCInternalState = struct {
    smc: SMCState,
    allocator: std.mem.Allocator,
    observation: ?ObservationGrid,
};

// Global allocator for C API - thread-safe initialization.
var gpa: std.heap.GeneralPurposeAllocator(.{}) = std.heap.GeneralPurposeAllocator(.{}){};

fn getAllocator() std.mem.Allocator {
    return gpa.allocator();
}

// =============================================================================
// World API
// =============================================================================

/// Create a new physics world.
export fn fizz_world_create(config: *const FizzPhysicsConfig) ?*FizzWorld {
    const allocator = getAllocator();

    const zig_config = PhysicsConfig{
        .gravity = Vec3.init(config.gravity_x, config.gravity_y, config.gravity_z),
        .dt = config.dt,
        .ground_height = config.ground_height,
        .bounds_min = Vec3.init(config.bounds_min_x, config.bounds_min_y, config.bounds_min_z),
        .bounds_max = Vec3.init(config.bounds_max_x, config.bounds_max_y, config.bounds_max_z),
        .process_noise = config.process_noise,
        .crp_alpha = config.crp_alpha,
        .survival_prob = config.survival_prob,
    };

    const state = allocator.create(WorldState) catch return null;
    state.* = .{
        .world = World.init(allocator, zig_config),
        .allocator = allocator,
        .rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp())),
    };

    return @ptrCast(state);
}

/// Destroy a physics world.
export fn fizz_world_destroy(world_ptr: ?*FizzWorld) void {
    const state = worldState(world_ptr) orelse return;
    state.world.deinit();
    state.allocator.destroy(state);
}

/// Add an entity to the world.
export fn fizz_entity_add(
    world_ptr: ?*FizzWorld,
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    physics_type: FizzPhysicsType,
) FizzEntityId {
    const state = worldState(world_ptr) orelse return 0;

    const zig_ptype: PhysicsType = @enumFromInt(@intFromEnum(physics_type));
    const entity = state.world.addEntity(
        Vec3.init(x, y, z),
        Vec3.init(vx, vy, vz),
        zig_ptype,
    ) catch return 0;

    return encodeEntityId(entity.label);
}

/// Remove an entity from the world (marks as dead).
export fn fizz_entity_remove(world_ptr: ?*FizzWorld, entity_id: FizzEntityId) FizzError {
    const state = worldState(world_ptr) orelse return .invalid_handle;
    const label = decodeEntityId(entity_id);

    for (state.world.entities.items) |*entity| {
        if (entity.label.birth_time == label.birth_time and
            entity.label.birth_index == label.birth_index)
        {
            entity.track_state = .dead;
            return .ok;
        }
    }
    return .invalid_entity;
}

/// Apply a force to an entity.
export fn fizz_entity_apply_force(
    world_ptr: ?*FizzWorld,
    entity_id: FizzEntityId,
    fx: f32,
    fy: f32,
    fz: f32,
) FizzError {
    const state = worldState(world_ptr) orelse return .invalid_handle;
    const label = decodeEntityId(entity_id);

    for (state.world.entities.items) |*entity| {
        if (entity.label.birth_time == label.birth_time and
            entity.label.birth_index == label.birth_index and
            entity.isAlive())
        {
            // Apply force as velocity change (F = ma, assume m=1)
            const force = Vec3.init(fx, fy, fz);
            entity.velocity.mean = entity.velocity.mean.add(force.scale(state.world.config.dt));
            return .ok;
        }
    }
    return .invalid_entity;
}

/// Set entity position directly.
export fn fizz_entity_set_position(
    world_ptr: ?*FizzWorld,
    entity_id: FizzEntityId,
    x: f32,
    y: f32,
    z: f32,
) FizzError {
    const state = worldState(world_ptr) orelse return .invalid_handle;
    const label = decodeEntityId(entity_id);

    for (state.world.entities.items) |*entity| {
        if (entity.label.birth_time == label.birth_time and
            entity.label.birth_index == label.birth_index and
            entity.isAlive())
        {
            entity.position.mean = Vec3.init(x, y, z);
            return .ok;
        }
    }
    return .invalid_entity;
}

/// Set entity velocity directly.
export fn fizz_entity_set_velocity(
    world_ptr: ?*FizzWorld,
    entity_id: FizzEntityId,
    vx: f32,
    vy: f32,
    vz: f32,
) FizzError {
    const state = worldState(world_ptr) orelse return .invalid_handle;
    const label = decodeEntityId(entity_id);

    for (state.world.entities.items) |*entity| {
        if (entity.label.birth_time == label.birth_time and
            entity.label.birth_index == label.birth_index and
            entity.isAlive())
        {
            entity.velocity.mean = Vec3.init(vx, vy, vz);
            return .ok;
        }
    }
    return .invalid_entity;
}

/// Step the physics simulation.
export fn fizz_world_step(world_ptr: ?*FizzWorld) FizzError {
    const state = worldState(world_ptr) orelse return .invalid_handle;
    state.world.step(state.rng.random());
    return .ok;
}

/// Get entity count (alive entities only).
export fn fizz_world_entity_count(world_ptr: ?*FizzWorld) u32 {
    const state = worldState(world_ptr) orelse return 0;
    return @intCast(state.world.aliveCount());
}

/// Get entity state by index (0 to entity_count-1).
export fn fizz_world_get_entity(world_ptr: ?*FizzWorld, index: u32, out: *FizzEntityState) FizzError {
    const state = worldState(world_ptr) orelse return .invalid_handle;

    var alive_idx: u32 = 0;
    for (state.world.entities.items) |entity| {
        if (entity.isAlive()) {
            if (alive_idx == index) {
                const pos = entity.positionMean();
                const vel = entity.velocityMean();
                out.* = .{
                    .id = encodeEntityId(entity.label),
                    .pos_x = pos.x,
                    .pos_y = pos.y,
                    .pos_z = pos.z,
                    .vel_x = vel.x,
                    .vel_y = vel.y,
                    .vel_z = vel.z,
                    .physics_type = @enumFromInt(@intFromEnum(entity.physics_type)),
                    .is_alive = true,
                };
                return .ok;
            }
            alive_idx += 1;
        }
    }
    return .invalid_entity;
}

// =============================================================================
// SMC Inference API
// =============================================================================

/// Create a new SMC inference state.
export fn fizz_smc_create(
    physics_config: *const FizzPhysicsConfig,
    smc_config: *const FizzSMCConfig,
) ?*FizzSMC {
    const allocator = getAllocator();

    const zig_physics = PhysicsConfig{
        .gravity = Vec3.init(physics_config.gravity_x, physics_config.gravity_y, physics_config.gravity_z),
        .dt = physics_config.dt,
        .ground_height = physics_config.ground_height,
        .bounds_min = Vec3.init(physics_config.bounds_min_x, physics_config.bounds_min_y, physics_config.bounds_min_z),
        .bounds_max = Vec3.init(physics_config.bounds_max_x, physics_config.bounds_max_y, physics_config.bounds_max_z),
        .process_noise = physics_config.process_noise,
        .crp_alpha = physics_config.crp_alpha,
        .survival_prob = physics_config.survival_prob,
    };

    const zig_smc_config = SMCConfig{
        .num_particles = smc_config.num_particles,
        .ess_threshold = smc_config.ess_threshold,
        .observation_noise = smc_config.observation_noise,
        .use_tempering = smc_config.use_tempering,
        .initial_temperature = smc_config.initial_temperature,
        .temperature_increment = smc_config.temperature_increment,
        .gibbs_sweeps = smc_config.gibbs_sweeps,
        .physics = zig_physics,
    };

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));

    const internal = allocator.create(SMCInternalState) catch return null;
    internal.* = .{
        .smc = SMCState.init(allocator, zig_smc_config, prng.random()) catch {
            allocator.destroy(internal);
            return null;
        },
        .allocator = allocator,
        .observation = null,
    };

    return @ptrCast(internal);
}

/// Destroy an SMC inference state.
export fn fizz_smc_destroy(smc_ptr: ?*FizzSMC) void {
    const state = smcState(smc_ptr) orelse return;
    if (state.observation) |*obs| {
        obs.deinit();
    }
    state.smc.deinit();
    state.allocator.destroy(state);
}

/// Initialize SMC with prior entity positions.
export fn fizz_smc_init_prior(
    smc_ptr: ?*FizzSMC,
    positions: [*]const FizzVec3,
    velocities: [*]const FizzVec3,
    count: u32,
) FizzError {
    const state = smcState(smc_ptr) orelse return .invalid_handle;

    var pos_slice = state.allocator.alloc(Vec3, count) catch return .out_of_memory;
    defer state.allocator.free(pos_slice);
    var vel_slice = state.allocator.alloc(Vec3, count) catch return .out_of_memory;
    defer state.allocator.free(vel_slice);

    for (0..count) |i| {
        pos_slice[i] = Vec3.init(positions[i].x, positions[i].y, positions[i].z);
        vel_slice[i] = Vec3.init(velocities[i].x, velocities[i].y, velocities[i].z);
    }

    state.smc.initializeWithPrior(pos_slice, vel_slice) catch return .out_of_memory;
    return .ok;
}

/// Step SMC inference with an observation from the world.
export fn fizz_smc_step_with_world(
    smc_ptr: ?*FizzSMC,
    world_ptr: ?*FizzWorld,
    camera: *const FizzCamera,
) FizzError {
    const smc_internal = smcState(smc_ptr) orelse return .invalid_handle;
    const world_state = worldState(world_ptr) orelse return .invalid_handle;

    const zig_camera = Camera{
        .position = Vec3.init(camera.pos_x, camera.pos_y, camera.pos_z),
        .target = Vec3.init(camera.target_x, camera.target_y, camera.target_z),
        .up = Vec3.init(camera.up_x, camera.up_y, camera.up_z),
        .fov = camera.fov,
        .aspect = camera.aspect,
        .near = camera.near,
        .far = camera.far,
    };

    // Render observation from world
    const observation = world_state.world.render(zig_camera, 16, 16, 32) catch return .out_of_memory;

    // Free previous observation
    if (smc_internal.observation) |*obs| {
        obs.deinit();
    }
    smc_internal.observation = observation;

    // Step SMC
    smc_internal.smc.step(observation, zig_camera) catch return .out_of_memory;
    return .ok;
}

/// Get number of entities tracked by SMC.
export fn fizz_smc_entity_count(smc_ptr: ?*FizzSMC) u32 {
    const state = smcState(smc_ptr) orelse return 0;
    if (state.smc.particles.len == 0) return 0;
    return @intCast(state.smc.particles[0].entities.items.len);
}

/// Get physics type posteriors for all entities.
export fn fizz_smc_get_posteriors(
    smc_ptr: ?*FizzSMC,
    out: [*]FizzPosterior,
    max_count: u32,
) u32 {
    const state = smcState(smc_ptr) orelse return 0;

    const posteriors = state.smc.getPhysicsTypePosterior() catch return 0;
    defer state.allocator.free(posteriors);

    const count = @min(@as(u32, @intCast(posteriors.len)), max_count);
    for (0..count) |i| {
        out[i] = .{
            .prob_standard = posteriors[i][0],
            .prob_bouncy = posteriors[i][1],
            .prob_sticky = posteriors[i][2],
            .prob_slippery = posteriors[i][3],
        };
    }
    return count;
}

/// Get effective sample size.
export fn fizz_smc_get_ess(smc_ptr: ?*FizzSMC) f32 {
    const state = smcState(smc_ptr) orelse return 0;
    return state.smc.effectiveSampleSize();
}

/// Get current temperature.
export fn fizz_smc_get_temperature(smc_ptr: ?*FizzSMC) f32 {
    const state = smcState(smc_ptr) orelse return 0;
    return state.smc.temperature;
}

// =============================================================================
// Utility Functions
// =============================================================================

fn worldState(ptr: ?*FizzWorld) ?*WorldState {
    return @ptrCast(@alignCast(ptr));
}

fn smcState(ptr: ?*FizzSMC) ?*SMCInternalState {
    return @ptrCast(@alignCast(ptr));
}

fn encodeEntityId(label: Label) FizzEntityId {
    // Use full u32 birth_time (upper 32 bits) and u16 birth_index (lower 16 bits).
    // Add 1 to ensure ID 0 is never valid (0 is reserved for errors).
    return ((@as(u64, label.birth_time) << 16) | @as(u64, label.birth_index)) + 1;
}

fn decodeEntityId(id: FizzEntityId) Label {
    // Subtract 1 to reverse the +1 offset from encoding.
    const adjusted = id -| 1; // Saturating subtract to handle 0 gracefully
    return .{
        .birth_time = @truncate(adjusted >> 16),
        .birth_index = @truncate(adjusted & 0xFFFF),
    };
}

// =============================================================================
// Version Info
// =============================================================================

/// Get library version string.
export fn fizz_version() [*:0]const u8 {
    return "0.1.0";
}
