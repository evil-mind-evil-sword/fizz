//! ParticleSwarm: SoA (Struct of Arrays) layout for SMC particles
//!
//! This module provides a Pytree-style data layout for particle filters.
//! All particles share contiguous arrays for each component, enabling:
//! - SIMD-friendly memory access patterns
//! - Cache-efficient iteration over all particles
//! - Zero per-step allocations (preallocated padded arrays)
//! - Fast resampling via memcpy
//!
//! Memory Layout:
//! ```
//! positions: [P0E0, P0E1, ..., P0E63, P1E0, P1E1, ..., P1E63, ...]
//!            └─── particle 0 ────────┘└─── particle 1 ────────┘
//! ```
//!
//! Indexing: `swarm.positions[particle * max_entities + entity]`

const std = @import("std");
const math = @import("../math.zig");
const types = @import("../types.zig");

const Vec3 = math.Vec3;
const Mat3 = math.Mat3;
const Label = types.Label;
const PhysicsParams = types.PhysicsParams;
const PhysicsParamsUncertainty = types.PhysicsParamsUncertainty;
const ContactMode = types.ContactMode;
const TrackState = types.TrackState;
const GoalType = types.GoalType;
const SpatialRelationType = types.SpatialRelationType;
const CameraPose = types.CameraPose;
const Gaussian6D = types.Gaussian6D;
const CovTriangle21 = types.CovTriangle21;
const Allocator = std.mem.Allocator;

/// Default maximum entities per particle
pub const DEFAULT_MAX_ENTITIES: usize = 64;

/// Default number of particles
pub const DEFAULT_NUM_PARTICLES: usize = 100;

/// Covariance matrix stored as upper triangle (6 elements for 3x3 symmetric)
pub const CovTriangle = [6]f32;

/// Convert full Mat3 covariance to triangle storage
pub fn covToTriangle(cov: Mat3) CovTriangle {
    return .{
        cov.get(0, 0), cov.get(0, 1), cov.get(0, 2),
        cov.get(1, 1), cov.get(1, 2),
        cov.get(2, 2),
    };
}

/// Convert triangle storage back to full Mat3 (column-major)
/// Triangle layout: [cov_00, cov_01, cov_02, cov_11, cov_12, cov_22]
pub fn triangleToCov(tri: CovTriangle) Mat3 {
    // Column-major: data = [col0..., col1..., col2...]
    return Mat3{
        .data = .{
            tri[0], tri[1], tri[2], // col 0: (0,0), (1,0), (2,0)
            tri[1], tri[3], tri[4], // col 1: (0,1), (1,1), (2,1)
            tri[2], tri[4], tri[5], // col 2: (0,2), (1,2), (2,2)
        },
    };
}

/// View of a single entity's state (for convenient access)
pub const EntityView = struct {
    position_mean: Vec3,
    position_cov: CovTriangle,
    velocity_mean: Vec3,
    velocity_cov: CovTriangle,
    physics_params: PhysicsParams,
    physics_params_uncertainty: PhysicsParamsUncertainty,
    contact_mode: ContactMode,
    track_state: TrackState,
    label: Label,
    occlusion_count: u32,
    alive: bool,
    // Appearance fields (for rendering/observation model)
    color: Vec3,
    opacity: f32,
    radius: f32,
    // Agency/Goal fields (for agent entities)
    goal_type: GoalType,
    target_label: ?Label, // Label of target entity (for track/avoid/acquire)
    target_position: ?Vec3, // Target position (for reach goals)
    // Spatial relation fields (for geometric constraints)
    spatial_relation_type: SpatialRelationType,
    spatial_reference: ?Label, // Reference entity for spatial relation
    spatial_distance: f32, // Expected distance for proximity relations
    spatial_tolerance: f32, // Tolerance for relation satisfaction
};

/// View of a single particle's metadata
pub const ParticleView = struct {
    camera_pose: CameraPose,
    log_weight: f64,
    entity_count: u32,
};

/// SoA particle swarm for SMC inference
///
/// All state is stored in contiguous arrays with layout [num_particles × max_entities].
/// This enables SIMD operations and cache-efficient iteration.
pub const ParticleSwarm = struct {
    // =========================================================================
    // Per-entity state (shape: [num_particles * max_entities])
    // =========================================================================

    /// Position means (Kalman state)
    position_mean: []Vec3,
    /// Position covariances (upper triangle)
    position_cov: []CovTriangle,
    /// Velocity means (Kalman state)
    velocity_mean: []Vec3,
    /// Velocity covariances (upper triangle)
    velocity_cov: []CovTriangle,

    // =========================================================================
    // Coupled 6D State (for cross-covariance tracking)
    // =========================================================================

    /// 6D state means [px, py, pz, vx, vy, vz] (coupled position+velocity)
    state_mean: [][6]f32,
    /// 6D state covariances (21-element upper triangle of 6x6)
    state_cov: []CovTriangle21,

    /// Continuous physics parameters (Spelke-aligned inference)
    physics_params: []PhysicsParams,
    /// Uncertainty in physics parameters (Beta distribution params for conjugate updates)
    physics_params_uncertainty: []PhysicsParamsUncertainty,
    /// Contact mode (discrete)
    contact_mode: []ContactMode,
    /// Track state (detected/occluded/tentative/dead)
    track_state: []TrackState,
    /// Entity label (birth_time, birth_index)
    label: []Label,
    /// Frames since last detection
    occlusion_count: []u32,
    /// Whether slot is active (for padded arrays)
    alive: []bool,
    /// Goal type (discrete, Gibbs-sampled for agent entities)
    goal_type: []GoalType,
    /// Target entity label (for track/avoid/acquire goals)
    target_label: []?Label,
    /// Target position (for reach goals)
    target_position: []?Vec3,
    /// Spatial relation type (for geometric constraints)
    spatial_relation_type: []SpatialRelationType,
    /// Reference entity for spatial relation
    spatial_reference: []?Label,
    /// Expected distance for proximity relations
    spatial_distance: []f32,
    /// Tolerance for relation satisfaction
    spatial_tolerance: []f32,

    // =========================================================================
    // Appearance state (for rendering/observation model)
    // =========================================================================

    /// Entity color (RGB)
    color: []Vec3,
    /// Entity opacity
    opacity: []f32,
    /// Entity radius for rendering
    radius: []f32,

    // =========================================================================
    // Previous state for Gibbs dynamics likelihood
    // =========================================================================

    /// Previous position means (before physics step)
    prev_position_mean: []Vec3,
    /// Previous velocity means (before physics step)
    prev_velocity_mean: []Vec3,
    /// Previous contact mode (before physics step) - critical for bounce detection
    prev_contact_mode: []ContactMode,

    // =========================================================================
    // Per-particle state (shape: [num_particles])
    // =========================================================================

    /// Camera poses (sampled, not Rao-Blackwellized)
    camera_poses: []CameraPose,
    /// Log weights (unnormalized)
    log_weights: []f64,
    /// Number of alive entities per particle
    entity_counts: []u32,

    // =========================================================================
    // Dimensions and allocator
    // =========================================================================

    num_particles: usize,
    max_entities: usize,
    allocator: Allocator,

    // =========================================================================
    // Double buffer for efficient resampling
    // =========================================================================

    /// Swap buffer (allocated lazily on first resample)
    swap_position_mean: ?[]Vec3,
    swap_position_cov: ?[]CovTriangle,
    swap_velocity_mean: ?[]Vec3,
    swap_velocity_cov: ?[]CovTriangle,
    swap_state_mean: ?[][6]f32,
    swap_state_cov: ?[]CovTriangle21,
    swap_physics_params: ?[]PhysicsParams,
    swap_physics_params_uncertainty: ?[]PhysicsParamsUncertainty,
    swap_contact_mode: ?[]ContactMode,
    swap_track_state: ?[]TrackState,
    swap_label: ?[]Label,
    swap_occlusion_count: ?[]u32,
    swap_alive: ?[]bool,
    swap_goal_type: ?[]GoalType,
    swap_target_label: ?[]?Label,
    swap_target_position: ?[]?Vec3,
    swap_spatial_relation_type: ?[]SpatialRelationType,
    swap_spatial_reference: ?[]?Label,
    swap_spatial_distance: ?[]f32,
    swap_spatial_tolerance: ?[]f32,
    swap_color: ?[]Vec3,
    swap_opacity: ?[]f32,
    swap_radius: ?[]f32,
    swap_camera_poses: ?[]CameraPose,
    swap_log_weights: ?[]f64,
    swap_entity_counts: ?[]u32,

    const Self = @This();

    /// Initialize a new particle swarm with given dimensions
    pub fn init(
        allocator: Allocator,
        num_particles: usize,
        max_entities: usize,
    ) !Self {
        const entity_slots = num_particles * max_entities;

        // Allocate all entity arrays
        const position_mean = try allocator.alloc(Vec3, entity_slots);
        errdefer allocator.free(position_mean);
        const position_cov = try allocator.alloc(CovTriangle, entity_slots);
        errdefer allocator.free(position_cov);
        const velocity_mean = try allocator.alloc(Vec3, entity_slots);
        errdefer allocator.free(velocity_mean);
        const velocity_cov = try allocator.alloc(CovTriangle, entity_slots);
        errdefer allocator.free(velocity_cov);

        // 6D coupled state arrays
        const state_mean = try allocator.alloc([6]f32, entity_slots);
        errdefer allocator.free(state_mean);
        const state_cov = try allocator.alloc(CovTriangle21, entity_slots);
        errdefer allocator.free(state_cov);

        const physics_params_arr = try allocator.alloc(PhysicsParams, entity_slots);
        errdefer allocator.free(physics_params_arr);
        const physics_params_uncertainty_arr = try allocator.alloc(PhysicsParamsUncertainty, entity_slots);
        errdefer allocator.free(physics_params_uncertainty_arr);
        const contact_mode = try allocator.alloc(ContactMode, entity_slots);
        errdefer allocator.free(contact_mode);
        const track_state_arr = try allocator.alloc(TrackState, entity_slots);
        errdefer allocator.free(track_state_arr);
        const label_arr = try allocator.alloc(Label, entity_slots);
        errdefer allocator.free(label_arr);
        const occlusion_count = try allocator.alloc(u32, entity_slots);
        errdefer allocator.free(occlusion_count);
        const alive_arr = try allocator.alloc(bool, entity_slots);
        errdefer allocator.free(alive_arr);
        const goal_type_arr = try allocator.alloc(GoalType, entity_slots);
        errdefer allocator.free(goal_type_arr);
        const target_label_arr = try allocator.alloc(?Label, entity_slots);
        errdefer allocator.free(target_label_arr);
        const target_position_arr = try allocator.alloc(?Vec3, entity_slots);
        errdefer allocator.free(target_position_arr);
        const spatial_relation_type_arr = try allocator.alloc(SpatialRelationType, entity_slots);
        errdefer allocator.free(spatial_relation_type_arr);
        const spatial_reference_arr = try allocator.alloc(?Label, entity_slots);
        errdefer allocator.free(spatial_reference_arr);
        const spatial_distance_arr = try allocator.alloc(f32, entity_slots);
        errdefer allocator.free(spatial_distance_arr);
        const spatial_tolerance_arr = try allocator.alloc(f32, entity_slots);
        errdefer allocator.free(spatial_tolerance_arr);

        // Appearance arrays
        const color_arr = try allocator.alloc(Vec3, entity_slots);
        errdefer allocator.free(color_arr);
        const opacity_arr = try allocator.alloc(f32, entity_slots);
        errdefer allocator.free(opacity_arr);
        const radius_arr = try allocator.alloc(f32, entity_slots);
        errdefer allocator.free(radius_arr);

        // Previous state arrays
        const prev_position_mean = try allocator.alloc(Vec3, entity_slots);
        errdefer allocator.free(prev_position_mean);
        const prev_velocity_mean = try allocator.alloc(Vec3, entity_slots);
        errdefer allocator.free(prev_velocity_mean);
        const prev_contact_mode = try allocator.alloc(ContactMode, entity_slots);
        errdefer allocator.free(prev_contact_mode);
        @memset(prev_contact_mode, .free);

        // Per-particle arrays
        const camera_poses = try allocator.alloc(CameraPose, num_particles);
        errdefer allocator.free(camera_poses);
        const log_weights = try allocator.alloc(f64, num_particles);
        errdefer allocator.free(log_weights);
        const entity_counts = try allocator.alloc(u32, num_particles);
        errdefer allocator.free(entity_counts);

        // Initialize to zero/default
        @memset(alive_arr, false);
        // Initialize 6D state arrays
        @memset(state_mean, .{ 0, 0, 0, 0, 0, 0 });
        @memset(state_cov, CovTriangle21.identity);
        // Initialize physics params with weak prior
        @memset(physics_params_arr, PhysicsParams.prior);
        @memset(physics_params_uncertainty_arr, PhysicsParamsUncertainty.weak_prior);
        @memset(goal_type_arr, .none);
        @memset(target_label_arr, null);
        @memset(target_position_arr, null);
        @memset(spatial_relation_type_arr, .none);
        @memset(spatial_reference_arr, null);
        @memset(spatial_distance_arr, 0);
        @memset(spatial_tolerance_arr, 1.0);
        // Appearance defaults
        for (color_arr) |*c| c.* = Vec3.init(0.5, 0.5, 0.5);
        @memset(opacity_arr, 1.0);
        @memset(radius_arr, 0.5);
        @memset(entity_counts, 0);
        @memset(log_weights, 0.0);
        for (camera_poses) |*cp| {
            cp.* = CameraPose.default;
        }

        return .{
            .position_mean = position_mean,
            .position_cov = position_cov,
            .velocity_mean = velocity_mean,
            .velocity_cov = velocity_cov,
            .state_mean = state_mean,
            .state_cov = state_cov,
            .physics_params = physics_params_arr,
            .physics_params_uncertainty = physics_params_uncertainty_arr,
            .contact_mode = contact_mode,
            .track_state = track_state_arr,
            .label = label_arr,
            .occlusion_count = occlusion_count,
            .alive = alive_arr,
            .goal_type = goal_type_arr,
            .target_label = target_label_arr,
            .target_position = target_position_arr,
            .spatial_relation_type = spatial_relation_type_arr,
            .spatial_reference = spatial_reference_arr,
            .spatial_distance = spatial_distance_arr,
            .spatial_tolerance = spatial_tolerance_arr,
            .color = color_arr,
            .opacity = opacity_arr,
            .radius = radius_arr,
            .prev_position_mean = prev_position_mean,
            .prev_velocity_mean = prev_velocity_mean,
            .prev_contact_mode = prev_contact_mode,
            .camera_poses = camera_poses,
            .log_weights = log_weights,
            .entity_counts = entity_counts,
            .num_particles = num_particles,
            .max_entities = max_entities,
            .allocator = allocator,
            // Swap buffers allocated lazily
            .swap_position_mean = null,
            .swap_position_cov = null,
            .swap_velocity_mean = null,
            .swap_velocity_cov = null,
            .swap_state_mean = null,
            .swap_state_cov = null,
            .swap_physics_params = null,
            .swap_physics_params_uncertainty = null,
            .swap_contact_mode = null,
            .swap_track_state = null,
            .swap_label = null,
            .swap_occlusion_count = null,
            .swap_alive = null,
            .swap_goal_type = null,
            .swap_target_label = null,
            .swap_target_position = null,
            .swap_spatial_relation_type = null,
            .swap_spatial_reference = null,
            .swap_spatial_distance = null,
            .swap_spatial_tolerance = null,
            .swap_color = null,
            .swap_opacity = null,
            .swap_radius = null,
            .swap_camera_poses = null,
            .swap_log_weights = null,
            .swap_entity_counts = null,
        };
    }

    /// Free all allocated memory
    pub fn deinit(self: *Self) void {
        // Entity arrays
        self.allocator.free(self.position_mean);
        self.allocator.free(self.position_cov);
        self.allocator.free(self.velocity_mean);
        self.allocator.free(self.velocity_cov);
        self.allocator.free(self.state_mean);
        self.allocator.free(self.state_cov);
        self.allocator.free(self.physics_params);
        self.allocator.free(self.physics_params_uncertainty);
        self.allocator.free(self.contact_mode);
        self.allocator.free(self.track_state);
        self.allocator.free(self.label);
        self.allocator.free(self.occlusion_count);
        self.allocator.free(self.alive);
        self.allocator.free(self.goal_type);
        self.allocator.free(self.target_label);
        self.allocator.free(self.target_position);
        self.allocator.free(self.spatial_relation_type);
        self.allocator.free(self.spatial_reference);
        self.allocator.free(self.spatial_distance);
        self.allocator.free(self.spatial_tolerance);
        self.allocator.free(self.color);
        self.allocator.free(self.opacity);
        self.allocator.free(self.radius);
        self.allocator.free(self.prev_position_mean);
        self.allocator.free(self.prev_velocity_mean);
        self.allocator.free(self.prev_contact_mode);

        // Particle arrays
        self.allocator.free(self.camera_poses);
        self.allocator.free(self.log_weights);
        self.allocator.free(self.entity_counts);

        // Swap buffers (if allocated)
        if (self.swap_position_mean) |buf| self.allocator.free(buf);
        if (self.swap_position_cov) |buf| self.allocator.free(buf);
        if (self.swap_velocity_mean) |buf| self.allocator.free(buf);
        if (self.swap_velocity_cov) |buf| self.allocator.free(buf);
        if (self.swap_state_mean) |buf| self.allocator.free(buf);
        if (self.swap_state_cov) |buf| self.allocator.free(buf);
        if (self.swap_physics_params) |buf| self.allocator.free(buf);
        if (self.swap_physics_params_uncertainty) |buf| self.allocator.free(buf);
        if (self.swap_contact_mode) |buf| self.allocator.free(buf);
        if (self.swap_track_state) |buf| self.allocator.free(buf);
        if (self.swap_label) |buf| self.allocator.free(buf);
        if (self.swap_occlusion_count) |buf| self.allocator.free(buf);
        if (self.swap_alive) |buf| self.allocator.free(buf);
        if (self.swap_goal_type) |buf| self.allocator.free(buf);
        if (self.swap_target_label) |buf| self.allocator.free(buf);
        if (self.swap_target_position) |buf| self.allocator.free(buf);
        if (self.swap_spatial_relation_type) |buf| self.allocator.free(buf);
        if (self.swap_spatial_reference) |buf| self.allocator.free(buf);
        if (self.swap_spatial_distance) |buf| self.allocator.free(buf);
        if (self.swap_spatial_tolerance) |buf| self.allocator.free(buf);
        if (self.swap_color) |buf| self.allocator.free(buf);
        if (self.swap_opacity) |buf| self.allocator.free(buf);
        if (self.swap_radius) |buf| self.allocator.free(buf);
        if (self.swap_camera_poses) |buf| self.allocator.free(buf);
        if (self.swap_log_weights) |buf| self.allocator.free(buf);
        if (self.swap_entity_counts) |buf| self.allocator.free(buf);
    }

    // =========================================================================
    // Indexing
    // =========================================================================

    /// Get flat index for (particle, entity) pair
    pub inline fn idx(self: Self, particle: usize, entity: usize) usize {
        std.debug.assert(particle < self.num_particles);
        std.debug.assert(entity < self.max_entities);
        return particle * self.max_entities + entity;
    }

    /// Get slice for all entities of a particle
    pub inline fn particleSlice(self: Self, particle: usize) struct { start: usize, end: usize } {
        const start = particle * self.max_entities;
        return .{ .start = start, .end = start + self.max_entities };
    }

    // =========================================================================
    // Entity Access
    // =========================================================================

    /// Get entity view (read-only snapshot)
    pub fn getEntity(self: Self, particle: usize, entity: usize) EntityView {
        const i = self.idx(particle, entity);
        return .{
            .position_mean = self.position_mean[i],
            .position_cov = self.position_cov[i],
            .velocity_mean = self.velocity_mean[i],
            .velocity_cov = self.velocity_cov[i],
            .physics_params = self.physics_params[i],
            .physics_params_uncertainty = self.physics_params_uncertainty[i],
            .contact_mode = self.contact_mode[i],
            .track_state = self.track_state[i],
            .label = self.label[i],
            .occlusion_count = self.occlusion_count[i],
            .alive = self.alive[i],
            .color = self.color[i],
            .opacity = self.opacity[i],
            .radius = self.radius[i],
            .goal_type = self.goal_type[i],
            .target_label = self.target_label[i],
            .target_position = self.target_position[i],
            .spatial_relation_type = self.spatial_relation_type[i],
            .spatial_reference = self.spatial_reference[i],
            .spatial_distance = self.spatial_distance[i],
            .spatial_tolerance = self.spatial_tolerance[i],
        };
    }

    /// Set entity from view
    pub fn setEntity(self: *Self, particle: usize, entity: usize, view: EntityView) void {
        const i = self.idx(particle, entity);
        self.position_mean[i] = view.position_mean;
        self.position_cov[i] = view.position_cov;
        self.velocity_mean[i] = view.velocity_mean;
        self.velocity_cov[i] = view.velocity_cov;
        self.physics_params[i] = view.physics_params;
        self.physics_params_uncertainty[i] = view.physics_params_uncertainty;
        self.contact_mode[i] = view.contact_mode;
        self.track_state[i] = view.track_state;
        self.label[i] = view.label;
        self.occlusion_count[i] = view.occlusion_count;
        self.alive[i] = view.alive;
        self.color[i] = view.color;
        self.opacity[i] = view.opacity;
        self.radius[i] = view.radius;
        self.goal_type[i] = view.goal_type;
        self.target_label[i] = view.target_label;
        self.target_position[i] = view.target_position;
        self.spatial_relation_type[i] = view.spatial_relation_type;
        self.spatial_reference[i] = view.spatial_reference;
        self.spatial_distance[i] = view.spatial_distance;
        self.spatial_tolerance[i] = view.spatial_tolerance;

        // Initialize 6D state from factored arrays (zero cross-covariance initially)
        const pos = view.position_mean;
        const vel = view.velocity_mean;
        self.state_mean[i] = .{ pos.x, pos.y, pos.z, vel.x, vel.y, vel.z };

        // Build 6D covariance from factored covariances with zero cross-covariance
        const P_pp = Mat3{
            .data = .{
                view.position_cov[0], view.position_cov[1], view.position_cov[2],
                view.position_cov[1], view.position_cov[3], view.position_cov[4],
                view.position_cov[2], view.position_cov[4], view.position_cov[5],
            },
        };
        const P_vv = Mat3{
            .data = .{
                view.velocity_cov[0], view.velocity_cov[1], view.velocity_cov[2],
                view.velocity_cov[1], view.velocity_cov[3], view.velocity_cov[4],
                view.velocity_cov[2], view.velocity_cov[4], view.velocity_cov[5],
            },
        };
        self.state_cov[i] = CovTriangle21.fromBlocks(P_pp, P_vv, Mat3.zero);
    }

    /// Check if entity slot is alive
    pub fn isAlive(self: Self, particle: usize, entity: usize) bool {
        return self.alive[self.idx(particle, entity)];
    }

    /// Mark entity slot as dead
    pub fn killEntity(self: *Self, particle: usize, entity: usize) void {
        const i = self.idx(particle, entity);
        self.alive[i] = false;
        self.track_state[i] = .dead;
        // Decrement count
        if (self.entity_counts[particle] > 0) {
            self.entity_counts[particle] -= 1;
        }
    }

    /// Add entity to particle (finds first free slot)
    /// Returns entity index or null if full
    pub fn addEntity(self: *Self, particle: usize, view: EntityView) ?usize {
        const slice = self.particleSlice(particle);
        for (slice.start..slice.end) |i| {
            if (!self.alive[i]) {
                const entity = i - slice.start;
                self.setEntity(particle, entity, view);
                self.alive[i] = true;
                self.entity_counts[particle] += 1;
                return entity;
            }
        }
        return null; // No free slots
    }

    // =========================================================================
    // 6D Coupled State Access
    // =========================================================================

    /// Get 6D coupled state (position + velocity with cross-covariance)
    pub fn getState6D(self: *const Self, entity_idx: usize) Gaussian6D {
        return Gaussian6D{
            .mean = self.state_mean[entity_idx],
            .cov = self.state_cov[entity_idx],
        };
    }

    /// Set 6D coupled state (position + velocity with cross-covariance)
    pub fn setState6D(self: *Self, entity_idx: usize, state: Gaussian6D) void {
        self.state_mean[entity_idx] = state.mean;
        self.state_cov[entity_idx] = state.cov;

        // Keep factored arrays in sync for backward compatibility
        self.position_mean[entity_idx] = state.position();
        self.velocity_mean[entity_idx] = state.velocity();

        // Extract position and velocity covariances (loses cross-covariance!)
        const pos_tri = state.cov.positionBlock();
        const vel_tri = state.cov.velocityBlock();
        self.position_cov[entity_idx] = pos_tri;
        self.velocity_cov[entity_idx] = vel_tri;
    }

    /// Get 6D state for (particle, entity) pair
    pub fn getState6DAt(self: *const Self, particle: usize, entity: usize) Gaussian6D {
        return self.getState6D(self.idx(particle, entity));
    }

    /// Set 6D state for (particle, entity) pair
    pub fn setState6DAt(self: *Self, particle: usize, entity: usize, state: Gaussian6D) void {
        self.setState6D(self.idx(particle, entity), state);
    }

    /// Initialize 6D state from position and velocity means (for new entities)
    /// Creates diagonal covariance with default variances (no cross-covariance)
    pub fn initState6D(self: *Self, entity_idx: usize, pos: Vec3, vel: Vec3) void {
        // Default variances for new entities
        const pos_var: f32 = 0.1;
        const vel_var: f32 = 0.01;

        self.state_mean[entity_idx] = .{ pos.x, pos.y, pos.z, vel.x, vel.y, vel.z };
        self.state_cov[entity_idx] = CovTriangle21.diagonal(
            Vec3.splat(pos_var),
            Vec3.splat(vel_var),
        );

        // Keep factored arrays in sync
        self.position_mean[entity_idx] = pos;
        self.velocity_mean[entity_idx] = vel;
        self.position_cov[entity_idx] = .{ pos_var, 0, 0, pos_var, 0, pos_var };
        self.velocity_cov[entity_idx] = .{ vel_var, 0, 0, vel_var, 0, vel_var };
    }

    /// Sync 6D state mean from factored position/velocity arrays
    /// Call this after directly modifying position_mean/velocity_mean (e.g. after bounce handling)
    /// Preserves the 6D covariance including cross-covariance
    pub fn syncFactoredTo6DMean(self: *Self, entity_idx: usize) void {
        const pos = self.position_mean[entity_idx];
        const vel = self.velocity_mean[entity_idx];
        self.state_mean[entity_idx] = .{ pos.x, pos.y, pos.z, vel.x, vel.y, vel.z };
    }

    // =========================================================================
    // Particle Access
    // =========================================================================

    /// Get particle metadata view
    pub fn getParticle(self: Self, particle: usize) ParticleView {
        return .{
            .camera_pose = self.camera_poses[particle],
            .log_weight = self.log_weights[particle],
            .entity_count = self.entity_counts[particle],
        };
    }

    /// Set particle camera pose
    pub fn setCameraPose(self: *Self, particle: usize, pose: CameraPose) void {
        self.camera_poses[particle] = pose;
    }

    /// Set particle weight
    pub fn setLogWeight(self: *Self, particle: usize, weight: f64) void {
        self.log_weights[particle] = weight;
    }

    // =========================================================================
    // Bulk Operations
    // =========================================================================

    /// Save current state to previous (before physics step)
    pub fn savePreviousState(self: *Self) void {
        @memcpy(self.prev_position_mean, self.position_mean);
        @memcpy(self.prev_velocity_mean, self.velocity_mean);
    }

    /// Normalize log weights (subtract max for numerical stability)
    /// If all weights are -inf, sets all to 0 (uniform)
    pub fn normalizeWeights(self: *Self) void {
        var max_weight: f64 = -std.math.inf(f64);
        for (self.log_weights) |w| {
            if (w > max_weight) max_weight = w;
        }

        // Handle degenerate case: all weights are -inf
        if (max_weight == -std.math.inf(f64)) {
            @memset(self.log_weights, 0.0);
            return;
        }

        for (self.log_weights) |*w| {
            w.* -= max_weight;
        }
    }

    /// Compute effective sample size (ESS)
    /// Note: Call normalizeWeights() first for numerical stability
    pub fn effectiveSampleSize(self: Self) f64 {
        var sum_w: f64 = 0;
        var sum_w2: f64 = 0;

        for (self.log_weights) |log_w| {
            const w = @exp(log_w);
            sum_w += w;
            sum_w2 += w * w;
        }

        if (sum_w2 < 1e-300) return 0;
        return (sum_w * sum_w) / sum_w2;
    }

    /// Compute alive entity count across all particles (for statistics)
    pub fn totalAliveEntities(self: Self) usize {
        var total: usize = 0;
        for (self.entity_counts) |c| {
            total += c;
        }
        return total;
    }

    /// Ensure swap buffers are allocated (lazy allocation for resampling)
    /// Call once before first resample, then reuse for zero-allocation resampling
    pub fn ensureSwapBuffers(self: *Self) !void {
        const entity_slots = self.num_particles * self.max_entities;
        const n = self.num_particles;

        if (self.swap_position_mean == null) {
            self.swap_position_mean = try self.allocator.alloc(Vec3, entity_slots);
        }
        if (self.swap_position_cov == null) {
            self.swap_position_cov = try self.allocator.alloc(CovTriangle, entity_slots);
        }
        if (self.swap_velocity_mean == null) {
            self.swap_velocity_mean = try self.allocator.alloc(Vec3, entity_slots);
        }
        if (self.swap_velocity_cov == null) {
            self.swap_velocity_cov = try self.allocator.alloc(CovTriangle, entity_slots);
        }
        if (self.swap_state_mean == null) {
            self.swap_state_mean = try self.allocator.alloc([6]f32, entity_slots);
        }
        if (self.swap_state_cov == null) {
            self.swap_state_cov = try self.allocator.alloc(CovTriangle21, entity_slots);
        }
        if (self.swap_physics_params == null) {
            self.swap_physics_params = try self.allocator.alloc(PhysicsParams, entity_slots);
        }
        if (self.swap_physics_params_uncertainty == null) {
            self.swap_physics_params_uncertainty = try self.allocator.alloc(PhysicsParamsUncertainty, entity_slots);
        }
        if (self.swap_contact_mode == null) {
            self.swap_contact_mode = try self.allocator.alloc(ContactMode, entity_slots);
        }
        if (self.swap_track_state == null) {
            self.swap_track_state = try self.allocator.alloc(TrackState, entity_slots);
        }
        if (self.swap_label == null) {
            self.swap_label = try self.allocator.alloc(Label, entity_slots);
        }
        if (self.swap_occlusion_count == null) {
            self.swap_occlusion_count = try self.allocator.alloc(u32, entity_slots);
        }
        if (self.swap_alive == null) {
            self.swap_alive = try self.allocator.alloc(bool, entity_slots);
        }
        if (self.swap_goal_type == null) {
            self.swap_goal_type = try self.allocator.alloc(GoalType, entity_slots);
        }
        if (self.swap_target_label == null) {
            self.swap_target_label = try self.allocator.alloc(?Label, entity_slots);
        }
        if (self.swap_target_position == null) {
            self.swap_target_position = try self.allocator.alloc(?Vec3, entity_slots);
        }
        if (self.swap_spatial_relation_type == null) {
            self.swap_spatial_relation_type = try self.allocator.alloc(SpatialRelationType, entity_slots);
        }
        if (self.swap_spatial_reference == null) {
            self.swap_spatial_reference = try self.allocator.alloc(?Label, entity_slots);
        }
        if (self.swap_spatial_distance == null) {
            self.swap_spatial_distance = try self.allocator.alloc(f32, entity_slots);
        }
        if (self.swap_spatial_tolerance == null) {
            self.swap_spatial_tolerance = try self.allocator.alloc(f32, entity_slots);
        }
        if (self.swap_color == null) {
            self.swap_color = try self.allocator.alloc(Vec3, entity_slots);
        }
        if (self.swap_opacity == null) {
            self.swap_opacity = try self.allocator.alloc(f32, entity_slots);
        }
        if (self.swap_radius == null) {
            self.swap_radius = try self.allocator.alloc(f32, entity_slots);
        }
        if (self.swap_camera_poses == null) {
            self.swap_camera_poses = try self.allocator.alloc(CameraPose, n);
        }
        if (self.swap_log_weights == null) {
            self.swap_log_weights = try self.allocator.alloc(f64, n);
        }
        if (self.swap_entity_counts == null) {
            self.swap_entity_counts = try self.allocator.alloc(u32, n);
        }
    }

    /// Get swap buffer pointers (assumes ensureSwapBuffers was called)
    /// Used by SMCState.resample() for zero-allocation resampling
    pub const SwapBuffers = struct {
        position_mean: []Vec3,
        position_cov: []CovTriangle,
        velocity_mean: []Vec3,
        velocity_cov: []CovTriangle,
        state_mean: [][6]f32,
        state_cov: []CovTriangle21,
        physics_params: []PhysicsParams,
        physics_params_uncertainty: []PhysicsParamsUncertainty,
        contact_mode: []ContactMode,
        track_state: []TrackState,
        label: []Label,
        occlusion_count: []u32,
        alive: []bool,
        goal_type: []GoalType,
        target_label: []?Label,
        target_position: []?Vec3,
        spatial_relation_type: []SpatialRelationType,
        spatial_reference: []?Label,
        spatial_distance: []f32,
        spatial_tolerance: []f32,
        color: []Vec3,
        opacity: []f32,
        radius: []f32,
        camera_poses: []CameraPose,
        log_weights: []f64,
        entity_counts: []u32,
    };

    pub fn getSwapBuffers(self: Self) ?SwapBuffers {
        return .{
            .position_mean = self.swap_position_mean orelse return null,
            .position_cov = self.swap_position_cov orelse return null,
            .velocity_mean = self.swap_velocity_mean orelse return null,
            .velocity_cov = self.swap_velocity_cov orelse return null,
            .state_mean = self.swap_state_mean orelse return null,
            .state_cov = self.swap_state_cov orelse return null,
            .physics_params = self.swap_physics_params orelse return null,
            .physics_params_uncertainty = self.swap_physics_params_uncertainty orelse return null,
            .contact_mode = self.swap_contact_mode orelse return null,
            .track_state = self.swap_track_state orelse return null,
            .label = self.swap_label orelse return null,
            .occlusion_count = self.swap_occlusion_count orelse return null,
            .alive = self.swap_alive orelse return null,
            .goal_type = self.swap_goal_type orelse return null,
            .target_label = self.swap_target_label orelse return null,
            .target_position = self.swap_target_position orelse return null,
            .spatial_relation_type = self.swap_spatial_relation_type orelse return null,
            .spatial_reference = self.swap_spatial_reference orelse return null,
            .spatial_distance = self.swap_spatial_distance orelse return null,
            .spatial_tolerance = self.swap_spatial_tolerance orelse return null,
            .color = self.swap_color orelse return null,
            .opacity = self.swap_opacity orelse return null,
            .radius = self.swap_radius orelse return null,
            .camera_poses = self.swap_camera_poses orelse return null,
            .log_weights = self.swap_log_weights orelse return null,
            .entity_counts = self.swap_entity_counts orelse return null,
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ParticleSwarm init and deinit" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 10, 8);
    defer swarm.deinit();

    try std.testing.expectEqual(@as(usize, 10), swarm.num_particles);
    try std.testing.expectEqual(@as(usize, 8), swarm.max_entities);
    try std.testing.expectEqual(@as(usize, 80), swarm.position_mean.len);
}

test "ParticleSwarm indexing" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 4, 16);
    defer swarm.deinit();

    // particle 0, entity 0 = index 0
    try std.testing.expectEqual(@as(usize, 0), swarm.idx(0, 0));
    // particle 0, entity 5 = index 5
    try std.testing.expectEqual(@as(usize, 5), swarm.idx(0, 5));
    // particle 1, entity 0 = index 16
    try std.testing.expectEqual(@as(usize, 16), swarm.idx(1, 0));
    // particle 2, entity 3 = index 35
    try std.testing.expectEqual(@as(usize, 35), swarm.idx(2, 3));
}

test "ParticleSwarm entity get/set round trip" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 2, 4);
    defer swarm.deinit();

    const test_entity = EntityView{
        .position_mean = Vec3.init(1.0, 2.0, 3.0),
        .position_cov = .{ 0.1, 0, 0, 0.1, 0, 0.1 },
        .velocity_mean = Vec3.init(0.5, -0.5, 0),
        .velocity_cov = .{ 0.01, 0, 0, 0.01, 0, 0.01 },
        .physics_params = PhysicsParams.prior,
        .physics_params_uncertainty = PhysicsParamsUncertainty.weak_prior,
        .contact_mode = .environment,
        .track_state = .detected,
        .label = .{ .birth_time = 42, .birth_index = 7 },
        .occlusion_count = 3,
        .alive = true,
        .color = Vec3.init(1.0, 0.0, 0.0),
        .opacity = 1.0,
        .radius = 0.5,
        .goal_type = .none,
        .target_label = null,
        .target_position = null,
        .spatial_relation_type = .none,
        .spatial_reference = null,
        .spatial_distance = 0,
        .spatial_tolerance = 1.0,
    };

    swarm.setEntity(1, 2, test_entity);
    const retrieved = swarm.getEntity(1, 2);

    try std.testing.expectEqual(test_entity.position_mean.x, retrieved.position_mean.x);
    try std.testing.expectEqual(test_entity.position_mean.y, retrieved.position_mean.y);
    try std.testing.expectEqual(test_entity.position_mean.z, retrieved.position_mean.z);
    try std.testing.expectEqual(test_entity.physics_params.elasticity, retrieved.physics_params.elasticity);
    try std.testing.expectEqual(test_entity.label.birth_time, retrieved.label.birth_time);
    try std.testing.expectEqual(test_entity.label.birth_index, retrieved.label.birth_index);
    try std.testing.expectEqual(test_entity.alive, retrieved.alive);
}

test "ParticleSwarm addEntity and killEntity" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 1, 4);
    defer swarm.deinit();

    try std.testing.expectEqual(@as(u32, 0), swarm.entity_counts[0]);

    const entity1 = EntityView{
        .position_mean = Vec3.init(0, 0, 0),
        .position_cov = .{ 1, 0, 0, 1, 0, 1 },
        .velocity_mean = Vec3.zero,
        .velocity_cov = .{ 1, 0, 0, 1, 0, 1 },
        .physics_params = PhysicsParams.prior,
        .physics_params_uncertainty = PhysicsParamsUncertainty.weak_prior,
        .contact_mode = .free,
        .track_state = .detected,
        .label = .{ .birth_time = 0, .birth_index = 0 },
        .occlusion_count = 0,
        .alive = true,
        .color = Vec3.init(0.5, 0.5, 0.5),
        .opacity = 1.0,
        .radius = 0.5,
        .goal_type = .none,
        .target_label = null,
        .target_position = null,
        .spatial_relation_type = .none,
        .spatial_reference = null,
        .spatial_distance = 0,
        .spatial_tolerance = 1.0,
    };

    const idx1 = swarm.addEntity(0, entity1);
    try std.testing.expectEqual(@as(?usize, 0), idx1);
    try std.testing.expectEqual(@as(u32, 1), swarm.entity_counts[0]);

    const idx2 = swarm.addEntity(0, entity1);
    try std.testing.expectEqual(@as(?usize, 1), idx2);
    try std.testing.expectEqual(@as(u32, 2), swarm.entity_counts[0]);

    swarm.killEntity(0, 0);
    try std.testing.expectEqual(@as(u32, 1), swarm.entity_counts[0]);
    try std.testing.expect(!swarm.isAlive(0, 0));
    try std.testing.expect(swarm.isAlive(0, 1));
}

test "ParticleSwarm ESS computation" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 4, 2);
    defer swarm.deinit();

    // Equal weights: ESS should equal num_particles
    swarm.log_weights[0] = 0;
    swarm.log_weights[1] = 0;
    swarm.log_weights[2] = 0;
    swarm.log_weights[3] = 0;

    const ess_equal = swarm.effectiveSampleSize();
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), ess_equal, 0.001);

    // One dominant particle: ESS should be close to 1
    swarm.log_weights[0] = 0;
    swarm.log_weights[1] = -100;
    swarm.log_weights[2] = -100;
    swarm.log_weights[3] = -100;

    const ess_degenerate = swarm.effectiveSampleSize();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), ess_degenerate, 0.01);
}

test "ParticleSwarm memory layout is contiguous" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 10, 8);
    defer swarm.deinit();

    // Verify arrays are the expected size
    try std.testing.expectEqual(@as(usize, 80), swarm.position_mean.len);
    try std.testing.expectEqual(@as(usize, 80), swarm.velocity_mean.len);
    try std.testing.expectEqual(@as(usize, 80), swarm.alive.len);

    // Verify per-particle arrays
    try std.testing.expectEqual(@as(usize, 10), swarm.camera_poses.len);
    try std.testing.expectEqual(@as(usize, 10), swarm.log_weights.len);
    try std.testing.expectEqual(@as(usize, 10), swarm.entity_counts.len);
}

test "covToTriangle and triangleToCov round trip" {
    // Create a symmetric covariance matrix
    const cov = Mat3.diagonal(Vec3.init(1.0, 2.0, 3.0));

    // Convert to triangle
    const tri = covToTriangle(cov);

    // Diagonal elements
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tri[0], 0.001); // (0,0)
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), tri[3], 0.001); // (1,1)
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), tri[5], 0.001); // (2,2)

    // Off-diagonal elements should be 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tri[1], 0.001); // (0,1)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tri[2], 0.001); // (0,2)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tri[4], 0.001); // (1,2)

    // Convert back to full matrix
    const recovered = triangleToCov(tri);

    // Verify diagonal
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), recovered.get(0, 0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), recovered.get(1, 1), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), recovered.get(2, 2), 0.001);

    // Verify symmetry
    try std.testing.expectApproxEqAbs(recovered.get(0, 1), recovered.get(1, 0), 0.001);
    try std.testing.expectApproxEqAbs(recovered.get(0, 2), recovered.get(2, 0), 0.001);
    try std.testing.expectApproxEqAbs(recovered.get(1, 2), recovered.get(2, 1), 0.001);
}

test "ParticleSwarm 6D state accessors" {
    const allocator = std.testing.allocator;

    var swarm = try ParticleSwarm.init(allocator, 2, 4);
    defer swarm.deinit();

    // Create a 6D state with cross-covariance
    var state = Gaussian6D.isotropic(
        Vec3.init(1, 2, 3), // position
        0.5, // pos variance
        Vec3.init(0.1, 0.2, 0.3), // velocity
        0.01, // vel variance
    );

    // Add some cross-covariance to test
    state.cov.set(0, 3, 0.05); // px-vx correlation

    // Set the state
    swarm.setState6D(swarm.idx(0, 0), state);

    // Get the state back
    const retrieved = swarm.getState6D(swarm.idx(0, 0));

    // Verify mean
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), retrieved.position().x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), retrieved.position().y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), retrieved.position().z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), retrieved.velocity().x, 1e-6);

    // Verify cross-covariance is preserved
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), retrieved.cov.get(0, 3), 1e-6);

    // Verify factored arrays are synced
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), swarm.position_mean[0].x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), swarm.velocity_mean[0].x, 1e-6);
}

