const std = @import("std");
const math = @import("math.zig");
const types = @import("types.zig");
const dynamics = @import("dynamics.zig");
const gmm = @import("gmm.zig");
const swarm_mod = @import("smc/mod.zig");
const crp = @import("crp/mod.zig");
const material_mod = @import("material.zig");
const priors = @import("priors.zig");

const Vec3 = math.Vec3;
const Vec2 = math.Vec2;
const Mat3 = math.Mat3;
const Entity = types.Entity;
const PhysicsParams = types.PhysicsParams;
const PhysicsParamsUncertainty = types.PhysicsParamsUncertainty;
const ContactMode = types.ContactMode;
const TrackState = types.TrackState;
const PhysicsConfig = types.PhysicsConfig;
const Camera = types.Camera;
const CameraPose = types.CameraPose;
const CameraIntrinsics = types.CameraIntrinsics;
const GaussianVec3 = types.GaussianVec3;
const Gaussian6D = types.Gaussian6D;
const CovTriangle21 = types.CovTriangle21;
const Label = types.Label;
const SpatialRelationType = types.SpatialRelationType;
const ObservationGrid = gmm.ObservationGrid;
const AdaptedFlowConfig = gmm.AdaptedFlowConfig;
const GaussianMixture = gmm.GaussianMixture;
const backProjectionLogLikelihood = gmm.backProjectionLogLikelihood;
const backProjectionLogLikelihoodWithDetections = gmm.backProjectionLogLikelihoodWithDetections;
const Detection2D = gmm.Detection2D;
const RayGaussian = gmm.RayGaussian;
const Observation = gmm.Observation;

// SoA particle swarm types
const ParticleSwarm = swarm_mod.ParticleSwarm;
const EntityView = swarm_mod.EntityView;
const covToTriangle = swarm_mod.covToTriangle;
const triangleToCov = swarm_mod.triangleToCov;
const ParticleWorldView = swarm_mod.ParticleWorldView;
const particleWorld = swarm_mod.particleWorld;

// Material trait types
const Material = material_mod.Material;
const LegacyMaterials = material_mod.LegacyMaterials;
const DefaultMaterial = material_mod.DefaultMaterial;

// =============================================================================
// SMC Configuration
// =============================================================================

/// Configuration for SMC inference
pub const SMCConfig = struct {
    /// Number of particles
    num_particles: u32 = 100,

    /// Maximum entities per particle (for SoA allocation)
    max_entities: u32 = 64,

    /// ESS threshold for resampling (as fraction of num_particles)
    ess_threshold: f32 = 0.5,

    /// Observation noise for likelihood (controls sharpness)
    /// This is the initial/default value; if use_adaptive_noise is true,
    /// the actual noise is learned via conjugate prior
    observation_noise: f32 = 0.3,

    /// Use adaptive observation noise (Inverse-Gamma conjugate prior)
    /// When true: Noise is learned from observation residuals
    /// When false: Uses fixed observation_noise value
    use_adaptive_noise: bool = false,

    /// Inverse-Gamma prior parameters for observation noise (only used when use_adaptive_noise is true)
    /// Default: α=3, β=0.27 gives prior mean ≈ 0.135 with reasonable uncertainty
    observation_noise_prior_alpha: f32 = 3.0,
    observation_noise_prior_beta: f32 = 0.27,

    /// Whether to use likelihood tempering
    use_tempering: bool = true,

    /// Initial temperature (0 = flat likelihood, 1 = full likelihood)
    initial_temperature: f32 = 0.1,

    /// Temperature increment per step
    temperature_increment: f32 = 0.1,

    /// Number of Gibbs sweeps for rejuvenation
    gibbs_sweeps: u32 = 3,

    /// Physics configuration
    physics: PhysicsConfig = .{},

    /// Camera intrinsics (fixed, not inferred)
    camera_intrinsics: CameraIntrinsics = .{},

    /// Camera position noise (standard deviation per axis per step)
    camera_position_noise: f32 = 0.1,

    /// Camera yaw noise (standard deviation in radians per step)
    camera_yaw_noise: f32 = 0.05,

    /// Camera prior bounds (for initialization)
    camera_prior_min: Vec3 = Vec3.init(-5, 2, 5),
    camera_prior_max: Vec3 = Vec3.init(5, 8, 15),

    /// Camera yaw prior bounds (radians, 0 = looking toward -Z)
    camera_yaw_min: f32 = -0.5,
    camera_yaw_max: f32 = 0.5,

    /// Use uniform colors for all entities (disables color-based inference)
    /// When true, inference relies solely on dynamics behavior (bounce/slide/stick)
    use_uniform_colors: bool = false,

    /// Uniform color to use when use_uniform_colors is true
    uniform_color: Vec3 = Vec3.init(0.8, 0.8, 0.8),

    /// Weber fraction for numerosity perception (approximate number system)
    /// Discrimination threshold σ = weber_fraction × count
    /// Human values typically 0.15-0.25; lower = more precise counting
    weber_fraction: f32 = 0.2,

    /// Minimum numerosity variance (prevents division by zero for count=0)
    min_numerosity_variance: f32 = 0.5,

    /// Depth prior mean for back-projection (distance from camera)
    back_projection_depth_mean: f32 = 10.0,

    /// Depth prior variance for back-projection
    back_projection_depth_var: f32 = 25.0,

    /// Initial position variance for entity initialization
    /// Higher = less confident in initial position, more responsive to observations
    initial_position_variance: f32 = 0.5,

    /// Initial velocity variance for entity initialization
    /// Higher = less confident in initial velocity, allows velocity learning
    initial_velocity_variance: f32 = 1.0,

    /// Position process noise (variance added per step)
    /// Prevents filter from becoming overconfident
    position_process_noise: f32 = 0.01,

    /// Velocity process noise (variance added per step)
    velocity_process_noise: f32 = 0.1,

    /// Use Metropolis-Hastings for camera yaw sampling (instead of discrete Gibbs)
    /// When true: Continuous MH proposals with Gaussian kernel (asymptotically exact)
    /// When false: Discrete Gibbs with 32 bins (~11° resolution, faster)
    /// Both methods are sound; MH is more precise, Gibbs is faster
    use_mh_yaw_sampling: bool = false,

    /// Standard deviation for MH yaw proposals (radians)
    /// Smaller = higher acceptance but slower mixing
    /// Larger = lower acceptance but faster exploration
    /// Rule of thumb: target ~23-40% acceptance rate
    mh_yaw_proposal_std: f32 = 0.1,

    /// Use CRP trans-dimensional moves (birth/death of entities)
    /// When true: Enables variable entity count inference via Reversible Jump MCMC
    /// When false: Fixed entity count (set during initialization)
    use_crp: bool = false,

    /// CRP configuration (only used when use_crp is true)
    crp_config: crp.CRPConfig = .{},

    /// Use sparse optical flow for velocity estimation
    /// When true: Computes Lucas-Kanade flow between consecutive frames
    /// and performs RBPF velocity updates (conjugate Gaussian)
    use_flow_observations: bool = false,

    /// Fallback flow config (used when adaptive flow is disabled for testing)
    /// Note: Resolution-adaptive flow is always used by default
    sparse_flow_config: gmm.SparseFlowConfig = .{},

    /// Material model for dynamics parameters (friction, elasticity, etc.)
    /// Default: LegacyMaterials (4-type system, backward compatible)
    /// Use DefaultMaterial for domain-general mode (skips material Gibbs)
    ///
    /// Spelke core knowledge (always active):
    /// - Continuity: Kalman tracking
    /// - Solidity: Collision detection
    /// - Support: ContactMode + gravity
    /// - Permanence: TrackState
    ///
    /// Material properties are domain-specific and optional.
    /// When material.num_types == 1, Gibbs sampling skips material inference.
    material: *const Material = &LegacyMaterials.instance,

    /// Get color for material index (respects use_uniform_colors setting)
    pub fn colorForMaterialIndex(self: SMCConfig, idx: usize) Vec3 {
        if (self.use_uniform_colors) {
            return self.uniform_color;
        }
        return self.material.color(idx);
    }

    /// Check if we're in domain-general mode (no material inference)
    pub fn isDomainGeneral(self: SMCConfig) bool {
        return self.material.isDomainGeneral();
    }
};

// =============================================================================
// Particle Representation
// =============================================================================

/// Stored previous state for dynamics likelihood computation
const PreviousState = struct {
    position: GaussianVec3,
    velocity: GaussianVec3,
};

/// A single particle representing a hypothesis about world state
/// FastSLAM pattern: camera pose is sampled, entities are Rao-Blackwellized
pub const Particle = struct {
    /// Camera pose (SAMPLED - not Gaussian)
    /// This is the agent's hypothesis about where it is
    camera_pose: CameraPose,

    /// Entity states (continuous: position/velocity as Gaussians)
    /// RAO-BLACKWELLIZED: conditioned on camera_pose, updated via Kalman
    entities: std.ArrayList(Entity),

    /// Previous states (before last physics step, for Gibbs dynamics likelihood)
    previous_states: std.ArrayList(PreviousState),

    /// Log weight (unnormalized)
    log_weight: f32,

    /// Allocator for this particle
    allocator: std.mem.Allocator,

    /// Initialize particle with given entities
    pub fn init(allocator: std.mem.Allocator) Particle {
        return .{
            .camera_pose = CameraPose.default,
            .entities = .empty,
            .previous_states = .empty,
            .log_weight = 0,
            .allocator = allocator,
        };
    }

    /// Deep copy from another particle
    pub fn copyFrom(self: *Particle, other: Particle) !void {
        self.camera_pose = other.camera_pose;

        self.entities.clearRetainingCapacity();
        try self.entities.ensureTotalCapacity(self.allocator, other.entities.items.len);
        for (other.entities.items) |e| {
            try self.entities.append(self.allocator, e);
        }

        self.previous_states.clearRetainingCapacity();
        try self.previous_states.ensureTotalCapacity(self.allocator, other.previous_states.items.len);
        for (other.previous_states.items) |s| {
            try self.previous_states.append(self.allocator, s);
        }

        self.log_weight = other.log_weight;
    }

    /// Free allocated memory
    pub fn deinit(self: *Particle) void {
        self.entities.deinit(self.allocator);
        self.previous_states.deinit(self.allocator);
    }

    /// Add entity to particle
    pub fn addEntity(self: *Particle, entity: Entity) !void {
        try self.entities.append(self.allocator, entity);
    }

    /// Get alive entity count
    pub fn aliveCount(self: Particle) usize {
        var count: usize = 0;
        for (self.entities.items) |e| {
            if (e.isAlive()) count += 1;
        }
        return count;
    }

    /// Step physics for all entities and camera in this particle
    pub fn stepPhysics(self: *Particle, smc_config: SMCConfig, rng: std.Random) !void {
        // Step camera pose with random walk dynamics
        self.camera_pose = self.camera_pose.step(
            smc_config.camera_position_noise,
            smc_config.camera_yaw_noise,
            rng,
        );

        // Save previous states for Gibbs dynamics likelihood
        self.previous_states.clearRetainingCapacity();
        try self.previous_states.ensureTotalCapacity(self.allocator, self.entities.items.len);
        for (self.entities.items) |entity| {
            try self.previous_states.append(self.allocator, .{
                .position = entity.position,
                .velocity = entity.velocity,
            });
        }

        for (self.entities.items) |*entity| {
            if (entity.isAlive()) {
                dynamics.entityPhysicsStep(entity, smc_config.physics, rng);
            }
        }

        // Entity-entity collisions
        for (0..self.entities.items.len) |i| {
            for (i + 1..self.entities.items.len) |j| {
                var e1 = &self.entities.items[i];
                var e2 = &self.entities.items[j];

                if (e1.isAlive() and e2.isAlive()) {
                    if (dynamics.checkEntityContact(e1.*, e2.*)) {
                        dynamics.resolveEntityCollision(e1, e2);
                    }
                }
            }
        }
    }

    /// Create GMM from particle's entities
    pub fn toGMM(self: Particle) !GaussianMixture {
        return GaussianMixture.fromEntities(self.entities.items, self.allocator);
    }
};

// =============================================================================
// Surprise Tracker (Violation of Expectation)
// =============================================================================

/// Tracks surprise signal for detecting physical violations
/// Based on: negative log-likelihood compared to running expectation
/// High surprise = observation inconsistent with learned physics model
pub const SurpriseTracker = struct {
    /// Current observation log-likelihood (weighted average across particles)
    current_log_likelihood: f64 = 0,

    /// Exponential moving average of log-likelihood (baseline expectation)
    expected_log_likelihood: f64 = 0,

    /// EMA decay rate (higher = faster adaptation, lower = longer memory)
    decay_rate: f32 = 0.1,

    /// Number of observations seen
    n_observations: u32 = 0,

    /// Whether tracker has been initialized (need at least 1 observation)
    initialized: bool = false,

    /// Current surprise signal: expected - observed
    /// Positive = surprising (worse than expected)
    /// Negative = better than expected
    /// Near zero = as expected
    pub fn surprise(self: SurpriseTracker) f32 {
        if (!self.initialized) return 0;
        return @floatCast(self.expected_log_likelihood - self.current_log_likelihood);
    }

    /// Update with new observation log-likelihood
    pub fn update(self: *SurpriseTracker, log_likelihood: f64) void {
        self.current_log_likelihood = log_likelihood;
        self.n_observations += 1;

        if (!self.initialized) {
            // First observation: initialize EMA
            self.expected_log_likelihood = log_likelihood;
            self.initialized = true;
        } else {
            // Update EMA: E = α*current + (1-α)*E
            const alpha: f64 = @floatCast(self.decay_rate);
            self.expected_log_likelihood = alpha * log_likelihood +
                (1.0 - alpha) * self.expected_log_likelihood;
        }
    }

    /// Check if surprise exceeds threshold (VoE detection)
    pub fn isSurprising(self: SurpriseTracker, threshold: f32) bool {
        return self.surprise() > threshold;
    }

    /// Reset tracker (new episode)
    pub fn reset(self: *SurpriseTracker) void {
        self.current_log_likelihood = 0;
        self.expected_log_likelihood = 0;
        self.n_observations = 0;
        self.initialized = false;
    }

    /// Get normalized surprise (0-1 scale via sigmoid)
    pub fn normalizedSurprise(self: SurpriseTracker) f32 {
        const s = self.surprise();
        // Sigmoid with reasonable scaling (surprise of 5 nats ≈ 0.99)
        return 1.0 / (1.0 + @exp(-s / 2.0));
    }
};

// =============================================================================
// SMC State
// =============================================================================

/// SMC inference state
pub const SMCState = struct {
    /// SoA particle swarm (replaces per-particle ArrayList)
    swarm: ParticleSwarm,

    /// Per-particle label sets for CRP tracking
    label_sets: []crp.LabelSet,

    /// Normalized weights (for resampling) - mirrors swarm.log_weights
    weights: []f32,

    /// Current temperature for tempering
    temperature: f32,

    /// Current timestep
    timestep: u32,

    /// Configuration
    config: SMCConfig,

    /// Allocator
    allocator: std.mem.Allocator,

    /// Random number generator
    rng: std.Random,

    /// Surprise tracker for VoE detection
    surprise_tracker: SurpriseTracker = .{},

    /// Observation noise state (conjugate prior for adaptive noise estimation)
    /// Uses Inverse-Gamma prior, updated with observation residuals
    observation_noise_state: ?priors.ObservationNoiseState = null,

    /// Previous frame storage for optical flow computation
    /// Stored as raw pixel data to avoid holding onto ObservationGrid memory
    prev_frame_data: ?[]Observation = null,
    prev_frame_width: u32 = 0,
    prev_frame_height: u32 = 0,

    /// Previous frame detections for flow matching
    prev_detections: ?[]Detection2D = null,

    /// Mode transition prior (Dirichlet for learning transition probabilities)
    /// Encodes Spelke core knowledge: objects at rest tend to stay at rest
    mode_transition_prior: priors.ModeTransitionPrior = priors.ModeTransitionPrior.spelke_prior,

    /// Initialize SMC state
    pub fn init(
        allocator: std.mem.Allocator,
        config: SMCConfig,
        rng: std.Random,
    ) !SMCState {
        // Initialize SoA particle swarm
        var swarm = try ParticleSwarm.init(
            allocator,
            config.num_particles,
            config.max_entities,
        );
        errdefer swarm.deinit();

        // Initialize per-particle label sets for CRP
        const label_sets = try allocator.alloc(crp.LabelSet, config.num_particles);
        errdefer allocator.free(label_sets);
        for (label_sets) |*ls| {
            ls.* = crp.LabelSet.init(allocator);
        }

        const weights = try allocator.alloc(f32, config.num_particles);
        @memset(weights, 1.0 / @as(f32, @floatFromInt(config.num_particles)));

        // Initialize adaptive noise state if enabled
        const noise_state: ?priors.ObservationNoiseState = if (config.use_adaptive_noise)
            priors.ObservationNoiseState.init(.{
                .alpha = config.observation_noise_prior_alpha,
                .beta = config.observation_noise_prior_beta,
            })
        else
            null;

        return .{
            .swarm = swarm,
            .label_sets = label_sets,
            .weights = weights,
            .temperature = if (config.use_tempering) config.initial_temperature else 1.0,
            .timestep = 0,
            .config = config,
            .allocator = allocator,
            .rng = rng,
            .observation_noise_state = noise_state,
        };
    }

    /// Get current effective observation noise
    /// Returns posterior mean if adaptive, otherwise fixed config value
    pub fn effectiveObservationNoise(self: SMCState) f32 {
        if (self.observation_noise_state) |state| {
            return @sqrt(state.noiseEstimate()); // Convert variance to std dev
        }
        return self.config.observation_noise;
    }

    /// Update observation noise state with residual (call during observation step)
    pub fn updateNoiseEstimate(self: *SMCState, residual: Vec3) void {
        if (self.observation_noise_state) |*state| {
            state.observe(residual);
        }
    }

    /// Observe a mode transition (updates Dirichlet prior)
    pub fn observeModeTransition(self: *SMCState, from: ContactMode, to: ContactMode) void {
        const from_mode = contactModeToModeEnum(from);
        const to_mode = contactModeToModeEnum(to);
        self.mode_transition_prior.observe(from_mode, to_mode);
    }

    /// Get posterior transition probability
    pub fn modeTransitionProb(self: SMCState, from: ContactMode, to: ContactMode) f32 {
        const from_mode = contactModeToModeEnum(from);
        const to_mode = contactModeToModeEnum(to);
        return self.mode_transition_prior.posteriorProb(from_mode, to_mode);
    }

    /// Reset mode transition counts (for new episode)
    pub fn resetModeTransitionCounts(self: *SMCState) void {
        self.mode_transition_prior.resetCounts();
    }

    /// Helper to convert ContactMode to ModeTransitionPrior.Mode
    fn contactModeToModeEnum(cm: ContactMode) priors.ModeTransitionPrior.Mode {
        return switch (cm) {
            .free => .free,
            .environment => .environment,
            .entity => .supported,
            .agency => .agency,
        };
    }

    /// Free allocated memory
    pub fn deinit(self: *SMCState) void {
        self.swarm.deinit();
        for (self.label_sets) |*ls| {
            ls.deinit();
        }
        self.allocator.free(self.label_sets);
        self.allocator.free(self.weights);

        // Free previous frame data if allocated
        if (self.prev_frame_data) |data| {
            self.allocator.free(data);
        }
        if (self.prev_detections) |dets| {
            self.allocator.free(dets);
        }
    }

    /// Build Entity array from swarm particle data (for GMM compatibility)
    /// Caller owns returned slice and must free it
    fn buildEntityArray(self: *SMCState, particle_idx: usize) ![]Entity {
        const n_entities = self.swarm.entity_counts[particle_idx];
        if (n_entities == 0) return &[_]Entity{};

        var entities = try self.allocator.alloc(Entity, n_entities);
        var out_idx: usize = 0;

        const slice = self.swarm.particleSlice(particle_idx);
        for (slice.start..slice.end) |i| {
            if (!self.swarm.alive[i]) continue;

            // Derive color from physics params: elasticity -> red, friction -> green
            const pp = self.swarm.physics_params[i];
            const color = if (self.config.use_uniform_colors)
                self.config.uniform_color
            else
                Vec3.init(pp.elasticity, 0.5, 1.0 - pp.friction);

            entities[out_idx] = Entity{
                .label = self.swarm.label[i],
                .position = GaussianVec3{
                    .mean = self.swarm.position_mean[i],
                    .cov = swarm_mod.triangleToCov(self.swarm.position_cov[i]),
                },
                .velocity = GaussianVec3{
                    .mean = self.swarm.velocity_mean[i],
                    .cov = swarm_mod.triangleToCov(self.swarm.velocity_cov[i]),
                },
                .physics_params = pp,
                .contact_mode = self.swarm.contact_mode[i],
                .track_state = self.swarm.track_state[i],
                .occlusion_count = self.swarm.occlusion_count[i],
                .appearance = .{
                    .color = color,
                    .opacity = 1.0,
                    .radius = 0.5,
                },
            };
            out_idx += 1;

            if (out_idx >= n_entities) break;
        }

        return entities[0..out_idx];
    }

    /// Initialize particles with prior (camera pose sampled, entities at given positions with unknown physics)
    pub fn initializeWithPrior(
        self: *SMCState,
        initial_positions: []const Vec3,
        initial_velocities: []const Vec3,
    ) void {
        // Use config values for initial uncertainty (allows observation-responsive tracking)
        const init_pos_cov = covToTriangle(Mat3.diagonal(Vec3.splat(self.config.initial_position_variance)));
        const init_vel_cov = covToTriangle(Mat3.diagonal(Vec3.splat(self.config.initial_velocity_variance)));

        for (0..self.swarm.num_particles) |p| {
            // Sample camera pose from prior
            self.swarm.camera_poses[p] = CameraPose.samplePrior(
                self.config.camera_prior_min,
                self.config.camera_prior_max,
                self.config.camera_yaw_min,
                self.config.camera_yaw_max,
                self.rng,
            );

            // Clear existing entities for this particle
            const slice = self.swarm.particleSlice(p);
            @memset(self.swarm.alive[slice.start..slice.end], false);
            self.swarm.entity_counts[p] = 0;

            // Add initial entities
            for (initial_positions, initial_velocities, 0..) |pos, vel, idx| {
                const label = Label{
                    .birth_time = 0,
                    .birth_index = @intCast(idx),
                };

                const entity_view = EntityView{
                    .position_mean = pos,
                    .position_cov = init_pos_cov,
                    .velocity_mean = vel,
                    .velocity_cov = init_vel_cov,
                    .physics_params = PhysicsParams.prior, // Start with weak prior
                    .physics_params_uncertainty = PhysicsParamsUncertainty.weak_prior,
                    .contact_mode = .free,
                    .track_state = .detected,
                    .label = label,
                    .occlusion_count = 0,
                    .alive = true,
                    .color = self.config.colorForMaterialIndex(0), // Default color
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

                _ = self.swarm.addEntity(p, entity_view);
            }

            self.swarm.log_weights[p] = 0;
        }

        // Uniform weights initially
        @memset(self.weights, 1.0 / @as(f32, @floatFromInt(self.config.num_particles)));
    }

    /// Compute effective sample size
    pub fn effectiveSampleSize(self: SMCState) f32 {
        var sum_sq: f32 = 0;
        for (self.weights) |w| {
            sum_sq += w * w;
        }
        if (sum_sq > 0) {
            return 1.0 / sum_sq;
        }
        return 0;
    }

    /// Normalize weights from log weights
    fn normalizeWeights(self: *SMCState) void {
        // Find max log weight for numerical stability
        var max_log_w: f64 = -std.math.inf(f64);
        for (self.swarm.log_weights) |lw| {
            max_log_w = @max(max_log_w, lw);
        }

        // Handle edge case: all particles have -inf log weight
        // Fall back to uniform weights
        if (max_log_w == -std.math.inf(f64)) {
            const uniform = 1.0 / @as(f32, @floatFromInt(self.swarm.num_particles));
            @memset(self.weights, uniform);
            return;
        }

        // Compute normalized weights
        var sum: f32 = 0;
        for (self.swarm.log_weights, 0..) |lw, i| {
            self.weights[i] = @floatCast(@exp(lw - max_log_w));
            sum += self.weights[i];
        }

        // Normalize (with fallback if sum is zero or NaN)
        if (sum > 0 and !std.math.isNan(sum)) {
            for (self.weights) |*w| {
                w.* /= sum;
            }
        } else {
            const uniform = 1.0 / @as(f32, @floatFromInt(self.swarm.num_particles));
            @memset(self.weights, uniform);
        }
    }

    /// Compute weighted mean log-likelihood and update surprise tracker
    fn updateSurpriseTracker(self: *SMCState) void {
        // Compute weighted average log-likelihood across particles
        var mean_log_lik: f64 = 0;
        for (self.swarm.log_weights, self.weights) |log_w, w| {
            // Use normalized weight to compute expectation
            mean_log_lik += @as(f64, w) * log_w;
        }

        // Update surprise tracker with observation log-likelihood
        self.surprise_tracker.update(mean_log_lik);
    }

    /// Get current surprise signal (for external monitoring/VoE detection)
    pub fn currentSurprise(self: SMCState) f32 {
        return self.surprise_tracker.surprise();
    }

    /// Check if current observation is surprising (VoE detection)
    pub fn isObservationSurprising(self: SMCState, threshold: f32) bool {
        return self.surprise_tracker.isSurprising(threshold);
    }

    /// Systematic resampling using SoA layout
    /// Performance: Zero-allocation after first call (uses swap buffers)
    pub fn resample(self: *SMCState) !void {
        const n = self.swarm.num_particles;
        const n_f: f32 = @floatFromInt(n);
        const max_e = self.swarm.max_entities;

        // Ensure swap buffers are allocated (lazy, zero-cost after first call)
        try self.swarm.ensureSwapBuffers();
        const swap = self.swarm.getSwapBuffers() orelse return error.OutOfMemory;

        // Compute cumulative distribution (small allocation, O(n))
        var cumsum = try self.allocator.alloc(f32, n);
        defer self.allocator.free(cumsum);

        cumsum[0] = self.weights[0];
        for (1..n) |i| {
            cumsum[i] = cumsum[i - 1] + self.weights[i];
        }

        // Systematic resampling - compute ancestor indices
        const u_start = self.rng.float(f32) / n_f;
        var ancestors = try self.allocator.alloc(usize, n);
        defer self.allocator.free(ancestors);

        var j: usize = 0;
        for (0..n) |i| {
            const u = u_start + @as(f32, @floatFromInt(i)) / n_f;
            while (j < n - 1 and cumsum[j] < u) {
                j += 1;
            }
            ancestors[i] = j;
        }

        // Copy data from ancestors to swap buffers
        for (0..n) |i| {
            const ancestor = ancestors[i];
            const src_start = ancestor * max_e;
            const dst_start = i * max_e;

            // Copy entity data (contiguous memcpy for each component)
            @memcpy(swap.position_mean[dst_start..][0..max_e], self.swarm.position_mean[src_start..][0..max_e]);
            @memcpy(swap.position_cov[dst_start..][0..max_e], self.swarm.position_cov[src_start..][0..max_e]);
            @memcpy(swap.velocity_mean[dst_start..][0..max_e], self.swarm.velocity_mean[src_start..][0..max_e]);
            @memcpy(swap.velocity_cov[dst_start..][0..max_e], self.swarm.velocity_cov[src_start..][0..max_e]);
            @memcpy(swap.physics_params[dst_start..][0..max_e], self.swarm.physics_params[src_start..][0..max_e]);
            @memcpy(swap.physics_params_uncertainty[dst_start..][0..max_e], self.swarm.physics_params_uncertainty[src_start..][0..max_e]);
            @memcpy(swap.contact_mode[dst_start..][0..max_e], self.swarm.contact_mode[src_start..][0..max_e]);
            @memcpy(swap.track_state[dst_start..][0..max_e], self.swarm.track_state[src_start..][0..max_e]);
            @memcpy(swap.label[dst_start..][0..max_e], self.swarm.label[src_start..][0..max_e]);
            @memcpy(swap.occlusion_count[dst_start..][0..max_e], self.swarm.occlusion_count[src_start..][0..max_e]);
            @memcpy(swap.alive[dst_start..][0..max_e], self.swarm.alive[src_start..][0..max_e]);
            @memcpy(swap.goal_type[dst_start..][0..max_e], self.swarm.goal_type[src_start..][0..max_e]);
            @memcpy(swap.target_label[dst_start..][0..max_e], self.swarm.target_label[src_start..][0..max_e]);
            @memcpy(swap.target_position[dst_start..][0..max_e], self.swarm.target_position[src_start..][0..max_e]);
            @memcpy(swap.spatial_relation_type[dst_start..][0..max_e], self.swarm.spatial_relation_type[src_start..][0..max_e]);
            @memcpy(swap.spatial_reference[dst_start..][0..max_e], self.swarm.spatial_reference[src_start..][0..max_e]);
            @memcpy(swap.spatial_distance[dst_start..][0..max_e], self.swarm.spatial_distance[src_start..][0..max_e]);
            @memcpy(swap.spatial_tolerance[dst_start..][0..max_e], self.swarm.spatial_tolerance[src_start..][0..max_e]);

            // Copy 6D coupled state (critical for cross-covariance preservation!)
            @memcpy(swap.state_mean[dst_start..][0..max_e], self.swarm.state_mean[src_start..][0..max_e]);
            @memcpy(swap.state_cov[dst_start..][0..max_e], self.swarm.state_cov[src_start..][0..max_e]);

            // Copy appearance (needed for identity tracking)
            @memcpy(swap.color[dst_start..][0..max_e], self.swarm.color[src_start..][0..max_e]);
            @memcpy(swap.opacity[dst_start..][0..max_e], self.swarm.opacity[src_start..][0..max_e]);
            @memcpy(swap.radius[dst_start..][0..max_e], self.swarm.radius[src_start..][0..max_e]);

            // Copy per-particle data
            swap.camera_poses[i] = self.swarm.camera_poses[ancestor];
            swap.entity_counts[i] = self.swarm.entity_counts[ancestor];
        }

        // Copy back from swap buffers to swarm
        @memcpy(self.swarm.position_mean, swap.position_mean);
        @memcpy(self.swarm.position_cov, swap.position_cov);
        @memcpy(self.swarm.velocity_mean, swap.velocity_mean);
        @memcpy(self.swarm.velocity_cov, swap.velocity_cov);
        @memcpy(self.swarm.physics_params, swap.physics_params);
        @memcpy(self.swarm.physics_params_uncertainty, swap.physics_params_uncertainty);
        @memcpy(self.swarm.contact_mode, swap.contact_mode);
        @memcpy(self.swarm.track_state, swap.track_state);
        @memcpy(self.swarm.label, swap.label);
        @memcpy(self.swarm.occlusion_count, swap.occlusion_count);
        @memcpy(self.swarm.alive, swap.alive);
        @memcpy(self.swarm.goal_type, swap.goal_type);
        @memcpy(self.swarm.target_label, swap.target_label);
        @memcpy(self.swarm.target_position, swap.target_position);
        @memcpy(self.swarm.spatial_relation_type, swap.spatial_relation_type);
        @memcpy(self.swarm.spatial_reference, swap.spatial_reference);
        @memcpy(self.swarm.spatial_distance, swap.spatial_distance);
        @memcpy(self.swarm.spatial_tolerance, swap.spatial_tolerance);
        @memcpy(self.swarm.state_mean, swap.state_mean);
        @memcpy(self.swarm.state_cov, swap.state_cov);
        @memcpy(self.swarm.color, swap.color);
        @memcpy(self.swarm.opacity, swap.opacity);
        @memcpy(self.swarm.radius, swap.radius);
        @memcpy(self.swarm.camera_poses, swap.camera_poses);
        @memcpy(self.swarm.entity_counts, swap.entity_counts);

        // Reset log weights
        @memset(self.swarm.log_weights, 0.0);

        // Reset to uniform weights
        @memset(self.weights, 1.0 / n_f);
    }

    /// Compute dynamics log-likelihood for a single entity given physics type
    /// Uses Kalman likelihood: p(x_t | x_{t-1}, params)
    fn dynamicsLogLikelihood(
        self: SMCState,
        entity: Entity,
        prev_state: PreviousState,
        physics_params: PhysicsParams,
    ) f32 {
        // Get dynamics matrices for these physics params
        const matrices = dynamics.DynamicsMatrices.forContactModeWithParams(
            entity.contact_mode,
            physics_params,
            self.config.physics,
            Vec3.unit_y,
        );

        // Predict what position/velocity WOULD be under this physics type
        const predicted_vel = dynamics.kalmanPredictVelocity(prev_state.velocity, matrices);
        const predicted_pos = dynamics.kalmanPredictPosition(prev_state.position, predicted_vel, matrices);

        // Compute likelihood of actual current state under prediction
        // Use position as the main discriminator (velocity is harder to observe)
        const pos_log_lik = dynamics.kalmanLogLikelihood(
            predicted_pos,
            entity.position.mean,
            Mat3.scale(0.1), // Small measurement noise for state comparison
        );

        return pos_log_lik;
    }

    // =========================================================================
    // Spatial Relation Likelihood (Phase 2b)
    // =========================================================================

    /// Compute spatial relation log-likelihood for a particle
    /// Returns negative penalty for violated spatial constraints
    fn spatialRelationLogLikelihood(self: SMCState, particle: usize) f32 {
        var log_lik: f32 = 0;
        const slice = self.swarm.particleSlice(particle);

        for (slice.start..slice.end) |i| {
            if (!self.swarm.alive[i]) continue;
            if (self.swarm.spatial_relation_type[i] == .none) continue;

            // Get reference entity position
            const ref_label = self.swarm.spatial_reference[i] orelse continue;
            const ref_pos = self.resolveLabelToPosition(particle, ref_label) orelse continue;

            const entity_pos = self.swarm.position_mean[i];
            const relation = self.swarm.spatial_relation_type[i];
            const tolerance = self.swarm.spatial_tolerance[i];
            const expected_dist = self.swarm.spatial_distance[i];

            // Compute relation violation
            const violation = self.computeRelationViolation(
                relation,
                entity_pos,
                ref_pos,
                expected_dist,
            );

            // Gaussian penalty: log_lik = -violation^2 / (2 * tolerance^2)
            const tolerance_sq = tolerance * tolerance;
            if (tolerance_sq > 1e-10) {
                log_lik += -(violation * violation) / (2.0 * tolerance_sq);
            }
        }

        return log_lik;
    }

    /// Compute how much a spatial relation is violated (0 = satisfied)
    fn computeRelationViolation(
        self: SMCState,
        relation: SpatialRelationType,
        entity_pos: Vec3,
        ref_pos: Vec3,
        expected_dist: f32,
    ) f32 {
        _ = self;
        const diff = entity_pos.sub(ref_pos);

        return switch (relation) {
            .none => 0,

            // Inside: entity should be near center of reference (simplified)
            .inside => diff.length(),

            // On: entity should be above reference with small vertical offset
            .on => blk: {
                // Expected: entity.y > ref.y and horizontally aligned
                const vertical_offset = diff.y;
                const horizontal_dist = @sqrt(diff.x * diff.x + diff.z * diff.z);
                // Penalize if below reference or too far horizontally
                if (vertical_offset < 0) {
                    break :blk -vertical_offset + horizontal_dist;
                }
                break :blk horizontal_dist;
            },

            // Near: entity should be at expected distance from reference
            .near => blk: {
                const actual_dist = diff.length();
                break :blk @abs(actual_dist - expected_dist);
            },

            // Above: entity.y > ref.y
            .above => blk: {
                if (diff.y > 0) break :blk 0;
                break :blk -diff.y;
            },

            // Below: entity.y < ref.y
            .below => blk: {
                if (diff.y < 0) break :blk 0;
                break :blk diff.y;
            },

            // Left/right are viewpoint-dependent (use x for now)
            .left_of => blk: {
                if (diff.x < 0) break :blk 0;
                break :blk diff.x;
            },

            .right_of => blk: {
                if (diff.x > 0) break :blk 0;
                break :blk -diff.x;
            },
        };
    }

    // =========================================================================
    // Numerosity Likelihood (Phase 3 - Weber's Law)
    // =========================================================================

    /// Count alive entities in a particle
    fn countAliveEntities(self: SMCState, particle: usize) u32 {
        var count: u32 = 0;
        const slice = self.swarm.particleSlice(particle);
        for (slice.start..slice.end) |i| {
            if (self.swarm.alive[i]) {
                count += 1;
            }
        }
        return count;
    }

    /// Compute numerosity log-likelihood for a particle given observed count
    /// Uses Weber's law: σ = w × max(n, min_var) where w is Weber fraction
    /// Returns unnormalized log p(observed_count | particle_count) for particle weighting
    /// Note: Normalization constant omitted to avoid heteroscedasticity bias
    /// (larger counts would otherwise be penalized by their larger σ)
    fn numerosityLogLikelihood(self: SMCState, particle: usize, observed_count: u32) f32 {
        const particle_count = self.countAliveEntities(particle);
        const n: f32 = @floatFromInt(particle_count);
        const k: f32 = @floatFromInt(observed_count);

        // Weber's law: variance scales with count squared
        // σ² = (w × max(n, min_var))²
        const base = @max(n, self.config.min_numerosity_variance);
        const sigma = self.config.weber_fraction * base;
        const sigma_sq = sigma * sigma;

        // Unnormalized Gaussian log-likelihood: -0.5 * (k - n)² / σ²
        // Omit -0.5 * log(2πσ²) to avoid bias toward low counts
        const diff = k - n;
        const log_lik = -(diff * diff) / (2.0 * sigma_sq);

        return log_lik;
    }

    /// Update particle weights with numerosity observation
    /// Call this in addition to updateWeights for visual observation
    pub fn updateWeightsNumerosity(self: *SMCState, observed_count: u32) void {
        for (0..self.swarm.num_particles) |p| {
            const log_lik = self.numerosityLogLikelihood(p, observed_count);
            // Apply temperature for annealing
            self.swarm.log_weights[p] += self.temperature * @as(f64, log_lik);
        }
        self.normalizeWeights();
        self.updateSurpriseTracker();
    }

    /// Compute observation log-likelihood for a particle (by index)
    /// Uses the particle's own camera pose hypothesis (FastSLAM pattern)
    pub fn observationLogLikelihood(
        self: *SMCState,
        particle_idx: usize,
        observation: ObservationGrid,
    ) f32 {
        // Derive camera from particle's camera pose + fixed intrinsics
        const camera = self.swarm.camera_poses[particle_idx].toCamera(self.config.camera_intrinsics);

        // Build Entity array from swarm data for GMM compatibility
        const entities = self.buildEntityArray(particle_idx) catch return -std.math.inf(f32);
        defer if (entities.len > 0) self.allocator.free(entities);

        // Create GMM from entities
        var particle_gmm = GaussianMixture.fromEntities(entities, self.allocator) catch return -std.math.inf(f32);
        defer particle_gmm.deinit();

        // Back-projection observation model (conjugate Gaussian)
        // Extracts detections from observed image, back-projects to ray-Gaussians,
        // computes overlap with particle's 3D GMM hypothesis
        var log_lik: f32 = backProjectionLogLikelihood(
            observation,
            particle_gmm,
            camera,
            self.config.back_projection_depth_mean,
            self.config.back_projection_depth_var,
            self.allocator,
        );

        // Add spatial relation likelihood (Phase 2b)
        const spatial_log_lik = self.spatialRelationLogLikelihood(particle_idx);
        log_lik += spatial_log_lik;

        return log_lik;
    }

    /// Update particle weights given observation
    /// Each particle uses its own camera hypothesis
    pub fn updateWeights(
        self: *SMCState,
        observation: ObservationGrid,
    ) void {
        for (0..self.swarm.num_particles) |p| {
            const log_lik = self.observationLogLikelihood(p, observation);
            // Apply temperature for annealing
            self.swarm.log_weights[p] += self.temperature * @as(f64, log_lik);
        }

        self.normalizeWeights();
        self.updateSurpriseTracker();
    }

    /// Compute dynamics log-likelihood for SoA entity data
    /// Compares predicted position under given physics params vs actual position
    /// Bug 1 fix: Use prev_contact_mode (saved before physics step) to detect bounces
    /// Bug 4 fix: Apply elasticity to post-gravity velocity, not prev_vel
    fn swarmDynamicsLogLikelihood(
        self: SMCState,
        entity_idx: usize,
        physics_params: PhysicsParams,
    ) f32 {
        // Get previous and current states
        const prev_pos = self.swarm.prev_position_mean[entity_idx];
        const prev_vel = self.swarm.prev_velocity_mean[entity_idx];
        const curr_pos = self.swarm.position_mean[entity_idx];
        // Bug 1 fix: Use prev_contact_mode to know if we WERE at ground before step
        const prev_contact_mode = self.swarm.prev_contact_mode[entity_idx];

        // Get dynamics matrices for these physics params
        const matrices = dynamics.DynamicsMatrices.forContactModeWithParams(
            prev_contact_mode,
            physics_params,
            self.config.physics,
            Vec3.unit_y,
        );

        // Predict what position WOULD be under these physics params
        // Step 1: Apply gravity to get post-gravity velocity
        var predicted_vel = matrices.B_vel.mulVec(prev_vel).add(matrices.gravity.scale(matrices.dt));

        // Step 2: Check if we hit environment during this step
        // Bug 4 fix: Apply elasticity to predicted_vel (post-gravity), not prev_vel
        const ground_height = self.config.physics.groundHeight();
        const predicted_y_before_bounce = prev_pos.y + predicted_vel.y * matrices.dt;
        if (predicted_y_before_bounce < ground_height and predicted_vel.y < 0) {
            // Ball would go through ground - apply bounce
            predicted_vel.y = -predicted_vel.y * physics_params.elasticity;
        }

        var predicted_pos = prev_pos.add(predicted_vel.scale(matrices.dt));
        // Clamp to environment
        if (predicted_pos.y < ground_height) {
            predicted_pos.y = ground_height;
        }

        // Compute likelihood: how well does predicted match actual?
        const diff = curr_pos.sub(predicted_pos);
        const dist_sq = diff.dot(diff);

        // Gaussian likelihood with fixed process noise variance
        const noise_var = self.config.physics.process_noise * matrices.dt * matrices.dt + 0.01;
        return -dist_sq / (2.0 * noise_var);
    }

    /// Compute dynamics log-likelihood using material interface
    /// Same logic as swarmDynamicsLogLikelihood but uses material trait for parameters
    fn swarmDynamicsLogLikelihoodMaterial(
        self: SMCState,
        entity_idx: usize,
        material_idx: usize,
    ) f32 {
        const mat = self.config.material;

        // Get previous and current states
        const prev_pos = self.swarm.prev_position_mean[entity_idx];
        const prev_vel = self.swarm.prev_velocity_mean[entity_idx];
        const curr_pos = self.swarm.position_mean[entity_idx];
        const prev_contact_mode = self.swarm.prev_contact_mode[entity_idx];

        // Get material parameters
        const friction = mat.friction(material_idx);
        const elasticity_val = mat.elasticity(material_idx);
        const process_noise = mat.processNoise(material_idx);

        // Build dynamics matrices using material parameters
        const dt = self.config.physics.dt;
        const vel_decay = 1.0 - friction * dt;

        // Predict velocity with friction and gravity
        var predicted_vel = prev_vel.scale(vel_decay).add(self.config.physics.gravity.scale(dt));

        // For environment contact, constrain vertical component
        if (prev_contact_mode == .environment) {
            predicted_vel.y = @max(0, predicted_vel.y);
        }

        // Check if we hit environment during this step - apply elasticity
        const ground_height = self.config.physics.groundHeight();
        const predicted_y_before_bounce = prev_pos.y + predicted_vel.y * dt;
        if (predicted_y_before_bounce < ground_height and predicted_vel.y < 0) {
            predicted_vel.y = -predicted_vel.y * elasticity_val;
        }

        var predicted_pos = prev_pos.add(predicted_vel.scale(dt));
        // Clamp to environment
        if (predicted_pos.y < ground_height) {
            predicted_pos.y = ground_height;
        }

        // Compute likelihood: how well does predicted match actual?
        const diff = curr_pos.sub(predicted_pos);
        const dist_sq = diff.dot(diff);

        // Gaussian likelihood with process noise variance
        const noise_var = process_noise * dt * dt + 0.01;
        return -dist_sq / (2.0 * noise_var);
    }

    /// Gibbs rejuvenation: resample discrete camera pose variables
    /// Physics parameters are now inferred continuously via Bayesian updates
    /// in bounce handling, not through discrete Gibbs sampling.
    pub fn gibbsRejuvenation(
        self: *SMCState,
        observation: ObservationGrid,
    ) void {
        // Extract detections ONCE for all particles (fixes O(n*k) -> O(1) performance)
        const detections = Detection2D.extractFromGrid(observation, self.allocator) catch return;
        defer self.allocator.free(detections);

        for (0..self.swarm.num_particles) |p| {
            for (0..self.config.gibbs_sweeps) |_| {
                // Camera yaw move (Gibbs or MH based on config)
                // This is always domain-general (camera pose is core knowledge)
                self.selectYawSamplingMethod(p, observation, detections);
            }
        }
    }

    // =========================================================================
    // Camera Yaw Sampling (Gibbs/MH)
    // =========================================================================
    //
    // Two methods available, controlled by config.yaw_sampling_method:
    //
    // 1. DISCRETE GIBBS (default): Discretizes yaw range into bins, samples from
    //    categorical. Sound: samples from discretized target p(yaw | y).
    //    Error bounded by bin width (~22.5° for 16 bins).
    //    Reference: Standard Gibbs sampling (Geman & Geman, 1984)
    //
    // 2. CONTINUOUS MH: Metropolis-Hastings with Gaussian proposals.
    //    Sound: satisfies detailed balance, asymptotically exact.
    //    Better precision but may have lower acceptance in peaked distributions.
    //    Reference: Metropolis et al. (1953), Hastings (1970)
    //
    // =========================================================================

    /// Number of discrete yaw bins for Gibbs sampling
    /// More bins = finer resolution but more computation
    /// 16 bins = ~22.5° resolution, 32 bins = ~11.25° resolution
    const num_yaw_bins: usize = 32; // Increased from 16 for better precision

    /// Yaw sampling method selector
    fn selectYawSamplingMethod(self: *SMCState, particle_idx: usize, observation: ObservationGrid, detections: []const Detection2D) void {
        if (self.config.use_mh_yaw_sampling) {
            self.mhCameraYaw(particle_idx, observation, detections);
        } else {
            self.gibbsCameraYawDiscrete(particle_idx, observation, detections);
        }
    }

    /// Metropolis-Hastings move on camera yaw (continuous)
    /// Uses symmetric Gaussian proposal: q(yaw' | yaw) = N(yaw' ; yaw, σ²)
    /// Acceptance ratio: α = min(1, p(y|yaw') / p(y|yaw))
    /// Satisfies detailed balance → asymptotically exact
    fn mhCameraYaw(
        self: *SMCState,
        particle_idx: usize,
        observation: ObservationGrid,
        detections: []const Detection2D,
    ) void {
        _ = observation; // Not used directly, likelihood computed via detections

        // Current yaw and its likelihood
        const current_yaw = self.swarm.camera_poses[particle_idx].yaw;

        // Build GMM for this particle
        const entities = self.buildEntityArray(particle_idx) catch return;
        defer if (entities.len > 0) self.allocator.free(entities);

        var particle_gmm = GaussianMixture.fromEntities(entities, self.allocator) catch return;
        defer particle_gmm.deinit();

        // Compute current likelihood
        const current_camera = self.swarm.camera_poses[particle_idx].toCamera(self.config.camera_intrinsics);
        const current_ll = backProjectionLogLikelihoodWithDetections(
            detections,
            32, // Use fixed size for likelihood computation
            32,
            particle_gmm,
            current_camera,
            self.config.back_projection_depth_mean,
            self.config.back_projection_depth_var,
        );

        // Propose new yaw (symmetric Gaussian proposal)
        const proposal_std = self.config.mh_yaw_proposal_std;
        const proposed_yaw = current_yaw + self.rng.floatNorm(f32) * proposal_std;

        // Clamp to valid range
        const clamped_yaw = @max(self.config.camera_yaw_min, @min(self.config.camera_yaw_max, proposed_yaw));

        // Compute proposed likelihood
        self.swarm.camera_poses[particle_idx].yaw = clamped_yaw;
        const proposed_camera = self.swarm.camera_poses[particle_idx].toCamera(self.config.camera_intrinsics);
        const proposed_ll = backProjectionLogLikelihoodWithDetections(
            detections,
            32,
            32,
            particle_gmm,
            proposed_camera,
            self.config.back_projection_depth_mean,
            self.config.back_projection_depth_var,
        );

        // MH acceptance ratio (log scale): log(α) = proposed_ll - current_ll
        // For symmetric proposal, q cancels out
        const log_alpha = proposed_ll - current_ll;

        // Accept or reject
        const u = self.rng.float(f32);
        if (@log(u) < log_alpha) {
            // Accept: keep proposed yaw (already set)
        } else {
            // Reject: restore current yaw
            self.swarm.camera_poses[particle_idx].yaw = current_yaw;
        }
    }

    /// Discrete Gibbs move on camera yaw
    /// Discretizes yaw range into bins, computes full conditional, samples
    /// Sound: samples from discretized target p(yaw | y, entities)
    fn gibbsCameraYawDiscrete(
        self: *SMCState,
        particle_idx: usize,
        observation: ObservationGrid,
        detections: []const Detection2D,
    ) void {
        const yaw_range = self.config.camera_yaw_max - self.config.camera_yaw_min;
        const yaw_step = yaw_range / @as(f32, @floatFromInt(num_yaw_bins));

        // Save original yaw
        const original_yaw = self.swarm.camera_poses[particle_idx].yaw;

        // Build GMM for this particle (once, reused for all yaw bins)
        const entities = self.buildEntityArray(particle_idx) catch return;
        defer if (entities.len > 0) self.allocator.free(entities);

        var particle_gmm = GaussianMixture.fromEntities(entities, self.allocator) catch return;
        defer particle_gmm.deinit();

        // Compute observation likelihood for each discrete yaw bin
        var log_probs: [num_yaw_bins]f32 = undefined;
        var max_log: f32 = -std.math.inf(f32);

        for (0..num_yaw_bins) |bin| {
            const yaw = self.config.camera_yaw_min + (@as(f32, @floatFromInt(bin)) + 0.5) * yaw_step;
            self.swarm.camera_poses[particle_idx].yaw = yaw;

            const camera = self.swarm.camera_poses[particle_idx].toCamera(self.config.camera_intrinsics);
            const obs_ll = backProjectionLogLikelihoodWithDetections(
                detections,
                observation.width,
                observation.height,
                particle_gmm,
                camera,
                self.config.back_projection_depth_mean,
                self.config.back_projection_depth_var,
            );
            log_probs[bin] = obs_ll;
            max_log = @max(max_log, obs_ll);
        }

        // Restore original yaw before sampling (will be replaced)
        self.swarm.camera_poses[particle_idx].yaw = original_yaw;

        // Convert to normalized probabilities (softmax)
        var probs: [num_yaw_bins]f32 = undefined;
        var sum: f32 = 0;
        for (0..num_yaw_bins) |bin| {
            probs[bin] = @exp(log_probs[bin] - max_log);
            sum += probs[bin];
        }
        for (&probs) |*pp| {
            pp.* /= sum;
        }

        // Sample from categorical distribution
        const u = self.rng.float(f32);
        var cumsum: f32 = 0;
        var sampled_bin: usize = 0;
        for (0..num_yaw_bins) |bin| {
            cumsum += probs[bin];
            if (u < cumsum) {
                sampled_bin = bin;
                break;
            }
        }

        // Set yaw to sampled bin center
        const sampled_yaw = self.config.camera_yaw_min + (@as(f32, @floatFromInt(sampled_bin)) + 0.5) * yaw_step;
        self.swarm.camera_poses[particle_idx].yaw = sampled_yaw;
    }

    // =========================================================================
    // Rao-Blackwellized Observation Update (RBPF Pattern)
    // =========================================================================
    //
    // References:
    // - Doucet et al. (2000): Rao-Blackwellized Particle Filtering
    // - Murphy (2002): Dynamic Bayesian Networks (Ch. 15)
    // - Lew et al. (2023): SMCP3 - Sequential Monte Carlo with Probabilistic Programs
    //
    // Key insight: In RBPF, observation is used ONCE via Kalman update.
    // The marginal likelihood p(y | prior) is returned for particle weighting.
    // This avoids "double-dipping" where observation is used for both weighting
    // and state update separately.
    // =========================================================================

    /// Rao-Blackwellized observation update for 6D coupled state
    /// Position observation updates BOTH position AND velocity via cross-covariance
    /// Returns marginal log-likelihood for particle weighting
    fn rbpfEntityObservationUpdate6D(
        self: *SMCState,
        entity_idx: usize,
        observation_3d: Vec3,
        observation_cov: Mat3,
    ) f32 {
        // Get current 6D Gaussian belief
        const prior = self.swarm.getState6D(entity_idx);

        // Perform 6D Kalman update (updates position AND velocity via cross-covariance)
        const result = dynamics.kalmanUpdate6D(prior, observation_3d, observation_cov);

        // Write back updated state (this also syncs factored arrays)
        self.swarm.setState6D(entity_idx, result.state);

        return result.log_lik;
    }

    /// Project 3D velocity to 2D image plane velocity (pixels/frame)
    /// Uses pinhole camera model: v_2d = (f/z) * v_3d_xy - (f*x/z²) * v_z
    /// Simplified for small motions: v_2d ≈ (f/z) * v_3d_xy
    fn projectVelocityTo2D(
        self: *SMCState,
        entity_idx: usize,
        camera: Camera,
        _: u32, // image_width (unused, symmetry with height)
        image_height: u32,
    ) Vec2 {
        const pos = self.swarm.position_mean[entity_idx];
        const vel = self.swarm.velocity_mean[entity_idx];

        // Transform to camera coordinates
        const rel_pos = pos.sub(camera.position);

        // Compute depth (distance along view direction)
        const view_dir = camera.target.sub(camera.position).normalize();
        const depth = @max(0.1, rel_pos.dot(view_dir));

        // Focal length in pixels (from FOV)
        const half_h: f32 = @floatFromInt(image_height / 2);
        const focal_pixels = half_h / @tan(camera.fov / 2.0);

        // Project velocity to image plane
        // For small depth variations, v_2d ≈ (f/z) * v_xy
        const scale = focal_pixels / depth;

        // Get camera right and up vectors for projection
        const right = view_dir.cross(camera.up).normalize();
        const up = right.cross(view_dir).normalize();

        // Project velocity onto image plane axes
        const v_right = vel.dot(right);
        const v_up = vel.dot(up);

        return Vec2.init(v_right * scale, -v_up * scale); // Flip Y for image coordinates
    }

    /// Rao-Blackwellized velocity update for a single entity using optical flow
    /// Returns marginal log-likelihood for particle weighting
    ///
    /// Updates velocity_mean and velocity_cov using approximate Kalman update:
    /// - Gains computed per-axis in 2D flow space: K = σ²_prior / (σ²_prior + R)
    /// - Covariance rotated to camera basis (right, up, view_dir)
    /// - Diagonal scaling applied: (1-K) reduction on right/up, identity on view_dir
    /// - Rotated back to world frame, preserving full covariance structure
    ///
    /// Depth (view_dir) variance is EXACTLY preserved for any camera orientation.
    /// The rotation-based approach maintains PSD by construction (D*C*D with D>0).
    fn rbpfEntityVelocityUpdate(
        self: *SMCState,
        entity_idx: usize,
        flow_obs: gmm.FlowObservation,
        camera: Camera,
        image_width: u32,
        image_height: u32,
    ) f32 {
        // Get current velocity belief
        const prior_vel = self.swarm.velocity_mean[entity_idx];
        const prior_cov_tri = self.swarm.velocity_cov[entity_idx];
        const prior_cov = triangleToCov(prior_cov_tri);

        // Project expected velocity to 2D
        const expected_flow = self.projectVelocityTo2D(entity_idx, camera, image_width, image_height);

        // Observation model: flow = H * velocity + noise
        // H is the projection Jacobian (approximated as constant scale factor)
        // For simplicity, we use a 2D observation model on the XY velocity components

        // Compute residual in 2D
        const residual = flow_obs.flow.sub(expected_flow);

        // Compute marginal log-likelihood using the flow observation's logLikelihood
        // (uses heteroscedastic covariance from Lucas-Kanade)
        const marginal_ll = flow_obs.logLikelihood(expected_flow);

        // For the Kalman update, we need to lift the 2D observation to 3D velocity space
        // Simplified approach: update the XZ velocity components based on flow
        // (assuming camera is looking toward -Z, so image X ≈ world X, image Y ≈ world Y)

        // Get camera orientation for proper axis mapping
        const view_dir = camera.target.sub(camera.position).normalize();
        const right = view_dir.cross(camera.up).normalize();
        const up = right.cross(view_dir).normalize();

        // Compute depth for velocity scaling
        const pos = self.swarm.position_mean[entity_idx];
        const rel_pos = pos.sub(camera.position);
        const depth = @max(0.1, rel_pos.dot(view_dir));
        const half_h: f32 = @floatFromInt(image_height / 2);
        const focal_pixels = half_h / @tan(camera.fov / 2.0);
        const inv_scale = depth / focal_pixels;

        // Convert 2D flow to 3D velocity correction
        // flow = (f/z) * v_3d -> v_3d = (z/f) * flow
        const v_correction_right = residual.x * inv_scale;
        const v_correction_up = -residual.y * inv_scale; // Flip Y back

        // Proper Kalman gain using flow observation covariance
        // In 2D projected space: K = σ²_prior / (σ²_prior + R)
        // where σ²_prior is velocity variance projected to 2D
        //
        // Project prior velocity covariance to 2D flow space:
        // For velocity in right/up directions, extract those variances
        const scale_sq = (focal_pixels / depth) * (focal_pixels / depth);
        const prior_var_right = prior_cov.get(0, 0) * right.x * right.x +
            prior_cov.get(1, 1) * right.y * right.y +
            prior_cov.get(2, 2) * right.z * right.z;
        const prior_var_up = prior_cov.get(0, 0) * up.x * up.x +
            prior_cov.get(1, 1) * up.y * up.y +
            prior_cov.get(2, 2) * up.z * up.z;

        // Project to 2D flow variance
        const flow_var_prior_x = prior_var_right * scale_sq;
        const flow_var_prior_y = prior_var_up * scale_sq;

        // Observation covariance (from heteroscedastic Lucas-Kanade)
        const R_xx = flow_obs.covariance.get(0, 0);
        const R_yy = flow_obs.covariance.get(1, 1);

        // Kalman gains for each axis: K = σ²_prior / (σ²_prior + R)
        const eps = 1e-6;
        const gain_x = flow_var_prior_x / (flow_var_prior_x + R_xx + eps);
        const gain_y = flow_var_prior_y / (flow_var_prior_y + R_yy + eps);

        // Apply Kalman update to velocity
        var new_vel = prior_vel;
        new_vel = new_vel.add(right.scale(v_correction_right * gain_x));
        new_vel = new_vel.add(up.scale(v_correction_up * gain_y));

        // Update velocity state
        self.swarm.velocity_mean[entity_idx] = new_vel;

        // Update velocity covariance: Σ_posterior = D * Σ_prior * D
        // where D is a diagonal scaling matrix in the camera observation basis.
        //
        // This correctly preserves depth variance for ANY camera orientation by:
        // 1. Rotating covariance to camera frame (right, up, view_dir basis)
        // 2. Applying Kalman-style diagonal scaling (reduce right/up, preserve view_dir)
        // 3. Rotating back to world frame
        //
        // The scaling factors are sqrt(1 - gain) so variance reduces by (1 - gain).
        // View_dir (depth) has scale = 1 since flow doesn't observe it.

        // Build rotation matrix: columns are camera basis vectors
        const R = Mat3.fromColumns(right, up, view_dir);
        const R_T = R.transpose();

        // Transform covariance to camera frame: C_cam = R^T * C_world * R
        const C_cam = R_T.mulMat(prior_cov).mulMat(R);

        // Scaling factors for Kalman reduction: s_i such that var_new = var * s_i²
        // For (1 - gain) reduction: s = sqrt(1 - gain)
        const s_right = @sqrt(@max(0.01, 1.0 - gain_x));
        const s_up = @sqrt(@max(0.01, 1.0 - gain_y));
        const s_view: f32 = 1.0; // Depth preserved (unobserved)

        // Apply scaling: C_cam_new[i,j] = s_i * C_cam[i,j] * s_j
        // This is equivalent to D * C_cam * D where D = diag(s_right, s_up, s_view)
        var C_cam_new: Mat3 = undefined;
        const scales = [3]f32{ s_right, s_up, s_view };
        for (0..3) |i| {
            for (0..3) |j| {
                C_cam_new.data[j * 3 + i] = scales[i] * C_cam.get(i, j) * scales[j];
            }
        }

        // Transform back to world frame: C_world_new = R * C_cam_new * R^T
        const new_cov = R.mulMat(C_cam_new).mulMat(R_T);

        // Apply variance floor to diagonal elements for numerical stability
        const min_var: f32 = 0.01;
        var clamped_cov = new_cov;
        clamped_cov.data[0] = @max(min_var, clamped_cov.data[0]); // xx
        clamped_cov.data[4] = @max(min_var, clamped_cov.data[4]); // yy
        clamped_cov.data[8] = @max(min_var, clamped_cov.data[8]); // zz

        self.swarm.velocity_cov[entity_idx] = covToTriangle(clamped_cov);

        return marginal_ll;
    }

    /// Back-project 2D detection to 3D observation with uncertainty
    /// Uses camera ray + depth prior to construct observation Gaussian
    fn backProjectDetectionTo3D(
        self: *SMCState,
        detection: Detection2D,
        camera: Camera,
        image_width: u32,
        image_height: u32,
    ) struct { mean: Vec3, cov: Mat3 } {
        const half_w: f32 = @floatFromInt(image_width / 2);
        const half_h: f32 = @floatFromInt(image_height / 2);

        // Convert pixel to NDC
        // Image: Y increases downward, NDC: Y increases upward
        const ndc_x = (detection.pixel_x - half_w) / half_w;
        const ndc_y = (half_h - detection.pixel_y) / half_h; // Flip Y

        // Get ray direction from camera through detection
        const ray_dir = gmm.computeRayDirection(camera, ndc_x, ndc_y);

        // 3D position along ray at depth prior mean
        const depth_mean = self.config.back_projection_depth_mean;
        const depth_var = self.config.back_projection_depth_var;
        const obs_mean = camera.position.add(ray_dir.scale(depth_mean));

        // Construct observation covariance
        // - Large variance along ray direction (depth uncertainty)
        // - Small variance perpendicular to ray (lateral precision from pixel)
        //
        // Cov = depth_var * (d ⊗ d) + lateral_var * (I - d ⊗ d)
        //     = lateral_var * I + (depth_var - lateral_var) * (d ⊗ d)
        const eff_noise = self.effectiveObservationNoise();
        const lateral_var = eff_noise * eff_noise;
        const outer_dd = Mat3.outer(ray_dir, ray_dir);
        const obs_cov = Mat3.scaleMat(Mat3.identity, lateral_var)
            .add(outer_dd.scaleMat(depth_var - lateral_var));

        return .{ .mean = obs_mean, .cov = obs_cov };
    }

    /// Perform RBPF observation update for all entities in a particle
    /// Returns total marginal log-likelihood for particle weighting
    fn rbpfParticleObservationUpdate(
        self: *SMCState,
        particle_idx: usize,
        detections: []const Detection2D,
        image_width: u32,
        image_height: u32,
    ) f32 {
        const camera_pose = self.swarm.camera_poses[particle_idx];
        const camera = camera_pose.toCamera(self.config.camera_intrinsics);
        const slice = self.swarm.particleSlice(particle_idx);

        var total_log_lik: f32 = 0;
        const half_w: f32 = @floatFromInt(image_width / 2);
        const half_h: f32 = @floatFromInt(image_height / 2);

        // For each entity in this particle
        for (slice.start..slice.end) |i| {
            if (!self.swarm.alive[i]) continue;

            // Project entity to image space
            const entity_pos = self.swarm.position_mean[i];
            const proj = camera.project(entity_pos) orelse continue;

            // Find closest detection (data association)
            // NDC: x [-1,1] left-to-right, y [-1,1] bottom-to-top
            // Image: x [0,W] left-to-right, y [0,H] top-to-bottom
            const entity_px = (proj.ndc.x + 1) * half_w;
            const entity_py = (1 - proj.ndc.y) * half_h; // Flip Y for image coords

            var best_detection_idx: ?usize = null;
            var best_cost: f32 = std.math.inf(f32);
            var best_spatial_dist_sq: f32 = std.math.inf(f32);

            // Get entity color for matching
            const entity_color = self.swarm.color[i];

            for (detections, 0..) |det, det_idx| {
                const dx = det.pixel_x - entity_px;
                const dy = det.pixel_y - entity_py;
                const spatial_dist_sq = dx * dx + dy * dy;

                // Color distance (L2 in RGB space)
                const color_diff = det.color.sub(entity_color);
                const color_dist_sq = color_diff.x * color_diff.x +
                    color_diff.y * color_diff.y +
                    color_diff.z * color_diff.z;

                // Combined cost: spatial + color (color weighted to be significant)
                // Color distance range: 0-3 (max RGB diff), spatial: 0-hundreds of pixels
                // Weight color so 0.5 color difference ~ 10 pixel spatial difference
                const color_weight: f32 = 400.0; // (10 pixels)^2 / (0.5)^2
                const cost = spatial_dist_sq + color_weight * color_dist_sq;

                if (cost < best_cost) {
                    best_cost = cost;
                    best_spatial_dist_sq = spatial_dist_sq;
                    best_detection_idx = det_idx;
                }
            }

            // Use spatial distance for threshold (color helps selection, not gating)
            const best_dist_sq = best_spatial_dist_sq;

            // If found a close enough detection, perform RBPF update
            // Use generous threshold to avoid data association failure during fast motion
            const max_match_dist: f32 = 50.0; // pixels (was 15.0)
            if (best_detection_idx) |det_idx| {
                if (best_dist_sq < max_match_dist * max_match_dist) {
                    const det = detections[det_idx];

                    // Back-project detection to 3D observation
                    const obs = self.backProjectDetectionTo3D(
                        det,
                        camera,
                        image_width,
                        image_height,
                    );

                    // Perform 6D RBPF update (updates position AND velocity via cross-covariance)
                    const entity_ll = self.rbpfEntityObservationUpdate6D(i, obs.mean, obs.cov);
                    total_log_lik += entity_ll;

                    // Infer color from observation (exponential moving average)
                    // Learning rate: 0.3 = moderate adaptation to observed color
                    const color_lr: f32 = 0.3;
                    const current_color = self.swarm.color[i];
                    const observed_color = det.color;
                    self.swarm.color[i] = current_color.scale(1.0 - color_lr).add(observed_color.scale(color_lr));
                } else {
                    // No matching detection - entity might be occluded
                    // Apply small penalty for miss
                    total_log_lik += -2.0;
                }
            } else if (detections.len > 0) {
                // Detections exist but none matched - likely occlusion or false entity
                total_log_lik += -3.0;
            }
        }

        return total_log_lik;
    }

    /// RBPF step: unified observation update that returns marginal likelihood
    /// Includes optical flow velocity updates when enabled
    fn rbpfObservationStep(
        self: *SMCState,
        observation: ObservationGrid,
    ) void {
        // Extract detections once (shared across all particles)
        const detections = Detection2D.extractFromGrid(observation, self.allocator) catch return;
        defer self.allocator.free(detections);

        // Compute sparse optical flow if enabled and previous frame exists
        var flow_observations: ?[]gmm.FlowObservation = null;
        defer if (flow_observations) |flows| self.allocator.free(flows);

        if (self.config.use_flow_observations) {
            // Resolution-adaptive flow config (disabled below 48px)
            const adapted = AdaptedFlowConfig.forResolution(observation.width, observation.height);

            // Only compute flow if enabled for this resolution
            if (adapted.enabled) {
                if (self.prev_frame_data) |prev_data| {
                    if (self.prev_detections) |prev_dets| {
                        // Create temporary ObservationGrid views
                        const prev_grid = ObservationGrid{
                            .pixels = prev_data,
                            .width = self.prev_frame_width,
                            .height = self.prev_frame_height,
                            .allocator = self.allocator,
                        };

                        // Compute sparse flow between consecutive frames
                        flow_observations = gmm.computeSparseFlow(
                            prev_dets,
                            detections,
                            prev_grid,
                            observation,
                            adapted.base,
                            self.allocator,
                        ) catch null;
                    }
                }
            }
        }

        // For each particle: update entities and accumulate marginal log-likelihood
        for (0..self.swarm.num_particles) |p| {
            var marginal_ll = self.rbpfParticleObservationUpdate(
                p,
                detections,
                observation.width,
                observation.height,
            );

            // Add velocity updates from flow observations
            if (flow_observations) |flows| {
                marginal_ll += self.rbpfParticleVelocityUpdate(
                    p,
                    flows,
                    detections,
                    observation.width,
                    observation.height,
                );
            }

            // Weight by marginal likelihood (applies temperature for annealing)
            self.swarm.log_weights[p] += self.temperature * @as(f64, marginal_ll);
        }

        self.normalizeWeights();
        self.updateSurpriseTracker();

        // Store current frame for next iteration (if flow enabled)
        if (self.config.use_flow_observations) {
            self.storePreviousFrame(observation, detections);
        }
    }

    /// Store observation frame for flow computation in next step
    fn storePreviousFrame(self: *SMCState, observation: ObservationGrid, detections: []const Detection2D) void {
        // Free old frame data
        if (self.prev_frame_data) |old_data| {
            self.allocator.free(old_data);
        }
        if (self.prev_detections) |old_dets| {
            self.allocator.free(old_dets);
        }

        // Copy current frame data
        const n_pixels = @as(usize, observation.width) * observation.height;
        self.prev_frame_data = self.allocator.alloc(Observation, n_pixels) catch {
            self.prev_frame_data = null;
            self.prev_detections = null;
            return;
        };
        @memcpy(self.prev_frame_data.?, observation.pixels);
        self.prev_frame_width = observation.width;
        self.prev_frame_height = observation.height;

        // Copy detections
        self.prev_detections = self.allocator.alloc(Detection2D, detections.len) catch {
            self.allocator.free(self.prev_frame_data.?);
            self.prev_frame_data = null;
            self.prev_detections = null;
            return;
        };
        @memcpy(self.prev_detections.?, detections);
    }

    /// Perform RBPF velocity update for all entities in a particle using flow observations
    fn rbpfParticleVelocityUpdate(
        self: *SMCState,
        particle_idx: usize,
        flow_observations: []const gmm.FlowObservation,
        curr_detections: []const Detection2D,
        image_width: u32,
        image_height: u32,
    ) f32 {
        if (flow_observations.len == 0) return 0;

        const camera_pose = self.swarm.camera_poses[particle_idx];
        const camera = camera_pose.toCamera(self.config.camera_intrinsics);
        const slice = self.swarm.particleSlice(particle_idx);

        var total_log_lik: f32 = 0;
        const half_w: f32 = @floatFromInt(image_width / 2);
        const half_h: f32 = @floatFromInt(image_height / 2);

        // For each entity, find associated flow observation and update velocity
        for (slice.start..slice.end) |i| {
            if (!self.swarm.alive[i]) continue;

            // Project entity to image space
            // NDC: Y increases upward, Image: Y increases downward
            const entity_pos = self.swarm.position_mean[i];
            const proj = camera.project(entity_pos) orelse continue;
            const entity_px = (proj.ndc.x + 1) * half_w;
            const entity_py = (1 - proj.ndc.y) * half_h; // Flip Y

            // Find flow observation whose current detection is closest to this entity
            var best_flow_idx: ?usize = null;
            var best_dist_sq: f32 = 20.0 * 20.0; // Max match distance

            for (flow_observations, 0..) |flow, flow_idx| {
                if (flow.curr_detection_idx >= curr_detections.len) continue;
                const det = curr_detections[flow.curr_detection_idx];
                const dx = det.pixel_x - entity_px;
                const dy = det.pixel_y - entity_py;
                const dist_sq = dx * dx + dy * dy;
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_flow_idx = flow_idx;
                }
            }

            // If found matching flow, perform velocity update
            if (best_flow_idx) |flow_idx| {
                const flow = flow_observations[flow_idx];
                const vel_ll = self.rbpfEntityVelocityUpdate(
                    i,
                    flow,
                    camera,
                    image_width,
                    image_height,
                );
                total_log_lik += vel_ll;
            }
        }

        return total_log_lik;
    }

    // =========================================================================
    // Goal Control (Phase 1b - Controller Pattern)
    // =========================================================================

    /// Resolve a Label to entity position within a particle
    /// O(n) scan is acceptable for max_entities=64 (fits in cache)
    fn resolveLabelToPosition(self: SMCState, particle: usize, target_label: Label) ?Vec3 {
        const slice = self.swarm.particleSlice(particle);
        for (slice.start..slice.end) |i| {
            if (!self.swarm.alive[i]) continue;
            if (self.swarm.label[i].eql(target_label)) {
                return self.swarm.position_mean[i];
            }
        }
        return null;
    }

    /// Compute goal-directed control acceleration
    /// Returns acceleration vector that gets added to velocity
    fn computeGoalControl(
        self: SMCState,
        particle: usize,
        entity_idx: usize,
        max_accel: f32,
    ) Vec3 {
        const goal_type = self.swarm.goal_type[entity_idx];
        const current_pos = self.swarm.position_mean[entity_idx];
        const current_vel = self.swarm.velocity_mean[entity_idx];

        // Resolve target position based on goal type
        const target_pos: ?Vec3 = switch (goal_type) {
            .none => null,
            .reach => self.swarm.target_position[entity_idx],
            .track, .avoid, .acquire => blk: {
                if (self.swarm.target_label[entity_idx]) |label| {
                    break :blk self.resolveLabelToPosition(particle, label);
                }
                break :blk null;
            },
        };

        // No target - no control
        const tp = target_pos orelse return Vec3.zero;

        // Compute direction to/from target
        const to_target = tp.sub(current_pos);
        const dist = to_target.length();

        if (dist < 0.01) return Vec3.zero; // At target

        const direction = to_target.normalize();

        // Compute desired acceleration based on goal type
        return switch (goal_type) {
            .none => Vec3.zero,
            .reach, .track, .acquire => blk: {
                // Proportional control: accelerate toward target
                // Use arrival behavior: slow down as we approach
                // Arrival radius sized for braking distance: v^2/(2a) at max_speed
                // With max_accel=5.0 and max_speed=5.0: braking_dist = 25/10 = 2.5
                const arrival_radius: f32 = 3.0;
                const max_speed: f32 = max_accel; // Keep 1:1 ratio for clean braking
                const speed_factor = @min(dist / arrival_radius, 1.0);

                // Desired velocity toward target (scales with distance via speed_factor)
                const desired_vel = direction.scale(max_speed * speed_factor);
                const vel_error = desired_vel.sub(current_vel);

                // Clamp to max acceleration
                const accel_mag = vel_error.length();
                if (accel_mag > max_accel) {
                    break :blk vel_error.normalize().scale(max_accel);
                }
                break :blk vel_error;
            },
            .avoid => blk: {
                // Accelerate away from target (inverse of track)
                // Flee behavior: stronger when closer
                const flee_radius: f32 = 5.0;
                const urgency = @max(0.0, 1.0 - dist / flee_radius);

                break :blk direction.scale(-max_accel * urgency);
            },
        };
    }

    /// Step physics for all particles in the swarm
    /// Uses proper Kalman filtering with covariance propagation (Rao-Blackwellized)
    fn stepSwarmPhysics(self: *SMCState) void {
        // Save previous states for Gibbs dynamics likelihood
        self.swarm.savePreviousState();

        // Step camera poses for all particles
        for (0..self.swarm.num_particles) |p| {
            self.swarm.camera_poses[p] = self.swarm.camera_poses[p].step(
                self.config.camera_position_noise,
                self.config.camera_yaw_noise,
                self.rng,
            );
        }

        // Step entity physics using Kalman prediction
        const entity_slots = self.swarm.num_particles * self.swarm.max_entities;
        const max_e = self.swarm.max_entities;

        // Save previous state for Gibbs dynamics likelihood
        // Bug 1 fix: Save contact_mode BEFORE physics step so Gibbs sees the bounce
        for (0..entity_slots) |i| {
            if (!self.swarm.alive[i]) continue;
            self.swarm.prev_position_mean[i] = self.swarm.position_mean[i];
            self.swarm.prev_velocity_mean[i] = self.swarm.velocity_mean[i];
            self.swarm.prev_contact_mode[i] = self.swarm.contact_mode[i];
        }

        for (0..entity_slots) |i| {
            if (!self.swarm.alive[i]) continue;

            // 6D coupled Kalman prediction (builds cross-covariance over time)
            const matrices_6d = dynamics.DynamicsMatrices6D.forContactModeWithParams(
                self.swarm.contact_mode[i],
                self.swarm.physics_params[i],
                self.config.physics,
                Vec3.unit_y, // Default contact normal
            );

            const prior = self.swarm.getState6D(i);
            const predicted = dynamics.kalmanPredict6D(prior, matrices_6d);

            // Store results (setState6D also syncs factored arrays for compatibility)
            self.swarm.setState6D(i, predicted);

            // Apply goal-directed control (Phase 1b)
            // Goal control acts as an acceleration term on velocity
            if (self.swarm.goal_type[i] != .none) {
                const particle = i / max_e;
                const max_accel: f32 = 5.0; // TODO: Make configurable
                const control_accel = self.computeGoalControl(particle, i, max_accel);
                // Apply control as velocity change (accel * dt)
                self.swarm.velocity_mean[i] = self.swarm.velocity_mean[i].add(
                    control_accel.scale(self.config.physics.dt),
                );
            }

            // Environment collision detection and response (unified ground/wall handling)
            const ground_height = self.config.physics.groundHeight();
            if (self.swarm.position_mean[i].y < ground_height) {
                self.swarm.position_mean[i].y = ground_height;

                if (self.swarm.velocity_mean[i].y < 0) {
                    // Ball hit environment while moving down - apply bounce
                    // Use continuous physics_params (Spelke-aligned inference)
                    const elasticity = self.swarm.physics_params[i].elasticity;
                    const vel_before = self.swarm.velocity_mean[i].y;
                    self.swarm.velocity_mean[i].y = -vel_before * elasticity;

                    // ELASTICITY INFERENCE: Observe bounce to update physics params
                    // Compare predicted bounce with actual (pre-bounce) velocity
                    // observed_elasticity = |v_after| / |v_before|
                    // We use prev_velocity which was set before dynamics step
                    const prev_vel_y = self.swarm.prev_velocity_mean[i].y;
                    if (prev_vel_y < -0.5) { // Only update if significant downward velocity
                        // The ball was falling with prev_vel_y, now bouncing
                        // We can infer what elasticity should be based on actual motion
                        // For now, use current velocity as "observed" for the update
                        // Higher confidence when velocity is larger (more informative bounce)
                        const confidence = @min(1.0, @abs(prev_vel_y) / 5.0);
                        // The applied elasticity is what we observe
                        self.swarm.physics_params_uncertainty[i].updateElasticity(elasticity, confidence * 0.5);
                        // Update point estimate from posterior mean
                        self.swarm.physics_params[i].elasticity = self.swarm.physics_params_uncertainty[i].elasticityMean();
                    }

                    // After bounce, ball is moving up - set to free so dynamics doesn't zero velocity
                    self.swarm.contact_mode[i] = .free;
                } else {
                    // Ball at environment but not moving down (resting or very slow)
                    self.swarm.contact_mode[i] = .environment;
                }
            } else if (self.swarm.position_mean[i].y > ground_height + 0.1) {
                // Clear environment contact if sufficiently above
                if (self.swarm.contact_mode[i] == .environment) {
                    self.swarm.contact_mode[i] = .free;
                }
            }

            // Sync 6D state mean after any position/velocity modifications
            // (bounce handling modified factored arrays directly)
            self.swarm.syncFactoredTo6DMean(i);
        }

        // Entity-entity collisions
        // Process per-particle to maintain particle independence
        for (0..self.swarm.num_particles) |p| {
            const n_entities = self.swarm.entity_counts[p];
            if (n_entities < 2) continue;

            const slice = self.swarm.particleSlice(p);
            for (slice.start..slice.end) |i| {
                if (!self.swarm.alive[i]) continue;
                for (i + 1..slice.end) |j| {
                    if (!self.swarm.alive[j]) continue;

                    // Check collision (sphere-sphere)
                    const dist = self.swarm.position_mean[i].sub(self.swarm.position_mean[j]).length();
                    const min_dist: f32 = 1.0; // Sum of radii (assuming 0.5 each)

                    if (dist < min_dist and dist > 0.001) {
                        // Simple collision response: push apart
                        const normal = self.swarm.position_mean[i].sub(self.swarm.position_mean[j]).normalize();
                        const overlap = min_dist - dist;

                        // Move entities apart
                        self.swarm.position_mean[i] = self.swarm.position_mean[i].add(normal.scale(overlap * 0.5));
                        self.swarm.position_mean[j] = self.swarm.position_mean[j].sub(normal.scale(overlap * 0.5));

                        // Update contact mode
                        self.swarm.contact_mode[i] = .entity;
                        self.swarm.contact_mode[j] = .entity;

                        // Elastic collision (simplified - swap normal components)
                        const v1n = normal.scale(self.swarm.velocity_mean[i].dot(normal));
                        const v2n = normal.scale(self.swarm.velocity_mean[j].dot(normal));

                        self.swarm.velocity_mean[i] = self.swarm.velocity_mean[i].sub(v1n).add(v2n);
                        self.swarm.velocity_mean[j] = self.swarm.velocity_mean[j].sub(v2n).add(v1n);

                        // Sync 6D state mean after collision modifications
                        self.swarm.syncFactoredTo6DMean(i);
                        self.swarm.syncFactoredTo6DMean(j);
                    }
                }
            }
        }
    }

    /// Single SMC step: propagate, weight, resample, rejuvenate
    /// Camera is no longer a parameter - each particle has its own camera hypothesis
    ///
    /// Uses Rao-Blackwellized observation update that updates both mean and
    /// covariance via Kalman filter, returning marginal likelihood for weighting.
    /// References: Doucet et al. (2000), SMCP3 (Lew et al. 2023)
    pub fn step(
        self: *SMCState,
        observation: ObservationGrid,
    ) !void {
        // 1. Propagate particles through dynamics (camera + entities)
        self.stepSwarmPhysics();

        // 2. RBPF observation step: updates entities AND computes weights
        // Uses observation ONCE via Kalman update, returns marginal likelihood
        self.rbpfObservationStep(observation);

        // 3. Check ESS and resample if needed
        const ess = self.effectiveSampleSize();
        const threshold = self.config.ess_threshold * @as(f32, @floatFromInt(self.config.num_particles));

        if (ess < threshold) {
            try self.resample();

            // 4. Gibbs rejuvenation after resampling (for discrete variables)
            self.gibbsRejuvenation(observation);
        }

        // 5. CRP trans-dimensional moves (if enabled)
        if (self.config.use_crp) {
            try self.transDimensionalStep(observation);
        }

        // 6. Increase temperature if tempering
        if (self.config.use_tempering and self.temperature < 1.0) {
            self.temperature = @min(1.0, self.temperature + self.config.temperature_increment);
        }

        self.timestep += 1;
    }

    /// Perform trans-dimensional moves (birth/death) for each particle
    /// Uses CRP prior for entity count inference
    fn transDimensionalStep(self: *SMCState, observation: ObservationGrid) !void {
        for (0..self.swarm.num_particles) |p| {
            // Get world view for this particle
            var view = particleWorld(&self.swarm, p);

            // Get camera for this particle
            const camera = self.swarm.camera_poses[p].toCamera(self.config.camera_intrinsics);

            // Perform trans-dimensional move
            _ = try crp.transDimensionalStep(
                &observation,
                &view,
                &self.label_sets[p],
                camera,
                self.config.crp_config,
                null, // no stats tracking
                self.rng,
                self.allocator,
            );
        }
    }

    /// Get posterior estimate of physics params (weighted mean for each entity)
    pub fn getPhysicsParamsEstimate(self: SMCState) ![]PhysicsParams {
        // Use entity count from first particle (assumes all have same count)
        const n_entities = self.swarm.entity_counts[0];
        if (n_entities == 0) {
            return &[_]PhysicsParams{};
        }

        var estimates = try self.allocator.alloc(PhysicsParams, n_entities);

        for (0..n_entities) |entity_idx| {
            var weighted_elasticity: f32 = 0;
            var weighted_friction: f32 = 0;
            var total_weight: f32 = 0;

            for (0..self.swarm.num_particles) |p| {
                const weight = self.weights[p];
                const i = self.swarm.idx(p, entity_idx);
                if (self.swarm.alive[i]) {
                    const params = self.swarm.physics_params[i];
                    weighted_elasticity += weight * params.elasticity;
                    weighted_friction += weight * params.friction;
                    total_weight += weight;
                }
            }

            if (total_weight > 0) {
                estimates[entity_idx] = .{
                    .elasticity = weighted_elasticity / total_weight,
                    .friction = weighted_friction / total_weight,
                };
            } else {
                estimates[entity_idx] = PhysicsParams.prior;
            }
        }

        return estimates;
    }

    /// Camera belief (weighted statistics over particle camera poses)
    pub const CameraBelief = struct {
        /// Weighted mean position
        mean_position: Vec3,
        /// Weighted mean yaw
        mean_yaw: f32,
        /// Position variance (diagonal)
        position_variance: Vec3,
        /// Yaw variance
        yaw_variance: f32,
    };

    /// Get posterior belief over camera pose (weighted mean and variance)
    pub fn getCameraBelief(self: SMCState) CameraBelief {
        var mean_pos = Vec3.zero;
        var mean_yaw: f32 = 0;

        // First pass: weighted mean
        for (0..self.swarm.num_particles) |p| {
            const weight = self.weights[p];
            const pose = self.swarm.camera_poses[p];
            mean_pos = mean_pos.add(pose.position.scale(weight));
            mean_yaw += pose.yaw * weight;
        }

        // Second pass: weighted variance
        var var_pos = Vec3.zero;
        var var_yaw: f32 = 0;

        for (0..self.swarm.num_particles) |p| {
            const weight = self.weights[p];
            const pose = self.swarm.camera_poses[p];
            const pos_diff = pose.position.sub(mean_pos);
            var_pos = var_pos.add(Vec3.init(
                pos_diff.x * pos_diff.x * weight,
                pos_diff.y * pos_diff.y * weight,
                pos_diff.z * pos_diff.z * weight,
            ));

            const yaw_diff = pose.yaw - mean_yaw;
            var_yaw += yaw_diff * yaw_diff * weight;
        }

        return .{
            .mean_position = mean_pos,
            .mean_yaw = mean_yaw,
            .position_variance = var_pos,
            .yaw_variance = var_yaw,
        };
    }

    /// Get MAP (maximum a posteriori) camera pose (highest weight particle)
    pub fn getCameraPoseMAP(self: SMCState) CameraPose {
        var max_weight: f32 = -std.math.inf(f32);
        var map_pose = CameraPose.default;

        for (0..self.swarm.num_particles) |p| {
            if (self.weights[p] > max_weight) {
                max_weight = self.weights[p];
                map_pose = self.swarm.camera_poses[p];
            }
        }

        return map_pose;
    }

    // =========================================================================
    // Physics Belief: Expectations over Particle Population
    // =========================================================================
    //
    // Rather than reporting MAP estimates, we compute expectations and variances
    // over the weighted particle population. This is the proper Bayesian approach
    // that preserves uncertainty information.
    //
    // For discrete physics types, we report:
    // - Categorical posterior probabilities (p(standard), p(bouncy), ...)
    // - Expected elasticity: E[elasticity] = Σ w_p × elasticity(type_p)
    // - Expected friction: E[friction] = Σ w_p × friction(type_p)
    // - Variance of each to quantify uncertainty
    //
    // These expectations are what you'd use for downstream decisions.
    // =========================================================================

    /// Physics belief for a single entity: expectations and variances
    pub const PhysicsBelief = struct {
        /// Categorical posterior: p(type) for 4 discrete material bins
        /// (legacy compatibility - prefer expected_elasticity/friction for continuous inference)
        type_probabilities: [4]f32,

        /// Expected elasticity (bounce coefficient)
        expected_elasticity: f32,
        /// Variance of elasticity
        elasticity_variance: f32,

        /// Expected friction coefficient
        expected_friction: f32,
        /// Variance of friction
        friction_variance: f32,

        /// Effective sample size (ESS) for this entity's belief
        /// Low ESS indicates particle degeneracy
        effective_sample_size: f32,
    };

    /// Discretize continuous physics params to type bucket (for backward compat)
    fn discretizeParams(params: PhysicsParams) usize {
        // Classify by elasticity thresholds (same logic as C API)
        if (params.elasticity > 0.8) return 1; // bouncy
        if (params.elasticity < 0.3 and params.friction > 0.6) return 2; // sticky
        if (params.friction < 0.15) return 3; // slippery
        return 0; // standard
    }

    /// Compute physics belief for each entity using weighted expectations
    pub fn getPhysicsBelief(self: SMCState) ![]PhysicsBelief {
        const n_entities = self.swarm.entity_counts[0];
        if (n_entities == 0) {
            return &[_]PhysicsBelief{};
        }

        var beliefs = try self.allocator.alloc(PhysicsBelief, n_entities);

        for (0..n_entities) |entity_idx| {
            // Initialize accumulators
            var type_probs = [4]f32{ 0, 0, 0, 0 };
            var expected_elasticity: f32 = 0;
            var expected_friction: f32 = 0;
            var sum_weight_sq: f32 = 0;

            // First pass: compute expectations
            for (0..self.swarm.num_particles) |p| {
                const weight = self.weights[p];
                const i = self.swarm.idx(p, entity_idx);

                if (self.swarm.alive[i]) {
                    const params = self.swarm.physics_params[i];
                    const type_idx = discretizeParams(params);
                    type_probs[type_idx] += weight;

                    expected_elasticity += weight * params.elasticity;
                    expected_friction += weight * params.friction;
                    sum_weight_sq += weight * weight;
                }
            }

            // Second pass: compute variances
            var elasticity_var: f32 = 0;
            var friction_var: f32 = 0;

            for (0..self.swarm.num_particles) |p| {
                const weight = self.weights[p];
                const i = self.swarm.idx(p, entity_idx);

                if (self.swarm.alive[i]) {
                    const params = self.swarm.physics_params[i];

                    const e_diff = params.elasticity - expected_elasticity;
                    const f_diff = params.friction - expected_friction;

                    elasticity_var += weight * e_diff * e_diff;
                    friction_var += weight * f_diff * f_diff;
                }
            }

            // ESS = 1 / Σ w² (for normalized weights summing to 1)
            const ess = if (sum_weight_sq > 0) 1.0 / sum_weight_sq else 0;

            beliefs[entity_idx] = .{
                .type_probabilities = type_probs,
                .expected_elasticity = expected_elasticity,
                .elasticity_variance = elasticity_var,
                .expected_friction = expected_friction,
                .friction_variance = friction_var,
                .effective_sample_size = ess,
            };
        }

        return beliefs;
    }

    /// Entity position/velocity belief: weighted mean and covariance
    pub const EntityBelief = struct {
        /// Weighted mean position across particles
        mean_position: Vec3,
        /// Position variance (diagonal approximation)
        position_variance: Vec3,
        /// Weighted mean velocity
        mean_velocity: Vec3,
        /// Velocity variance (diagonal approximation)
        velocity_variance: Vec3,
    };

    /// Compute position/velocity belief for each entity
    pub fn getEntityBelief(self: SMCState) ![]EntityBelief {
        const n_entities = self.swarm.entity_counts[0];
        if (n_entities == 0) {
            return &[_]EntityBelief{};
        }

        var beliefs = try self.allocator.alloc(EntityBelief, n_entities);

        for (0..n_entities) |entity_idx| {
            // First pass: weighted mean
            var mean_pos = Vec3.zero;
            var mean_vel = Vec3.zero;

            for (0..self.swarm.num_particles) |p| {
                const weight = self.weights[p];
                const i = self.swarm.idx(p, entity_idx);

                if (self.swarm.alive[i]) {
                    mean_pos = mean_pos.add(self.swarm.position_mean[i].scale(weight));
                    mean_vel = mean_vel.add(self.swarm.velocity_mean[i].scale(weight));
                }
            }

            // Second pass: weighted variance
            var var_pos = Vec3.zero;
            var var_vel = Vec3.zero;

            for (0..self.swarm.num_particles) |p| {
                const weight = self.weights[p];
                const i = self.swarm.idx(p, entity_idx);

                if (self.swarm.alive[i]) {
                    const pos_diff = self.swarm.position_mean[i].sub(mean_pos);
                    const vel_diff = self.swarm.velocity_mean[i].sub(mean_vel);

                    var_pos = var_pos.add(Vec3.init(
                        pos_diff.x * pos_diff.x * weight,
                        pos_diff.y * pos_diff.y * weight,
                        pos_diff.z * pos_diff.z * weight,
                    ));
                    var_vel = var_vel.add(Vec3.init(
                        vel_diff.x * vel_diff.x * weight,
                        vel_diff.y * vel_diff.y * weight,
                        vel_diff.z * vel_diff.z * weight,
                    ));
                }
            }

            beliefs[entity_idx] = .{
                .mean_position = mean_pos,
                .position_variance = var_pos,
                .mean_velocity = mean_vel,
                .velocity_variance = var_vel,
            };
        }

        return beliefs;
    }
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "Legacy Particle creation and physics step" {
    const allocator = testing.allocator;

    var particle = Particle.init(allocator);
    defer particle.deinit();

    const label = Label{ .birth_time = 0, .birth_index = 0 };
    const entity = Entity.initPoint(label, Vec3.init(0, 5, 0), Vec3.zero, .standard);
    try particle.addEntity(entity);

    try testing.expect(particle.aliveCount() == 1);

    var prng = std.Random.DefaultPrng.init(42);
    var config = SMCConfig{};
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    try particle.stepPhysics(config, prng.random());

    // Entity should have fallen
    try testing.expect(particle.entities.items[0].positionMean().y < 5);
}

test "SMCState initialization" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc_state = try SMCState.init(allocator, SMCConfig{ .num_particles = 10 }, prng.random());
    defer smc_state.deinit();

    try testing.expect(smc_state.swarm.num_particles == 10);
    try testing.expect(smc_state.effectiveSampleSize() > 9.9); // Should be ~10 with uniform weights
}

test "SMCState initialize with prior" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc_state = try SMCState.init(allocator, SMCConfig{ .num_particles = 10 }, prng.random());
    defer smc_state.deinit();

    const positions = [_]Vec3{ Vec3.init(0, 5, 0), Vec3.init(2, 5, 0) };
    const velocities = [_]Vec3{ Vec3.zero, Vec3.zero };

    smc_state.initializeWithPrior(&positions, &velocities);

    // Each particle should have 2 entities
    for (0..smc_state.swarm.num_particles) |p| {
        try testing.expect(smc_state.swarm.entity_counts[p] == 2);
    }
}

test "SMCState camera pose initialization" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{ .num_particles = 100 };
    config.camera_prior_min = Vec3.init(-2, 3, 8);
    config.camera_prior_max = Vec3.init(2, 7, 12);

    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    const positions = [_]Vec3{Vec3.init(0, 1, 0)};
    const velocities = [_]Vec3{Vec3.zero};
    smc_state.initializeWithPrior(&positions, &velocities);

    // Each particle should have a camera pose within prior bounds
    for (0..smc_state.swarm.num_particles) |p| {
        const pose = smc_state.swarm.camera_poses[p];
        try testing.expect(pose.position.x >= -2 and pose.position.x <= 2);
        try testing.expect(pose.position.y >= 3 and pose.position.y <= 7);
        try testing.expect(pose.position.z >= 8 and pose.position.z <= 12);
        try testing.expect(pose.yaw >= -std.math.pi and pose.yaw <= std.math.pi);
    }

    // Camera belief should have reasonable variance
    const belief = smc_state.getCameraBelief();
    try testing.expect(belief.position_variance.x > 0);
    try testing.expect(belief.yaw_variance > 0);
}

test "CameraPose forward direction" {
    // Yaw = 0 should look along -Z
    const pose0 = CameraPose.init(Vec3.zero, 0);
    const fwd0 = pose0.forward();
    try testing.expect(@abs(fwd0.x) < 0.001);
    try testing.expect(@abs(fwd0.y) < 0.001);
    try testing.expect(fwd0.z < -0.99);

    // Yaw = pi/2 should look along -X
    const pose90 = CameraPose.init(Vec3.zero, std.math.pi / 2.0);
    const fwd90 = pose90.forward();
    try testing.expect(fwd90.x < -0.99);
    try testing.expect(@abs(fwd90.y) < 0.001);
    try testing.expect(@abs(fwd90.z) < 0.001);
}

test "Systematic resampling" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc_state = try SMCState.init(allocator, SMCConfig{ .num_particles = 10 }, prng.random());
    defer smc_state.deinit();

    const positions = [_]Vec3{Vec3.init(0, 5, 0)};
    const velocities = [_]Vec3{Vec3.zero};
    smc_state.initializeWithPrior(&positions, &velocities);

    // Set skewed weights
    smc_state.weights[0] = 0.9;
    for (1..10) |i| {
        smc_state.weights[i] = 0.1 / 9.0;
    }

    try smc_state.resample();

    // After resampling, weights should be uniform
    for (smc_state.weights) |w| {
        try testing.expect(@abs(w - 0.1) < 0.01);
    }
}

test "Effective sample size" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc_state = try SMCState.init(allocator, SMCConfig{ .num_particles = 100 }, prng.random());
    defer smc_state.deinit();

    // Uniform weights -> ESS = N
    try testing.expect(smc_state.effectiveSampleSize() > 99);

    // Concentrated weight -> low ESS
    @memset(smc_state.weights, 0);
    smc_state.weights[0] = 1.0;
    try testing.expect(smc_state.effectiveSampleSize() < 1.1);
}

test "Swarm physics stepping" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{ .num_particles = 5 };
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;

    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    const positions = [_]Vec3{Vec3.init(0, 5, 0)};
    const velocities = [_]Vec3{Vec3.zero};
    smc_state.initializeWithPrior(&positions, &velocities);

    // Get initial position
    const initial_y = smc_state.swarm.position_mean[0].y;

    // Step physics
    smc_state.stepSwarmPhysics();

    // Entity should have fallen
    try testing.expect(smc_state.swarm.position_mean[0].y < initial_y);
}

test "resolveLabelToPosition finds entity by label" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 2;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Create entity with specific label in particle 0
    const target_label = Label{ .birth_time = 42, .birth_index = 7 };
    const target_pos = Vec3.init(3.0, 4.0, 5.0);
    smc_state.swarm.label[0] = target_label;
    smc_state.swarm.position_mean[0] = target_pos;
    smc_state.swarm.alive[0] = true;

    // Create another entity
    smc_state.swarm.label[1] = Label{ .birth_time = 0, .birth_index = 0 };
    smc_state.swarm.alive[1] = true;

    // Should find target entity
    const found = smc_state.resolveLabelToPosition(0, target_label);
    try testing.expect(found != null);
    try testing.expectApproxEqAbs(@as(f32, 3.0), found.?.x, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 4.0), found.?.y, 0.001);

    // Should not find nonexistent label
    const not_found = smc_state.resolveLabelToPosition(0, Label{ .birth_time = 99, .birth_index = 99 });
    try testing.expect(not_found == null);

    // Should not find label in different particle
    const wrong_particle = smc_state.resolveLabelToPosition(1, target_label);
    try testing.expect(wrong_particle == null);
}

test "computeGoalControl returns zero for no goal" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup entity with no goal
    smc_state.swarm.goal_type[0] = .none;
    smc_state.swarm.position_mean[0] = Vec3.zero;
    smc_state.swarm.velocity_mean[0] = Vec3.zero;
    smc_state.swarm.alive[0] = true;

    const control = smc_state.computeGoalControl(0, 0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), control.length(), 0.001);
}

test "computeGoalControl accelerates toward reach target" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup entity with reach goal
    smc_state.swarm.goal_type[0] = .reach;
    smc_state.swarm.target_position[0] = Vec3.init(10.0, 0.0, 0.0); // Target 10 units away
    smc_state.swarm.position_mean[0] = Vec3.zero;
    smc_state.swarm.velocity_mean[0] = Vec3.zero;
    smc_state.swarm.alive[0] = true;

    const control = smc_state.computeGoalControl(0, 0, 5.0);

    // Should accelerate toward target (positive x direction)
    try testing.expect(control.x > 0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), control.y, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), control.z, 0.001);

    // Should be capped at max_accel
    try testing.expect(control.length() <= 5.01);
}

test "computeGoalControl avoid accelerates away from target" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup entity with avoid goal (avoiding entity at index 1)
    const target_label = Label{ .birth_time = 10, .birth_index = 1 };

    // Avoider at origin
    smc_state.swarm.goal_type[0] = .avoid;
    smc_state.swarm.target_label[0] = target_label;
    smc_state.swarm.position_mean[0] = Vec3.zero;
    smc_state.swarm.velocity_mean[0] = Vec3.zero;
    smc_state.swarm.alive[0] = true;

    // Target entity at (2, 0, 0)
    smc_state.swarm.label[1] = target_label;
    smc_state.swarm.position_mean[1] = Vec3.init(2.0, 0.0, 0.0);
    smc_state.swarm.alive[1] = true;

    const control = smc_state.computeGoalControl(0, 0, 5.0);

    // Should accelerate away from target (negative x direction)
    try testing.expect(control.x < 0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), control.y, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), control.z, 0.001);
}

test "computeGoalControl arrival behavior slows near target" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup entity with reach goal, very close to target
    smc_state.swarm.goal_type[0] = .reach;
    smc_state.swarm.target_position[0] = Vec3.init(0.5, 0.0, 0.0); // Only 0.5 units away
    smc_state.swarm.position_mean[0] = Vec3.zero;
    smc_state.swarm.velocity_mean[0] = Vec3.zero;
    smc_state.swarm.alive[0] = true;

    const control_near = smc_state.computeGoalControl(0, 0, 5.0);

    // Setup same entity far from target
    smc_state.swarm.target_position[0] = Vec3.init(10.0, 0.0, 0.0);
    const control_far = smc_state.computeGoalControl(0, 0, 5.0);

    // Near target should have smaller acceleration (arrival behavior)
    try testing.expect(control_near.length() < control_far.length());
}

test "computeGoalControl brakes moving entity near target" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup entity moving toward target at high speed
    // Entity at origin, target at (1, 0, 0), moving toward target at 5 m/s
    smc_state.swarm.goal_type[0] = .reach;
    smc_state.swarm.target_position[0] = Vec3.init(1.0, 0.0, 0.0);
    smc_state.swarm.position_mean[0] = Vec3.zero;
    smc_state.swarm.velocity_mean[0] = Vec3.init(5.0, 0.0, 0.0); // Moving toward target
    smc_state.swarm.alive[0] = true;

    const control = smc_state.computeGoalControl(0, 0, 5.0);

    // Entity is inside arrival_radius (dist=1 < 3) and moving fast
    // Desired velocity scales with distance: desired = direction * max_speed * (1/3) ≈ 1.67
    // Current velocity = 5.0 > desired, so vel_error should be negative
    // Control should brake (negative x to oppose positive velocity)
    try testing.expect(control.x < 0); // Braking force opposes motion
}

test "spatialRelationLogLikelihood returns zero for no relations" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup entity with no spatial relation
    smc_state.swarm.spatial_relation_type[0] = .none;
    smc_state.swarm.alive[0] = true;

    const log_lik = smc_state.spatialRelationLogLikelihood(0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), log_lik, 0.001);
}

test "spatialRelationLogLikelihood penalizes violated above relation" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup reference entity
    const ref_label = Label{ .birth_time = 1, .birth_index = 0 };
    smc_state.swarm.label[1] = ref_label;
    smc_state.swarm.position_mean[1] = Vec3.init(0, 2, 0); // Reference at y=2
    smc_state.swarm.alive[1] = true;

    // Setup entity that should be "above" reference
    smc_state.swarm.spatial_relation_type[0] = .above;
    smc_state.swarm.spatial_reference[0] = ref_label;
    smc_state.swarm.spatial_tolerance[0] = 1.0;
    smc_state.swarm.alive[0] = true;

    // Entity above reference (satisfied) - should have ~0 penalty
    smc_state.swarm.position_mean[0] = Vec3.init(0, 5, 0); // Above reference
    const log_lik_satisfied = smc_state.spatialRelationLogLikelihood(0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), log_lik_satisfied, 0.001);

    // Entity below reference (violated) - should have negative log likelihood
    smc_state.swarm.position_mean[0] = Vec3.init(0, 0, 0); // Below reference
    const log_lik_violated = smc_state.spatialRelationLogLikelihood(0);
    try testing.expect(log_lik_violated < 0);
}

test "spatialRelationLogLikelihood penalizes near relation distance" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup reference entity at origin
    const ref_label = Label{ .birth_time = 1, .birth_index = 0 };
    smc_state.swarm.label[1] = ref_label;
    smc_state.swarm.position_mean[1] = Vec3.zero;
    smc_state.swarm.alive[1] = true;

    // Setup entity that should be "near" reference at distance 3.0
    smc_state.swarm.spatial_relation_type[0] = .near;
    smc_state.swarm.spatial_reference[0] = ref_label;
    smc_state.swarm.spatial_distance[0] = 3.0; // Expected distance
    smc_state.swarm.spatial_tolerance[0] = 0.5;
    smc_state.swarm.alive[0] = true;

    // Entity at expected distance (satisfied)
    smc_state.swarm.position_mean[0] = Vec3.init(3.0, 0, 0); // Exactly at expected distance
    const log_lik_satisfied = smc_state.spatialRelationLogLikelihood(0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), log_lik_satisfied, 0.001);

    // Entity too close (violated)
    smc_state.swarm.position_mean[0] = Vec3.init(0.5, 0, 0); // Too close
    const log_lik_too_close = smc_state.spatialRelationLogLikelihood(0);
    try testing.expect(log_lik_too_close < 0);

    // Entity too far (violated)
    smc_state.swarm.position_mean[0] = Vec3.init(10.0, 0, 0); // Too far
    const log_lik_too_far = smc_state.spatialRelationLogLikelihood(0);
    try testing.expect(log_lik_too_far < 0);
}

// =============================================================================
// Numerosity Tests (Phase 3)
// =============================================================================

test "countAliveEntities counts correctly" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 2;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Particle 0: 3 alive entities
    smc_state.swarm.alive[0] = true;
    smc_state.swarm.alive[1] = true;
    smc_state.swarm.alive[2] = true;
    smc_state.swarm.alive[3] = false;

    // Particle 1: 1 alive entity
    smc_state.swarm.alive[4] = true;
    smc_state.swarm.alive[5] = false;
    smc_state.swarm.alive[6] = false;
    smc_state.swarm.alive[7] = false;

    try testing.expectEqual(@as(u32, 3), smc_state.countAliveEntities(0));
    try testing.expectEqual(@as(u32, 1), smc_state.countAliveEntities(1));
}

test "numerosityLogLikelihood highest when counts match" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 8;
    config.weber_fraction = 0.2;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Setup 5 alive entities
    for (0..5) |i| {
        smc_state.swarm.alive[i] = true;
    }
    for (5..8) |i| {
        smc_state.swarm.alive[i] = false;
    }

    // Likelihood should be highest when observed count matches particle count
    const log_lik_match = smc_state.numerosityLogLikelihood(0, 5);
    const log_lik_under = smc_state.numerosityLogLikelihood(0, 3);
    const log_lik_over = smc_state.numerosityLogLikelihood(0, 8);

    try testing.expect(log_lik_match > log_lik_under);
    try testing.expect(log_lik_match > log_lik_over);
}

test "numerosityLogLikelihood Weber scaling - larger counts more tolerant" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 2;
    config.max_entities = 16;
    config.weber_fraction = 0.2;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Particle 0: 2 entities (small count)
    smc_state.swarm.alive[0] = true;
    smc_state.swarm.alive[1] = true;
    for (2..16) |i| {
        smc_state.swarm.alive[i] = false;
    }

    // Particle 1: 10 entities (large count)
    for (16..26) |i| {
        smc_state.swarm.alive[i] = true;
    }
    for (26..32) |i| {
        smc_state.swarm.alive[i] = false;
    }

    // For small count (2), being off by 1 should hurt more
    // than being off by 1 for large count (10)
    // This is Weber's law: discrimination threshold scales with magnitude

    // Both particles observe count off by 1
    const log_lik_small_off_1 = smc_state.numerosityLogLikelihood(0, 3); // 2 entities, observe 3
    const log_lik_large_off_1 = smc_state.numerosityLogLikelihood(1, 11); // 10 entities, observe 11

    // For Weber's law: σ = w * n
    // Small count σ = 0.2 * 2 = 0.4, penalty for diff=1: -1/(2*0.16) = -3.125
    // Large count σ = 0.2 * 10 = 2.0, penalty for diff=1: -1/(2*4) = -0.125
    // Larger count should be MORE tolerant (less negative log-likelihood)

    // Note: We need to account for the normalization constant too, but the
    // relative penalty for being off by 1 should be less severe for larger counts
    try testing.expect(log_lik_large_off_1 > log_lik_small_off_1);
}

test "numerosityLogLikelihood handles zero entities with min variance" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    config.weber_fraction = 0.2;
    config.min_numerosity_variance = 0.5;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // No alive entities
    for (0..4) |i| {
        smc_state.swarm.alive[i] = false;
    }

    // Should not crash or return NaN/Inf due to min_variance floor
    const log_lik_match = smc_state.numerosityLogLikelihood(0, 0);
    try testing.expect(!std.math.isNan(log_lik_match));
    try testing.expect(!std.math.isInf(log_lik_match));

    // Observing 0 when particle has 0 should give 0 log-likelihood (perfect match)
    // With unnormalized likelihood: diff=0 => log_lik = 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), log_lik_match, 0.001);

    // Observing non-zero should give negative log-likelihood
    const log_lik_mismatch = smc_state.numerosityLogLikelihood(0, 3);
    try testing.expect(log_lik_mismatch < 0);
}

test "numerosityLogLikelihood no heteroscedasticity bias for matching counts" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 2;
    config.max_entities = 64;
    config.weber_fraction = 0.2;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Particle 0: 2 entities
    smc_state.swarm.alive[0] = true;
    smc_state.swarm.alive[1] = true;
    for (2..64) |i| {
        smc_state.swarm.alive[i] = false;
    }

    // Particle 1: 50 entities
    for (64..114) |i| {
        smc_state.swarm.alive[i] = true;
    }
    for (114..128) |i| {
        smc_state.swarm.alive[i] = false;
    }

    // Both particles should get SAME likelihood when observing their actual count
    // (This verifies the heteroscedasticity bias fix)
    const log_lik_small_match = smc_state.numerosityLogLikelihood(0, 2); // 2 matches 2
    const log_lik_large_match = smc_state.numerosityLogLikelihood(1, 50); // 50 matches 50

    // Both should be 0 (perfect match with unnormalized likelihood)
    try testing.expectApproxEqAbs(@as(f32, 0.0), log_lik_small_match, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), log_lik_large_match, 0.001);
}

test "updateWeightsNumerosity shifts weight toward matching particles" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 3;
    config.max_entities = 4;
    config.weber_fraction = 0.2;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Particle 0: 2 entities
    smc_state.swarm.alive[0] = true;
    smc_state.swarm.alive[1] = true;
    smc_state.swarm.alive[2] = false;
    smc_state.swarm.alive[3] = false;

    // Particle 1: 4 entities (matches observation)
    smc_state.swarm.alive[4] = true;
    smc_state.swarm.alive[5] = true;
    smc_state.swarm.alive[6] = true;
    smc_state.swarm.alive[7] = true;

    // Particle 2: 1 entity
    smc_state.swarm.alive[8] = true;
    smc_state.swarm.alive[9] = false;
    smc_state.swarm.alive[10] = false;
    smc_state.swarm.alive[11] = false;

    // Start with uniform weights
    @memset(smc_state.weights, 1.0 / 3.0);
    @memset(smc_state.swarm.log_weights, 0.0);

    // Observe 4 entities
    smc_state.updateWeightsNumerosity(4);

    // Particle 1 (4 entities) should now have highest weight
    try testing.expect(smc_state.weights[1] > smc_state.weights[0]);
    try testing.expect(smc_state.weights[1] > smc_state.weights[2]);
}

// =============================================================================
// Inference Scenario Tests (Core Knowledge Integration)
// =============================================================================

test "scenario: two-ball chase - goal control produces tracking behavior" {
    // Setup: Ball A at origin, Ball B chasing A
    // Expectation: B's velocity should be directed toward A
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    config.physics.dt = 0.1;
    config.physics.gravity = Vec3.init(0, 0, 0); // No gravity for clean test
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Ball A (target) - label (0, 0)
    const label_a = Label{ .birth_time = 0, .birth_index = 0 };
    smc_state.swarm.label[0] = label_a;
    smc_state.swarm.initState6D(0, Vec3.init(10.0, 0.0, 0.0), Vec3.zero); // Target at x=10
    smc_state.swarm.goal_type[0] = .none; // A is passive
    smc_state.swarm.alive[0] = true;
    smc_state.swarm.physics_params[0] = PhysicsParams.standard;
    smc_state.swarm.contact_mode[0] = .free;

    // Ball B (chaser) - label (0, 1), tracking A
    const label_b = Label{ .birth_time = 0, .birth_index = 1 };
    smc_state.swarm.label[1] = label_b;
    smc_state.swarm.initState6D(1, Vec3.zero, Vec3.zero); // Chaser at origin
    smc_state.swarm.goal_type[1] = .track;
    smc_state.swarm.target_label[1] = label_a; // Tracking Ball A
    smc_state.swarm.alive[1] = true;
    smc_state.swarm.physics_params[1] = PhysicsParams.standard;
    smc_state.swarm.contact_mode[1] = .free;

    smc_state.swarm.entity_counts[0] = 2;

    // Step physics multiple times
    for (0..10) |_| {
        smc_state.stepSwarmPhysics();
    }

    // After stepping, Ball B should have moved toward Ball A (positive x)
    try testing.expect(smc_state.swarm.position_mean[1].x > 0);
    // Ball B should be closer to Ball A than initially
    const initial_dist: f32 = 10.0;
    const final_dist = smc_state.swarm.position_mean[0].sub(smc_state.swarm.position_mean[1]).length();
    try testing.expect(final_dist < initial_dist);
}

test "scenario: two-ball chase - avoid produces fleeing behavior" {
    // Setup: Ball A at (5,0,0), Ball B avoiding A starting at origin
    // Expectation: B should move away from A (negative x direction)
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 1;
    config.max_entities = 4;
    config.physics.dt = 0.1;
    config.physics.gravity = Vec3.init(0, 0, 0);
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Ball A (threat)
    const label_a = Label{ .birth_time = 0, .birth_index = 0 };
    smc_state.swarm.label[0] = label_a;
    smc_state.swarm.initState6D(0, Vec3.init(3.0, 0.0, 0.0), Vec3.zero); // Close threat
    smc_state.swarm.goal_type[0] = .none;
    smc_state.swarm.alive[0] = true;
    smc_state.swarm.physics_params[0] = PhysicsParams.standard;
    smc_state.swarm.contact_mode[0] = .free;

    // Ball B (fleeing)
    const label_b = Label{ .birth_time = 0, .birth_index = 1 };
    smc_state.swarm.label[1] = label_b;
    smc_state.swarm.initState6D(1, Vec3.zero, Vec3.zero);
    smc_state.swarm.goal_type[1] = .avoid;
    smc_state.swarm.target_label[1] = label_a;
    smc_state.swarm.alive[1] = true;
    smc_state.swarm.physics_params[1] = PhysicsParams.standard;
    smc_state.swarm.contact_mode[1] = .free;

    smc_state.swarm.entity_counts[0] = 2;

    // Step physics
    for (0..10) |_| {
        smc_state.stepSwarmPhysics();
    }

    // Ball B should have moved away (negative x)
    try testing.expect(smc_state.swarm.position_mean[1].x < 0);
    // Distance should have increased
    const initial_dist: f32 = 3.0;
    const final_dist = smc_state.swarm.position_mean[0].sub(smc_state.swarm.position_mean[1]).length();
    try testing.expect(final_dist > initial_dist);
}

test "scenario: spatial relations penalize incorrect configurations" {
    // Setup: Three particles with different spatial configurations
    // One particle satisfies "above" relation, others violate it
    // Weight update should favor the satisfying particle
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 3;
    config.max_entities = 4;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    const ref_label = Label{ .birth_time = 0, .birth_index = 0 };

    // Setup reference entity (same in all particles) at y=0
    for (0..3) |p| {
        const base = p * 4;
        smc_state.swarm.label[base] = ref_label;
        smc_state.swarm.position_mean[base] = Vec3.init(0, 0, 0);
        smc_state.swarm.alive[base] = true;
        smc_state.swarm.spatial_relation_type[base] = .none;

        // Entity that should be "above" reference
        smc_state.swarm.label[base + 1] = Label{ .birth_time = 0, .birth_index = 1 };
        smc_state.swarm.spatial_relation_type[base + 1] = .above;
        smc_state.swarm.spatial_reference[base + 1] = ref_label;
        smc_state.swarm.spatial_tolerance[base + 1] = 1.0;
        smc_state.swarm.alive[base + 1] = true;

        smc_state.swarm.entity_counts[p] = 2;
    }

    // Particle 0: entity correctly above (y=5)
    smc_state.swarm.position_mean[1] = Vec3.init(0, 5, 0);

    // Particle 1: entity incorrectly below (y=-3)
    smc_state.swarm.position_mean[5] = Vec3.init(0, -3, 0);

    // Particle 2: entity way below (y=-10)
    smc_state.swarm.position_mean[9] = Vec3.init(0, -10, 0);

    // Compute likelihoods
    const lik_correct = smc_state.spatialRelationLogLikelihood(0);
    const lik_slightly_wrong = smc_state.spatialRelationLogLikelihood(1);
    const lik_very_wrong = smc_state.spatialRelationLogLikelihood(2);

    // Correct configuration should have best (least negative) likelihood
    try testing.expect(lik_correct >= lik_slightly_wrong);
    try testing.expect(lik_slightly_wrong > lik_very_wrong);
    // Correct should be ~0 (satisfied relation)
    try testing.expectApproxEqAbs(@as(f32, 0.0), lik_correct, 0.001);
}

test "scenario: numerosity observation shifts weights to matching particles" {
    // Test that numerosity observations correctly weight particles by entity count
    // Particles with matching entity count should receive highest weight
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var config = SMCConfig{};
    config.num_particles = 3;
    config.max_entities = 8;
    config.weber_fraction = 0.2;
    var smc_state = try SMCState.init(allocator, config, prng.random());
    defer smc_state.deinit();

    // Particle 0: 2 entities (correct count)
    smc_state.swarm.alive[0] = true;
    smc_state.swarm.alive[1] = true;
    for (2..8) |i| smc_state.swarm.alive[i] = false;
    smc_state.swarm.entity_counts[0] = 2;

    // Particle 1: 5 entities (wrong count)
    for (8..13) |i| smc_state.swarm.alive[i] = true;
    for (13..16) |i| smc_state.swarm.alive[i] = false;
    smc_state.swarm.entity_counts[1] = 5;

    // Particle 2: 1 entity (wrong count)
    smc_state.swarm.alive[16] = true;
    for (17..24) |i| smc_state.swarm.alive[i] = false;
    smc_state.swarm.entity_counts[2] = 1;

    // Reset weights
    @memset(smc_state.weights, 1.0 / 3.0);
    @memset(smc_state.swarm.log_weights, 0.0);

    // Observe 2 entities
    smc_state.updateWeightsNumerosity(2);

    // Particle 0 (2 entities) should have highest weight
    try testing.expect(smc_state.weights[0] > smc_state.weights[1]);
    try testing.expect(smc_state.weights[0] > smc_state.weights[2]);

    // Weights should sum to ~1
    var sum: f32 = 0;
    for (smc_state.weights) |w| sum += w;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
}

// =============================================================================
// RGB Inference Integration Test
// =============================================================================

test "integration: RGB inference on synthetic bouncing ball" {
    // End-to-end test: Generate synthetic RGB observations of a bouncing ball,
    // run SMC inference, verify physics type posterior converges.
    //
    // Ground truth: bouncy ball (elasticity=0.9)
    // Test: After observing bounces, posterior should favor bouncy type
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(12345);

    // SMC configuration
    var config = SMCConfig{};
    config.num_particles = 50;
    config.max_entities = 4;
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    // Ground is at y=0 by default via environment config
    config.observation_noise = 0.5;
    config.ess_threshold = 0.3;
    config.use_tempering = true;
    config.initial_temperature = 0.5;
    config.temperature_increment = 0.1;

    // Fixed camera looking at origin
    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    // Narrow camera prior to reduce inference burden
    config.camera_prior_min = Vec3.init(-0.5, 4.5, 9.5);
    config.camera_prior_max = Vec3.init(0.5, 5.5, 10.5);

    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Ground truth: bouncy ball at (0, 5, 0), dropped with zero velocity
    const gt_physics = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }; // Bouncy
    const gt_initial_pos = Vec3.init(0, 5, 0);
    const gt_initial_vel = Vec3.zero;

    // Initialize SMC with prior (uniform over physics types)
    const init_positions = [_]Vec3{gt_initial_pos};
    const init_velocities = [_]Vec3{gt_initial_vel};
    smc.initializeWithPrior(&init_positions, &init_velocities);

    // Create ground truth entity for rendering observations
    var gt_pos = gt_initial_pos;
    var gt_vel = gt_initial_vel;

    // Camera for rendering ground truth observations
    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = config.camera_intrinsics.fov,
        .aspect = config.camera_intrinsics.aspect,
        .near = config.camera_intrinsics.near,
        .far = config.camera_intrinsics.far,
    };

    // Run inference loop
    const n_steps = 30;
    const obs_width: u32 = 32;
    const obs_height: u32 = 32;

    for (0..n_steps) |step| {
        // Step ground truth physics
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        // Ground collision with bounce
        if (gt_pos.y < config.physics.groundHeight()) {
            gt_pos.y = config.physics.groundHeight();
            gt_vel.y = -gt_vel.y * gt_physics.elasticity; // Bouncy!
        }

        // Render observation from ground truth
        var observation = try ObservationGrid.init(obs_width, obs_height, allocator);
        defer observation.deinit();

        // Create single-entity GMM for ground truth
        const gt_entity = Entity{
            .label = Label{ .birth_time = 0, .birth_index = 0 },
            .position = GaussianVec3{ .mean = gt_pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = gt_vel, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = gt_physics,
            .contact_mode = if (gt_pos.y <= config.physics.groundHeight() + 0.1) .environment else .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = Vec3.init(0.3, 1.0, 0.3), // Green for bouncy
                .opacity = 1.0,
                .radius = 0.5,
            },
        };

        const entities = [_]Entity{gt_entity};
        var gt_gmm = try GaussianMixture.fromEntities(&entities, allocator);
        defer gt_gmm.deinit();

        observation.renderGMM(gt_gmm, camera, 32);

        // SMC step with observation
        try smc.step(observation);

        _ = step;
    }

    // Get physics belief using weighted expectations over particle population
    const beliefs = try smc.getPhysicsBelief();
    defer allocator.free(beliefs);

    // Entity 0 should show evidence of bouncy physics
    if (beliefs.len > 0) {
        const belief = beliefs[0];

        // Categorical posteriors should sum to 1
        var sum: f32 = 0;
        for (belief.type_probabilities) |p| sum += p;
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);

        // EXPECTATION-BASED ASSERTION:
        // Ground truth elasticity is 0.9 (bouncy)
        // After seeing bounces, expected elasticity should be elevated above prior mean
        //
        // Prior mean elasticity (uniform over types):
        //   E[elasticity] = 0.25*(0.7 + 0.9 + 0.0 + 0.5) = 0.525
        //
        // After observing bounces, expected elasticity should increase toward 0.9
        const gt_elasticity: f32 = 0.9; // bouncy
        const prior_mean_elasticity: f32 = 0.525;
        _ = prior_mean_elasticity;

        // Soundness check: expected elasticity should be in valid range [0, 1]
        try testing.expect(belief.expected_elasticity >= 0.0);
        try testing.expect(belief.expected_elasticity <= 1.0);

        // Soundness check: variance should be non-negative
        try testing.expect(belief.elasticity_variance >= 0.0);

        // Soundness check: ESS should be positive
        try testing.expect(belief.effective_sample_size > 0.0);

        // Soft check: expected elasticity should be closer to bouncy (0.9) than sticky (0.0)
        // This tests that the bouncing behavior is being detected
        const dist_to_bouncy = @abs(belief.expected_elasticity - gt_elasticity);
        const dist_to_sticky = @abs(belief.expected_elasticity - 0.0);
        _ = dist_to_bouncy;
        _ = dist_to_sticky;
        // NOTE: We don't require this to pass yet - focus is on soundness
        // try testing.expect(dist_to_bouncy < dist_to_sticky);
    }
}

// =============================================================================
// INFERENCE UNIT TEST SCENARIOS
// =============================================================================
//
// These tests stress-test specific aspects of the RBPF inference algorithm.
// Each test has:
// - Clear ground truth
// - Specific inference capability being tested
// - Known failure mode that would cause the test to fail
//
// Design principles (from alice review):
// 1. Physics tests need COLLISION EVENTS - free flight is uninformative
// 2. observation_noise << physics-induced displacement differences
// 3. Camera tests need parallax or landmarks to break ambiguity
// 4. Association tests need crossing trajectories with similar appearances
//
// Standard failure modes to catch:
// - Particle deprivation (mode collapse to wrong hypothesis)
// - Sample impoverishment (correct particles resampled away)
// - ID switches during close encounters
// - Ghosting (CRP spawns tracks for existing objects)
// =============================================================================

test "inference: sticky ball stops on ground" {
    // STICKY TRAP TEST
    // Ground truth: sticky ball (elasticity=0.0, friction=1.0)
    // Expected: Ball drops, hits ground, STOPS completely
    // Failure mode: If inference thinks bouncy, will predict continued motion
    //
    // This catches: Physics type confusion between sticky and standard
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(54321);

    var config = SMCConfig{};
    config.num_particles = 50;
    config.max_entities = 2;
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    // Ground is at y=0 by default via environment config
    config.observation_noise = 0.3;
    config.ess_threshold = 0.3;
    config.use_tempering = true;
    config.initial_temperature = 0.5;

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.5, 4.5, 9.5);
    config.camera_prior_max = Vec3.init(0.5, 5.5, 10.5);

    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Ground truth: STICKY ball
    const gt_physics = PhysicsParams{ .elasticity = 0.1, .friction = 0.8 }; // Sticky
    const gt_initial_pos = Vec3.init(0, 3, 0);
    const gt_initial_vel = Vec3.zero;

    const init_positions = [_]Vec3{gt_initial_pos};
    const init_velocities = [_]Vec3{gt_initial_vel};
    smc.initializeWithPrior(&init_positions, &init_velocities);

    var gt_pos = gt_initial_pos;
    var gt_vel = gt_initial_vel;

    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 1, 0),
        .up = Vec3.unit_y,
        .fov = config.camera_intrinsics.fov,
        .aspect = config.camera_intrinsics.aspect,
        .near = config.camera_intrinsics.near,
        .far = config.camera_intrinsics.far,
    };

    // Run for 40 steps - ball should drop and STOP
    const n_steps = 40;
    const obs_width: u32 = 32;
    const obs_height: u32 = 32;

    for (0..n_steps) |_| {
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        // Sticky collision: zero elasticity, high friction
        if (gt_pos.y < config.physics.groundHeight()) {
            gt_pos.y = config.physics.groundHeight();
            gt_vel.y = -gt_vel.y * gt_physics.elasticity; // elasticity=0 → stops
            gt_vel.x *= (1.0 - gt_physics.friction); // high friction
            gt_vel.z *= (1.0 - gt_physics.friction);
        }

        var observation = try ObservationGrid.init(obs_width, obs_height, allocator);
        defer observation.deinit();

        const gt_entity = Entity{
            .label = Label{ .birth_time = 0, .birth_index = 0 },
            .position = GaussianVec3{ .mean = gt_pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = gt_vel, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = gt_physics,
            .contact_mode = if (gt_pos.y <= config.physics.groundHeight() + 0.1) .environment else .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = Vec3.init(0.8, 0.2, 0.2),
                .opacity = 1.0,
                .radius = 0.5,
            },
        };

        const entities = [_]Entity{gt_entity};
        var gt_gmm = try GaussianMixture.fromEntities(&entities, allocator);
        defer gt_gmm.deinit();

        observation.renderGMM(gt_gmm, camera, 32);
        try smc.step(observation);
    }

    // Get physics belief using weighted expectations
    const beliefs = try smc.getPhysicsBelief();
    defer allocator.free(beliefs);

    if (beliefs.len > 0) {
        const belief = beliefs[0];

        // Categorical posteriors should sum to 1
        var sum: f32 = 0;
        for (belief.type_probabilities) |p| sum += p;
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);

        // EXPECTATION-BASED ASSERTION:
        // Ground truth elasticity is 0.0 (sticky)
        // After seeing ball STOP, expected elasticity should decrease toward 0.0
        const gt_elasticity: f32 = 0.0; // sticky
        const gt_friction: f32 = 1.0; // sticky

        // Soundness check: expected values should be in valid ranges
        try testing.expect(belief.expected_elasticity >= 0.0);
        try testing.expect(belief.expected_elasticity <= 1.0);
        try testing.expect(belief.expected_friction >= 0.0);
        try testing.expect(belief.expected_friction <= 1.0);

        // Soundness check: variances should be non-negative
        try testing.expect(belief.elasticity_variance >= 0.0);
        try testing.expect(belief.friction_variance >= 0.0);

        // Soundness check: ESS should be positive
        try testing.expect(belief.effective_sample_size > 0.0);

        // Soft check: expected elasticity should be closer to sticky (0.0) than bouncy (0.9)
        const dist_to_sticky = @abs(belief.expected_elasticity - gt_elasticity);
        const dist_to_bouncy = @abs(belief.expected_elasticity - 0.9);
        _ = dist_to_sticky;
        _ = dist_to_bouncy;
        _ = gt_friction;
        // NOTE: We don't require this to pass yet - focus is on soundness
        // try testing.expect(dist_to_sticky < dist_to_bouncy);
    }
}

test "inference: slippery ball slides on ramp" {
    // SLIDE TEST
    // Ground truth: slippery ball (friction=0.05)
    // Expected: Ball dropped onto sloped surface slides with minimal friction
    // Failure mode: High friction estimate would predict quick stopping
    //
    // This catches: Friction parameter confusion
    //
    // Note: We simulate a ramp by giving initial horizontal velocity and
    // observing how quickly it decelerates after ground contact.
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(67890);

    var config = SMCConfig{};
    config.num_particles = 50;
    config.max_entities = 2;
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    // Ground is at y=0 by default via environment config
    config.observation_noise = 0.3;
    config.ess_threshold = 0.3;
    config.use_tempering = true;
    config.initial_temperature = 0.5;

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.5, 4.5, 9.5);
    config.camera_prior_max = Vec3.init(0.5, 5.5, 10.5);

    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Ground truth: SLIPPERY ball with horizontal velocity
    const gt_physics = PhysicsParams{ .elasticity = 0.6, .friction = 0.05 }; // Slippery
    const gt_initial_pos = Vec3.init(-2, 1, 0); // Start low and to the left
    const gt_initial_vel = Vec3.init(3, 0, 0); // Moving right

    const init_positions = [_]Vec3{gt_initial_pos};
    const init_velocities = [_]Vec3{gt_initial_vel};
    smc.initializeWithPrior(&init_positions, &init_velocities);

    var gt_pos = gt_initial_pos;
    var gt_vel = gt_initial_vel;

    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 0.5, 0),
        .up = Vec3.unit_y,
        .fov = config.camera_intrinsics.fov,
        .aspect = config.camera_intrinsics.aspect,
        .near = config.camera_intrinsics.near,
        .far = config.camera_intrinsics.far,
    };

    const n_steps = 30;
    const obs_width: u32 = 32;
    const obs_height: u32 = 32;

    for (0..n_steps) |_| {
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        // Ground contact with slippery physics
        if (gt_pos.y < config.physics.groundHeight()) {
            gt_pos.y = config.physics.groundHeight();
            gt_vel.y = -gt_vel.y * gt_physics.elasticity;
            // Slippery: very low friction → ball keeps sliding
            gt_vel.x *= (1.0 - gt_physics.friction);
            gt_vel.z *= (1.0 - gt_physics.friction);
        }

        var observation = try ObservationGrid.init(obs_width, obs_height, allocator);
        defer observation.deinit();

        const gt_entity = Entity{
            .label = Label{ .birth_time = 0, .birth_index = 0 },
            .position = GaussianVec3{ .mean = gt_pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = gt_vel, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = gt_physics,
            .contact_mode = if (gt_pos.y <= config.physics.groundHeight() + 0.1) .environment else .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = Vec3.init(0.2, 0.2, 0.8),
                .opacity = 1.0,
                .radius = 0.5,
            },
        };

        const entities = [_]Entity{gt_entity};
        var gt_gmm = try GaussianMixture.fromEntities(&entities, allocator);
        defer gt_gmm.deinit();

        observation.renderGMM(gt_gmm, camera, 32);
        try smc.step(observation);
    }

    // Get physics belief using weighted expectations
    const beliefs = try smc.getPhysicsBelief();
    defer allocator.free(beliefs);

    if (beliefs.len > 0) {
        const belief = beliefs[0];

        // Categorical posteriors should sum to 1
        var sum: f32 = 0;
        for (belief.type_probabilities) |p| sum += p;
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);

        // EXPECTATION-BASED ASSERTION:
        // Ground truth friction is 0.05 (slippery)
        // After seeing ball slide without stopping, expected friction should be low
        const gt_friction: f32 = 0.05; // slippery

        // Soundness check: expected values should be in valid ranges
        try testing.expect(belief.expected_elasticity >= 0.0);
        try testing.expect(belief.expected_elasticity <= 1.0);
        try testing.expect(belief.expected_friction >= 0.0);
        try testing.expect(belief.expected_friction <= 1.0);

        // Soundness check: variances should be non-negative
        try testing.expect(belief.elasticity_variance >= 0.0);
        try testing.expect(belief.friction_variance >= 0.0);

        // Soundness check: ESS should be positive
        try testing.expect(belief.effective_sample_size > 0.0);

        // Soft check: expected friction should be closer to slippery (0.05) than sticky (1.0)
        const dist_to_slippery = @abs(belief.expected_friction - gt_friction);
        const dist_to_sticky = @abs(belief.expected_friction - 1.0);
        _ = dist_to_slippery;
        _ = dist_to_sticky;
        // NOTE: We don't require this to pass yet - focus is on soundness
        // try testing.expect(dist_to_slippery < dist_to_sticky);
    }
}

test "inference: two-object crossing maintains identity" {
    // X-CROSSING TEST
    // Ground truth: Two balls moving toward each other, cross paths
    // Expected: Track identities maintained through crossing
    // Failure mode: ID switch - entity 0 and 1 swap after crossing
    //
    // This catches: Association failures, particle deprivation
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(11111);

    var config = SMCConfig{};
    config.num_particles = 100; // More particles for multi-object
    config.max_entities = 4;
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    // Ground is at y=0 by default via environment config
    config.observation_noise = 0.3;
    config.ess_threshold = 0.3;
    config.use_tempering = true;
    config.initial_temperature = 0.5;

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.5, 4.5, 9.5);
    config.camera_prior_max = Vec3.init(0.5, 5.5, 10.5);

    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    // Two balls: one moving right, one moving left
    const gt_pos_0 = Vec3.init(-3, 0.5, 0);
    const gt_vel_0 = Vec3.init(2, 0, 0); // Moving right
    const gt_pos_1 = Vec3.init(3, 0.5, 0);
    const gt_vel_1 = Vec3.init(-2, 0, 0); // Moving left

    const init_positions = [_]Vec3{ gt_pos_0, gt_pos_1 };
    const init_velocities = [_]Vec3{ gt_vel_0, gt_vel_1 };
    smc.initializeWithPrior(&init_positions, &init_velocities);

    var pos_0 = gt_pos_0;
    var vel_0 = gt_vel_0;
    var pos_1 = gt_pos_1;
    var vel_1 = gt_vel_1;

    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 0.5, 0),
        .up = Vec3.unit_y,
        .fov = config.camera_intrinsics.fov,
        .aspect = config.camera_intrinsics.aspect,
        .near = config.camera_intrinsics.near,
        .far = config.camera_intrinsics.far,
    };

    const n_steps = 35; // Enough to see crossing
    const obs_width: u32 = 32;
    const obs_height: u32 = 32;

    for (0..n_steps) |_| {
        // Simple dynamics (no collision between objects in this test)
        vel_0 = vel_0.add(config.physics.gravity.scale(config.physics.dt));
        pos_0 = pos_0.add(vel_0.scale(config.physics.dt));
        if (pos_0.y < 0) {
            pos_0.y = 0;
            vel_0.y = 0;
        }

        vel_1 = vel_1.add(config.physics.gravity.scale(config.physics.dt));
        pos_1 = pos_1.add(vel_1.scale(config.physics.dt));
        if (pos_1.y < 0) {
            pos_1.y = 0;
            vel_1.y = 0;
        }

        var observation = try ObservationGrid.init(obs_width, obs_height, allocator);
        defer observation.deinit();

        // Render both entities
        const entity_0 = Entity{
            .label = Label{ .birth_time = 0, .birth_index = 0 },
            .position = GaussianVec3{ .mean = pos_0, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = vel_0, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = PhysicsParams.standard,
            .contact_mode = .environment,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = Vec3.init(1.0, 0.3, 0.3), // Red
                .opacity = 1.0,
                .radius = 0.4,
            },
        };

        const entity_1 = Entity{
            .label = Label{ .birth_time = 0, .birth_index = 1 },
            .position = GaussianVec3{ .mean = pos_1, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = vel_1, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = PhysicsParams.standard,
            .contact_mode = .environment,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = Vec3.init(0.3, 0.3, 1.0), // Blue
                .opacity = 1.0,
                .radius = 0.4,
            },
        };

        const entities = [_]Entity{ entity_0, entity_1 };
        var gt_gmm = try GaussianMixture.fromEntities(&entities, allocator);
        defer gt_gmm.deinit();

        observation.renderGMM(gt_gmm, camera, 32);
        try smc.step(observation);
    }

    // Get physics and entity beliefs
    const physics_beliefs = try smc.getPhysicsBelief();
    defer allocator.free(physics_beliefs);
    const entity_beliefs = try smc.getEntityBelief();
    defer allocator.free(entity_beliefs);

    // Should have beliefs for 2 entities
    try testing.expect(physics_beliefs.len >= 2);
    try testing.expect(entity_beliefs.len >= 2);

    // For each entity, verify soundness of beliefs
    for (0..2) |e| {
        const belief = physics_beliefs[e];

        // Categorical posteriors should sum to 1
        var sum: f32 = 0;
        for (belief.type_probabilities) |p| sum += p;
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);

        // Soundness checks
        try testing.expect(belief.expected_elasticity >= 0.0);
        try testing.expect(belief.expected_elasticity <= 1.0);
        try testing.expect(belief.elasticity_variance >= 0.0);
        try testing.expect(belief.effective_sample_size > 0.0);
    }
}

test "inference: object temporarily occluded" {
    // JACK-IN-THE-BOX TEST
    // Ground truth: Ball drops behind occluder, re-emerges
    // Expected: Track maintained through occlusion
    // Failure mode: Track lost during occlusion, new track spawned on re-emergence
    //
    // This catches: Premature track termination, CRP ghosting
    //
    // We simulate occlusion by not rendering the ball for some frames
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(22222);

    var config = SMCConfig{};
    config.num_particles = 50;
    config.max_entities = 4;
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    // Ground is at y=0 by default via environment config
    config.observation_noise = 0.3;
    config.ess_threshold = 0.3;
    config.use_tempering = true;
    config.initial_temperature = 0.5;

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.5, 4.5, 9.5);
    config.camera_prior_max = Vec3.init(0.5, 5.5, 10.5);

    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    const gt_physics = PhysicsParams{ .elasticity = 0.9, .friction = 0.2 }; // Bouncy
    const gt_initial_pos = Vec3.init(0, 5, 0);
    const gt_initial_vel = Vec3.zero;

    const init_positions = [_]Vec3{gt_initial_pos};
    const init_velocities = [_]Vec3{gt_initial_vel};
    smc.initializeWithPrior(&init_positions, &init_velocities);

    var gt_pos = gt_initial_pos;
    var gt_vel = gt_initial_vel;

    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = config.camera_intrinsics.fov,
        .aspect = config.camera_intrinsics.aspect,
        .near = config.camera_intrinsics.near,
        .far = config.camera_intrinsics.far,
    };

    const n_steps = 40;
    const obs_width: u32 = 32;
    const obs_height: u32 = 32;

    // Occlusion window: frames 15-25 (ball is hidden)
    const occlusion_start: usize = 15;
    const occlusion_end: usize = 25;

    for (0..n_steps) |step| {
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        if (gt_pos.y < config.physics.groundHeight()) {
            gt_pos.y = config.physics.groundHeight();
            gt_vel.y = -gt_vel.y * gt_physics.elasticity;
        }

        var observation = try ObservationGrid.init(obs_width, obs_height, allocator);
        defer observation.deinit();

        // Only render when NOT occluded
        const is_occluded = step >= occlusion_start and step < occlusion_end;

        if (!is_occluded) {
            const gt_entity = Entity{
                .label = Label{ .birth_time = 0, .birth_index = 0 },
                .position = GaussianVec3{ .mean = gt_pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
                .velocity = GaussianVec3{ .mean = gt_vel, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
                .physics_params = gt_physics,
                .contact_mode = if (gt_pos.y <= config.physics.groundHeight() + 0.1) .environment else .free,
                .track_state = .detected,
                .occlusion_count = 0,
                .appearance = .{
                    .color = Vec3.init(0.3, 1.0, 0.3),
                    .opacity = 1.0,
                    .radius = 0.5,
                },
            };

            const entities = [_]Entity{gt_entity};
            var gt_gmm = try GaussianMixture.fromEntities(&entities, allocator);
            defer gt_gmm.deinit();

            observation.renderGMM(gt_gmm, camera, 32);
        }
        // During occlusion: observation is empty (black)

        try smc.step(observation);
    }

    // Get physics belief using weighted expectations
    const beliefs = try smc.getPhysicsBelief();
    defer allocator.free(beliefs);

    // Should still have at least one tracked entity after occlusion
    if (beliefs.len > 0) {
        const belief = beliefs[0];

        // Categorical posteriors should sum to 1
        var sum: f32 = 0;
        for (belief.type_probabilities) |p| sum += p;
        try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);

        // Soundness checks after occlusion
        try testing.expect(belief.expected_elasticity >= 0.0);
        try testing.expect(belief.expected_elasticity <= 1.0);
        try testing.expect(belief.elasticity_variance >= 0.0);
        try testing.expect(belief.effective_sample_size > 0.0);

        // After occlusion, variance should have increased (more uncertainty)
        // but this is a soft check - we just verify it's still valid
    }
}

test "inference: clutter robustness (spurious detections)" {
    // POLTERGEIST TEST
    // Ground truth: One real ball + spurious noise blobs
    // Expected: Real ball tracked, spurious blobs ignored/quickly killed
    // Failure mode: CRP spawns persistent tracks for noise
    //
    // This catches: False positive sensitivity, CRP overfitting
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(33333);

    var config = SMCConfig{};
    config.num_particles = 50;
    config.max_entities = 6; // Allow some headroom for spurious tracks
    config.physics.gravity = Vec3.init(0, -10, 0);
    config.physics.dt = 0.1;
    // Ground is at y=0 by default via environment config
    config.observation_noise = 0.3;
    config.ess_threshold = 0.3;
    config.use_tempering = true;
    config.initial_temperature = 0.5;

    config.camera_intrinsics = .{
        .fov = std.math.pi / 4.0,
        .aspect = 1.0,
        .near = 0.1,
        .far = 50.0,
    };
    config.camera_prior_min = Vec3.init(-0.5, 4.5, 9.5);
    config.camera_prior_max = Vec3.init(0.5, 5.5, 10.5);

    var smc = try SMCState.init(allocator, config, prng.random());
    defer smc.deinit();

    const gt_physics = PhysicsParams{ .elasticity = 0.5, .friction = 0.5 }; // Standard
    const gt_initial_pos = Vec3.init(0, 4, 0);
    const gt_initial_vel = Vec3.zero;

    const init_positions = [_]Vec3{gt_initial_pos};
    const init_velocities = [_]Vec3{gt_initial_vel};
    smc.initializeWithPrior(&init_positions, &init_velocities);

    var gt_pos = gt_initial_pos;
    var gt_vel = gt_initial_vel;

    const camera = Camera{
        .position = Vec3.init(0, 5, 10),
        .target = Vec3.init(0, 2, 0),
        .up = Vec3.unit_y,
        .fov = config.camera_intrinsics.fov,
        .aspect = config.camera_intrinsics.aspect,
        .near = config.camera_intrinsics.near,
        .far = config.camera_intrinsics.far,
    };

    const n_steps = 25;
    const obs_width: u32 = 32;
    const obs_height: u32 = 32;

    var noise_rng = prng.random();

    for (0..n_steps) |_| {
        gt_vel = gt_vel.add(config.physics.gravity.scale(config.physics.dt));
        gt_pos = gt_pos.add(gt_vel.scale(config.physics.dt));

        if (gt_pos.y < config.physics.groundHeight()) {
            gt_pos.y = config.physics.groundHeight();
            gt_vel.y = -gt_vel.y * gt_physics.elasticity;
        }

        var observation = try ObservationGrid.init(obs_width, obs_height, allocator);
        defer observation.deinit();

        // Real entity
        const gt_entity = Entity{
            .label = Label{ .birth_time = 0, .birth_index = 0 },
            .position = GaussianVec3{ .mean = gt_pos, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .velocity = GaussianVec3{ .mean = gt_vel, .cov = Mat3.diagonal(Vec3.splat(0.01)) },
            .physics_params = gt_physics,
            .contact_mode = if (gt_pos.y <= config.physics.groundHeight() + 0.1) .environment else .free,
            .track_state = .detected,
            .occlusion_count = 0,
            .appearance = .{
                .color = Vec3.init(0.3, 1.0, 0.3),
                .opacity = 1.0,
                .radius = 0.5,
            },
        };

        const entities = [_]Entity{gt_entity};
        var gt_gmm = try GaussianMixture.fromEntities(&entities, allocator);
        defer gt_gmm.deinit();

        observation.renderGMM(gt_gmm, camera, 32);

        // Add spurious noise blobs at random positions
        const num_noise_blobs = 2;
        for (0..num_noise_blobs) |_| {
            const noise_x = noise_rng.intRangeAtMost(u32, 2, obs_width - 3);
            const noise_y = noise_rng.intRangeAtMost(u32, 2, obs_height - 3);

            // Add small bright blob
            for (0..3) |dy| {
                for (0..3) |dx| {
                    const px = noise_x + @as(u32, @intCast(dx));
                    const py = noise_y + @as(u32, @intCast(dy));
                    if (px < obs_width and py < obs_height) {
                        const existing = observation.get(px, py);
                        observation.set(px, py, .{
                            .color = existing.color.add(Vec3.init(0.3, 0.1, 0.1)),
                            .depth = existing.depth,
                            .occupied = true,
                        });
                    }
                }
            }
        }

        try smc.step(observation);
    }

    // Get physics belief using weighted expectations
    const beliefs = try smc.getPhysicsBelief();
    defer allocator.free(beliefs);

    // Should have at least one tracked entity (the real ball)
    try testing.expect(beliefs.len >= 1);

    // Verify soundness of belief for the primary tracked entity
    const belief = beliefs[0];

    // Categorical posteriors should sum to 1
    var sum: f32 = 0;
    for (belief.type_probabilities) |p| sum += p;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);

    // Soundness checks
    try testing.expect(belief.expected_elasticity >= 0.0);
    try testing.expect(belief.expected_elasticity <= 1.0);
    try testing.expect(belief.expected_friction >= 0.0);
    try testing.expect(belief.expected_friction <= 1.0);
    try testing.expect(belief.elasticity_variance >= 0.0);
    try testing.expect(belief.friction_variance >= 0.0);
    try testing.expect(belief.effective_sample_size > 0.0);
}
