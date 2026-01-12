const std = @import("std");
const math = @import("math.zig");
const types = @import("types.zig");
const dynamics = @import("dynamics.zig");
const gmm = @import("gmm.zig");

const Vec3 = math.Vec3;
const Mat3 = math.Mat3;
const Entity = types.Entity;
const PhysicsType = types.PhysicsType;
const PhysicsConfig = types.PhysicsConfig;
const Camera = types.Camera;
const GaussianVec3 = types.GaussianVec3;
const Label = types.Label;
const ObservationGrid = gmm.ObservationGrid;
const GaussianMixture = gmm.GaussianMixture;

// =============================================================================
// SMC Configuration
// =============================================================================

/// Configuration for SMC inference
pub const SMCConfig = struct {
    /// Number of particles
    num_particles: u32 = 100,

    /// ESS threshold for resampling (as fraction of num_particles)
    ess_threshold: f32 = 0.5,

    /// Observation noise for likelihood (controls sharpness)
    observation_noise: f32 = 0.3,

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

    /// Use uniform colors for all entities (disables color-based inference)
    /// When true, inference relies solely on dynamics behavior (bounce/slide/stick)
    use_uniform_colors: bool = false,

    /// Uniform color to use when use_uniform_colors is true
    uniform_color: Vec3 = Vec3.init(0.8, 0.8, 0.8),

    /// Get color for physics type (respects use_uniform_colors setting)
    pub fn colorForPhysicsType(self: SMCConfig, ptype: PhysicsType) Vec3 {
        if (self.use_uniform_colors) {
            return self.uniform_color;
        }
        return switch (ptype) {
            .standard => Vec3.init(1.0, 0.3, 0.3),
            .bouncy => Vec3.init(0.3, 1.0, 0.3),
            .sticky => Vec3.init(0.3, 0.3, 1.0),
            .slippery => Vec3.init(1.0, 1.0, 0.3),
        };
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
pub const Particle = struct {
    /// Entity states (continuous: position/velocity as Gaussians)
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
            .entities = .empty,
            .previous_states = .empty,
            .log_weight = 0,
            .allocator = allocator,
        };
    }

    /// Deep copy from another particle
    pub fn copyFrom(self: *Particle, other: Particle) !void {
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

    /// Step physics for all entities in this particle
    pub fn stepPhysics(self: *Particle, config: PhysicsConfig, rng: std.Random) !void {
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
                dynamics.entityPhysicsStep(entity, config, rng);
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
// SMC State
// =============================================================================

/// SMC inference state
pub const SMCState = struct {
    /// Particle population
    particles: []Particle,

    /// Normalized weights (for resampling)
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

    /// Initialize SMC state
    pub fn init(
        allocator: std.mem.Allocator,
        config: SMCConfig,
        rng: std.Random,
    ) !SMCState {
        const particles = try allocator.alloc(Particle, config.num_particles);
        errdefer allocator.free(particles);

        for (particles) |*p| {
            p.* = Particle.init(allocator);
        }

        const weights = try allocator.alloc(f32, config.num_particles);
        @memset(weights, 1.0 / @as(f32, @floatFromInt(config.num_particles)));

        return .{
            .particles = particles,
            .weights = weights,
            .temperature = if (config.use_tempering) config.initial_temperature else 1.0,
            .timestep = 0,
            .config = config,
            .allocator = allocator,
            .rng = rng,
        };
    }

    /// Free allocated memory
    pub fn deinit(self: *SMCState) void {
        for (self.particles) |*p| {
            p.deinit();
        }
        self.allocator.free(self.particles);
        self.allocator.free(self.weights);
    }

    /// Initialize particles with prior (entities at given positions with unknown physics)
    pub fn initializeWithPrior(
        self: *SMCState,
        initial_positions: []const Vec3,
        initial_velocities: []const Vec3,
    ) !void {
        const physics_types = [_]PhysicsType{ .standard, .bouncy, .sticky, .slippery };

        for (self.particles) |*particle| {
            particle.entities.clearRetainingCapacity();

            for (initial_positions, initial_velocities, 0..) |pos, vel, idx| {
                // Sample physics type uniformly from prior
                const ptype_idx = self.rng.intRangeAtMost(usize, 0, physics_types.len - 1);
                const ptype = physics_types[ptype_idx];

                const label = Label{
                    .birth_time = 0,
                    .birth_index = @intCast(idx),
                };

                var entity = Entity.initPoint(label, pos, vel, ptype);
                entity.appearance.radius = 0.5;
                entity.appearance.color = self.config.colorForPhysicsType(ptype);

                try particle.addEntity(entity);
            }

            particle.log_weight = 0;
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
        var max_log_w: f32 = -std.math.inf(f32);
        for (self.particles) |p| {
            max_log_w = @max(max_log_w, p.log_weight);
        }

        // Handle edge case: all particles have -inf log weight
        // Fall back to uniform weights
        if (max_log_w == -std.math.inf(f32)) {
            const uniform = 1.0 / @as(f32, @floatFromInt(self.particles.len));
            @memset(self.weights, uniform);
            return;
        }

        // Compute normalized weights
        var sum: f32 = 0;
        for (self.particles, 0..) |p, i| {
            self.weights[i] = @exp(p.log_weight - max_log_w);
            sum += self.weights[i];
        }

        // Normalize (with fallback if sum is zero or NaN)
        if (sum > 0 and !std.math.isNan(sum)) {
            for (self.weights) |*w| {
                w.* /= sum;
            }
        } else {
            const uniform = 1.0 / @as(f32, @floatFromInt(self.particles.len));
            @memset(self.weights, uniform);
        }
    }

    /// Systematic resampling
    pub fn resample(self: *SMCState) !void {
        const n = self.particles.len;
        const n_f: f32 = @floatFromInt(n);

        // Compute cumulative distribution
        var cumsum = try self.allocator.alloc(f32, n);
        defer self.allocator.free(cumsum);

        cumsum[0] = self.weights[0];
        for (1..n) |i| {
            cumsum[i] = cumsum[i - 1] + self.weights[i];
        }

        // Systematic resampling
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

        // Create new particles by copying from ancestors
        var new_particles = try self.allocator.alloc(Particle, n);
        for (new_particles) |*p| {
            p.* = Particle.init(self.allocator);
        }

        for (0..n) |i| {
            try new_particles[i].copyFrom(self.particles[ancestors[i]]);
            new_particles[i].log_weight = 0; // Reset weights after resampling
        }

        // Swap and free old particles
        for (self.particles) |*p| {
            p.deinit();
        }
        self.allocator.free(self.particles);
        self.particles = new_particles;

        // Reset to uniform weights
        @memset(self.weights, 1.0 / n_f);
    }

    /// Compute dynamics log-likelihood for a single entity given physics type
    /// Uses Kalman likelihood: p(x_t | x_{t-1}, type)
    fn dynamicsLogLikelihood(
        self: SMCState,
        entity: Entity,
        prev_state: PreviousState,
        physics_type: PhysicsType,
    ) f32 {
        // Get dynamics matrices for this physics type
        const matrices = dynamics.DynamicsMatrices.forContactMode(
            entity.contact_mode,
            physics_type,
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

    /// Compute observation log-likelihood for a particle
    fn observationLogLikelihood(
        self: SMCState,
        particle: Particle,
        observation: ObservationGrid,
        camera: Camera,
    ) f32 {
        // Create GMM from particle's entities
        var particle_gmm = particle.toGMM() catch return -std.math.inf(f32);
        defer particle_gmm.deinit();

        // Render expected observation
        var expected = ObservationGrid.init(
            observation.width,
            observation.height,
            self.allocator,
        ) catch return -std.math.inf(f32);
        defer expected.deinit();

        expected.renderGMM(particle_gmm, camera, 64);

        // Compute pixel-wise log-likelihood
        var log_lik: f32 = 0;
        const noise = self.config.observation_noise;
        const noise_sq = noise * noise;

        for (0..observation.height) |yi| {
            for (0..observation.width) |xi| {
                const x: u32 = @intCast(xi);
                const y: u32 = @intCast(yi);

                const obs = observation.get(x, y);
                const exp = expected.get(x, y);

                // Color likelihood (Gaussian)
                const color_diff = obs.color.sub(exp.color);
                const color_sq_dist = color_diff.dot(color_diff);
                log_lik += -color_sq_dist / (2.0 * noise_sq);
            }
        }

        return log_lik;
    }

    /// Update particle weights given observation
    pub fn updateWeights(
        self: *SMCState,
        observation: ObservationGrid,
        camera: Camera,
    ) void {
        for (self.particles) |*particle| {
            const log_lik = self.observationLogLikelihood(particle.*, observation, camera);
            // Apply temperature for annealing
            particle.log_weight += self.temperature * log_lik;
        }

        self.normalizeWeights();
    }

    /// Gibbs rejuvenation: resample discrete physics types
    /// Targets: p(type | x_t, x_{t-1}, y_t) ∝ p(y_t|x_t,type) * p(x_t|x_{t-1},type) * p(type)
    pub fn gibbsRejuvenation(
        self: *SMCState,
        observation: ObservationGrid,
        camera: Camera,
    ) void {
        const physics_types = [_]PhysicsType{ .standard, .bouncy, .sticky, .slippery };

        for (self.particles) |*particle| {
            // Skip if no previous states (first step)
            if (particle.previous_states.items.len != particle.entities.items.len) continue;

            for (0..self.config.gibbs_sweeps) |_| {
                for (particle.entities.items, 0..) |*entity, entity_idx| {
                    if (!entity.isAlive()) continue;

                    const prev_state = particle.previous_states.items[entity_idx];

                    // Compute conditional for each physics type
                    // log p(type|...) ∝ log p(y|x,type) + log p(x|x_{-1},type) + log p(type)
                    var log_probs: [4]f32 = undefined;

                    for (physics_types, 0..) |ptype, i| {
                        entity.physics_type = ptype;
                        // Update color to match physics type
                        entity.appearance.color = self.config.colorForPhysicsType(ptype);

                        // Observation likelihood (tempered)
                        const obs_log_lik = self.temperature * self.observationLogLikelihood(
                            particle.*,
                            observation,
                            camera,
                        );

                        // Dynamics likelihood: p(x_t | x_{t-1}, type)
                        const dyn_log_lik = self.dynamicsLogLikelihood(
                            entity.*,
                            prev_state,
                            ptype,
                        );

                        // Prior is uniform (log 0.25 for each type)
                        const prior_log = -1.386294; // log(0.25)

                        log_probs[i] = obs_log_lik + dyn_log_lik + prior_log;
                    }

                    // Sample from conditional (log-domain)
                    const max_log = @max(@max(log_probs[0], log_probs[1]), @max(log_probs[2], log_probs[3]));
                    var probs: [4]f32 = undefined;
                    var sum: f32 = 0;
                    for (log_probs, 0..) |lp, i| {
                        probs[i] = @exp(lp - max_log);
                        sum += probs[i];
                    }

                    // Normalize and sample (with fallback for degenerate case)
                    if (sum <= 0 or std.math.isNan(sum)) {
                        // Fall back to uniform sampling
                        const sampled_idx = self.rng.intRangeAtMost(usize, 0, 3);
                        entity.physics_type = physics_types[sampled_idx];
                    } else {
                        const u = self.rng.float(f32) * sum;
                        var cumsum: f32 = 0;
                        var sampled_idx: usize = 0;
                        for (probs, 0..) |p, i| {
                            cumsum += p;
                            if (u <= cumsum) {
                                sampled_idx = i;
                                break;
                            }
                        }
                        entity.physics_type = physics_types[sampled_idx];
                    }

                    // Update color to match sampled physics type
                    entity.appearance.color = self.config.colorForPhysicsType(entity.physics_type);
                }
            }
        }
    }

    /// Single SMC step: propagate, weight, resample, rejuvenate
    pub fn step(
        self: *SMCState,
        observation: ObservationGrid,
        camera: Camera,
    ) !void {
        // 1. Propagate particles through dynamics
        for (self.particles) |*particle| {
            try particle.stepPhysics(self.config.physics, self.rng);
        }

        // 2. Update weights with observation likelihood
        self.updateWeights(observation, camera);

        // 3. Check ESS and resample if needed
        const ess = self.effectiveSampleSize();
        const threshold = self.config.ess_threshold * @as(f32, @floatFromInt(self.config.num_particles));

        if (ess < threshold) {
            try self.resample();

            // 4. Gibbs rejuvenation after resampling
            self.gibbsRejuvenation(observation, camera);
        }

        // 5. Increase temperature if tempering
        if (self.config.use_tempering and self.temperature < 1.0) {
            self.temperature = @min(1.0, self.temperature + self.config.temperature_increment);
        }

        self.timestep += 1;
    }

    /// Get posterior estimate of physics types (mode for each entity)
    pub fn getPhysicsTypeEstimate(self: SMCState) ![]PhysicsType {
        if (self.particles.len == 0 or self.particles[0].entities.items.len == 0) {
            return &[_]PhysicsType{};
        }

        const n_entities = self.particles[0].entities.items.len;
        var estimates = try self.allocator.alloc(PhysicsType, n_entities);

        for (0..n_entities) |entity_idx| {
            // Count weighted votes for each physics type
            var votes: [4]f32 = .{ 0, 0, 0, 0 };

            for (self.particles, self.weights) |particle, weight| {
                if (entity_idx < particle.entities.items.len) {
                    const ptype = particle.entities.items[entity_idx].physics_type;
                    votes[@intFromEnum(ptype)] += weight;
                }
            }

            // Find mode
            var max_votes: f32 = 0;
            var max_idx: usize = 0;
            for (votes, 0..) |v, i| {
                if (v > max_votes) {
                    max_votes = v;
                    max_idx = i;
                }
            }

            estimates[entity_idx] = @enumFromInt(max_idx);
        }

        return estimates;
    }

    /// Get posterior probability distribution over physics types for each entity
    pub fn getPhysicsTypePosterior(self: SMCState) ![][4]f32 {
        if (self.particles.len == 0 or self.particles[0].entities.items.len == 0) {
            return &[_][4]f32{};
        }

        const n_entities = self.particles[0].entities.items.len;
        var posteriors = try self.allocator.alloc([4]f32, n_entities);

        for (0..n_entities) |entity_idx| {
            posteriors[entity_idx] = .{ 0, 0, 0, 0 };

            for (self.particles, self.weights) |particle, weight| {
                if (entity_idx < particle.entities.items.len) {
                    const ptype = particle.entities.items[entity_idx].physics_type;
                    posteriors[entity_idx][@intFromEnum(ptype)] += weight;
                }
            }
        }

        return posteriors;
    }
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "Particle creation and physics step" {
    const allocator = testing.allocator;

    var particle = Particle.init(allocator);
    defer particle.deinit();

    const label = Label{ .birth_time = 0, .birth_index = 0 };
    const entity = Entity.initPoint(label, Vec3.init(0, 5, 0), Vec3.zero, .standard);
    try particle.addEntity(entity);

    try testing.expect(particle.aliveCount() == 1);

    var prng = std.Random.DefaultPrng.init(42);
    try particle.stepPhysics(PhysicsConfig{ .gravity = Vec3.init(0, -10, 0), .dt = 0.1 }, prng.random());

    // Entity should have fallen
    try testing.expect(particle.entities.items[0].positionMean().y < 5);
}

test "SMCState initialization" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc = try SMCState.init(allocator, SMCConfig{ .num_particles = 10 }, prng.random());
    defer smc.deinit();

    try testing.expect(smc.particles.len == 10);
    try testing.expect(smc.effectiveSampleSize() > 9.9); // Should be ~10 with uniform weights
}

test "SMCState initialize with prior" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc = try SMCState.init(allocator, SMCConfig{ .num_particles = 10 }, prng.random());
    defer smc.deinit();

    const positions = [_]Vec3{ Vec3.init(0, 5, 0), Vec3.init(2, 5, 0) };
    const velocities = [_]Vec3{ Vec3.zero, Vec3.zero };

    try smc.initializeWithPrior(&positions, &velocities);

    // Each particle should have 2 entities
    for (smc.particles) |p| {
        try testing.expect(p.entities.items.len == 2);
    }
}

test "Systematic resampling" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc = try SMCState.init(allocator, SMCConfig{ .num_particles = 10 }, prng.random());
    defer smc.deinit();

    const positions = [_]Vec3{Vec3.init(0, 5, 0)};
    const velocities = [_]Vec3{Vec3.zero};
    try smc.initializeWithPrior(&positions, &velocities);

    // Set skewed weights
    smc.weights[0] = 0.9;
    for (1..10) |i| {
        smc.weights[i] = 0.1 / 9.0;
    }

    try smc.resample();

    // After resampling, weights should be uniform
    for (smc.weights) |w| {
        try testing.expect(@abs(w - 0.1) < 0.01);
    }
}

test "Effective sample size" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);

    var smc = try SMCState.init(allocator, SMCConfig{ .num_particles = 100 }, prng.random());
    defer smc.deinit();

    // Uniform weights -> ESS = N
    try testing.expect(smc.effectiveSampleSize() > 99);

    // Concentrated weight -> low ESS
    @memset(smc.weights, 0);
    smc.weights[0] = 1.0;
    try testing.expect(smc.effectiveSampleSize() < 1.1);
}
