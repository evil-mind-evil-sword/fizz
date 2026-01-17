//! Conjugate Priors for Rao-Blackwellized Inference
//!
//! This module provides conjugate prior distributions that enable
//! exact or near-exact Bayesian inference in the intuitive physics model.
//!
//! Key conjugacies:
//! - Gaussian likelihood × Inverse-Gamma prior → Inverse-Gamma posterior (variance)
//! - Gaussian likelihood × Inverse-Wishart prior → Inverse-Wishart posterior (covariance)
//! - Categorical likelihood × Dirichlet prior → Dirichlet posterior (transitions)
//! - Bernoulli likelihood × Beta prior → Beta posterior (contact probability)
//!
//! References:
//! - Gelman et al. (2013) "Bayesian Data Analysis" Ch. 2-3
//! - Murphy (2012) "Machine Learning: A Probabilistic Perspective" Ch. 4

const std = @import("std");
const math = @import("math.zig");
const Vec3 = math.Vec3;
const Mat3 = math.Mat3;

// =============================================================================
// Inverse-Gamma Distribution (for variance parameters)
// =============================================================================

/// Inverse-Gamma distribution for variance parameters
/// Conjugate prior for Gaussian variance with known mean
/// σ² ~ IG(α, β) means 1/σ² ~ Gamma(α, β)
pub const InverseGamma = struct {
    /// Shape parameter (α > 0, need α > 1 for finite mean)
    alpha: f32,
    /// Rate parameter (β > 0)
    beta: f32,

    /// Default prior: weakly informative
    pub const default = InverseGamma{
        .alpha = 2.0, // Ensures finite mean
        .beta = 0.1,
    };

    /// Prior for observation noise (tighter)
    pub const observation_noise = InverseGamma{
        .alpha = 3.0,
        .beta = 0.3, // Prior mean = 0.15
    };

    /// Prior for process noise (smaller)
    pub const process_noise = InverseGamma{
        .alpha = 3.0,
        .beta = 0.03, // Prior mean = 0.015
    };

    /// Posterior mean: E[σ²] = β / (α - 1) for α > 1
    pub fn mean(self: InverseGamma) f32 {
        if (self.alpha <= 1.0) return std.math.inf(f32);
        return self.beta / (self.alpha - 1.0);
    }

    /// Posterior mode: β / (α + 1)
    pub fn mode(self: InverseGamma) f32 {
        return self.beta / (self.alpha + 1.0);
    }

    /// Posterior variance: β² / ((α-1)²(α-2)) for α > 2
    pub fn variance(self: InverseGamma) f32 {
        if (self.alpha <= 2.0) return std.math.inf(f32);
        const num = self.beta * self.beta;
        const denom = (self.alpha - 1.0) * (self.alpha - 1.0) * (self.alpha - 2.0);
        return num / denom;
    }

    /// Log PDF: log IG(x; α, β)
    pub fn logPdf(self: InverseGamma, x: f32) f32 {
        if (x <= 0) return -std.math.inf(f32);
        // log IG(x; α, β) = α log(β) - log Γ(α) - (α+1) log(x) - β/x
        const log_beta = @log(self.beta);
        const log_gamma_alpha = std.math.lgamma(self.alpha);
        const log_x = @log(x);
        return self.alpha * log_beta - log_gamma_alpha - (self.alpha + 1.0) * log_x - self.beta / x;
    }

    /// Update with n observations having sum of squared residuals SSR
    /// Posterior: IG(α + n/2, β + SSR/2)
    pub fn update(self: InverseGamma, n: u32, sum_sq_residuals: f32) InverseGamma {
        return .{
            .alpha = self.alpha + @as(f32, @floatFromInt(n)) / 2.0,
            .beta = self.beta + sum_sq_residuals / 2.0,
        };
    }

    /// Sample from Inverse-Gamma (via Gamma)
    pub fn sample(self: InverseGamma, rng: std.Random) f32 {
        // If X ~ Gamma(α, β), then 1/X ~ IG(α, β)
        const gamma_sample = gammaSample(self.alpha, self.beta, rng);
        return 1.0 / gamma_sample;
    }
};

// =============================================================================
// Beta Distribution (for probabilities)
// =============================================================================

/// Beta distribution for probability parameters
/// Conjugate prior for Bernoulli/Binomial likelihood
pub const Beta = struct {
    /// Shape parameter α > 0 (pseudo-count of successes + 1)
    alpha: f32,
    /// Shape parameter β > 0 (pseudo-count of failures + 1)
    beta: f32,

    /// Uniform prior
    pub const uniform = Beta{ .alpha = 1.0, .beta = 1.0 };

    /// Jeffreys prior (uninformative)
    pub const jeffreys = Beta{ .alpha = 0.5, .beta = 0.5 };

    /// Prior for contact probability (slightly biased toward no contact)
    pub const contact = Beta{ .alpha = 1.0, .beta = 2.0 };

    /// Prior for friction (moderate values)
    pub const friction = Beta{ .alpha = 2.0, .beta = 5.0 }; // Mean ≈ 0.29

    /// Prior for elasticity (centered)
    pub const elasticity = Beta{ .alpha = 2.0, .beta = 2.0 }; // Mean = 0.5

    /// Posterior mean: E[p] = α / (α + β)
    pub fn mean(self: Beta) f32 {
        return self.alpha / (self.alpha + self.beta);
    }

    /// Posterior mode: (α - 1) / (α + β - 2) for α, β > 1
    pub fn mode(self: Beta) f32 {
        if (self.alpha <= 1.0 or self.beta <= 1.0) return self.mean();
        return (self.alpha - 1.0) / (self.alpha + self.beta - 2.0);
    }

    /// Posterior variance
    pub fn variance(self: Beta) f32 {
        const sum = self.alpha + self.beta;
        return (self.alpha * self.beta) / (sum * sum * (sum + 1.0));
    }

    /// Log PDF: log Beta(x; α, β)
    pub fn logPdf(self: Beta, x: f32) f32 {
        if (x <= 0 or x >= 1) return -std.math.inf(f32);
        // log Beta(x; α, β) = (α-1)log(x) + (β-1)log(1-x) - log B(α, β)
        const log_x = @log(x);
        const log_1_minus_x = @log(1.0 - x);
        const log_beta_fn = std.math.lgamma(self.alpha) + std.math.lgamma(self.beta) -
            std.math.lgamma(self.alpha + self.beta);
        return (self.alpha - 1.0) * log_x + (self.beta - 1.0) * log_1_minus_x - log_beta_fn;
    }

    /// Update with observed successes and failures
    /// Posterior: Beta(α + successes, β + failures)
    pub fn update(self: Beta, successes: u32, failures: u32) Beta {
        return .{
            .alpha = self.alpha + @as(f32, @floatFromInt(successes)),
            .beta = self.beta + @as(f32, @floatFromInt(failures)),
        };
    }

    /// Update with single Bernoulli observation
    pub fn updateSingle(self: Beta, success: bool) Beta {
        return if (success)
            .{ .alpha = self.alpha + 1.0, .beta = self.beta }
        else
            .{ .alpha = self.alpha, .beta = self.beta + 1.0 };
    }

    /// Sample from Beta distribution
    pub fn sample(self: Beta, rng: std.Random) f32 {
        const x = gammaSample(self.alpha, 1.0, rng);
        const y = gammaSample(self.beta, 1.0, rng);
        return x / (x + y);
    }
};

// =============================================================================
// Dirichlet Distribution (for categorical probabilities)
// =============================================================================

/// Dirichlet distribution for probability vectors
/// Conjugate prior for Categorical/Multinomial likelihood
pub const Dirichlet = struct {
    /// Concentration parameters (α_1, ..., α_K)
    alpha: []f32,
    allocator: std.mem.Allocator,

    /// Create with uniform concentration
    pub fn uniform(allocator: std.mem.Allocator, k: usize) !Dirichlet {
        const alpha = try allocator.alloc(f32, k);
        @memset(alpha, 1.0);
        return .{ .alpha = alpha, .allocator = allocator };
    }

    /// Create from slice (copies)
    pub fn fromSlice(allocator: std.mem.Allocator, alpha: []const f32) !Dirichlet {
        const alpha_copy = try allocator.alloc(f32, alpha.len);
        @memcpy(alpha_copy, alpha);
        return .{ .alpha = alpha_copy, .allocator = allocator };
    }

    pub fn deinit(self: *Dirichlet) void {
        self.allocator.free(self.alpha);
    }

    /// Posterior mean: E[p_i] = α_i / Σα
    pub fn mean(self: Dirichlet, allocator: std.mem.Allocator) ![]f32 {
        var sum: f32 = 0;
        for (self.alpha) |a| sum += a;

        const result = try allocator.alloc(f32, self.alpha.len);
        for (self.alpha, 0..) |a, i| {
            result[i] = a / sum;
        }
        return result;
    }

    /// Update with observed counts
    /// Posterior: Dirichlet(α + counts)
    pub fn update(self: *Dirichlet, counts: []const u32) void {
        for (counts, 0..) |c, i| {
            self.alpha[i] += @floatFromInt(c);
        }
    }

    /// Update with single observation
    pub fn updateSingle(self: *Dirichlet, category: usize) void {
        self.alpha[category] += 1.0;
    }

    /// Sample from Dirichlet (via Gamma)
    pub fn sample(self: Dirichlet, allocator: std.mem.Allocator, rng: std.Random) ![]f32 {
        const result = try allocator.alloc(f32, self.alpha.len);
        var sum: f32 = 0;

        for (self.alpha, 0..) |a, i| {
            result[i] = gammaSample(a, 1.0, rng);
            sum += result[i];
        }

        // Normalize
        for (result) |*r| {
            r.* /= sum;
        }

        return result;
    }
};

// =============================================================================
// Mode Transition Prior (5x5 Dirichlet for each source mode)
// =============================================================================

/// Mode transition matrix with Dirichlet priors per row
/// Encodes Spelke core knowledge as prior pseudo-counts
pub const ModeTransitionPrior = struct {
    /// Dirichlet concentration for each row: alpha[from][to]
    /// Pseudo-counts encode prior beliefs about physics
    alpha: [5][5]f32,

    /// Observed transition counts
    counts: [5][5]u32,

    /// Contact modes (for reference)
    pub const Mode = enum(u3) {
        free = 0,
        environment = 1,
        supported = 2,
        attached = 3,
        agency = 4,
    };

    /// Default prior encoding Spelke core knowledge
    /// Higher pseudo-counts = stronger prior belief
    pub const spelke_prior = ModeTransitionPrior{
        .alpha = .{
            // From FREE: objects in motion tend to stay in motion (inertia)
            .{ 9.5, 0.2, 0.1, 0.1, 0.1 }, // P(stay free) high
            // From ENVIRONMENT: objects at rest tend to stay at rest (stability)
            .{ 0.1, 9.7, 0.1, 0.05, 0.05 }, // P(stay on ground) very high
            // From SUPPORTED: stacked objects are stable
            .{ 0.2, 0.3, 9.3, 0.1, 0.1 },
            // From ATTACHED: sticky things stay stuck
            .{ 0.2, 0.2, 0.1, 9.4, 0.1 },
            // From AGENCY: agents are somewhat persistent
            .{ 1.0, 2.0, 0.5, 0.5, 6.0 },
        },
        .counts = std.mem.zeroes([5][5]u32),
    };

    /// Weak prior (more learnable)
    pub const weak_prior = ModeTransitionPrior{
        .alpha = .{
            .{ 2.0, 0.5, 0.5, 0.5, 0.5 },
            .{ 0.5, 2.0, 0.5, 0.5, 0.5 },
            .{ 0.5, 0.5, 2.0, 0.5, 0.5 },
            .{ 0.5, 0.5, 0.5, 2.0, 0.5 },
            .{ 0.5, 0.5, 0.5, 0.5, 2.0 },
        },
        .counts = std.mem.zeroes([5][5]u32),
    };

    /// Observe a mode transition
    pub fn observe(self: *ModeTransitionPrior, from: Mode, to: Mode) void {
        self.counts[@intFromEnum(from)][@intFromEnum(to)] += 1;
    }

    /// Get posterior transition probability P(to | from)
    pub fn posteriorProb(self: ModeTransitionPrior, from: Mode, to: Mode) f32 {
        const f = @intFromEnum(from);
        const t = @intFromEnum(to);

        var sum: f32 = 0;
        for (0..5) |j| {
            sum += self.alpha[f][j] + @as(f32, @floatFromInt(self.counts[f][j]));
        }
        return (self.alpha[f][t] + @as(f32, @floatFromInt(self.counts[f][t]))) / sum;
    }

    /// Get full posterior row (all transition probs from a mode)
    pub fn posteriorRow(self: ModeTransitionPrior, from: Mode) [5]f32 {
        const f = @intFromEnum(from);
        var result: [5]f32 = undefined;

        var sum: f32 = 0;
        for (0..5) |j| {
            sum += self.alpha[f][j] + @as(f32, @floatFromInt(self.counts[f][j]));
        }

        for (0..5) |j| {
            result[j] = (self.alpha[f][j] + @as(f32, @floatFromInt(self.counts[f][j]))) / sum;
        }

        return result;
    }

    /// Sample transition probabilities for a row (for Gibbs)
    pub fn sampleRow(self: ModeTransitionPrior, from: Mode, rng: std.Random) [5]f32 {
        const f = @intFromEnum(from);
        var result: [5]f32 = undefined;
        var sum: f32 = 0;

        for (0..5) |j| {
            const alpha_post = self.alpha[f][j] + @as(f32, @floatFromInt(self.counts[f][j]));
            result[j] = gammaSample(alpha_post, 1.0, rng);
            sum += result[j];
        }

        // Normalize
        for (&result) |*r| {
            r.* /= sum;
        }

        return result;
    }

    /// Log probability of observed transitions (for model comparison)
    pub fn logMarginalLikelihood(self: ModeTransitionPrior) f32 {
        var log_ml: f32 = 0;

        for (0..5) |f| {
            // Dirichlet-Multinomial: log P(counts | alpha)
            var alpha_sum: f32 = 0;
            var count_sum: u32 = 0;
            var alpha_count_sum: f32 = 0;

            for (0..5) |t| {
                alpha_sum += self.alpha[f][t];
                count_sum += self.counts[f][t];
                alpha_count_sum += self.alpha[f][t] + @as(f32, @floatFromInt(self.counts[f][t]));

                // log Γ(α_t + n_t) - log Γ(α_t)
                log_ml += std.math.lgamma(self.alpha[f][t] + @as(f32, @floatFromInt(self.counts[f][t])));
                log_ml -= std.math.lgamma(self.alpha[f][t]);
            }

            // log Γ(Σα) - log Γ(Σα + Σn)
            log_ml += std.math.lgamma(alpha_sum);
            log_ml -= std.math.lgamma(alpha_count_sum);
        }

        return log_ml;
    }

    /// Reset observed counts (for new inference episode)
    pub fn resetCounts(self: *ModeTransitionPrior) void {
        self.counts = std.mem.zeroes([5][5]u32);
    }
};

// =============================================================================
// Material Parameters (continuous, grid-based Gibbs)
// =============================================================================

/// Material parameters with Beta priors and grid-based inference
pub const MaterialPrior = struct {
    /// Beta prior on friction ∈ [0, 1]
    friction_prior: Beta = Beta.friction,

    /// Beta prior on elasticity ∈ [0, 1]
    elasticity_prior: Beta = Beta.elasticity,

    /// Discrete grid for tractable Gibbs sampling
    pub const friction_grid = [_]f32{ 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9 };
    pub const elasticity_grid = [_]f32{ 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9 };

    /// Compute posterior over friction grid given data
    pub fn frictionPosterior(
        self: MaterialPrior,
        elasticity: f32,
        dynamics_log_likelihood: *const fn (f32, f32) f32,
    ) [friction_grid.len]f32 {
        var log_probs: [friction_grid.len]f32 = undefined;
        var max_log: f32 = -std.math.inf(f32);

        for (friction_grid, 0..) |f, i| {
            const log_prior = self.friction_prior.logPdf(f);
            const log_lik = dynamics_log_likelihood(f, elasticity);
            log_probs[i] = log_prior + log_lik;
            max_log = @max(max_log, log_probs[i]);
        }

        // Normalize (softmax)
        var sum: f32 = 0;
        for (&log_probs) |*lp| {
            lp.* = @exp(lp.* - max_log);
            sum += lp.*;
        }
        for (&log_probs) |*lp| {
            lp.* /= sum;
        }

        return log_probs;
    }

    /// Sample friction from posterior
    pub fn sampleFriction(
        self: MaterialPrior,
        elasticity: f32,
        dynamics_log_likelihood: *const fn (f32, f32) f32,
        rng: std.Random,
    ) f32 {
        const probs = self.frictionPosterior(elasticity, dynamics_log_likelihood);
        const idx = categoricalSample(&probs, rng);
        return friction_grid[idx];
    }

    /// Sample elasticity from posterior
    pub fn sampleElasticity(
        self: MaterialPrior,
        friction: f32,
        dynamics_log_likelihood: *const fn (f32, f32) f32,
        rng: std.Random,
    ) f32 {
        var log_probs: [elasticity_grid.len]f32 = undefined;
        var max_log: f32 = -std.math.inf(f32);

        for (elasticity_grid, 0..) |e, i| {
            const log_prior = self.elasticity_prior.logPdf(e);
            const log_lik = dynamics_log_likelihood(friction, e);
            log_probs[i] = log_prior + log_lik;
            max_log = @max(max_log, log_probs[i]);
        }

        // Normalize
        var sum: f32 = 0;
        for (&log_probs) |*lp| {
            lp.* = @exp(lp.* - max_log);
            sum += lp.*;
        }
        for (&log_probs) |*lp| {
            lp.* /= sum;
        }

        return elasticity_grid[categoricalSample(&log_probs, rng)];
    }
};

// =============================================================================
// Soft Contact State (Beta-Bernoulli)
// =============================================================================

/// Contact state with uncertainty
pub const SoftContact = struct {
    /// Beta posterior on P(in contact)
    contact_belief: Beta,

    /// Point estimate for downstream use
    pub fn probability(self: SoftContact) f32 {
        return self.contact_belief.mean();
    }

    /// Initialize with prior
    pub fn init(prior: Beta) SoftContact {
        return .{ .contact_belief = prior };
    }

    /// Update with contact observation (e.g., from penetration depth)
    pub fn update(self: *SoftContact, in_contact: bool) void {
        self.contact_belief = self.contact_belief.updateSingle(in_contact);
    }

    /// Update with continuous signal (penetration depth)
    /// Treats depth > threshold as contact observation
    pub fn updateFromPenetration(self: *SoftContact, penetration: f32, threshold: f32) void {
        self.update(penetration > threshold);
    }

    /// Decay belief toward prior over time (for non-stationary contact)
    pub fn decay(self: *SoftContact, rate: f32, prior: Beta) void {
        self.contact_belief.alpha = self.contact_belief.alpha * (1 - rate) + prior.alpha * rate;
        self.contact_belief.beta = self.contact_belief.beta * (1 - rate) + prior.beta * rate;
    }
};

// =============================================================================
// Observation Noise State (tracks sufficient statistics)
// =============================================================================

/// Observation noise with Inverse-Gamma prior and online updates
pub const ObservationNoiseState = struct {
    /// Prior/posterior parameters
    prior: InverseGamma,

    /// Sufficient statistics
    n_observations: u32 = 0,
    sum_sq_residuals: f32 = 0,

    /// Initialize with prior
    pub fn init(prior: InverseGamma) ObservationNoiseState {
        return .{ .prior = prior };
    }

    /// Current posterior
    pub fn posterior(self: ObservationNoiseState) InverseGamma {
        return self.prior.update(self.n_observations, self.sum_sq_residuals);
    }

    /// Current noise estimate (posterior mean)
    pub fn noiseEstimate(self: ObservationNoiseState) f32 {
        return self.posterior().mean();
    }

    /// Update with observation residual
    pub fn observe(self: *ObservationNoiseState, residual: Vec3) void {
        self.n_observations += 1;
        self.sum_sq_residuals += residual.dot(residual);
    }

    /// Update with scalar residual
    pub fn observeScalar(self: *ObservationNoiseState, residual: f32) void {
        self.n_observations += 1;
        self.sum_sq_residuals += residual * residual;
    }

    /// Reset statistics (new episode)
    pub fn reset(self: *ObservationNoiseState) void {
        self.n_observations = 0;
        self.sum_sq_residuals = 0;
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Sample from Gamma(α, β) distribution using Marsaglia and Tsang's method
pub fn gammaSample(alpha: f32, beta: f32, rng: std.Random) f32 {
    if (alpha < 1.0) {
        // Use Ahrens-Dieter method for α < 1
        const u = rng.float(f32);
        return gammaSample(alpha + 1.0, beta, rng) * std.math.pow(f32, u, 1.0 / alpha);
    }

    // Marsaglia and Tsang's method for α >= 1
    const d = alpha - 1.0 / 3.0;
    const c = 1.0 / @sqrt(9.0 * d);

    while (true) {
        var x: f32 = undefined;
        var v: f32 = undefined;

        while (true) {
            x = sampleStdNormal(rng);
            v = 1.0 + c * x;
            if (v > 0) break;
        }

        v = v * v * v;
        const u = rng.float(f32);

        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return d * v / beta;
        }

        if (@log(u) < 0.5 * x * x + d * (1.0 - v + @log(v))) {
            return d * v / beta;
        }
    }
}

/// Sample from standard normal using Box-Muller
fn sampleStdNormal(rng: std.Random) f32 {
    const r1 = rng.float(f32);
    const r2 = rng.float(f32);
    return @sqrt(-2.0 * @log(r1 + 1e-10)) * @cos(2.0 * std.math.pi * r2);
}

/// Sample from categorical distribution
fn categoricalSample(probs: []const f32, rng: std.Random) usize {
    const u = rng.float(f32);
    var cumsum: f32 = 0;

    for (probs, 0..) |p, i| {
        cumsum += p;
        if (u < cumsum) return i;
    }

    return probs.len - 1;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "InverseGamma mean and update" {
    const prior = InverseGamma{ .alpha = 3.0, .beta = 1.0 };
    try testing.expectApproxEqAbs(prior.mean(), 0.5, 0.01);

    // Update with 10 observations, SSR = 2.0
    const posterior = prior.update(10, 2.0);
    try testing.expect(posterior.alpha == 8.0); // 3 + 10/2
    try testing.expect(posterior.beta == 2.0); // 1 + 2/2

    // Posterior mean should be smaller (more data constrains variance)
    try testing.expect(posterior.mean() < prior.mean());
}

test "Beta mean and update" {
    const prior = Beta.uniform;
    try testing.expectApproxEqAbs(prior.mean(), 0.5, 0.01);

    // Update with 8 successes, 2 failures
    const posterior = prior.update(8, 2);
    try testing.expectApproxEqAbs(posterior.mean(), 9.0 / 12.0, 0.01);
}

test "ModeTransitionPrior posterior" {
    var prior = ModeTransitionPrior.spelke_prior;

    // Initially, P(stay on ground) should be high
    const p_stay = prior.posteriorProb(.environment, .environment);
    try testing.expect(p_stay > 0.9);

    // After observing many transitions off ground, probability should decrease
    for (0..20) |_| {
        prior.observe(.environment, .free);
    }

    const p_stay_after = prior.posteriorProb(.environment, .environment);
    try testing.expect(p_stay_after < p_stay);
}

test "Gamma sampling produces positive values" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    for (0..100) |_| {
        const sample = gammaSample(2.0, 1.0, rng);
        try testing.expect(sample > 0);
    }
}

test "InverseGamma sampling" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = InverseGamma{ .alpha = 3.0, .beta = 1.0 };

    // Sample mean should be close to theoretical mean
    var sum: f32 = 0;
    const n = 1000;
    for (0..n) |_| {
        sum += dist.sample(rng);
    }
    const sample_mean = sum / @as(f32, n);
    try testing.expectApproxEqAbs(sample_mean, dist.mean(), 0.1);
}

test "Beta sampling" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const dist = Beta{ .alpha = 2.0, .beta = 5.0 };

    var sum: f32 = 0;
    const n = 1000;
    for (0..n) |_| {
        const s = dist.sample(rng);
        try testing.expect(s >= 0 and s <= 1);
        sum += s;
    }
    const sample_mean = sum / @as(f32, n);
    try testing.expectApproxEqAbs(sample_mean, dist.mean(), 0.05);
}

test "SoftContact update" {
    var contact = SoftContact.init(Beta.contact);

    // Initially low contact probability
    const initial_prob = contact.probability();
    try testing.expect(initial_prob < 0.5);

    // After many contact observations, probability should increase
    for (0..10) |_| {
        contact.update(true);
    }

    try testing.expect(contact.probability() > initial_prob);
    try testing.expect(contact.probability() > 0.7);
}

test "ObservationNoiseState" {
    var state = ObservationNoiseState.init(InverseGamma.observation_noise);

    _ = state.noiseEstimate(); // Initial estimate exists

    // Observe some small residuals
    for (0..20) |_| {
        state.observe(Vec3.init(0.1, 0.1, 0.1));
    }

    // Posterior should be tighter and potentially different
    const posterior = state.posterior();
    try testing.expect(posterior.alpha > state.prior.alpha);
}
