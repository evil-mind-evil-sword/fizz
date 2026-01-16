const std = @import("std");

// =============================================================================
// CRP Configuration
// =============================================================================

/// Configuration for CRP-based entity inference
pub const CRPConfig = struct {
    /// CRP concentration parameter (higher = more entities expected)
    alpha: f32 = 1.0,

    /// Base survival probability per timestep
    survival_prob: f32 = 0.99,

    /// Probability of proposing a birth move
    birth_proposal_prob: f32 = 0.3,

    /// Probability of proposing a death move
    death_proposal_prob: f32 = 0.2,

    /// Minimum cluster size for birth proposal
    min_cluster_size: u32 = 10,

    /// Maximum entities per particle (computational limit)
    max_entities: u32 = 50,

    /// Initial velocity variance for new entities
    birth_velocity_variance: f32 = 1.0,

    /// Gating threshold for unexplained observations
    unexplained_threshold: f32 = 0.5,

    /// Birth likelihood threshold (accept if above)
    birth_acceptance_threshold: f32 = 0.1,

    /// Death likelihood threshold (accept if below)
    death_acceptance_threshold: f32 = 0.1,

    /// Maximum occlusion frames before death consideration
    max_occlusion_before_death: u32 = 30,
};

/// Statistics for CRP inference (for debugging/analysis)
pub const CRPStats = struct {
    /// Total birth proposals attempted
    birth_attempts: u64 = 0,
    /// Successful birth proposals
    birth_accepts: u64 = 0,
    /// Total death proposals attempted
    death_attempts: u64 = 0,
    /// Successful death proposals
    death_accepts: u64 = 0,
    /// Current entity count (across all particles)
    total_entities: u64 = 0,

    pub fn birthAcceptRate(self: CRPStats) f32 {
        if (self.birth_attempts == 0) return 0;
        return @as(f32, @floatFromInt(self.birth_accepts)) / @as(f32, @floatFromInt(self.birth_attempts));
    }

    pub fn deathAcceptRate(self: CRPStats) f32 {
        if (self.death_attempts == 0) return 0;
        return @as(f32, @floatFromInt(self.death_accepts)) / @as(f32, @floatFromInt(self.death_attempts));
    }
};

// =============================================================================
// Tests
// =============================================================================

test "CRPConfig defaults" {
    const config = CRPConfig{};
    try std.testing.expect(config.alpha > 0);
    try std.testing.expect(config.survival_prob > 0.9);
}

test "CRPStats rates" {
    var stats = CRPStats{};
    stats.birth_attempts = 100;
    stats.birth_accepts = 25;

    try std.testing.expect(@abs(stats.birthAcceptRate() - 0.25) < 0.01);
}
