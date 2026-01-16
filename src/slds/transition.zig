const std = @import("std");
const math = @import("../math.zig");
const Vec3 = math.Vec3;

const ecs = @import("../ecs/mod.zig");
const ContactMode = ecs.ContactMode;

// =============================================================================
// Mode Transition Prior
// =============================================================================

/// Spelke core knowledge encoded as mode transition probabilities
/// Key principle: Objects at rest tend to stay at rest (stability prior)
pub const ModeTransitionPrior = struct {
    /// Base transition probabilities (before conditioning on observations)
    /// Encodes folk physics intuitions:
    /// - Objects don't spontaneously start moving
    /// - Objects in contact tend to stay in contact
    /// - Agency is rare (sparse prior)

    /// P(mode_t+1 | mode_t, contact_detected, speed)
    pub fn transitionProb(
        from: ContactMode,
        to: ContactMode,
        contact_detected: bool,
        speed: f32,
    ) f32 {
        // Speed thresholds
        const low_speed = speed < 0.1;
        const high_speed = speed > 0.5;

        return switch (from) {
            .free => transitionFromFree(to, contact_detected, high_speed),
            .ground => transitionFromGround(to, contact_detected, low_speed, high_speed),
            .supported => transitionFromSupported(to, low_speed),
            .attached => transitionFromAttached(to),
            .agency => transitionFromAgency(to, low_speed),
        };
    }

    /// Transitions from free flight
    fn transitionFromFree(to: ContactMode, contact_detected: bool, high_speed: bool) f32 {
        if (contact_detected) {
            return switch (to) {
                .free => 0.1, // Bounce off
                .ground => 0.7, // Land on ground
                .supported => 0.15, // Land on object
                .attached => 0.04, // Stick (rare)
                .agency => 0.01, // Spontaneous agency (very rare)
            };
        } else {
            return switch (to) {
                .free => 0.95, // Continue flying
                .ground => 0.02, // Unlikely without contact
                .supported => 0.01,
                .attached => 0.01,
                .agency => 0.01, // Objects don't spontaneously become agents
            };
        }
        _ = high_speed;
    }

    /// Transitions from ground contact
    /// Core knowledge: Objects at rest stay at rest
    fn transitionFromGround(
        to: ContactMode,
        contact_detected: bool,
        low_speed: bool,
        high_speed: bool,
    ) f32 {
        _ = contact_detected;

        if (low_speed) {
            // Very stable when at rest - "island of stability"
            return switch (to) {
                .free => 0.01, // Almost never leaves ground spontaneously
                .ground => 0.97, // Very likely to stay
                .supported => 0.01, // Would need something on top
                .attached => 0.005, // Could become sticky
                .agency => 0.005, // Could become agent
            };
        } else if (high_speed) {
            // More likely to leave if moving fast
            return switch (to) {
                .free => 0.3, // Can launch
                .ground => 0.65, // Still likely to stay
                .supported => 0.02,
                .attached => 0.02,
                .agency => 0.01,
            };
        } else {
            // Medium speed
            return switch (to) {
                .free => 0.1,
                .ground => 0.85,
                .supported => 0.02,
                .attached => 0.02,
                .agency => 0.01,
            };
        }
    }

    /// Transitions from supported (on another entity)
    fn transitionFromSupported(to: ContactMode, low_speed: bool) f32 {
        if (low_speed) {
            return switch (to) {
                .free => 0.02, // Supporter might move
                .ground => 0.03, // Might fall to ground
                .supported => 0.93, // Very stable stack
                .attached => 0.01,
                .agency => 0.01,
            };
        } else {
            return switch (to) {
                .free => 0.2,
                .ground => 0.1,
                .supported => 0.65,
                .attached => 0.03,
                .agency => 0.02,
            };
        }
    }

    /// Transitions from attached (stuck)
    fn transitionFromAttached(to: ContactMode) f32 {
        // Very stable - hard to unstick
        return switch (to) {
            .free => 0.02,
            .ground => 0.02,
            .supported => 0.01,
            .attached => 0.94, // Stay attached
            .agency => 0.01,
        };
    }

    /// Transitions from agency (self-propelled)
    fn transitionFromAgency(to: ContactMode, low_speed: bool) f32 {
        if (low_speed) {
            // Agent might stop
            return switch (to) {
                .free => 0.1,
                .ground => 0.2, // Might rest
                .supported => 0.05,
                .attached => 0.05,
                .agency => 0.6, // Often continues as agent
            };
        } else {
            return switch (to) {
                .free => 0.15,
                .ground => 0.1,
                .supported => 0.05,
                .attached => 0.05,
                .agency => 0.65, // Agents tend to stay agents
            };
        }
    }

    /// Sample next mode given current state
    pub fn sampleTransition(
        from: ContactMode,
        contact_detected: bool,
        speed: f32,
        rng: std.Random,
    ) ContactMode {
        const u = rng.float(f32);
        var cumulative: f32 = 0;

        const modes = [_]ContactMode{ .free, .ground, .supported, .attached, .agency };
        for (modes) |to_mode| {
            cumulative += transitionProb(from, to_mode, contact_detected, speed);
            if (u < cumulative) {
                return to_mode;
            }
        }

        return from; // Fallback
    }

    /// Log probability of transition
    pub fn logTransitionProb(
        from: ContactMode,
        to: ContactMode,
        contact_detected: bool,
        speed: f32,
    ) f32 {
        const p = transitionProb(from, to, contact_detected, speed);
        if (p <= 0) return -std.math.inf(f32);
        return @log(p);
    }
};

// =============================================================================
// Permanence Prior
// =============================================================================

/// Object permanence prior - probability entity still exists after occlusion
/// Core knowledge: Objects persist even when not visible
pub const PermanencePrior = struct {
    /// Base survival probability per timestep
    base_survival: f32 = 0.99,
    /// Decay rate for extended occlusion
    occlusion_decay: f32 = 0.01,
    /// Minimum existence probability
    min_prob: f32 = 0.1,

    /// P(exists | occluded for t timesteps)
    pub fn existenceProb(self: PermanencePrior, occlusion_duration: u32) f32 {
        // Exponential decay with floor
        const decay = @exp(-self.occlusion_decay * @as(f32, @floatFromInt(occlusion_duration)));
        return @max(self.min_prob, self.base_survival * decay);
    }

    /// Log probability of existence
    pub fn logExistenceProb(self: PermanencePrior, occlusion_duration: u32) f32 {
        return @log(self.existenceProb(occlusion_duration));
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ModeTransitionPrior sums to 1" {
    const modes = [_]ContactMode{ .free, .ground, .supported, .attached, .agency };

    for (modes) |from| {
        var sum: f32 = 0;
        for (modes) |to| {
            sum += ModeTransitionPrior.transitionProb(from, to, false, 0.05);
        }
        // Allow small numerical error
        try std.testing.expect(@abs(sum - 1.0) < 0.01);
    }
}

test "Stability prior - low speed ground stays ground" {
    const p_stay = ModeTransitionPrior.transitionProb(.ground, .ground, false, 0.01);
    const p_leave = ModeTransitionPrior.transitionProb(.ground, .free, false, 0.01);

    // Should strongly prefer staying on ground when at rest
    try std.testing.expect(p_stay > 0.9);
    try std.testing.expect(p_leave < 0.05);
}

test "PermanencePrior decays with occlusion" {
    const prior = PermanencePrior{};

    const p0 = prior.existenceProb(0);
    const p10 = prior.existenceProb(10);
    const p100 = prior.existenceProb(100);

    // Should decay over time
    try std.testing.expect(p0 > p10);
    try std.testing.expect(p10 > p100);
    // But never below minimum
    try std.testing.expect(p100 >= prior.min_prob);
}
