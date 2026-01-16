//! Switching Linear Dynamical System (SLDS) for Spelke Core Knowledge
//!
//! This module implements the "base system" - an SLDS with folk physics priors
//! that captures Spelke's core knowledge principles:
//!
//! - Object permanence: Entities persist through occlusion
//! - Solidity: Objects cannot pass through each other (mode switching)
//! - Continuity: Smooth trajectories (low process noise)
//! - Cohesion: Entities move as unified wholes
//! - Support: Objects at rest stay at rest (stability islands)
//! - Agency: Self-propelled entities have different dynamics
//!
//! The SLDS has:
//! - Continuous state: Position, Velocity (Kalman-filtered)
//! - Discrete mode: ContactMode (sampled)
//! - Mode-dependent dynamics: Different A, B, Q matrices per mode
//! - Mode transitions: Prior biases toward stability

pub const dynamics = @import("dynamics.zig");
pub const transition = @import("transition.zig");
pub const kalman = @import("kalman.zig");

// Re-export main types
pub const SLDSMatrices = dynamics.SLDSMatrices;
pub const PhysicsConfig = dynamics.PhysicsConfig;

pub const ModeTransitionPrior = transition.ModeTransitionPrior;

pub const kalmanPredict = kalman.kalmanPredict;
pub const kalmanUpdate = kalman.kalmanUpdate;
pub const kalmanLogLikelihood = kalman.kalmanLogLikelihood;

// =============================================================================
// Tests
// =============================================================================

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("dynamics.zig");
    _ = @import("transition.zig");
    _ = @import("kalman.zig");
}
