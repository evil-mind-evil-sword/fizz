const std = @import("std");

// =============================================================================
// Data Association Configuration
// =============================================================================

/// Configuration for data association and occlusion reasoning
pub const AssociationConfig = struct {
    /// Observation noise (pixel-space variance)
    observation_noise: f32 = 0.1,

    /// Detection probability (P(observe | entity exists, visible))
    detection_prob: f32 = 0.95,

    /// False alarm rate (clutter density per pixel)
    clutter_rate: f32 = 0.001,

    /// Gating threshold for association (Mahalanobis distance)
    gating_threshold: f32 = 9.21, // chi2(3, 0.99)

    /// Maximum assignment hypotheses to maintain per particle
    max_hypotheses: u32 = 10,

    /// Pruning threshold for low-probability hypotheses
    pruning_threshold: f32 = 1e-4,

    /// Gibbs sampling iterations for assignment refinement
    gibbs_iterations: u32 = 5,

    /// Occlusion model parameters
    occlusion: OcclusionConfig = .{},

    /// Track state transition parameters
    track_transition: TrackTransitionConfig = .{},
};

/// Configuration for occlusion reasoning
pub const OcclusionConfig = struct {
    /// Minimum overlap ratio to consider occlusion
    overlap_threshold: f32 = 0.3,

    /// Depth difference threshold for occlusion (closer occluder)
    depth_threshold: f32 = 0.1,

    /// Maximum frames to maintain occluded track
    max_occlusion_frames: u32 = 30,

    /// Decay rate for existence probability during occlusion
    existence_decay: f32 = 0.95,
};

/// Configuration for track state transitions
pub const TrackTransitionConfig = struct {
    /// P(detected -> detected | matched)
    detected_to_detected_matched: f32 = 0.95,

    /// P(detected -> detected | not matched)
    detected_to_detected_unmatched: f32 = 0.1,

    /// P(detected -> occluded | not matched)
    detected_to_occluded: f32 = 0.7,

    /// P(detected -> dead)
    detected_to_dead: f32 = 0.01,

    /// P(occluded -> detected | matched)
    occluded_to_detected_matched: f32 = 0.8,

    /// P(occluded -> occluded | not matched)
    occluded_to_occluded: f32 = 0.9,

    /// P(occluded -> dead)
    occluded_to_dead: f32 = 0.01,

    /// P(tentative -> detected | matched)
    tentative_to_detected: f32 = 0.7,

    /// P(tentative -> dead | not matched)
    tentative_to_dead: f32 = 0.5,
};

// =============================================================================
// Tests
// =============================================================================

test "AssociationConfig defaults" {
    const config = AssociationConfig{};
    try std.testing.expect(config.detection_prob > 0.9);
    try std.testing.expect(config.max_hypotheses > 0);
}
