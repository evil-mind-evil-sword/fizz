//! Data Association Module
//!
//! Handles observation-to-entity assignment with occlusion reasoning.
//! Key components:
//! - Assignment: observation -> entity mapping
//! - AssignmentHypotheses: multiple hypotheses per particle (GLMB-style)
//! - OcclusionGraph: depth-ordered occlusion reasoning
//! - GibbsSampler: Gibbs sampling for data association
//!
//! This module implements the data association layer of the multi-object
//! tracking system. It determines which observations belong to which entities,
//! accounting for occlusion, clutter, and missed detections.

pub const config = @import("config.zig");
pub const assignment = @import("assignment.zig");
pub const occlusion = @import("occlusion.zig");
pub const likelihood = @import("likelihood.zig");
pub const gibbs = @import("gibbs.zig");

// Re-export commonly used types
pub const AssociationConfig = config.AssociationConfig;
pub const OcclusionConfig = config.OcclusionConfig;
pub const TrackTransitionConfig = config.TrackTransitionConfig;

pub const Assignment = assignment.Assignment;
pub const AssignmentHypotheses = assignment.AssignmentHypotheses;
pub const trackTransitionProb = assignment.trackTransitionProb;

pub const OcclusionGraph = occlusion.OcclusionGraph;
pub const ProjectedEntity = occlusion.ProjectedEntity;
pub const PixelOwnership = occlusion.PixelOwnership;
pub const renderPixelOwnership = occlusion.renderPixelOwnership;

pub const assignmentLogLikelihood = likelihood.assignmentLogLikelihood;
pub const singleAssignmentLikelihood = likelihood.singleAssignmentLikelihood;
pub const assignmentLikelihoodRatio = likelihood.assignmentLikelihoodRatio;
pub const detectionLikelihood = likelihood.detectionLikelihood;

pub const GibbsSampler = gibbs.GibbsSampler;
pub const generateHypotheses = gibbs.generateHypotheses;

// =============================================================================
// Tests
// =============================================================================

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("config.zig");
    _ = @import("assignment.zig");
    _ = @import("occlusion.zig");
    _ = @import("likelihood.zig");
    _ = @import("gibbs.zig");
}
