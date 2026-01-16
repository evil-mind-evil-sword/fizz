//! CRP Entity Inference Module
//!
//! Implements variable entity count inference using Chinese Restaurant Process (CRP).
//! Enables trans-dimensional moves (birth/death) to discover the unknown number
//! of entities from observations.
//!
//! Key components:
//! - LabelSet: Per-particle entity label management
//! - BirthProposal: Create new entities at unexplained observations
//! - DeathProposal: Remove entities with no recent observations
//! - transDimensionalStep: Combined birth/death MH step
//!
//! The CRP prior naturally handles:
//! - Unknown number of entities (grows as needed)
//! - Entity persistence (survival probability)
//! - New entity discovery (concentration parameter α)

pub const config = @import("config.zig");
pub const label_set = @import("label_set.zig");
pub const birth = @import("birth.zig");
pub const death = @import("death.zig");
pub const moves = @import("moves.zig");

// Re-export commonly used types
pub const CRPConfig = config.CRPConfig;
pub const CRPStats = config.CRPStats;

pub const LabelSet = label_set.LabelSet;
pub const crpPrior = label_set.crpPrior;
pub const crpNewEntityProb = label_set.crpNewEntityProb;
pub const crpExistingEntityProb = label_set.crpExistingEntityProb;

pub const BirthProposal = birth.BirthProposal;
pub const UnexplainedCluster = birth.UnexplainedCluster;
pub const findUnexplainedPixels = birth.findUnexplainedPixels;
pub const clusterUnexplained = birth.clusterUnexplained;
pub const proposeBirth = birth.proposeBirth;
pub const executeBirth = birth.executeBirth;

pub const DeathProposal = death.DeathProposal;
pub const proposeDeath = death.proposeDeath;
pub const executeDeath = death.executeDeath;
pub const updateOcclusionStates = death.updateOcclusionStates;
pub const shouldAutoDeath = death.shouldAutoDeath;

pub const MoveResult = moves.MoveResult;
pub const EntityCountPosterior = moves.EntityCountPosterior;
pub const transDimensionalStep = moves.transDimensionalStep;

// =============================================================================
// Tests
// =============================================================================

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("config.zig");
    _ = @import("label_set.zig");
    _ = @import("birth.zig");
    _ = @import("death.zig");
    _ = @import("moves.zig");
}
