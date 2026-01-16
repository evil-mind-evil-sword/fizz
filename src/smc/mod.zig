//! SMC (Sequential Monte Carlo) module
//!
//! Provides particle filter inference for the fizz physics engine.
//!
//! Key components:
//! - ParticleSwarm: SoA data layout for efficient particle operations
//! - ParticleWorldView: World-like facade for per-particle entity access
//! - (Future) SMCState: Main inference state machine
//! - (Future) Resampling: Multinomial, systematic, stratified
//! - (Future) Rejuvenation: Gibbs sweeps for particle diversity

pub const swarm = @import("swarm.zig");
pub const world_view = @import("world_view.zig");

// Re-export main types
pub const ParticleSwarm = swarm.ParticleSwarm;
pub const EntityView = swarm.EntityView;
pub const ParticleView = swarm.ParticleView;
pub const CovTriangle = swarm.CovTriangle;
pub const covToTriangle = swarm.covToTriangle;
pub const triangleToCov = swarm.triangleToCov;

pub const ParticleWorldView = world_view.ParticleWorldView;
pub const particleWorld = world_view.particleWorld;

pub const DEFAULT_MAX_ENTITIES = swarm.DEFAULT_MAX_ENTITIES;
pub const DEFAULT_NUM_PARTICLES = swarm.DEFAULT_NUM_PARTICLES;

test {
    _ = swarm;
    _ = world_view;
}
