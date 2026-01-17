//! Fizz: Probabilistic Physics Engine for Inverse Simulation
//!
//! A generative model for 3D physics that can be run "backward" via SMC
//! to infer world dynamics from observations.
//!
//! Architecture:
//! - ECS (Entity-Component-System) for flexible entity composition
//! - SLDS (Switching Linear Dynamical System) with Spelke core knowledge priors
//! - CRP/IBP for unknown entity count and component membership
//! - Rao-Blackwellized particle filter (Kalman for continuous, sample discrete)
//! - 3D Gaussian mixture observation model (entities = mixture components)
//!
//! Core concepts:
//! - Entity: Just an ID (generational for safe reuse)
//! - Component: Data attached to entities (Position, Velocity, Physics, etc.)
//! - System: Global program that queries entities by component signature
//! - Mode: Discrete contact state (free, ground, supported, attached, agency)
//!
//! Spelke core knowledge encoded as:
//! - Object permanence: Entities persist through occlusion
//! - Solidity: Objects cannot pass through each other (mode switching)
//! - Continuity: Smooth trajectories (low process noise)
//! - Support: Objects at rest stay at rest (stability islands)
//! - Agency: Self-propelled entities have different dynamics

pub const math = @import("math.zig");
pub const ecs = @import("ecs/mod.zig");
pub const slds = @import("slds/mod.zig");
pub const association = @import("association/mod.zig");
pub const crp = @import("crp/mod.zig");
pub const smc_swarm = @import("smc/mod.zig");

// Material trait system (domain-general vs domain-specific)
pub const material = @import("material.zig");
pub const Material = material.Material;
pub const DefaultMaterial = material.DefaultMaterial;
pub const LegacyMaterials = material.LegacyMaterials;

// Conjugate priors for Rao-Blackwellized inference
pub const priors = @import("priors.zig");

// Legacy modules (being deprecated)
pub const types = @import("types.zig");
pub const dynamics = @import("dynamics.zig");
pub const gmm = @import("gmm.zig");
pub const smc = @import("smc.zig");

// Re-export commonly used types
pub const Vec3 = math.Vec3;
pub const Vec2 = math.Vec2;
pub const Mat3 = math.Mat3;
pub const Mat2 = math.Mat2;

// ============================================================================
// New ECS types (preferred)
// ============================================================================
pub const EntityId = ecs.EntityId;
pub const ECSWorld = ecs.World;
pub const Position = ecs.Position;
pub const Velocity = ecs.Velocity;
pub const Physics = ecs.Physics;
pub const ECSContactMode = ecs.ContactMode;
pub const Contact = ecs.Contact;
pub const Support = ecs.Support;
pub const ECSLabel = ecs.Label;
pub const Agency = ecs.Agency;

// SLDS types
pub const SLDSMatrices = slds.SLDSMatrices;
pub const SLDSConfig = slds.PhysicsConfig;
pub const ModeTransitionPrior = slds.ModeTransitionPrior;

// Association types
pub const Assignment = association.Assignment;
pub const AssignmentHypotheses = association.AssignmentHypotheses;
pub const OcclusionGraph = association.OcclusionGraph;
pub const AssociationConfig = association.AssociationConfig;
pub const GibbsSampler = association.GibbsSampler;

// CRP types
pub const CRPConfig = crp.CRPConfig;
pub const LabelSet = crp.LabelSet;
pub const BirthProposal = crp.BirthProposal;
pub const DeathProposal = crp.DeathProposal;
pub const transDimensionalStep = crp.transDimensionalStep;
pub const EntityCountPosterior = crp.EntityCountPosterior;

// ============================================================================
// Legacy types (DEPRECATED - use ECS equivalents above)
// These will be removed in the next major version.
// Migration guide:
//   Entity -> ECSWorld.spawnPhysics() returns EntityId
//   PhysicsType -> Physics.standard, Physics.bouncy, etc.
//   ContactMode -> ecs.ContactMode (enum in builtin.zig)
//   Label -> ecs.Label
//   TrackState -> ecs.TrackState
// ============================================================================
pub const Entity = types.Entity;
pub const Label = types.Label;
pub const PhysicsParams = types.PhysicsParams;
pub const PhysicsParamsUncertainty = types.PhysicsParamsUncertainty;
pub const ContactMode = types.ContactMode;
pub const TrackState = types.TrackState;
pub const GaussianVec3 = types.GaussianVec3;
pub const Appearance = types.Appearance;
pub const PhysicsConfig = types.PhysicsConfig;
pub const Camera = types.Camera;
pub const ProjectionResult = types.ProjectionResult;

// Environment types (static geometry)
pub const EnvironmentEntity = types.EnvironmentEntity;
pub const EnvironmentConfig = types.EnvironmentConfig;

pub const DynamicsMatrices = dynamics.DynamicsMatrices;
pub const kalmanPredictPosition = dynamics.kalmanPredictPosition;
pub const kalmanPredictVelocity = dynamics.kalmanPredictVelocity;
pub const kalmanUpdate = dynamics.kalmanUpdate;
pub const kalmanLogLikelihood = dynamics.kalmanLogLikelihood;
pub const physicsStepPoint = dynamics.physicsStepPoint;
pub const entityPhysicsStep = dynamics.entityPhysicsStep;
pub const checkEntityContact = dynamics.checkEntityContact;
pub const resolveEntityCollision = dynamics.resolveEntityCollision;

pub const GaussianComponent = gmm.GaussianComponent;
pub const GaussianMixture = gmm.GaussianMixture;
pub const Observation = gmm.Observation;
pub const ObservationGrid = gmm.ObservationGrid;
pub const imageLogLikelihood = gmm.imageLogLikelihood;
pub const entityLogLikelihood = gmm.entityLogLikelihood;

// Back-projection observation model (conjugate)
pub const Detection2D = gmm.Detection2D;
pub const RayGaussian = gmm.RayGaussian;
pub const backProjectionLogLikelihood = gmm.backProjectionLogLikelihood;

// Sparse optical flow observation model (conjugate velocity)
pub const FlowObservation = gmm.FlowObservation;
pub const SparseFlowConfig = gmm.SparseFlowConfig;
pub const computeSparseFlow = gmm.computeSparseFlow;

pub const SMCConfig = smc.SMCConfig;
pub const Particle = smc.Particle;
pub const SMCState = smc.SMCState;
pub const SurpriseTracker = smc.SurpriseTracker;

// Conjugate prior types
pub const InverseGamma = priors.InverseGamma;
pub const Beta = priors.Beta;
pub const Dirichlet = priors.Dirichlet;
pub const PriorModeTransition = priors.ModeTransitionPrior;
pub const MaterialPrior = priors.MaterialPrior;
pub const SoftContact = priors.SoftContact;
pub const ObservationNoiseState = priors.ObservationNoiseState;

// =============================================================================
// World State (Collection of Entities)
// =============================================================================

/// World state containing all entities
pub const World = struct {
    /// All entities (both alive and dead)
    entities: std.ArrayList(Entity),
    /// Physics configuration
    config: PhysicsConfig,
    /// Current timestep
    timestep: u32,
    /// Next birth index for labels
    next_birth_index: u16,
    /// Allocator
    allocator: std.mem.Allocator,

    const std = @import("std");

    /// Initialize empty world
    pub fn init(allocator: std.mem.Allocator, config: PhysicsConfig) World {
        return .{
            .entities = .empty,
            .config = config,
            .timestep = 0,
            .next_birth_index = 0,
            .allocator = allocator,
        };
    }

    /// Free allocated memory
    pub fn deinit(self: *World) void {
        self.entities.deinit(self.allocator);
    }

    /// Add new entity to world (birth)
    pub fn addEntity(
        self: *World,
        position: Vec3,
        velocity: Vec3,
        physics_params: PhysicsParams,
    ) !*Entity {
        const label = Label{
            .birth_time = self.timestep,
            .birth_index = self.next_birth_index,
        };
        self.next_birth_index += 1;

        const entity = Entity.initPoint(label, position, velocity, physics_params);
        try self.entities.append(self.allocator, entity);

        return &self.entities.items[self.entities.items.len - 1];
    }

    /// Step physics for all entities
    pub fn step(self: *World, rng: ?std.Random) void {
        // Update physics for each entity
        for (self.entities.items) |*entity| {
            if (entity.isAlive()) {
                entityPhysicsStep(entity, self.config, rng);
            }
        }

        // Check entity-entity collisions
        for (0..self.entities.items.len) |i| {
            for (i + 1..self.entities.items.len) |j| {
                var e1 = &self.entities.items[i];
                var e2 = &self.entities.items[j];

                if (e1.isAlive() and e2.isAlive()) {
                    if (checkEntityContact(e1.*, e2.*)) {
                        resolveEntityCollision(e1, e2);
                    }
                }
            }
        }

        self.timestep += 1;
    }

    /// Get alive entity count
    pub fn aliveCount(self: World) usize {
        var count: usize = 0;
        for (self.entities.items) |e| {
            if (e.isAlive()) count += 1;
        }
        return count;
    }

    /// Create GMM from current world state
    pub fn toGMM(self: World) !GaussianMixture {
        return GaussianMixture.fromEntities(self.entities.items, self.allocator);
    }

    /// Render world to observation grid
    pub fn render(
        self: World,
        camera: Camera,
        width: u32,
        height: u32,
        samples_per_ray: u32,
    ) !ObservationGrid {
        var grid = try ObservationGrid.init(width, height, self.allocator);
        errdefer grid.deinit();

        var gmm_model = try self.toGMM();
        defer gmm_model.deinit();

        grid.renderGMM(gmm_model, camera, samples_per_ray);

        return grid;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "World creation and entity addition" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var world = World.init(allocator, PhysicsConfig{});
    defer world.deinit();

    _ = try world.addEntity(Vec3.init(0, 5, 0), Vec3.zero, .standard);
    _ = try world.addEntity(Vec3.init(2, 5, 0), Vec3.zero, .bouncy);

    try std.testing.expect(world.aliveCount() == 2);
    try std.testing.expect(world.entities.items[0].label.birth_index == 0);
    try std.testing.expect(world.entities.items[1].label.birth_index == 1);
}

test "World physics step" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var world = World.init(allocator, PhysicsConfig{
        .gravity = Vec3.init(0, -10, 0),
        .dt = 0.1,
    });
    defer world.deinit();

    _ = try world.addEntity(Vec3.init(0, 5, 0), Vec3.zero, .standard);

    const initial_y = world.entities.items[0].positionMean().y;

    world.step(null);

    // Entity should fall
    try std.testing.expect(world.entities.items[0].positionMean().y < initial_y);
}

test "World to GMM" {
    const std = @import("std");
    const allocator = std.testing.allocator;

    var world = World.init(allocator, PhysicsConfig{});
    defer world.deinit();

    _ = try world.addEntity(Vec3.init(0, 0, 0), Vec3.zero, .standard);
    _ = try world.addEntity(Vec3.init(2, 0, 0), Vec3.zero, .standard);

    var gmm_model = try world.toGMM();
    defer gmm_model.deinit();

    try std.testing.expect(gmm_model.components.len == 2);
}

// Run all module tests
test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("math.zig");
    _ = @import("types.zig");
    _ = @import("dynamics.zig");
    _ = @import("gmm.zig");
    _ = @import("smc.zig");
    _ = @import("material.zig");
    _ = @import("priors.zig");
    _ = @import("scenario_tests.zig");
    // New modules
    _ = @import("ecs/mod.zig");
    _ = @import("slds/mod.zig");
    _ = @import("association/mod.zig");
    _ = @import("crp/mod.zig");
    _ = @import("smc/mod.zig");
}
