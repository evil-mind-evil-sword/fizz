import Foundation
import simd

/// Swift wrapper for FizzWorld C API
@MainActor
@Observable
final class FizzWorldWrapper {
    // nonisolated to allow deinit access - C library is thread-safe for distinct handles
    private nonisolated(unsafe) var world: OpaquePointer?
    private(set) var entities: [EntityState] = []
    private(set) var timestep: UInt32 = 0

    struct EntityState: Identifiable {
        let id: UInt64
        var position: SIMD3<Float>
        var velocity: SIMD3<Float>
        var physicsType: PhysicsType
        var isAlive: Bool
    }

    enum PhysicsType: UInt32, CaseIterable, CustomStringConvertible {
        case standard = 0
        case bouncy = 1
        case sticky = 2
        case slippery = 3

        var description: String {
            switch self {
            case .standard: return "Standard"
            case .bouncy: return "Bouncy"
            case .sticky: return "Sticky"
            case .slippery: return "Slippery"
            }
        }

        var color: (r: Float, g: Float, b: Float) {
            switch self {
            case .standard: return (0.5, 0.5, 0.5)
            case .bouncy: return (0.2, 0.8, 0.2)
            case .sticky: return (0.8, 0.2, 0.2)
            case .slippery: return (0.2, 0.6, 1.0)
            }
        }
    }

    init() {
        var config = fizz_physics_config_default()
        world = fizz_world_create(&config)
    }

    deinit {
        fizz_world_destroy(world)
    }

    /// Add a new entity to the world
    @discardableResult
    func addEntity(
        position: SIMD3<Float>,
        velocity: SIMD3<Float> = .zero,
        physicsType: PhysicsType = .standard
    ) -> UInt64 {
        let id = fizz_entity_add(
            world,
            position.x, position.y, position.z,
            velocity.x, velocity.y, velocity.z,
            FizzPhysicsType(rawValue: physicsType.rawValue)
        )
        refreshEntities()
        return id
    }

    /// Remove an entity from the world
    func removeEntity(id: UInt64) {
        _ = fizz_entity_remove(world, id)
        refreshEntities()
    }

    /// Apply a force to an entity
    func applyForce(to id: UInt64, force: SIMD3<Float>) {
        _ = fizz_entity_apply_force(world, id, force.x, force.y, force.z)
    }

    /// Set entity position directly
    func setPosition(of id: UInt64, to position: SIMD3<Float>) {
        _ = fizz_entity_set_position(world, id, position.x, position.y, position.z)
        refreshEntities()
    }

    /// Set entity velocity directly
    func setVelocity(of id: UInt64, to velocity: SIMD3<Float>) {
        _ = fizz_entity_set_velocity(world, id, velocity.x, velocity.y, velocity.z)
        refreshEntities()
    }

    /// Step the physics simulation
    func step() {
        _ = fizz_world_step(world)
        timestep += 1
        refreshEntities()
    }

    /// Refresh entity list from C world
    private func refreshEntities() {
        let count = fizz_world_entity_count(world)
        var newEntities: [EntityState] = []

        for i in 0..<count {
            var state = FizzEntityState()
            if fizz_world_get_entity(world, i, &state) == FIZZ_OK {
                newEntities.append(EntityState(
                    id: state.id,
                    position: SIMD3(state.pos_x, state.pos_y, state.pos_z),
                    velocity: SIMD3(state.vel_x, state.vel_y, state.vel_z),
                    physicsType: PhysicsType(rawValue: state.physics_type.rawValue) ?? .standard,
                    isAlive: state.is_alive
                ))
            }
        }

        entities = newEntities
    }
}

// MARK: - SMC Inference Wrapper

/// Swift wrapper for FizzSMC C API (FastSLAM inference)
@MainActor
@Observable
final class FizzSMCWrapper {
    private nonisolated(unsafe) var smc: OpaquePointer?

    private(set) var posteriors: [[Float]] = []  // [entity_idx][physics_type]
    private(set) var cameraBelief: CameraBelief = .init()
    private(set) var ess: Float = 0
    private(set) var temperature: Float = 0
    private(set) var entityCount: UInt32 = 0

    struct CameraBelief {
        var meanPosition: SIMD3<Float> = .zero
        var meanYaw: Float = 0
        var positionVariance: SIMD3<Float> = .zero
        var yawVariance: Float = 0
    }

    struct EntityPosterior: Identifiable {
        let id: Int
        var standard: Float
        var bouncy: Float
        var sticky: Float
        var slippery: Float

        var mostLikely: FizzWorldWrapper.PhysicsType {
            let probs = [standard, bouncy, sticky, slippery]
            let maxIdx = probs.enumerated().max(by: { $0.1 < $1.1 })?.0 ?? 0
            return FizzWorldWrapper.PhysicsType(rawValue: UInt32(maxIdx)) ?? .standard
        }
    }

    init() {
        var physicsConfig = fizz_physics_config_default()
        var smcConfig = fizz_smc_config_default()
        smcConfig.num_particles = 50
        smc = fizz_smc_create(&physicsConfig, &smcConfig)
    }

    deinit {
        fizz_smc_destroy(smc)
    }

    /// Initialize SMC with observed entity positions
    func initWithPrior(positions: [SIMD3<Float>], velocities: [SIMD3<Float>]) {
        let posVecs = positions.map { FizzVec3(x: $0.x, y: $0.y, z: $0.z) }
        let velVecs = velocities.map { FizzVec3(x: $0.x, y: $0.y, z: $0.z) }

        posVecs.withUnsafeBufferPointer { posPtr in
            velVecs.withUnsafeBufferPointer { velPtr in
                _ = fizz_smc_init_prior(smc, posPtr.baseAddress, velPtr.baseAddress, UInt32(positions.count))
            }
        }

        refreshState()
    }

    /// Step inference with RGB image observation
    func stepWithImage(rgbData: [UInt8], width: UInt32, height: UInt32) {
        rgbData.withUnsafeBufferPointer { ptr in
            _ = fizz_smc_step_with_image(smc, ptr.baseAddress, width, height)
        }
        refreshState()
    }

    /// Get posteriors as structured array
    var entityPosteriors: [EntityPosterior] {
        posteriors.enumerated().map { idx, probs in
            EntityPosterior(
                id: idx,
                standard: probs.count > 0 ? probs[0] : 0,
                bouncy: probs.count > 1 ? probs[1] : 0,
                sticky: probs.count > 2 ? probs[2] : 0,
                slippery: probs.count > 3 ? probs[3] : 0
            )
        }
    }

    /// Refresh all state from C API
    private func refreshState() {
        // Get posteriors
        entityCount = fizz_smc_entity_count(smc)
        var posteriorArray = [FizzPosterior](repeating: FizzPosterior(), count: Int(entityCount))
        let written = posteriorArray.withUnsafeMutableBufferPointer { ptr in
            fizz_smc_get_posteriors(smc, ptr.baseAddress, entityCount)
        }

        posteriors = (0..<written).map { i in
            [posteriorArray[Int(i)].prob_standard,
             posteriorArray[Int(i)].prob_bouncy,
             posteriorArray[Int(i)].prob_sticky,
             posteriorArray[Int(i)].prob_slippery]
        }

        // Get camera belief
        var belief = FizzCameraBelief()
        if fizz_smc_get_camera_belief(smc, &belief) == FIZZ_OK {
            cameraBelief = CameraBelief(
                meanPosition: SIMD3(belief.mean_pos_x, belief.mean_pos_y, belief.mean_pos_z),
                meanYaw: belief.mean_yaw,
                positionVariance: SIMD3(belief.var_pos_x, belief.var_pos_y, belief.var_pos_z),
                yawVariance: belief.var_yaw
            )
        }

        // Get ESS and temperature
        ess = fizz_smc_get_ess(smc)
        temperature = fizz_smc_get_temperature(smc)
    }
}
