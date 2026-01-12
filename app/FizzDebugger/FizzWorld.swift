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
