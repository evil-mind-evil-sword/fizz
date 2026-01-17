//! Fizz Demo: ECS-based Generative Physics Simulation
//!
//! This demo uses the new ECS architecture:
//! - Entities are IDs with component composition
//! - Physics parameters are continuous (not discrete enum)
//! - SLDS provides mode-dependent dynamics with Spelke core knowledge priors
//!
//! The base system encodes folk physics:
//! - Objects persist (permanence)
//! - Objects don't pass through each other (solidity)
//! - Objects at rest stay at rest (support/stability)
//! - Some objects are self-propelled (agency)

const std = @import("std");
const fizz = @import("fizz");

const Vec3 = fizz.Vec3;
const ECSWorld = fizz.ECSWorld;
const Physics = fizz.Physics;
const Position = fizz.Position;
const Velocity = fizz.Velocity;
const Contact = fizz.Contact;
const ContactMode = fizz.ecs.ContactMode;
const SLDSMatrices = fizz.SLDSMatrices;
const SLDSConfig = fizz.SLDSConfig;
const ModeTransitionPrior = fizz.ModeTransitionPrior;

const print = std.debug.print;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Fizz: ECS-based Probabilistic Physics ===\n\n", .{});

    // Initialize ECS world
    var world = ECSWorld.init(allocator);
    defer world.deinit();

    // Physics configurations (replacing old PhysicsType enum)
    const physics_configs = [_]struct { name: []const u8, physics: Physics, color: Vec3 }{
        .{ .name = "standard", .physics = Physics.standard, .color = Vec3.init(1.0, 0.3, 0.3) },
        .{ .name = "bouncy", .physics = Physics.bouncy, .color = Vec3.init(0.3, 1.0, 0.3) },
        .{ .name = "sticky", .physics = Physics.sticky, .color = Vec3.init(0.3, 0.3, 1.0) },
        .{ .name = "slippery", .physics = Physics.slippery, .color = Vec3.init(1.0, 1.0, 0.3) },
    };

    print("Creating entities with ECS:\n", .{});

    for (physics_configs, 0..) |cfg, i| {
        const x = @as(f32, @floatFromInt(i)) * 2.0 - 3.0;
        const id = try world.spawnPhysics(
            Vec3.init(x, 5.0, 0),
            Vec3.zero,
            cfg.physics,
        );

        // Set appearance color
        if (world.appearances.getMut(id.index)) |app| {
            app.color = cfg.color;
            app.radius = 0.5;
        }

        print("  Entity {d} ({s}): friction={d:.2}, elasticity={d:.2}\n", .{
            id.index,
            cfg.name,
            cfg.physics.friction,
            cfg.physics.elasticity,
        });
    }

    print("\nEntity count: {d}\n", .{world.entityCount()});

    // SLDS configuration (ground at y=0 by default via environment config)
    const slds_config = SLDSConfig{
        .gravity = Vec3.init(0, -9.81, 0),
        .dt = 1.0 / 60.0,
    };

    print("\n=== SLDS Mode-Dependent Dynamics ===\n", .{});

    // Demonstrate SLDS matrices for different modes
    const modes = [_]ContactMode{ .free, .environment, .supported, .agency };
    for (modes) |mode| {
        const matrices = SLDSMatrices.forMode(mode, Physics.standard, slds_config);
        print("\n  Mode: {s}\n", .{@tagName(mode)});
        print("    Gravity effect (b[4]): {d:.4}\n", .{matrices.b[4]});
        print("    Position noise Q[0,0]: {d:.6}\n", .{matrices.Q.get(0, 0)});
    }

    print("\n=== Mode Transition Priors (Spelke Core Knowledge) ===\n", .{});

    // Demonstrate mode transition probabilities
    print("\n  From ENVIRONMENT mode, low speed (stability prior):\n", .{});
    const stay_env = ModeTransitionPrior.transitionProb(.environment, .environment, false, 0.01);
    const leave_env = ModeTransitionPrior.transitionProb(.environment, .free, false, 0.01);
    print("    P(stay on environment) = {d:.3}\n", .{stay_env});
    print("    P(leave environment)   = {d:.3}\n", .{leave_env});
    print("    => Objects at rest tend to stay at rest\n", .{});

    print("\n  From FREE mode, contact detected:\n", .{});
    const land = ModeTransitionPrior.transitionProb(.free, .environment, true, 0.5);
    const bounce = ModeTransitionPrior.transitionProb(.free, .free, true, 0.5);
    print("    P(land on environment) = {d:.3}\n", .{land});
    print("    P(bounce off)          = {d:.3}\n", .{bounce});

    print("\n=== Entity Queries ===\n", .{});

    // Query physics entities
    var phys_count: usize = 0;
    var query = world.queryPhysicsEntities();
    while (query.next()) |id| {
        if (world.getPosition(id)) |pos| {
            if (world.getPhysics(id)) |phys| {
                print("  Entity {d}: pos=({d:.1}, {d:.1}, {d:.1}) friction={d:.2}\n", .{
                    id.index,
                    pos.mean.x,
                    pos.mean.y,
                    pos.mean.z,
                    phys.friction,
                });
            }
        }
        phys_count += 1;
    }
    print("  Total physics entities: {d}\n", .{phys_count});

    // Query agents (none yet)
    var agent_query = world.queryAgents();
    var agent_count: usize = 0;
    while (agent_query.next()) |_| {
        agent_count += 1;
    }
    print("  Total agent entities: {d}\n", .{agent_count});

    // Spawn an agent to demonstrate
    print("\n=== Spawning Agent Entity ===\n", .{});
    const agent_id = try world.spawnAgent(
        Vec3.init(0, 3, 2),
        Vec3.init(1, 0, 0),
        Physics.standard,
    );
    print("  Created agent entity {d}\n", .{agent_id.index});
    print("  Has Agency component: {}\n", .{world.hasAgency(agent_id)});

    // Show agent dynamics have higher noise
    const agent_matrices = SLDSMatrices.forMode(.agency, Physics.standard, slds_config);
    const free_matrices = SLDSMatrices.forMode(.free, Physics.standard, slds_config);
    print("  Agency mode noise: {d:.6}\n", .{agent_matrices.Q.get(0, 0)});
    print("  Free mode noise:   {d:.6}\n", .{free_matrices.Q.get(0, 0)});
    print("  => Agents are less predictable (Spelke: agency detection)\n", .{});

    print("\n=== ECS Demo Complete ===\n", .{});
    print("\nArchitecture:\n", .{});
    print("  - Entities: generational IDs with component composition\n", .{});
    print("  - Components: Position, Velocity, Physics, Contact, Agency\n", .{});
    print("  - Systems: Global programs querying by component signature\n", .{});
    print("  - SLDS: Mode-dependent dynamics with folk physics priors\n", .{});
    print("\nNext: Integrate with SMC inference pipeline\n", .{});
}

test "main compiles" {
    // Verify compilation
}
