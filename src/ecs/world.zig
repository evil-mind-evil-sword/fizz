const std = @import("std");
const Allocator = std.mem.Allocator;

const entity_mod = @import("entity.zig");
const EntityId = entity_mod.EntityId;
const EntityAllocator = entity_mod.EntityAllocator;

const component_mod = @import("component.zig");
const ComponentStorage = component_mod.ComponentStorage;
const ComponentId = component_mod.ComponentId;

const builtin = @import("builtin.zig");
const Position = builtin.Position;
const Velocity = builtin.Velocity;
const Physics = builtin.Physics;
const Contact = builtin.Contact;
const Support = builtin.Support;
const Occlusion = builtin.Occlusion;
const Appearance = builtin.Appearance;
const Label = builtin.Label;
const Agency = builtin.Agency;
const ContactMode = builtin.ContactMode;

const math = @import("../math.zig");
const Vec3 = math.Vec3;

// =============================================================================
// World - Central ECS Manager
// =============================================================================

/// ECS World - manages entities and their components
pub const World = struct {
    /// Entity allocator with generational indices
    entities: EntityAllocator,

    // Built-in component storage
    positions: ComponentStorage(Position),
    velocities: ComponentStorage(Velocity),
    physics: ComponentStorage(Physics),
    contacts: ComponentStorage(Contact),
    supports: ComponentStorage(Support),
    occlusions: ComponentStorage(Occlusion),
    appearances: ComponentStorage(Appearance),
    labels: ComponentStorage(Label),
    agencies: ComponentStorage(Agency),

    /// Current timestep
    timestep: u32,
    /// Next birth index for labels
    next_birth_index: u16,
    /// Allocator
    allocator: Allocator,

    pub fn init(allocator: Allocator) World {
        return .{
            .entities = EntityAllocator.init(allocator),
            .positions = ComponentStorage(Position).init(allocator),
            .velocities = ComponentStorage(Velocity).init(allocator),
            .physics = ComponentStorage(Physics).init(allocator),
            .contacts = ComponentStorage(Contact).init(allocator),
            .supports = ComponentStorage(Support).init(allocator),
            .occlusions = ComponentStorage(Occlusion).init(allocator),
            .appearances = ComponentStorage(Appearance).init(allocator),
            .labels = ComponentStorage(Label).init(allocator),
            .agencies = ComponentStorage(Agency).init(allocator),
            .timestep = 0,
            .next_birth_index = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *World) void {
        self.entities.deinit();
        self.positions.deinit();
        self.velocities.deinit();
        self.physics.deinit();
        self.contacts.deinit();
        self.supports.deinit();
        self.occlusions.deinit();
        self.appearances.deinit();
        self.labels.deinit();
        self.agencies.deinit();
    }

    /// Spawn a new entity (returns EntityId)
    pub fn spawn(self: *World) !EntityId {
        return self.entities.alloc();
    }

    /// Spawn entity with physics bundle
    pub fn spawnPhysics(
        self: *World,
        position: Vec3,
        velocity: Vec3,
        phys: Physics,
    ) !EntityId {
        const id = try self.spawn();
        const idx = id.index;

        try self.positions.set(idx, Position.point(position));
        try self.velocities.set(idx, Velocity.point(velocity));
        try self.physics.set(idx, phys);
        try self.contacts.set(idx, Contact.free());
        try self.occlusions.set(idx, Occlusion.detected());
        try self.appearances.set(idx, Appearance.default);
        try self.labels.set(idx, Label{
            .birth_time = self.timestep,
            .birth_index = self.next_birth_index,
        });
        self.next_birth_index += 1;

        return id;
    }

    /// Spawn entity with agency (self-propelled)
    pub fn spawnAgent(
        self: *World,
        position: Vec3,
        velocity: Vec3,
        phys: Physics,
    ) !EntityId {
        const id = try self.spawnPhysics(position, velocity, phys);
        try self.agencies.set(id.index, Agency.basic());
        return id;
    }

    /// Despawn entity (marks as dead, removes all components)
    pub fn despawn(self: *World, id: EntityId) void {
        if (!self.entities.isValid(id)) return;

        const idx = id.index;
        self.positions.remove(idx);
        self.velocities.remove(idx);
        self.physics.remove(idx);
        self.contacts.remove(idx);
        self.supports.remove(idx);
        self.occlusions.remove(idx);
        self.appearances.remove(idx);
        self.labels.remove(idx);
        self.agencies.remove(idx);

        self.entities.free(id);
    }

    /// Check if entity is alive
    pub fn isAlive(self: *const World, id: EntityId) bool {
        return self.entities.isValid(id);
    }

    /// Get count of alive entities
    pub fn entityCount(self: *const World) usize {
        return self.entities.count();
    }

    // =========================================================================
    // Component Accessors
    // =========================================================================

    pub fn getPosition(self: *const World, id: EntityId) ?*const Position {
        if (!self.entities.isValid(id)) return null;
        return self.positions.get(id.index);
    }

    pub fn getPositionMut(self: *World, id: EntityId) ?*Position {
        if (!self.entities.isValid(id)) return null;
        return self.positions.getMut(id.index);
    }

    pub fn getVelocity(self: *const World, id: EntityId) ?*const Velocity {
        if (!self.entities.isValid(id)) return null;
        return self.velocities.get(id.index);
    }

    pub fn getVelocityMut(self: *World, id: EntityId) ?*Velocity {
        if (!self.entities.isValid(id)) return null;
        return self.velocities.getMut(id.index);
    }

    pub fn getPhysics(self: *const World, id: EntityId) ?*const Physics {
        if (!self.entities.isValid(id)) return null;
        return self.physics.get(id.index);
    }

    pub fn getContact(self: *const World, id: EntityId) ?*const Contact {
        if (!self.entities.isValid(id)) return null;
        return self.contacts.get(id.index);
    }

    pub fn getContactMut(self: *World, id: EntityId) ?*Contact {
        if (!self.entities.isValid(id)) return null;
        return self.contacts.getMut(id.index);
    }

    pub fn hasAgency(self: *const World, id: EntityId) bool {
        if (!self.entities.isValid(id)) return false;
        return self.agencies.has(id.index);
    }

    pub fn getAgency(self: *const World, id: EntityId) ?*const Agency {
        if (!self.entities.isValid(id)) return null;
        return self.agencies.get(id.index);
    }

    // =========================================================================
    // Value-returning accessors (for generic world interface compatibility)
    // =========================================================================

    /// Get appearance component as value (for generic interface)
    pub fn getAppearance(self: *const World, id: EntityId) ?Appearance {
        if (!self.entities.isValid(id)) return null;
        const ptr = self.appearances.get(id.index) orelse return null;
        return ptr.*;
    }

    /// Set appearance component (for generic interface)
    pub fn setAppearance(self: *World, id: EntityId, app: Appearance) void {
        if (!self.entities.isValid(id)) return;
        self.appearances.set(id.index, app) catch {};
    }

    /// Get occlusion component as value (for generic interface)
    pub fn getOcclusion(self: *const World, id: EntityId) ?Occlusion {
        if (!self.entities.isValid(id)) return null;
        const ptr = self.occlusions.get(id.index) orelse return null;
        return ptr.*;
    }

    /// Set occlusion component (for generic interface)
    pub fn setOcclusion(self: *World, id: EntityId, occ: Occlusion) void {
        if (!self.entities.isValid(id)) return;
        self.occlusions.set(id.index, occ) catch {};
    }

    /// Get label component as value (for generic interface)
    pub fn getLabel(self: *const World, id: EntityId) ?Label {
        if (!self.entities.isValid(id)) return null;
        const ptr = self.labels.get(id.index) orelse return null;
        return ptr.*;
    }

    /// Set label component (for generic interface)
    pub fn setLabel(self: *World, id: EntityId, label: Label) void {
        if (!self.entities.isValid(id)) return;
        self.labels.set(id.index, label) catch {};
    }

    // =========================================================================
    // Queries - Find entities with specific components
    // =========================================================================

    /// Query for entities with Position and Velocity (basic physics entities)
    pub fn queryPhysicsEntities(self: *const World) QueryIterator {
        return .{
            .world = self,
            .entity_iter = self.entities.aliveIterator(),
        };
    }

    pub const QueryIterator = struct {
        world: *const World,
        entity_iter: EntityAllocator.AliveIterator,

        pub fn next(self: *QueryIterator) ?EntityId {
            while (self.entity_iter.next()) |id| {
                // Check required components
                if (self.world.positions.has(id.index) and
                    self.world.velocities.has(id.index))
                {
                    return id;
                }
            }
            return null;
        }
    };

    /// Query for agent entities
    pub fn queryAgents(self: *const World) AgentQueryIterator {
        return .{
            .world = self,
            .entity_iter = self.entities.aliveIterator(),
        };
    }

    pub const AgentQueryIterator = struct {
        world: *const World,
        entity_iter: EntityAllocator.AliveIterator,

        pub fn next(self: *AgentQueryIterator) ?EntityId {
            while (self.entity_iter.next()) |id| {
                if (self.world.agencies.has(id.index)) {
                    return id;
                }
            }
            return null;
        }
    };

    // =========================================================================
    // Utility
    // =========================================================================

    /// Advance timestep counter
    pub fn tick(self: *World) void {
        self.timestep += 1;
    }

    /// Clone world state (for particle filter)
    pub fn clone(self: *const World, allocator: Allocator) !World {
        var new_world = World.init(allocator);
        errdefer new_world.deinit();

        // Clone entities
        var iter = self.entities.aliveIterator();
        while (iter.next()) |id| {
            const new_id = try new_world.entities.alloc();
            std.debug.assert(new_id.index == id.index);

            const idx = id.index;
            if (self.positions.get(idx)) |p| try new_world.positions.set(idx, p.*);
            if (self.velocities.get(idx)) |v| try new_world.velocities.set(idx, v.*);
            if (self.physics.get(idx)) |p| try new_world.physics.set(idx, p.*);
            if (self.contacts.get(idx)) |c| try new_world.contacts.set(idx, c.*);
            if (self.supports.get(idx)) |s| try new_world.supports.set(idx, s.*);
            if (self.occlusions.get(idx)) |o| try new_world.occlusions.set(idx, o.*);
            if (self.appearances.get(idx)) |a| try new_world.appearances.set(idx, a.*);
            if (self.labels.get(idx)) |l| try new_world.labels.set(idx, l.*);
            if (self.agencies.get(idx)) |a| try new_world.agencies.set(idx, a.*);
        }

        new_world.timestep = self.timestep;
        new_world.next_birth_index = self.next_birth_index;

        return new_world;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "World spawn and despawn" {
    const allocator = std.testing.allocator;

    var world = World.init(allocator);
    defer world.deinit();

    const e1 = try world.spawnPhysics(
        Vec3.init(0, 5, 0),
        Vec3.zero,
        Physics.standard,
    );

    const e2 = try world.spawnPhysics(
        Vec3.init(2, 5, 0),
        Vec3.zero,
        Physics.bouncy,
    );

    try std.testing.expectEqual(@as(usize, 2), world.entityCount());
    try std.testing.expect(world.isAlive(e1));
    try std.testing.expect(world.isAlive(e2));

    // Check components
    const pos = world.getPosition(e1).?;
    try std.testing.expect(pos.mean.y == 5);

    world.despawn(e1);
    try std.testing.expect(!world.isAlive(e1));
    try std.testing.expectEqual(@as(usize, 1), world.entityCount());
}

test "World queries" {
    const allocator = std.testing.allocator;

    var world = World.init(allocator);
    defer world.deinit();

    _ = try world.spawnPhysics(Vec3.zero, Vec3.zero, Physics.standard);
    _ = try world.spawnAgent(Vec3.unit_x, Vec3.zero, Physics.standard);

    // Query all physics entities
    var phys_query = world.queryPhysicsEntities();
    var phys_count: usize = 0;
    while (phys_query.next()) |_| {
        phys_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), phys_count);

    // Query agents only
    var agent_query = world.queryAgents();
    var agent_count: usize = 0;
    while (agent_query.next()) |_| {
        agent_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), agent_count);
}

test "World clone" {
    const allocator = std.testing.allocator;

    var world = World.init(allocator);
    defer world.deinit();

    _ = try world.spawnPhysics(Vec3.init(1, 2, 3), Vec3.zero, Physics.standard);
    _ = try world.spawnAgent(Vec3.init(4, 5, 6), Vec3.zero, Physics.bouncy);

    var cloned = try world.clone(allocator);
    defer cloned.deinit();

    try std.testing.expectEqual(world.entityCount(), cloned.entityCount());
    try std.testing.expectEqual(world.timestep, cloned.timestep);
}
