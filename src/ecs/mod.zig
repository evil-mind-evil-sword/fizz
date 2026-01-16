//! Entity-Component-System (ECS) Architecture for Fizz
//!
//! This module provides a flexible ECS framework for representing entities
//! with dynamic component composition. Designed to support:
//!
//! - Hierarchical CRP/IBP inference over entity structure
//! - Spelke core knowledge priors via built-in components
//! - System modifications (programs that alter base dynamics)
//!
//! Core concepts:
//! - Entity: Just an ID (generational for safe reuse)
//! - Component: Data attached to entities (Position, Velocity, Physics, etc.)
//! - System: Global program that queries entities by component signature
//!
//! Built-in components encode Spelke core knowledge:
//! - Position/Velocity: Objects exist in space, move continuously
//! - Physics: Objects have physical properties
//! - Contact: Solidity (objects don't pass through each other)
//! - Support: Objects at rest stay at rest
//! - Occlusion: Objects persist through occlusion
//! - Agency: Some objects are self-propelled

pub const component = @import("component.zig");
pub const entity = @import("entity.zig");
pub const world = @import("world.zig");
pub const system = @import("system.zig");
pub const builtin = @import("builtin.zig");

// Re-export commonly used types
pub const ComponentId = component.ComponentId;
pub const ComponentStorage = component.ComponentStorage;

pub const EntityId = entity.EntityId;
pub const EntityAllocator = entity.EntityAllocator;

pub const World = world.World;

pub const System = system.System;
pub const Query = system.Query;
pub const Scheduler = system.Scheduler;

// Built-in components
pub const Position = builtin.Position;
pub const Velocity = builtin.Velocity;
pub const Physics = builtin.Physics;
pub const ContactMode = builtin.ContactMode;
pub const Contact = builtin.Contact;
pub const Support = builtin.Support;
pub const Occlusion = builtin.Occlusion;
pub const Appearance = builtin.Appearance;
pub const Label = builtin.Label;
pub const EntityRef = builtin.EntityRef;
pub const GoalType = builtin.GoalType;
pub const Agency = builtin.Agency;
pub const TrackState = builtin.TrackState;
pub const SpatialRelationType = builtin.SpatialRelationType;
pub const SpatialRelation = builtin.SpatialRelation;

// Bundles
pub const PhysicsBundle = builtin.PhysicsBundle;
pub const AgentBundle = builtin.AgentBundle;

// =============================================================================
// Tests
// =============================================================================

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("component.zig");
    _ = @import("entity.zig");
    _ = @import("world.zig");
    _ = @import("system.zig");
    _ = @import("builtin.zig");
}
