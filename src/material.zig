//! Material Trait System
//!
//! This module provides a domain-specific, pluggable interface for material properties.
//! The base RBPF algorithm encodes only Spelke core knowledge principles:
//! - Continuity (Kalman tracking)
//! - Solidity (collision detection)
//! - Support (ContactMode + gravity)
//! - Permanence (TrackState)
//!
//! Material properties (friction, elasticity, etc.) are domain-specific and optional.
//! When using DefaultMaterial (1 type), Gibbs sampling skips material inference entirely.
//! When using LegacyMaterials (4 types), full Gibbs enumeration is performed.

const std = @import("std");
const math = @import("math.zig");

const Vec3 = math.Vec3;

/// Maximum number of material types supported
/// Used for static array sizing in Gibbs enumeration
pub const MAX_MATERIAL_TYPES: usize = 16;

/// Generic material interface - domain-specific, pluggable
///
/// Materials define dynamics parameters that vary by type:
/// - friction: velocity decay rate
/// - elasticity: bounce coefficient
/// - process_noise: dynamics uncertainty
///
/// When num_types == 1, Gibbs sampling is skipped (domain-general mode).
pub const Material = struct {
    /// Number of discrete material types (for Gibbs enumeration)
    /// When == 1, Gibbs material sampling is skipped
    num_types: usize,

    /// Get friction coefficient for a material type index
    getFriction: *const fn (type_idx: usize) f32,

    /// Get elasticity (coefficient of restitution) for a material type index
    getElasticity: *const fn (type_idx: usize) f32,

    /// Get process noise scale for a material type index
    getProcessNoise: *const fn (type_idx: usize) f32,

    /// Get prior probability for each type (unnormalized)
    getPrior: *const fn (type_idx: usize) f32,

    /// Get display name for debugging/visualization
    getName: *const fn (type_idx: usize) []const u8,

    /// Get color for visualization (optional, defaults to gray)
    getColor: ?*const fn (type_idx: usize) Vec3 = null,

    /// Helper: get friction for index
    pub fn friction(self: Material, idx: usize) f32 {
        return self.getFriction(idx);
    }

    /// Helper: get elasticity for index
    pub fn elasticity(self: Material, idx: usize) f32 {
        return self.getElasticity(idx);
    }

    /// Helper: get process noise for index
    pub fn processNoise(self: Material, idx: usize) f32 {
        return self.getProcessNoise(idx);
    }

    /// Helper: get prior for index
    pub fn prior(self: Material, idx: usize) f32 {
        return self.getPrior(idx);
    }

    /// Helper: get name for index
    pub fn name(self: Material, idx: usize) []const u8 {
        return self.getName(idx);
    }

    /// Helper: get color for index (falls back to gray)
    pub fn color(self: Material, idx: usize) Vec3 {
        if (self.getColor) |f| {
            return f(idx);
        }
        return Vec3.init(0.5, 0.5, 0.5);
    }

    /// Check if this is domain-general mode (no material inference)
    pub fn isDomainGeneral(self: Material) bool {
        return self.num_types <= 1;
    }
};

// =============================================================================
// DefaultMaterial: Domain-General Mode (1 type = skip Gibbs)
// =============================================================================

/// Domain-general material: single default type
/// When used, Gibbs sampling is skipped entirely for materials.
/// This encodes pure core knowledge without domain-specific material inference.
pub const DefaultMaterial = struct {
    pub const instance = Material{
        .num_types = 1,
        .getFriction = defaultFriction,
        .getElasticity = defaultElasticity,
        .getProcessNoise = defaultProcessNoise,
        .getPrior = defaultPrior,
        .getName = defaultName,
        .getColor = defaultColor,
    };

    fn defaultFriction(_: usize) f32 {
        return 0.3;
    }

    fn defaultElasticity(_: usize) f32 {
        return 0.5;
    }

    fn defaultProcessNoise(_: usize) f32 {
        return 0.01;
    }

    fn defaultPrior(_: usize) f32 {
        return 1.0;
    }

    fn defaultName(_: usize) []const u8 {
        return "default";
    }

    fn defaultColor(_: usize) Vec3 {
        return Vec3.init(0.5, 0.5, 0.5);
    }
};

// =============================================================================
// LegacyMaterials: Backward-Compatible 4-Type System
// =============================================================================

/// Legacy 4-type material system (backward compatible with PhysicsType enum)
/// Maps to: standard (0), bouncy (1), sticky (2), slippery (3)
pub const LegacyMaterials = struct {
    pub const instance = Material{
        .num_types = 4,
        .getFriction = legacyFriction,
        .getElasticity = legacyElasticity,
        .getProcessNoise = legacyProcessNoise,
        .getPrior = legacyPrior,
        .getName = legacyName,
        .getColor = legacyColor,
    };

    fn legacyFriction(idx: usize) f32 {
        // Values aligned with PhysicsParams presets
        return switch (idx) {
            0 => 0.5, // standard
            1 => 0.2, // bouncy
            2 => 0.8, // sticky
            3 => 0.1, // slippery
            else => 0.5,
        };
    }

    fn legacyElasticity(idx: usize) f32 {
        // Values aligned with PhysicsParams presets
        return switch (idx) {
            0 => 0.5, // standard
            1 => 0.9, // bouncy
            2 => 0.2, // sticky
            3 => 0.7, // slippery
            else => 0.5,
        };
    }

    fn legacyProcessNoise(idx: usize) f32 {
        return switch (idx) {
            0 => 0.01, // standard
            1 => 0.02, // bouncy
            2 => 0.005, // sticky
            3 => 0.015, // slippery
            else => 0.01,
        };
    }

    fn legacyPrior(idx: usize) f32 {
        _ = idx;
        return 0.25; // Uniform prior
    }

    fn legacyName(idx: usize) []const u8 {
        return switch (idx) {
            0 => "standard",
            1 => "bouncy",
            2 => "sticky",
            3 => "slippery",
            else => "unknown",
        };
    }

    fn legacyColor(idx: usize) Vec3 {
        return switch (idx) {
            0 => Vec3.init(1.0, 0.3, 0.3), // standard - red
            1 => Vec3.init(0.3, 1.0, 0.3), // bouncy - green
            2 => Vec3.init(0.3, 0.3, 1.0), // sticky - blue
            3 => Vec3.init(1.0, 1.0, 0.3), // slippery - yellow
            else => Vec3.init(0.5, 0.5, 0.5),
        };
    }
};


// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "DefaultMaterial is domain-general" {
    const mat = DefaultMaterial.instance;
    try testing.expect(mat.isDomainGeneral());
    try testing.expectEqual(@as(usize, 1), mat.num_types);
}

test "LegacyMaterials has 4 types" {
    const mat = LegacyMaterials.instance;
    try testing.expect(!mat.isDomainGeneral());
    try testing.expectEqual(@as(usize, 4), mat.num_types);

    // Verify friction values (standard=0.5, bouncy=0.2, sticky=0.8, slippery=0.1)
    try testing.expectApproxEqAbs(@as(f32, 0.5), mat.friction(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), mat.friction(1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), mat.friction(2), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), mat.friction(3), 1e-6);

    // Verify elasticity values (standard=0.5, bouncy=0.9, sticky=0.2, slippery=0.7)
    try testing.expectApproxEqAbs(@as(f32, 0.5), mat.elasticity(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.9), mat.elasticity(1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), mat.elasticity(2), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.7), mat.elasticity(3), 1e-6);

    // Verify process noise values (standard=0.01, bouncy=0.02, sticky=0.005, slippery=0.015)
    try testing.expectApproxEqAbs(@as(f32, 0.01), mat.processNoise(0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.02), mat.processNoise(1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.005), mat.processNoise(2), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.015), mat.processNoise(3), 1e-6);
}

test "Material color helper" {
    const default_mat = DefaultMaterial.instance;
    const default_color = default_mat.color(0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), default_color.x, 1e-6);

    const legacy_mat = LegacyMaterials.instance;
    const bouncy_color = legacy_mat.color(1);
    try testing.expectApproxEqAbs(@as(f32, 0.3), bouncy_color.x, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), bouncy_color.y, 1e-6);
}
