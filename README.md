# fizz

A physics simulation library with Sequential Monte Carlo (SMC) inference for inverse simulation.

## Overview

fizz provides two main capabilities:

1. **Forward simulation**: Run physics with gravity, collisions, and ground contacts
2. **Inverse inference**: Given observations, infer the physics properties (bouncy, sticky, slippery, standard) of entities using SMC

The inference engine uses Rao-Blackwellized particle filtering with Gibbs rejuvenation to maintain discrete physics type hypotheses while tracking continuous state (position, velocity) with Kalman filtering.

## Building

```bash
zig build              # Build library and demos
zig build test         # Run tests
zig build run          # Run basic demo
zig build infer        # Run inference demo
```

Requires Zig 0.15.2 or later.

## Usage

### Forward Simulation

```zig
const fizz = @import("fizz");

var world = fizz.World.init(allocator, .{
    .gravity = fizz.Vec3.init(0, -9.81, 0),
    .dt = 1.0 / 60.0,
});
defer world.deinit();

// Add entities
_ = try world.addEntity(
    fizz.Vec3.init(0, 5, 0),  // position
    fizz.Vec3.zero,           // velocity
    .bouncy,                  // physics type
);

// Step simulation
world.step(rng);
```

### SMC Inference

```zig
const fizz = @import("fizz");

var smc = try fizz.SMCState.init(allocator, .{
    .num_particles = 100,
    .ess_threshold = 0.5,
    .observation_noise = 0.3,
}, rng);
defer smc.deinit();

// Initialize with observed positions (unknown physics types)
try smc.initializeWithPrior(&positions, &velocities);

// Run inference loop
while (running) {
    const observation = world.render(camera, 16, 16, 32);
    try smc.step(observation, camera);
}

// Get posterior estimates
const posteriors = try smc.getPhysicsTypePosterior();
```

## C API

fizz exports a C-compatible API for integration with Swift, GTK, or other frameworks.

```c
#include "libfizz.h"

FizzPhysicsConfig config = {
    .gravity_y = -9.81,
    .dt = 1.0 / 60.0,
};

FizzWorld* world = fizz_world_create(&config);
FizzEntityId ball = fizz_entity_add(world, 0, 5, 0, 0, 0, 0, FIZZ_STANDARD);
fizz_world_step(world);
fizz_world_destroy(world);
```

## Architecture

| Module | Purpose |
|--------|---------|
| `math.zig` | Vec3, Mat3 linear algebra |
| `types.zig` | Entity, Label, PhysicsType, Camera |
| `dynamics.zig` | Kalman prediction, collision resolution |
| `gmm.zig` | Gaussian mixture observation model |
| `smc.zig` | Particle filter, resampling, Gibbs rejuvenation |
| `cabi.zig` | C ABI exports |

## Physics Types

| Type | Friction | Elasticity | Behavior |
|------|----------|------------|----------|
| standard | 0.3 | 0.5 | Default dynamics |
| bouncy | 0.2 | 0.9 | High restitution |
| sticky | 0.8 | 0.1 | High friction, low bounce |
| slippery | 0.05 | 0.6 | Low friction (ice-like) |

## License

AGPL-3.0
