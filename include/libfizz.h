/**
 * libfizz - Probabilistic Physics Engine for Inverse Simulation
 *
 * A generative model for 3D physics that can be run "backward" via SMC
 * to infer world dynamics from observations.
 *
 * Architecture follows the Ghostty pattern:
 * - Opaque handles for internal state (FizzWorld, FizzSMC)
 * - C-compatible structs for data exchange
 * - Explicit memory management (create/destroy pairs)
 *
 * Thread safety: All functions are thread-safe for distinct handles.
 * Do not call functions on the same handle from multiple threads.
 */

#ifndef LIBFIZZ_H
#define LIBFIZZ_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * Opaque Handles
 * ========================================================================== */

/** Opaque handle to a Fizz world. */
typedef struct FizzWorld FizzWorld;

/** Opaque handle to an SMC inference state. */
typedef struct FizzSMC FizzSMC;

/* ==========================================================================
 * Basic Types
 * ========================================================================== */

/**
 * Entity identifier (birth_time << 16 | birth_index + 1).
 * Uses uint64_t to preserve full 32-bit birth_time without truncation.
 * Value 0 is reserved as invalid/error indicator.
 */
typedef uint64_t FizzEntityId;

/** Invalid entity ID (returned on error). */
#define FIZZ_INVALID_ENTITY_ID ((FizzEntityId)0)

/** 3D vector for C interop. */
typedef struct {
    float x;
    float y;
    float z;
} FizzVec3;

/** Physics type enumeration. */
typedef enum {
    FIZZ_PHYSICS_STANDARD = 0,
    FIZZ_PHYSICS_BOUNCY = 1,
    FIZZ_PHYSICS_STICKY = 2,
    FIZZ_PHYSICS_SLIPPERY = 3
} FizzPhysicsType;

/** Error codes. */
typedef enum {
    FIZZ_OK = 0,
    FIZZ_OUT_OF_MEMORY = -1,
    FIZZ_INVALID_HANDLE = -2,
    FIZZ_INVALID_ENTITY = -3,
    FIZZ_INVALID_ARGUMENT = -4
} FizzError;

/* ==========================================================================
 * Configuration Structs
 * ========================================================================== */

/** Physics configuration. */
typedef struct {
    float gravity_x;
    float gravity_y;
    float gravity_z;
    float dt;
    float ground_height;
    float bounds_min_x;
    float bounds_min_y;
    float bounds_min_z;
    float bounds_max_x;
    float bounds_max_y;
    float bounds_max_z;
    float process_noise;
    float crp_alpha;
    float survival_prob;
} FizzPhysicsConfig;

/** SMC configuration. */
typedef struct {
    uint32_t num_particles;
    float ess_threshold;
    float observation_noise;
    bool use_tempering;
    float initial_temperature;
    float temperature_increment;
    uint32_t gibbs_sweeps;
} FizzSMCConfig;

/** Camera configuration for rendering observations. */
typedef struct {
    float pos_x;
    float pos_y;
    float pos_z;
    float target_x;
    float target_y;
    float target_z;
    float up_x;
    float up_y;
    float up_z;
    float fov;
    float aspect;
    float near;
    float far;
} FizzCamera;

/** Entity state for queries. */
typedef struct {
    FizzEntityId id;
    float pos_x;
    float pos_y;
    float pos_z;
    float vel_x;
    float vel_y;
    float vel_z;
    FizzPhysicsType physics_type;
    bool is_alive;
} FizzEntityState;

/** Physics type posterior for a single entity. */
typedef struct {
    float prob_standard;
    float prob_bouncy;
    float prob_sticky;
    float prob_slippery;
} FizzPosterior;

/** Camera belief (posterior over agent's pose). */
typedef struct {
    float mean_pos_x;
    float mean_pos_y;
    float mean_pos_z;
    float mean_yaw;       /**< Mean yaw in radians */
    float var_pos_x;
    float var_pos_y;
    float var_pos_z;
    float var_yaw;
} FizzCameraBelief;

/* ==========================================================================
 * Default Configuration Helpers
 * ========================================================================== */

/** Create default physics configuration. */
static inline FizzPhysicsConfig fizz_physics_config_default(void) {
    return (FizzPhysicsConfig){
        .gravity_x = 0.0f,
        .gravity_y = -9.81f,
        .gravity_z = 0.0f,
        .dt = 1.0f / 60.0f,
        .ground_height = 0.0f,
        .bounds_min_x = -10.0f,
        .bounds_min_y = -10.0f,
        .bounds_min_z = -10.0f,
        .bounds_max_x = 10.0f,
        .bounds_max_y = 10.0f,
        .bounds_max_z = 10.0f,
        .process_noise = 0.01f,
        .crp_alpha = 1.0f,
        .survival_prob = 0.99f
    };
}

/** Create default SMC configuration. */
static inline FizzSMCConfig fizz_smc_config_default(void) {
    return (FizzSMCConfig){
        .num_particles = 100,
        .ess_threshold = 0.5f,
        .observation_noise = 0.1f,
        .use_tempering = true,
        .initial_temperature = 0.1f,
        .temperature_increment = 0.1f,
        .gibbs_sweeps = 1
    };
}

/** Create default camera configuration. */
static inline FizzCamera fizz_camera_default(void) {
    return (FizzCamera){
        .pos_x = 0.0f,
        .pos_y = 5.0f,
        .pos_z = 10.0f,
        .target_x = 0.0f,
        .target_y = 0.0f,
        .target_z = 0.0f,
        .up_x = 0.0f,
        .up_y = 1.0f,
        .up_z = 0.0f,
        .fov = 0.785398f,  /* pi/4 */
        .aspect = 1.0f,
        .near = 0.1f,
        .far = 100.0f
    };
}

/* ==========================================================================
 * World API
 * ========================================================================== */

/**
 * Create a new physics world.
 * @param config Physics configuration
 * @return World handle, or NULL on failure
 */
FizzWorld* fizz_world_create(const FizzPhysicsConfig* config);

/**
 * Destroy a physics world.
 * @param world World handle (NULL is safe)
 */
void fizz_world_destroy(FizzWorld* world);

/**
 * Add an entity to the world.
 * @param world World handle
 * @param x,y,z Initial position
 * @param vx,vy,vz Initial velocity
 * @param physics_type Entity physics behavior
 * @return Entity ID, or FIZZ_INVALID_ENTITY_ID (0) on failure
 */
FizzEntityId fizz_entity_add(
    FizzWorld* world,
    float x, float y, float z,
    float vx, float vy, float vz,
    FizzPhysicsType physics_type);

/**
 * Remove an entity from the world (marks as dead).
 * @param world World handle
 * @param entity_id Entity to remove
 * @return FIZZ_OK on success
 */
FizzError fizz_entity_remove(FizzWorld* world, FizzEntityId entity_id);

/**
 * Apply a force to an entity.
 * @param world World handle
 * @param entity_id Target entity
 * @param fx,fy,fz Force vector
 * @return FIZZ_OK on success
 */
FizzError fizz_entity_apply_force(
    FizzWorld* world,
    FizzEntityId entity_id,
    float fx, float fy, float fz);

/**
 * Set entity position directly.
 * @param world World handle
 * @param entity_id Target entity
 * @param x,y,z New position
 * @return FIZZ_OK on success
 */
FizzError fizz_entity_set_position(
    FizzWorld* world,
    FizzEntityId entity_id,
    float x, float y, float z);

/**
 * Set entity velocity directly.
 * @param world World handle
 * @param entity_id Target entity
 * @param vx,vy,vz New velocity
 * @return FIZZ_OK on success
 */
FizzError fizz_entity_set_velocity(
    FizzWorld* world,
    FizzEntityId entity_id,
    float vx, float vy, float vz);

/**
 * Step the physics simulation.
 * @param world World handle
 * @return FIZZ_OK on success
 */
FizzError fizz_world_step(FizzWorld* world);

/**
 * Get entity count (alive entities only).
 * @param world World handle
 * @return Number of alive entities
 */
uint32_t fizz_world_entity_count(FizzWorld* world);

/**
 * Get entity state by index (0 to entity_count-1).
 * @param world World handle
 * @param index Entity index
 * @param out Output entity state
 * @return FIZZ_OK on success
 */
FizzError fizz_world_get_entity(
    FizzWorld* world,
    uint32_t index,
    FizzEntityState* out);

/* ==========================================================================
 * SMC Inference API
 * ========================================================================== */

/**
 * Create a new SMC inference state.
 * @param physics_config Physics configuration for simulation
 * @param smc_config SMC algorithm configuration
 * @return SMC handle, or NULL on failure
 */
FizzSMC* fizz_smc_create(
    const FizzPhysicsConfig* physics_config,
    const FizzSMCConfig* smc_config);

/**
 * Destroy an SMC inference state.
 * @param smc SMC handle (NULL is safe)
 */
void fizz_smc_destroy(FizzSMC* smc);

/**
 * Initialize SMC with prior entity positions.
 * @param smc SMC handle
 * @param positions Array of initial positions
 * @param velocities Array of initial velocities
 * @param count Number of entities
 * @return FIZZ_OK on success
 */
FizzError fizz_smc_init_prior(
    FizzSMC* smc,
    const FizzVec3* positions,
    const FizzVec3* velocities,
    uint32_t count);

/**
 * Step SMC inference with an observation from the world.
 * @param smc SMC handle
 * @param world World handle (used to generate observation)
 * @param camera Camera configuration for observation
 * @return FIZZ_OK on success
 */
FizzError fizz_smc_step_with_world(
    FizzSMC* smc,
    FizzWorld* world,
    const FizzCamera* camera);

/**
 * Step SMC inference with an RGB image observation.
 * Primary API for external renderers (SwiftUI, GTK, etc).
 * The image represents what the agent "sees" from an unknown camera position.
 * FastSLAM pattern: each particle has its own camera hypothesis.
 * @param smc SMC handle
 * @param rgb_data RGB24 pixel data (row-major, 3 bytes per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @return FIZZ_OK on success
 */
FizzError fizz_smc_step_with_image(
    FizzSMC* smc,
    const uint8_t* rgb_data,
    uint32_t width,
    uint32_t height);

/**
 * Get number of entities tracked by SMC.
 * @param smc SMC handle
 * @return Number of tracked entities
 */
uint32_t fizz_smc_entity_count(FizzSMC* smc);

/**
 * Get physics type posteriors for all entities.
 * @param smc SMC handle
 * @param out Output array for posteriors
 * @param max_count Maximum number of posteriors to write
 * @return Number of posteriors written
 */
uint32_t fizz_smc_get_posteriors(
    FizzSMC* smc,
    FizzPosterior* out,
    uint32_t max_count);

/**
 * Get effective sample size.
 * @param smc SMC handle
 * @return ESS value (0 to num_particles)
 */
float fizz_smc_get_ess(FizzSMC* smc);

/**
 * Get current temperature.
 * @param smc SMC handle
 * @return Current temperature value
 */
float fizz_smc_get_temperature(FizzSMC* smc);

/**
 * Get camera belief (posterior over agent's pose).
 * FastSLAM pattern: agent doesn't know where it is, infers from observations.
 * @param smc SMC handle
 * @param out Output camera belief struct
 * @return FIZZ_OK on success
 */
FizzError fizz_smc_get_camera_belief(FizzSMC* smc, FizzCameraBelief* out);

/**
 * Get MAP (maximum a posteriori) camera pose.
 * Returns the camera pose from the highest-weight particle.
 * @param smc SMC handle
 * @param out_x Output X position
 * @param out_y Output Y position
 * @param out_z Output Z position
 * @param out_yaw Output yaw angle (radians)
 * @return FIZZ_OK on success
 */
FizzError fizz_smc_get_camera_map(
    FizzSMC* smc,
    float* out_x,
    float* out_y,
    float* out_z,
    float* out_yaw);

/* ==========================================================================
 * Version Info
 * ========================================================================== */

/**
 * Get library version string.
 * @return Version string (e.g., "0.1.0")
 */
const char* fizz_version(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBFIZZ_H */
