/**
 * C ABI test harness for libfizz
 *
 * Validates the C API works correctly before building GUI clients.
 * Compile: clang -o test_cabi test_cabi.c -I../include -L../zig-out/lib -lfizz -Wl,-rpath,../zig-out/lib
 * Run: ./test_cabi
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "libfizz.h"

#define TEST(name) printf("Testing %s... ", name)
#define PASS() printf("PASS\n")
#define FAIL(msg) do { printf("FAIL: %s\n", msg); exit(1); } while(0)

static void test_version(void) {
    TEST("fizz_version");
    const char* version = fizz_version();
    if (version == NULL) FAIL("version is NULL");
    if (version[0] == '\0') FAIL("version is empty");
    printf("(v%s) ", version);
    PASS();
}

static void test_world_lifecycle(void) {
    TEST("world create/destroy");

    FizzPhysicsConfig config = fizz_physics_config_default();
    FizzWorld* world = fizz_world_create(&config);
    if (world == NULL) FAIL("world creation failed");

    fizz_world_destroy(world);
    fizz_world_destroy(NULL);  // Should be safe
    PASS();
}

static void test_entity_add(void) {
    TEST("entity add");

    FizzPhysicsConfig config = fizz_physics_config_default();
    FizzWorld* world = fizz_world_create(&config);

    // Add first entity
    FizzEntityId id1 = fizz_entity_add(world, 0, 5, 0, 0, 0, 0, FIZZ_PHYSICS_STANDARD);
    if (id1 == FIZZ_INVALID_ENTITY_ID) FAIL("first entity add failed");

    // Add second entity with different physics
    FizzEntityId id2 = fizz_entity_add(world, 2, 5, 0, 0, 0, 0, FIZZ_PHYSICS_BOUNCY);
    if (id2 == FIZZ_INVALID_ENTITY_ID) FAIL("second entity add failed");

    // IDs should be different
    if (id1 == id2) FAIL("entity IDs should be unique");

    // Count should be 2
    uint32_t count = fizz_world_entity_count(world);
    if (count != 2) FAIL("expected 2 entities");

    fizz_world_destroy(world);
    PASS();
}

static void test_entity_query(void) {
    TEST("entity query");

    FizzPhysicsConfig config = fizz_physics_config_default();
    FizzWorld* world = fizz_world_create(&config);

    FizzEntityId id = fizz_entity_add(world, 1.5f, 3.0f, -0.5f, 0.1f, 0.2f, 0.3f, FIZZ_PHYSICS_SLIPPERY);

    FizzEntityState state;
    FizzError err = fizz_world_get_entity(world, 0, &state);
    if (err != FIZZ_OK) FAIL("get_entity failed");

    if (state.id != id) FAIL("entity ID mismatch");
    if (fabsf(state.pos_x - 1.5f) > 0.001f) FAIL("pos_x mismatch");
    if (fabsf(state.pos_y - 3.0f) > 0.001f) FAIL("pos_y mismatch");
    if (fabsf(state.pos_z - (-0.5f)) > 0.001f) FAIL("pos_z mismatch");
    if (state.physics_type != FIZZ_PHYSICS_SLIPPERY) FAIL("physics_type mismatch");
    if (!state.is_alive) FAIL("entity should be alive");

    fizz_world_destroy(world);
    PASS();
}

static void test_physics_step(void) {
    TEST("physics step (gravity)");

    FizzPhysicsConfig config = fizz_physics_config_default();
    FizzWorld* world = fizz_world_create(&config);

    // Add entity at y=5, should fall due to gravity
    fizz_entity_add(world, 0, 5, 0, 0, 0, 0, FIZZ_PHYSICS_STANDARD);

    FizzEntityState before, after;
    fizz_world_get_entity(world, 0, &before);

    // Step physics 10 times
    for (int i = 0; i < 10; i++) {
        FizzError err = fizz_world_step(world);
        if (err != FIZZ_OK) FAIL("step failed");
    }

    fizz_world_get_entity(world, 0, &after);

    // Entity should have fallen (y decreased)
    if (after.pos_y >= before.pos_y) FAIL("entity should have fallen");

    // Should have downward velocity
    if (after.vel_y >= 0) FAIL("entity should have downward velocity");

    fizz_world_destroy(world);
    PASS();
}

static void test_entity_manipulation(void) {
    TEST("entity manipulation");

    FizzPhysicsConfig config = fizz_physics_config_default();
    FizzWorld* world = fizz_world_create(&config);

    FizzEntityId id = fizz_entity_add(world, 0, 0, 0, 0, 0, 0, FIZZ_PHYSICS_STANDARD);

    // Set position
    FizzError err = fizz_entity_set_position(world, id, 10, 20, 30);
    if (err != FIZZ_OK) FAIL("set_position failed");

    FizzEntityState state;
    fizz_world_get_entity(world, 0, &state);
    if (fabsf(state.pos_x - 10) > 0.001f) FAIL("position not set");

    // Set velocity
    err = fizz_entity_set_velocity(world, id, 1, 2, 3);
    if (err != FIZZ_OK) FAIL("set_velocity failed");

    fizz_world_get_entity(world, 0, &state);
    if (fabsf(state.vel_x - 1) > 0.001f) FAIL("velocity not set");

    // Apply force
    err = fizz_entity_apply_force(world, id, 100, 0, 0);
    if (err != FIZZ_OK) FAIL("apply_force failed");

    fizz_world_get_entity(world, 0, &state);
    if (state.vel_x <= 1) FAIL("force not applied");

    // Remove entity
    err = fizz_entity_remove(world, id);
    if (err != FIZZ_OK) FAIL("remove failed");

    if (fizz_world_entity_count(world) != 0) FAIL("entity should be removed");

    fizz_world_destroy(world);
    PASS();
}

static void test_invalid_handles(void) {
    TEST("invalid handle safety");

    // All these should return errors, not crash
    FizzError err;

    err = fizz_world_step(NULL);
    if (err != FIZZ_INVALID_HANDLE) FAIL("expected INVALID_HANDLE for NULL world");

    FizzEntityId id = fizz_entity_add(NULL, 0, 0, 0, 0, 0, 0, FIZZ_PHYSICS_STANDARD);
    if (id != FIZZ_INVALID_ENTITY_ID) FAIL("expected INVALID_ENTITY_ID for NULL world");

    if (fizz_world_entity_count(NULL) != 0) FAIL("expected 0 count for NULL world");

    PASS();
}

static void test_smc_lifecycle(void) {
    TEST("SMC create/destroy");

    FizzPhysicsConfig physics = fizz_physics_config_default();
    FizzSMCConfig smc_config = fizz_smc_config_default();

    FizzSMC* smc = fizz_smc_create(&physics, &smc_config);
    if (smc == NULL) FAIL("SMC creation failed");

    float ess = fizz_smc_get_ess(smc);
    float temp = fizz_smc_get_temperature(smc);
    printf("(ESS=%.1f, temp=%.2f) ", ess, temp);

    fizz_smc_destroy(smc);
    fizz_smc_destroy(NULL);  // Should be safe
    PASS();
}

int main(void) {
    printf("=== libfizz C ABI Test Suite ===\n\n");

    test_version();
    test_world_lifecycle();
    test_entity_add();
    test_entity_query();
    test_physics_step();
    test_entity_manipulation();
    test_invalid_handles();
    test_smc_lifecycle();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
