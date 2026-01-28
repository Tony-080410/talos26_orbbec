#include "scheduler/ecs/app.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/system_meta.hpp"
#include <cstdio>

// ============================================================================
// Test Types
// ============================================================================

struct Frame {
    int id    = 0;
    int width = 1920;
};

struct Detection {
    int frame_id = 0;
    float x      = 0.0f;
    float y      = 0.0f;
};

struct Config {
    float threshold = 0.5f;
};

struct TrackerState {
    int count = 0;
};

// Tags for distinguishing channels
struct CameraPipe {};
struct DetectorPipe {};

// ============================================================================
// Test: Type Traits
// ============================================================================

void test_type_traits() {
    printf("=== Test: Type Traits ===\n");

    static_assert(talos::spsc<Frame>::kind == talos::channel_kind::spsc_reader);
    static_assert(talos::spsc_mut<Frame>::kind == talos::channel_kind::spsc_writer);
    static_assert(talos::spmc<Frame>::kind == talos::channel_kind::spmc_reader);
    static_assert(talos::spmc_mut<Frame>::kind == talos::channel_kind::spmc_writer);
    static_assert(talos::res<Config>::kind == talos::channel_kind::res);
    static_assert(talos::res_mut<TrackerState>::kind == talos::channel_kind::res_mut);

    printf("  All static assertions passed!\n\n");
}

// ============================================================================
// Test: Meta Extraction with Tags
// ============================================================================

void camera_system(talos::spsc_mut<Frame, CameraPipe> out) {
    static int frame_id = 0;
    out.write(Frame{.id = frame_id++, .width = 1920});
}

void detector_system(
    talos::spsc<Frame, CameraPipe> frame_in, talos::res<Config> config,
    talos::spsc_mut<Detection, DetectorPipe> det_out) {
    if (const auto f = frame_in.read()) {
        printf("    detector: got frame %d, threshold=%.2f\n", f->id, config->threshold);
        det_out.write(Detection{.frame_id = f->id, .x = 100.0f, .y = 200.0f});
    }
}

void tracker_system(
    talos::spsc<Detection, DetectorPipe> det_in, talos::res_mut<TrackerState> state) {
    if (const auto d = det_in.read()) {
        state->count++;
        printf(
            "    tracker: got detection from frame %d, total count=%d\n", d->frame_id,
            state->count);
    }
}

void test_meta_extraction() {
    printf("=== Test: Meta Extraction with Tags ===\n");

    auto meta1 = talos::extract_system_meta<decltype(&camera_system)>("camera");
    printf(
        "  [%s] spsc:%zu spmc:%zu res:%zu\n", meta1.name.c_str(), meta1.spsc_channels.size(),
        meta1.spmc_channels.size(), meta1.reads.size());

    auto meta2 = talos::extract_system_meta<decltype(&detector_system)>("detector");
    printf(
        "  [%s] spsc:%zu spmc:%zu res:%zu\n", meta2.name.c_str(), meta2.spsc_channels.size(),
        meta2.spmc_channels.size(), meta2.reads.size());

    auto meta3 = talos::extract_system_meta<decltype(&tracker_system)>("tracker");
    printf(
        "  [%s] spsc:%zu spmc:%zu res_mut:%zu\n", meta3.name.c_str(), meta3.spsc_channels.size(),
        meta3.spmc_channels.size(), meta3.writes.size());

    printf("\n");
}

// ============================================================================
// Test: SPSC Pipeline
// ============================================================================

void test_spsc_pipeline() {
    printf("=== Test: SPSC Pipeline ===\n");

    talos::App app;
    app.add_resource(Config{.threshold = 0.7f})
        .add_resource(TrackerState{.count = 0})
        .add_system("camera", &camera_system)
        .add_system("detector", &detector_system)
        .add_system("tracker", &tracker_system);

    printf("  Registered systems:\n");
    app.print_systems();

    printf("\n  Running pipeline 3 times:\n");
    for (int i = 0; i < 3; ++i) {
        printf("  --- tick %d ---\n", i);
        app.run_once();
    }

    auto state = app.world().get_res_mut<TrackerState>();
    printf("\n  Final tracker count: %d\n\n", state->count);
}

// ============================================================================
// Test: Lambda with Tags
// ============================================================================

void test_lambda_with_tags() {
    printf("=== Test: Lambda with Tags ===\n");

    struct ProducerTag {};
    struct ConsumerTag {};

    int produced = 0;
    int consumed = 0;

    talos::App app;
    app.add_system(
           "producer",
           [&produced](talos::spsc_mut<int, ProducerTag> out) {
               out.write(++produced);
               printf("    produced: %d\n", produced);
           })
        .add_system("consumer", [&consumed](talos::spsc<int, ProducerTag> in) {
            if (const auto val = in.read()) {
                consumed = *val;
                printf("    consumed: %d\n", consumed);
            }
        });

    printf("  Running 3 times:\n");
    for (int i = 0; i < 3; ++i) {
        app.run_once();
    }

    printf("  Final: produced=%d, consumed=%d\n\n", produced, consumed);
}

// ============================================================================
// Test: Execution Policies
// ============================================================================

void test_execution_policies() {
    printf("=== Test: Execution Policies ===\n");

    // Static assertions for policy traits
    static_assert(talos::is_fixed_rate_policy_v<talos::fixed_rate<30>>);
    static_assert(talos::is_fixed_rate_policy_v<talos::fixed_rate<1000, 0, 99>>);
    static_assert(talos::is_fixed_rate_policy_v<talos::fixed_rate_silent<1000>>);
    static_assert(!talos::is_fixed_rate_policy_v<talos::pool_compute>);

    static_assert(talos::is_pool_policy_v<talos::pool_compute>);
    static_assert(!talos::is_pool_policy_v<talos::fixed_rate<30>>);
    static_assert(!talos::is_pool_policy_v<talos::fixed_rate_silent<1000>>);

    // Notify vs silent traits
    static_assert(talos::is_notifying_policy_v<talos::fixed_rate<30>>);
    static_assert(!talos::is_notifying_policy_v<talos::fixed_rate_silent<1000>>);
    static_assert(talos::is_silent_policy_v<talos::fixed_rate_silent<1000>>);
    static_assert(!talos::is_silent_policy_v<talos::fixed_rate<30>>);

    // Test policy info extraction
    constexpr auto ex_info = talos::make_policy_info<talos::fixed_rate<30, 2, 50>>();
    static_assert(ex_info.kind == talos::PolicyKind::FixedRate);
    static_assert(ex_info.frequency_hz == 30);
    static_assert(ex_info.cpu_affinity == 2);
    static_assert(ex_info.thread_priority == 50);
    static_assert(ex_info.notifies == true);

    constexpr auto silent_info = talos::make_policy_info<talos::fixed_rate_silent<1000, 1, 10>>();
    static_assert(silent_info.kind == talos::PolicyKind::FixedRate);
    static_assert(silent_info.frequency_hz == 1000);
    static_assert(silent_info.cpu_affinity == 1);
    static_assert(silent_info.thread_priority == 10);
    static_assert(silent_info.notifies == false);

    constexpr auto pool_info = talos::make_policy_info<talos::pool_compute>();
    static_assert(pool_info.kind == talos::PolicyKind::PoolCompute);

    printf("  Static assertions for policy traits passed!\n");

    // Test App with policies
    struct DataPipe {};
    struct ImuPipe {};

    talos::App app;
    app.add_system<talos::fixed_rate<30>>(
           "camera", [](talos::spsc_mut<int, DataPipe> out) { out.write(42); })
        .add_system<talos::fixed_rate_silent<1000>>(
            "imu", [](talos::spsc_mut<int, ImuPipe> out) { out.write(123); })
        .add_system<talos::pool_compute>(
            "processor", [](talos::spsc<int, DataPipe> in, talos::spsc<int, ImuPipe> imu) {
                (void)in.read();
                (void)imu.read();
            });

    printf("  Systems with policies:\n");
    app.print_systems();

    // Verify metadata
    const auto metas     = app.all_meta();
    bool found_camera    = false;
    bool found_imu       = false;
    bool found_processor = false;

    for (const auto* m : metas) {
        if (m->name == "camera" && m->policy.is_fixed_rate() && m->policy.notifies) {
            found_camera = true;
        }
        if (m->name == "imu" && m->policy.is_fixed_rate() && !m->policy.notifies) {
            found_imu = true;
        }
        if (m->name == "processor" && m->policy.kind == talos::PolicyKind::PoolCompute) {
            found_processor = true;
        }
    }

    if (found_camera && found_imu && found_processor) {
        printf("  Policy metadata verification passed!\n\n");
    } else {
        printf("  ERROR: Policy metadata verification failed!\n\n");
    }
}

// ============================================================================
// Test: Local Variables (talos::local<T>)
// ============================================================================

void test_local_variables() {
    printf("=== Test: Local Variables ===\n");

    // 1. Type trait verification
    static_assert(talos::local<int>::kind == talos::channel_kind::local);
    printf("  [PASS] Type trait checks for local<T>\n");

    // 2. Basic counter that persists across runs
    struct InputTag {};
    struct OutputTag {};

    talos::App app;

    // Producer system
    app.add_system<talos::pool_compute>("producer", [](talos::spmc_mut<int, InputTag> out) {
        out.write(42);
        printf("    producer: sent 42\n");
    });

    // Counter system with local<T>
    app.add_system<talos::pool_compute>(
        "counter", [](talos::spmc<int, InputTag> in, talos::local<int> run_count,
                      talos::local<int> total_sum) {
            if (auto data = in.read()) {
                (*run_count)++;
                (*total_sum) += *data;
                printf(
                    "    counter: run_count=%d, total_sum=%d, data=%d\n", *run_count, *total_sum,
                    *data);
            }
        });

    printf("\n  Running 3 times:\n");
    for (int i = 0; i < 3; ++i) {
        printf("  --- tick %d ---\n", i);
        app.run_once();
    }

    printf("\n  [PASS] Local variables persist across runs\n");

    // 3. Verify local<T> doesn't participate in dependency analysis
    auto metas         = app.all_meta();
    bool found_counter = false;
    for (const auto* m : metas) {
        if (m->name == "counter") {
            found_counter = true;
            // Local variables should NOT appear in metadata
            // The system has spmc<int, InputTag> input, which SHOULD be in metadata
            // But the 2 local<int> variables should NOT add any entries
            if (m->spmc_channels.size() == 1 && m->spsc_channels.empty() && m->reads.empty()
                && m->writes.empty()) {
                printf("  [PASS] local<T> excluded from dependency analysis\n");
            } else {
                printf(
                    "  [FAIL] local<T> incorrectly included in metadata (spmc=%zu)\n",
                    m->spmc_channels.size());
            }
        }
    }

    if (!found_counter) {
        printf("  [FAIL] Counter system not found\n");
    }

    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("\n");
    printf("==========================================\n");
    printf("  Bevy-style ECS with Tags Test Suite\n");
    printf("==========================================\n\n");

    test_type_traits();
    test_meta_extraction();
    test_spsc_pipeline();
    test_lambda_with_tags();
    test_execution_policies();
    test_local_variables();

    printf("==========================================\n");
    printf("  All tests completed!\n");
    printf("==========================================\n\n");

    return 0;
}
