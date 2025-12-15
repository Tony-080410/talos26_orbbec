#include <benchmark/benchmark.h>
#include "ukf_logic.hpp"

namespace spike::bench {
    // ============================================================================
    // Benchmark: UKF Predict (Target: < 200ns)
    // ============================================================================

    static void BM_UKFPredict(benchmark::State &state) {
        TrackerSoA tracker{};
        UKFState ukf{};

        // Initialize with typical values
        const Vec3 pos(1.0f, 2.0f, 5.0f);
        init_from_detection(tracker, ukf, pos, 0.5f);
        tracker.vel[0] = 0.5f;
        tracker.vel[1] = 0.3f;
        tracker.v_yaw = 1.0f;

        for (auto _: state) {
            predict(tracker, ukf, 2.0f); // 2ms timestep
            benchmark::DoNotOptimize(tracker);
            benchmark::DoNotOptimize(ukf);
        }
    }

    BENCHMARK(BM_UKFPredict);

    // ============================================================================
    // Benchmark: UKF Update (Target: < 300ns)
    // ============================================================================

    static void BM_UKFUpdate(benchmark::State &state) {
        TrackerSoA tracker{};
        UKFState ukf{};

        const Vec3 pos(1.0f, 2.0f, 5.0f);
        init_from_detection(tracker, ukf, pos, 0.5f);

        const Vec3 measurement(1.01f, 2.01f, 5.01f);

        for (auto _: state) {
            constexpr float meas_yaw = 0.51f;
            update(tracker, ukf, measurement, meas_yaw);
            benchmark::DoNotOptimize(tracker);
            benchmark::DoNotOptimize(ukf);
        }
    }

    BENCHMARK(BM_UKFUpdate);

    // ============================================================================
    // Benchmark: Scalar Update (single dimension)
    // ============================================================================

    static void BM_ScalarUpdate(benchmark::State &state) {
        TrackerSoA tracker{};
        UKFState ukf{};

        const Vec3 pos(0.0f, 0.0f, 5.0f);
        init_from_detection(tracker, ukf, pos, 0.0f);

        for (auto _: state) {
            scalar_update(tracker, ukf, 0, 0.01f, 0.05f);
            benchmark::DoNotOptimize(tracker);
            benchmark::DoNotOptimize(ukf);
        }
    }

    BENCHMARK(BM_ScalarUpdate);

    // ============================================================================
    // Benchmark: Mahalanobis Distance (Data Association)
    // ============================================================================

    static void BM_MahalanobisDistance(benchmark::State &state) {
        TrackerSoA tracker{};
        UKFState ukf{};

        const Vec3 pos(1.0f, 2.0f, 5.0f);
        init_from_detection(tracker, ukf, pos, 0.5f);

        const Vec3 candidate(1.1f, 2.1f, 5.1f);

        for (auto _: state) {
            auto d2 = mahalanobis_distance_sq(tracker, ukf, candidate);
            benchmark::DoNotOptimize(d2);
        }
    }

    BENCHMARK(BM_MahalanobisDistance);

    // ============================================================================
    // Benchmark: Full Predict-Update Cycle
    // ============================================================================

    static void BM_PredictUpdateCycle(benchmark::State &state) {
        TrackerSoA tracker{};
        UKFState ukf{};

        const Vec3 pos(1.0f, 2.0f, 5.0f);
        init_from_detection(tracker, ukf, pos, 0.5f);
        tracker.vel[0] = 0.5f;

        const Vec3 measurement(1.01f, 2.0f, 5.0f);

        for (auto _: state) {
            predict(tracker, ukf, 2.0f);
            update(tracker, ukf, measurement, 0.5f);
            benchmark::DoNotOptimize(tracker);
            benchmark::DoNotOptimize(ukf);
        }
    }

    BENCHMARK(BM_PredictUpdateCycle);

    // ============================================================================
    // Benchmark: Init from Detection
    // ============================================================================

    static void BM_InitFromDetection(benchmark::State &state) {
        TrackerSoA tracker{};
        UKFState ukf{};
        const Vec3 pos(1.0f, 2.0f, 5.0f);

        for (auto _: state) {
            init_from_detection(tracker, ukf, pos, 0.5f);
            benchmark::DoNotOptimize(tracker);
            benchmark::DoNotOptimize(ukf);
        }
    }

    BENCHMARK(BM_InitFromDetection);
} // namespace spike::bench

BENCHMARK_MAIN();
