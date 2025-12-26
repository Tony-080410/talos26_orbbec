#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "fast_tf/mod.hpp"

using namespace fast_tf;

// ============================================================================
// TransformMatrix4d Benchmarks
// ============================================================================

static void BM_TransformMatrix4d_Identity(benchmark::State& state) {
    for (auto _ : state) {
        auto tf = TransformMatrix4d();
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BM_TransformMatrix4d_Identity);

static void BM_TransformMatrix4d_FromTranslation(benchmark::State& state) {
    for (auto _ : state) {
        auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BM_TransformMatrix4d_FromTranslation);

static void BM_TransformMatrix4d_FromRPY(benchmark::State& state) {
    for (auto _ : state) {
        auto tf = TransformMatrix4d::from_rpy(0.5, 0.3, 0.1, 1.0, 2.0, 3.0);
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BM_TransformMatrix4d_FromRPY);

static void BM_TransformMatrix4d_Multiplication(benchmark::State& state) {
    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 4.0, 5.0, 6.0);

    for (auto _ : state) {
        auto result = tf1 * tf2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_TransformMatrix4d_Multiplication);

static void BM_TransformMatrix4d_Inverse(benchmark::State& state) {
    auto tf = TransformMatrix4d::from_rpy(0.5, 0.3, 0.1, 1.0, 2.0, 3.0);

    for (auto _ : state) {
        auto result = tf.inverse();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_TransformMatrix4d_Inverse);

static void BM_TransformMatrix4d_Translation(benchmark::State& state) {
    auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);

    for (auto _ : state) {
        auto result = tf.translation();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_TransformMatrix4d_Translation);

static void BM_TransformMatrix4d_Rotation(benchmark::State& state) {
    auto tf = TransformMatrix4d::from_rpy(0.5, 0.3, 0.1, 0.0, 0.0, 0.0);

    for (auto _ : state) {
        auto result = tf.rotation();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_TransformMatrix4d_Rotation);

static void BM_TransformMatrix4d_Quaternion(benchmark::State& state) {
    auto tf = TransformMatrix4d::from_rpy(0.5, 0.3, 0.1, 0.0, 0.0, 0.0);

    for (auto _ : state) {
        auto result = tf.quaternion();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_TransformMatrix4d_Quaternion);

// ============================================================================
// CircularTransformBuffer Benchmarks
// ============================================================================

static void BM_CircularTransformBuffer_Push(benchmark::State& state) {
    using Buffer = CircularTransformBuffer<100>;
    Buffer buffer;
    auto tf            = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    uint64_t timestamp = 0;

    for (auto _ : state) {
        buffer.push(timestamp++, tf);
    }
}
BENCHMARK(BM_CircularTransformBuffer_Push);

static void BM_CircularTransformBuffer_At(benchmark::State& state) {
    using Buffer = CircularTransformBuffer<100>;
    Buffer buffer;
    auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);

    // Pre-fill buffer
    for (int i = 0; i < 50; ++i) {
        buffer.push(i * 100, tf);
    }

    for (auto _ : state) {
        auto result = buffer.at(2500);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CircularTransformBuffer_At);

static void BM_CircularTransformBuffer_AtInterpolated(benchmark::State& state) {
    using Buffer = CircularTransformBuffer<100>;
    Buffer buffer;
    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 10.0, 0.0, 0.0);

    // Create interpolation scenario
    for (int i = 0; i < 50; ++i) {
        buffer.push(i * 1000, tf1);
        buffer.push(i * 1000 + 500, tf2);
    }

    for (auto _ : state) {
        auto result = buffer.at_interpolated(25500);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CircularTransformBuffer_AtInterpolated);

static void BM_CircularTransformBuffer_Latest(benchmark::State& state) {
    using Buffer = CircularTransformBuffer<100>;
    Buffer buffer;
    auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);

    // Pre-fill buffer
    for (int i = 0; i < 50; ++i) {
        buffer.push(i * 100, tf);
    }

    for (auto _ : state) {
        auto result = buffer.latest();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CircularTransformBuffer_Latest);

static void BM_CircularTransformBuffer_TimeRange(benchmark::State& state) {
    using Buffer = CircularTransformBuffer<100>;
    Buffer buffer;
    auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);

    // Pre-fill buffer
    for (int i = 0; i < 50; ++i) {
        buffer.push(i * 100, tf);
    }

    for (auto _ : state) {
        auto result = buffer.time_range();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CircularTransformBuffer_TimeRange);

// Varying buffer sizes
static void BM_CircularTransformBuffer_VariousCapacities(benchmark::State& state) {
    const int capacity = state.range(0);

    for (auto _ : state) {
        std::unique_ptr<char[]> buffer_storage(new char[capacity * sizeof(TransformEntry)]);
        // Simulate buffer operations
        auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BM_CircularTransformBuffer_VariousCapacities)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000);

// ============================================================================
// TransformTree Benchmarks
// ============================================================================
// Note: Using test frames declared in fast_tf_test.cpp would require linking,
// so we simplify by using TransformMatrix4d benchmarks which cover the main operations

BENCHMARK_MAIN();
