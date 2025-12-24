/**
 * Performance Benchmarks for Fast TF
 *
 * Benchmarks:
 * - Transform creation (RPY and quaternion)
 * - Matrix composition
 * - Circular buffer insertion and query
 * - Static vs dynamic transform access
 * - Interpolation performance
 */

#include "fast_tf/core.hpp"
#include <benchmark/benchmark.h>
#include <vector>

using namespace fast_tf;

// Define frames for benchmarking
DECLARE_FRAME(bench_world);
DECLARE_FRAME(bench_base, bench_world);
DECLARE_FRAME(bench_camera, bench_base);

// ============================================================================
// Transform Creation Benchmarks
// ============================================================================

static void BenchmarkCreateFromRPY(benchmark::State& state) {
    for (auto _ : state) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BenchmarkCreateFromRPY);

static void BenchmarkCreateFromQuaternion(benchmark::State& state) {
    for (auto _ : state) {
        TransformMatrix4d tf =
            TransformMatrix4d::from_quaternion(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BenchmarkCreateFromQuaternion);

static void BenchmarkCreateFromIdentity(benchmark::State& state) {
    for (auto _ : state) {
        TransformMatrix4d tf;
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BenchmarkCreateFromIdentity);

// ============================================================================
// Transform Composition Benchmarks
// ============================================================================

static void BenchmarkMatrixComposition(benchmark::State& state) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 0.0, 0.0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    for (auto _ : state) {
        TransformMatrix4d result = tf1 * tf2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BenchmarkMatrixComposition);

static void BenchmarkMatrixInverse(benchmark::State& state) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);

    for (auto _ : state) {
        TransformMatrix4d inv = tf.inverse();
        benchmark::DoNotOptimize(inv);
    }
}
BENCHMARK(BenchmarkMatrixInverse);

static void BenchmarkExtractTranslation(benchmark::State& state) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);

    for (auto _ : state) {
        Eigen::Vector3d trans = tf.translation();
        benchmark::DoNotOptimize(trans);
    }
}
BENCHMARK(BenchmarkExtractTranslation);

static void BenchmarkExtractQuaternion(benchmark::State& state) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);

    for (auto _ : state) {
        Eigen::Quaterniond quat = tf.quaternion();
        benchmark::DoNotOptimize(quat);
    }
}
BENCHMARK(BenchmarkExtractQuaternion);

// ============================================================================
// Interpolation Benchmarks
// ============================================================================

static void BenchmarkLinearInterpolation(benchmark::State& state) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);

    for (auto _ : state) {
        TransformMatrix4d lerped = TransformMatrix4d::lerp(tf1, tf2, 0.5);
        benchmark::DoNotOptimize(lerped);
    }
}
BENCHMARK(BenchmarkLinearInterpolation);

// ============================================================================
// Circular Buffer Benchmarks
// ============================================================================

static void BenchmarkBufferInsertion(benchmark::State& state) {
    CircularTransformBuffer<1000> buffer;
    uint64_t timestamp = 1000000;

    for (auto _ : state) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
        buffer.push(timestamp++, tf);
    }
}
BENCHMARK(BenchmarkBufferInsertion);

static void BenchmarkBufferQuery(benchmark::State& state) {
    CircularTransformBuffer<1000> buffer;

    // Pre-populate buffer
    for (int i = 0; i < 100; ++i) {
        TransformMatrix4d tf =
            TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, static_cast<double>(i), 0.0, 0.0);
        buffer.push(1000000 + i * 1000, tf);
    }

    uint64_t query_ts = 1000000;
    for (auto _ : state) {
        auto result = buffer.at(query_ts);
        query_ts += 1000;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BenchmarkBufferQuery);

static void BenchmarkSmallBufferInsertion(benchmark::State& state) {
    CircularTransformBuffer<10> buffer;
    uint64_t timestamp = 1000000;

    for (auto _ : state) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
        buffer.push(timestamp++, tf);
    }
}
BENCHMARK(BenchmarkSmallBufferInsertion);

static void BenchmarkLargeBufferInsertion(benchmark::State& state) {
    CircularTransformBuffer<10000> buffer;
    uint64_t timestamp = 1000000;

    for (auto _ : state) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
        buffer.push(timestamp++, tf);
    }
}
BENCHMARK(BenchmarkLargeBufferInsertion);

// ============================================================================
// Static Transform Benchmarks
// ============================================================================

static void BenchmarkStaticTransformRegistration(benchmark::State& state) {
    for (auto _ : state) {
        StaticTransformStore::clear();
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
        StaticTransformStore::register_transform<bench_camera_Frame, bench_base_Frame>(tf);
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BenchmarkStaticTransformRegistration);

static void BenchmarkStaticTransformRetrieval(benchmark::State& state) {
    StaticTransformStore::clear();
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
    StaticTransformStore::register_transform<bench_camera_Frame, bench_base_Frame>(tf);

    for (auto _ : state) {
        const TransformMatrix4d* result =
            StaticTransformStore::get_transform<bench_camera_Frame, bench_base_Frame>();
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BenchmarkStaticTransformRetrieval);

// ============================================================================
// Frame Configuration Benchmarks
// ============================================================================

static void BenchmarkBufferConfigSet(benchmark::State& state) {
    for (auto _ : state) {
        FrameBufferConfig::set_buffer_size<bench_world_Frame>(100);
    }
}
BENCHMARK(BenchmarkBufferConfigSet);

static void BenchmarkBufferConfigGet(benchmark::State& state) {
    FrameBufferConfig::set_buffer_size<bench_world_Frame>(100);

    for (auto _ : state) {
        std::size_t size = FrameBufferConfig::get_buffer_size<bench_world_Frame>();
        benchmark::DoNotOptimize(size);
    }
}
BENCHMARK(BenchmarkBufferConfigGet);

// ============================================================================
// Comparison Benchmarks
// ============================================================================

// Static vs Dynamic Transform Access
static void BenchmarkStaticVsDynamicCreation(benchmark::State& state) {
    // Create via static method
    for (auto _ : state) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);
        benchmark::DoNotOptimize(tf);
    }
}
BENCHMARK(BenchmarkStaticVsDynamicCreation);

// Matrix operation performance
static void BenchmarkMatrixMultiply3Levels(benchmark::State& state) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 0.0, 0.0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    TransformMatrix4d tf3 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    for (auto _ : state) {
        TransformMatrix4d result = tf1 * tf2 * tf3;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BenchmarkMatrixMultiply3Levels);

// ============================================================================
// Registration Data Structure Tests
// ============================================================================

static void BenchmarkFrameRegistryRegister(benchmark::State& state) {
    for (auto _ : state) {
        FrameRegistry::clear();
        FrameRegistry::register_frame("world", 1);
        FrameRegistry::register_frame("base", 2);
        FrameRegistry::register_frame("camera", 3);
    }
}
BENCHMARK(BenchmarkFrameRegistryRegister);

static void BenchmarkFrameRegistryLookup(benchmark::State& state) {
    FrameRegistry::clear();
    FrameRegistry::register_frame("camera", 12345);

    for (auto _ : state) {
        uint64_t id = FrameRegistry::get_frame_id("camera");
        benchmark::DoNotOptimize(id);
    }
}
BENCHMARK(BenchmarkFrameRegistryLookup);

BENCHMARK_MAIN();
