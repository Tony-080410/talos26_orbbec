/**
 * Fast TF 性能基准测试
 * 与ROS2 tf2进行比较（如果可用）
 */

#include <benchmark/benchmark.h>
#include <chrono>
#include <random>
#include "fast_tf/core.hpp"

using namespace fast_tf;

// ============================================================================
// 性能测试工具
// ============================================================================

class FastTFBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State&) override {
        // 初始化缓冲区
        for (int i = 0; i < 1000; ++i) {
            double x = static_cast<double>(i) * 0.1;
            TransformMatrix4d tf =
                TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, x, 0, 0);
            buffer.push(i * 1000, tf);
        }
    }

    CircularTransformBuffer<1000> buffer;
};

// ============================================================================
// 基准测试：缓冲区操作
// ============================================================================

BENCHMARK_F(FastTFBenchmark, PushOperation)(benchmark::State& state) {
    TransformMatrix4d tf =
        TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1, 2, 3);

    for (auto _ : state) {
        buffer.push(state.iterations() * 1000, tf);
    }
}

BENCHMARK_F(FastTFBenchmark, ExactLookup)(benchmark::State& state) {
    for (auto _ : state) {
        auto entry = buffer.at(500000);
        benchmark::DoNotOptimize(entry);
    }
}

BENCHMARK_F(FastTFBenchmark, InterpolatedLookup)(benchmark::State& state) {
    for (auto _ : state) {
        auto entry = buffer.at_interpolated(500000);
        benchmark::DoNotOptimize(entry);
    }
}

// ============================================================================
// 基准测试：矩阵运算
// ============================================================================

BENCHMARK(TransformMatrixMultiplication)(benchmark::State& state) {
    TransformMatrix4d tf1 =
        TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1, 2, 3);
    TransformMatrix4d tf2 =
        TransformMatrix4d::from_rpy(0.2, 0.3, 0.4, 2, 3, 4);

    for (auto _ : state) {
        auto result = tf1 * tf2;
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(TransformMatrixInverse)(benchmark::State& state) {
    TransformMatrix4d tf =
        TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1, 2, 3);

    for (auto _ : state) {
        auto inv = tf.inverse();
        benchmark::DoNotOptimize(inv);
    }
}

BENCHMARK(TransformMatrixInterpolation)(benchmark::State& state) {
    TransformMatrix4d tf1 =
        TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0, 0, 0);
    TransformMatrix4d tf2 =
        TransformMatrix4d::from_rpy(0.1, 0.1, 0.1, 1, 1, 1);

    for (auto _ : state) {
        auto result = TransformMatrix4d::lerp(tf1, tf2, 0.5);
        benchmark::DoNotOptimize(result);
    }
}

// ============================================================================
// 基准测试：编译期字符串
// ============================================================================

BENCHMARK(CompileTimeStringHash)(benchmark::State& state) {
    constexpr CompileTimeString<5> world("world");
    constexpr CompileTimeString<9> base_link("base_link");

    for (auto _ : state) {
        auto h1 = world.hash();
        auto h2 = base_link.hash();
        benchmark::DoNotOptimize(h1);
        benchmark::DoNotOptimize(h2);
    }
}

// ============================================================================
// 基准测试：树结构
// ============================================================================

void TreeStructureCreation(benchmark::State& state) {
    for (auto _ : state) {
        using SimpleTree = TransformTreeStructure<
            TreeNode<5, NullParent, BufferCapacity<100>>,
            TreeNode<9, ParentRef<5>, BufferCapacity<50>>>;

        SimpleTree tree;
        benchmark::DoNotOptimize(tree);
    }
}

BENCHMARK(TreeStructureCreation);

// ============================================================================
// 基准测试：框架注册
// ============================================================================

void FrameRegistration(benchmark::State& state) {
    for (auto _ : state) {
        FrameRegistry::clear();

        for (int i = 0; i < 100; ++i) {
            std::string name = "frame_" + std::to_string(i);
            uint64_t id = hash_string_view(name);
            FrameRegistry::register_frame(name, id);
        }

        benchmark::DoNotOptimize(FrameRegistry::get_frame_name(0));
    }
}

BENCHMARK(FrameRegistration);

// ============================================================================
// 基准测试：并行操作
// ============================================================================

void ParallelBufferQueries(benchmark::State& state) {
    CircularTransformBuffer<100> buffer;
    TransformMatrix4d tf =
        TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1, 2, 3);

    for (int i = 0; i < 100; ++i) {
        buffer.push(i * 10000, tf);
    }

    for (auto _ : state) {
        // 模拟并行查询（在真实场景中使用TBB）
        for (int i = 0; i < 100; ++i) {
            auto entry = buffer.at_interpolated(i * 10000 + 5000);
            benchmark::DoNotOptimize(entry);
        }
    }
}

BENCHMARK(ParallelBufferQueries)->Threads(4);

// ============================================================================
// 基准测试：大规模场景
// ============================================================================

void LargeCachedLookup(benchmark::State& state) {
    // 模拟大型变换树（10000个变换）
    CircularTransformBuffer<10000> buffer;

    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < 10000; ++i) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(
            dis(gen), dis(gen), dis(gen),
            dis(gen), dis(gen), dis(gen));
        buffer.push(i * 1000, tf);
    }

    std::uniform_int_distribution<> time_dis(0, 9999000);

    for (auto _ : state) {
        uint64_t ts = time_dis(gen);
        auto entry = buffer.at_interpolated(ts);
        benchmark::DoNotOptimize(entry);
    }
}

BENCHMARK(LargeCachedLookup);

// ============================================================================
// 基准测试：内存使用
// ============================================================================

void MemoryUsage(benchmark::State& state) {
    for (auto _ : state) {
        CircularTransformBuffer<1000> buffer;

        TransformMatrix4d tf =
            TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);

        for (int i = 0; i < 1000; ++i) {
            buffer.push(i * 1000, tf);
        }

        benchmark::DoNotOptimize(buffer.size());
    }

    // 报告内存使用
    state.SetItemsProcessed(1000 * state.iterations());
}

BENCHMARK(MemoryUsage)->Unit(benchmark::kMillisecond);

// ============================================================================
// 基准测试：缓存效率
// ============================================================================

void CacheEfficiency(benchmark::State& state) {
    // 小缓冲区应该更高效
    CircularTransformBuffer<64> buffer;

    TransformMatrix4d tf =
        TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);

    for (int i = 0; i < 64; ++i) {
        buffer.push(i * 1000, tf);
    }

    std::mt19937 gen(0);
    std::uniform_int_distribution<> dis(0, 63000);

    for (auto _ : state) {
        auto entry = buffer.at_interpolated(dis(gen));
        benchmark::DoNotOptimize(entry);
    }
}

BENCHMARK(CacheEfficiency)->Name("SmallBufferLookup");

// ============================================================================
// 基准测试：线程安全操作
// ============================================================================

void ThreadSafeRead(benchmark::State& state) {
    CircularTransformBuffer<100> buffer;

    for (int i = 0; i < 100; ++i) {
        TransformMatrix4d tf =
            TransformMatrix4d::from_rpy(0, 0, 0, i * 0.1, 0, 0);
        buffer.push(i * 1000, tf);
    }

    for (auto _ : state) {
        auto entry = buffer.at_interpolated(50000);
        benchmark::DoNotOptimize(entry);
    }
}

BENCHMARK(ThreadSafeRead)->ThreadRange(1, 8);

BENCHMARK_MAIN();
