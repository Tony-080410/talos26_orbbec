/**
 * @file integration_scheduler_benchmark.cpp
 * @brief Scheduler 综合性能基准测试
 *
 * 测试场景:
 * 1. 0-copy 开销对比
 * 2. 完整流水线吞吐量
 * 3. 真实负载模拟 (160Hz)
 * 4. 端到端延迟分布
 */

#include <benchmark/benchmark.h>
#include <oneapi/tbb/parallel_for.h>
#include "scheduler/scheduler.hpp"

#include <atomic>
#include <thread>

#include "test_strategy.hpp"

using namespace scheduler;
using namespace scheduler::test;

// ============ 0-Copy 开销测试 ============

// 测量 ImageBuffer::wrap (外部缓冲) 开销
static void BM_ImageBuffer_Wrap(benchmark::State &state) {
    std::vector<uint8_t> buffer(480 * 640 * 3);

    for (auto _: state) {
        auto img = ImageBuffer::wrap(buffer.data(), 480, 640, CV_8UC3);
        benchmark::DoNotOptimize(img);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ImageBuffer_Wrap);

// 测量 ImageBuffer::from_mat 开销
static void BM_ImageBuffer_FromMat(benchmark::State &state) {
    for (auto _: state) {
        cv::Mat mat(480, 640, CV_8UC3);
        auto img = ImageBuffer::from_mat(std::move(mat));
        benchmark::DoNotOptimize(img);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ImageBuffer_FromMat);

// 测量 Frame::wrap 完整开销
static void BM_Frame_Wrap(benchmark::State &state) {
    std::vector<uint8_t> buffer(480 * 640 * 3);
    uint64_t seq = 0;

    for (auto _: state) {
        auto frame = Frame::wrap(seq++, 1000000, buffer.data(), 480, 640, CV_8UC3);
        benchmark::DoNotOptimize(frame);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Frame_Wrap);

// 测量 Frame::from_mat 完整开销
static void BM_Frame_FromMat(benchmark::State &state) {
    uint64_t seq = 0;

    for (auto _: state) {
        cv::Mat mat(480, 640, CV_8UC3);
        auto frame = Frame::from_mat(seq++, 1000000, std::move(mat));
        benchmark::DoNotOptimize(frame);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Frame_FromMat);

// ============ 完整流水线吞吐量 ============

// 空回调 - 纯调度开销
static void BM_Pipeline_EmptyCallbacks(benchmark::State &state) {
    std::atomic<uint64_t> processed{0};
    TestStrategy strategy;

    strategy.set_detect([&](const SharedFeature &sf) {
        processed.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    strategy.set_track([](const Detection &det, const SharedFeature &) {
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        return tr;
    });

    strategy.set_solve([](const TrackResult &tr) {
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        return out;
    });

    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);


    sched.start();

    uint64_t seq = 0;
    for (auto _: state) {
        auto frame = Frame::wrap(seq++, 1000000, nullptr, 0, 0, CV_8UC1);
        sched.push_frame(std::move(frame));
    }

    // 等待处理完成
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (processed.load() < seq && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    sched.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Pipeline_EmptyCallbacks);

// 带真实图像的流水线
static void BM_Pipeline_WithImage(benchmark::State &state) {
    TestStrategy strategy;

    std::atomic<uint64_t> processed{0};

    strategy.set_detect([&](const SharedFeature &sf) {
        processed.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);

    std::vector<uint8_t> camera_buffer(480 * 640 * 3, 128);

    sched.start();

    uint64_t seq = 0;
    for (auto _: state) {
        auto frame = Frame::wrap(seq++, 1000000, camera_buffer.data(), 480, 640, CV_8UC3);
        sched.push_frame(std::move(frame));
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (processed.load() < seq && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    sched.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Pipeline_WithImage);

// 带预处理的流水线
static void BM_Pipeline_WithPreprocess(benchmark::State &state) {
    std::atomic<uint64_t> processed{0};
    TestStrategy strategy;

    strategy.set_preprocess([](SharedFeature &sf) {
        if (sf.image && sf.image->rows() > 0) {
            sf.preprocessed = ImageBuffer::create(sf.image->rows() / 2, sf.image->cols() / 2, CV_8UC1);
        }
    });

    strategy.set_detect([&](const SharedFeature &sf) {
        processed.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);

    sched.start();

    uint64_t seq = 0;
    for (auto _: state) {
        cv::Mat mat(480, 640, CV_8UC3);
        auto frame = Frame::from_mat(seq++, 1000000, std::move(mat));
        sched.push_frame(std::move(frame));
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (processed.load() < seq && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    sched.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Pipeline_WithPreprocess);

// 带 TBB 并行预处理
static void BM_Pipeline_WithParallelPreprocess(benchmark::State &state) {
    TestStrategy strategy;

    std::atomic<uint64_t> processed{0};

    strategy.set_preprocess([](SharedFeature &sf) {
        if (sf.image && sf.image->rows() > 0) {
            sf.preprocessed = ImageBuffer::create(sf.image->rows() / 2, sf.image->cols() / 2, CV_8UC1);
            auto &src = sf.image->mat();
            auto &dst = sf.preprocessed->mat();

            // 并行下采样
            tbb::parallel_for(0, dst.rows, [&](const int y) {
                auto *dst_row = dst.ptr<uint8_t>(y);
                auto *src_row = src.ptr<cv::Vec3b>(y * 2);
                for (int x = 0; x < dst.cols; ++x) {
                    dst_row[x] = src_row[x * 2][0];
                }
            });
        }
    });

    strategy.set_detect([&](const SharedFeature &sf) {
        processed.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);

    sched.start();

    uint64_t seq = 0;
    for (auto _: state) {
        cv::Mat mat(480, 640, CV_8UC3);
        auto frame = Frame::from_mat(seq++, 1000000, std::move(mat));
        sched.push_frame(std::move(frame));
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (processed.load() < seq && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    sched.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Pipeline_WithParallelPreprocess);

// ============ 真实负载模拟 ============

// 模拟 160Hz 相机输入，测量丢帧率
static void BM_RealWorld_160Hz(benchmark::State &state) {
    TestStrategy strategy;

    std::atomic<uint64_t> processed{0};
    std::vector<uint8_t> camera_buffer(480 * 640 * 3, 100);

    strategy.set_detect([&](const SharedFeature &sf) {
        processed.fetch_add(1, std::memory_order_relaxed);
        // 模拟 1ms 检测延迟
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    strategy.set_track([](const Detection &det, const SharedFeature &) {
        // 模拟 500us 跟踪延迟
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        return tr;
    });

    strategy.set_solve([](const TrackResult &tr) {
        // 模拟 200us 解算延迟
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        return out;
    });

    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);

    sched.start();

    uint64_t seq = 0;
    for (auto _: state) {
        auto frame = Frame::wrap(seq++, 1000000, camera_buffer.data(), 480, 640, CV_8UC3);
        sched.push_frame(std::move(frame));
        // 160Hz = 6.25ms 间隔
        std::this_thread::sleep_for(std::chrono::microseconds(6250));
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (processed.load() < seq && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    sched.stop();

    state.counters["processed"] = processed.load();
    state.counters["dropped"] = sched.stats().frame_drop.load();
    state.counters["rate"] = processed.load() * 100.0 / seq;
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RealWorld_160Hz)->Iterations(100);

// ============ 端到端延迟测量 ============

// 测量从 push_frame 到 output 的完整延迟
static void BM_E2E_Latency_Measurement(benchmark::State &state) {
    TestStrategy strategy;

    std::atomic<uint64_t> last_output_ts{0};
    std::atomic last_seq{UINT64_MAX};

    strategy.set_detect([](const SharedFeature &sf) {
        Detection det;
        det.frame_seq = sf.frame_seq;
        det.timestamp_ns = sf.timestamp_ns;
        return det;
    });

    strategy.set_track([](const Detection &det, const SharedFeature &) {
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        tr.timestamp_ns = det.timestamp_ns;
        return tr;
    });

    strategy.set_solve([](const TrackResult &tr) {
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        out.timestamp_ns = tr.timestamp_ns;
        return out;
    });

    strategy.set_output([&](const SolverOutput &out) {
        last_output_ts.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
        last_seq.store(out.frame_seq, std::memory_order_release);
    });

    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);

    sched.start();

    uint64_t seq = 0;
    uint64_t total_latency = 0;
    uint64_t count = 0;

    for (auto _: state) {
        Frame frame;
        frame.seq = seq;
        frame.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();

        sched.push_frame(std::move(frame));

        // 等待这一帧处理完成
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (last_seq.load(std::memory_order_acquire) < seq && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }

        if (last_seq.load() == seq) {
            const auto latency = last_output_ts.load() - frame.timestamp_ns;
            total_latency += latency;
            ++count;
        }

        ++seq;
    }

    sched.stop();

    if (count > 0) {
        state.counters["avg_latency_us"] = total_latency / count / 1000.0;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_E2E_Latency_Measurement)->Iterations(100);

// ============ 内存分配测试 ============

// 测量 ImageBuffer::create 分配开销
static void BM_ImageBuffer_Create_Small(benchmark::State &state) {
    for (auto _: state) {
        auto img = ImageBuffer::create(120, 160, CV_8UC1);
        benchmark::DoNotOptimize(img);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ImageBuffer_Create_Small);

static void BM_ImageBuffer_Create_Medium(benchmark::State &state) {
    for (auto _: state) {
        auto img = ImageBuffer::create(480, 640, CV_8UC3);
        benchmark::DoNotOptimize(img);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ImageBuffer_Create_Medium);

static void BM_ImageBuffer_Create_Large(benchmark::State &state) {
    for (auto _: state) {
        auto img = ImageBuffer::create(1080, 1920, CV_8UC3);
        benchmark::DoNotOptimize(img);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ImageBuffer_Create_Large);

// ============ SharedPtr 开销测试 ============

static void BM_SharedPtr_Copy(benchmark::State &state) {
    const auto img = ImageBuffer::create(480, 640, CV_8UC3);

    for (auto _: state) {
        auto copy = img;
        benchmark::DoNotOptimize(copy);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SharedPtr_Copy);

static void BM_SharedPtr_Move(benchmark::State &state) {
    for (auto _: state) {
        auto img = ImageBuffer::create(480, 640, CV_8UC3);
        auto moved = std::move(img);
        benchmark::DoNotOptimize(moved);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SharedPtr_Move);

BENCHMARK_MAIN();
