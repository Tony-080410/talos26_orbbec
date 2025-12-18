#include <benchmark/benchmark.h>
#include "scheduler/primitives/latency_histogram.hpp"
#include "scheduler/primitives/latest_value.hpp"
#include "scheduler/primitives/native_event.hpp"
#include "scheduler/primitives/triple_buffer.hpp"
#include "scheduler/scheduler.hpp"

#include <atomic>
#include <thread>

#include "test_strategy.hpp"

using namespace scheduler;
using namespace scheduler::test;

// ============ 基础组件 Benchmark ============

// TripleBuffer 写入开销
static void BM_TripleBuffer_Write(benchmark::State &state) {
    spike::TripleBuffer<uint64_t> buffer;
    uint64_t seq = 0;

    for ([[maybe_unused]] auto _: state) {
        auto &slot = buffer.borrow_mut();
        slot = seq++;
        buffer.publish();
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TripleBuffer_Write);

// TripleBuffer 读取开销
static void BM_TripleBuffer_Read(benchmark::State &state) {
    spike::TripleBuffer<uint64_t> buffer;

    // 预写入数据
    auto &slot = buffer.borrow_mut();
    slot = 42;
    buffer.publish();

    for ([[maybe_unused]] auto _: state) {
        auto result = buffer.borrow();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TripleBuffer_Read);

// LatestValue 写入开销
static void BM_LatestValue_Write(benchmark::State &state) {
    LatestValue<uint64_t> lv;
    uint64_t seq = 0;

    for ([[maybe_unused]] auto _: state) {
        lv.write(seq++);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LatestValue_Write);

// LatestValue 读取开销
static void BM_LatestValue_Read(benchmark::State &state) {
    LatestValue<uint64_t> lv;
    lv.write(42);

    for ([[maybe_unused]] auto _: state) {
        auto val = lv.read();
        benchmark::DoNotOptimize(val);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LatestValue_Read);

// LatestValue try_read 开销
static void BM_LatestValue_TryRead(benchmark::State &state) {
    LatestValue<uint64_t> lv;
    uint64_t version = 0;
    uint64_t value = 0;

    // Writer thread
    std::atomic running{true};
    std::thread writer([&] {
        uint64_t seq = 0;
        while (running.load(std::memory_order_relaxed)) {
            lv.write(seq++);
        }
    });

    for ([[maybe_unused]] auto _: state) {
        [[maybe_unused]] auto _2 = lv.try_read(value, version);
        benchmark::DoNotOptimize(value);
    }

    running = false;
    writer.join();

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LatestValue_TryRead);

// EventFlag notify 开销 (无等待者)
static void BM_EventFlag_Notify_NoWaiter(benchmark::State &state) {
    NativeEvent flag;

    for ([[maybe_unused]] auto _: state) {
        flag.notify_one();
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_EventFlag_Notify_NoWaiter);

// LatencyHistogram record 开销
static void BM_LatencyHistogram_Record(benchmark::State &state) {
    LatencyHistogram hist;
    uint64_t latency = 1000;

    for ([[maybe_unused]] auto _: state) {
        hist.record(latency);
        latency = (latency + 1) % 100000;
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LatencyHistogram_Record);

// ============ Scheduler Benchmark ============

// push_frame 调度开销 (空回调)
static void BM_Scheduler_PushFrame(benchmark::State &state) {
    auto strategy = TestStrategy{};
    // 空回调，只测调度开销
    strategy.set_detect([](const SharedFeature &sf) {
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });
    Scheduler<TestStrategy> sched;
    RuntimeConfig config(1, 4, 10);
    sched.init(config, &strategy);

    sched.start();

    uint64_t seq = 0;
    for ([[maybe_unused]] auto _: state) {
        Frame frame;
        frame.seq = seq++;
        frame.timestamp_ns = seq * 1000000;

        auto t0 = std::chrono::steady_clock::now();
        sched.push_frame(std::move(frame));
        auto t1 = std::chrono::steady_clock::now();

        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        benchmark::DoNotOptimize(ns);
    }

    sched.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scheduler_PushFrame);

// 端到端延迟测量 (简化版，避免死锁)
static void BM_Scheduler_E2E_Latency(benchmark::State &state) {
    std::atomic<uint64_t> processed{0};
    auto strategy = TestStrategy{};
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
    strategy.set_output([&](const SolverOutput &) { processed.fetch_add(1, std::memory_order_release); });

    Scheduler<TestStrategy> sched;
    RuntimeConfig config;
    config.detector_threads = 2;

    sched.init(config, &strategy);
    sched.start();

    uint64_t seq = 0;
    for ([[maybe_unused]] auto _: state) {
        Frame frame;
        frame.seq = seq++;
        frame.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
        sched.push_frame(std::move(frame));
    }

    // 等待处理完成 (带超时)
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (processed.load(std::memory_order_acquire) < seq && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    sched.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scheduler_E2E_Latency);

// 高吞吐量测试
static void BM_Scheduler_Throughput(benchmark::State &state) {
    std::atomic<uint64_t> processed{0};
    auto strategy = TestStrategy{};
    strategy.set_detect([&](const SharedFeature &sf) {
        processed.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    Scheduler<TestStrategy> sched;
    RuntimeConfig config;
    config.detector_threads = 2;

    sched.init(config, &strategy);
    sched.start();

    uint64_t seq = 0;
    for ([[maybe_unused]] auto _: state) {
        Frame frame;
        frame.seq = seq++;
        sched.push_frame(std::move(frame));
    }

    // 等待处理完成 (带超时)
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (processed.load(std::memory_order_acquire) < seq && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    sched.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scheduler_Throughput);

// TBB concurrent_queue push/pop 开销
static void BM_TBB_ConcurrentQueue_Push(benchmark::State &state) {
    tbb::concurrent_queue<uint64_t> queue;
    uint64_t seq = 0;

    for ([[maybe_unused]] auto _: state) {
        queue.push(seq++);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TBB_ConcurrentQueue_Push);

static void BM_TBB_ConcurrentQueue_TryPop(benchmark::State &state) {
    tbb::concurrent_queue<uint64_t> queue;

    // 预填充
    for (int i = 0; i < 1000; ++i) {
        queue.push(i);
    }

    uint64_t value;
    for ([[maybe_unused]] auto _: state) {
        if (!queue.try_pop(value)) {
            queue.push(value); // 保持队列有数据
        }
        benchmark::DoNotOptimize(value);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TBB_ConcurrentQueue_TryPop);

BENCHMARK_MAIN();
