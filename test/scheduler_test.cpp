#include "scheduler/scheduler.hpp"
#include <gtest/gtest.h>
#include "scheduler/primitives/latency_histogram.hpp"
#include "test_strategy.hpp"

#include <atomic>
#include <chrono>
#include <oneapi/tbb/parallel_for.h>
#include <random>
#include <thread>

using namespace scheduler;
using namespace scheduler::test;
using namespace std::chrono_literals;

// ============ Scheduler Tests ============

TEST(SchedulerTest, InitStartStop) {
    TestStrategy strategy;
    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;

    sched.init(config, &strategy);
    EXPECT_FALSE(sched.is_running());

    sched.start();
    EXPECT_TRUE(sched.is_running());

    sched.stop();
    EXPECT_FALSE(sched.is_running());
}

TEST(SchedulerTest, PushFrame) {
    TestStrategy strategy;
    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);
    sched.start();

    Frame frame;

    EXPECT_TRUE(sched.push_frame(std::move(frame)));
    EXPECT_EQ(sched.stats().frame_in.load(), 1);

    sched.stop();
}

TEST(SchedulerTest, FullPipeline) {
    TestStrategy strategy;

    std::atomic detect_count{0};
    std::atomic track_count{0};
    std::atomic solve_count{0};
    std::atomic output_count{0};

    strategy.set_detect([&](const SharedFeature &sf) {
        detect_count.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        det.timestamp_ns = sf.timestamp_ns;
        return det;
    });

    strategy.set_track([&](const Detection &det, const SharedFeature &) {
        track_count.fetch_add(1, std::memory_order_relaxed);
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        tr.timestamp_ns = det.timestamp_ns;
        return tr;
    });

    strategy.set_solve([&](const TrackResult &tr) {
        solve_count.fetch_add(1, std::memory_order_relaxed);
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        out.timestamp_ns = tr.timestamp_ns;
        return out;
    });

    strategy.set_output([&](const SolverOutput &) { output_count.fetch_add(1, std::memory_order_relaxed); });

    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);
    sched.start();

    // 推送一些帧
    for (int i = 0; i < 10; ++i) {
        Frame frame;
        frame.seq = i;
        frame.timestamp_ns = i * 1000000;
        sched.push_frame(std::move(frame));
    }

    // 等待处理完成
    std::this_thread::sleep_for(500ms);

    sched.stop();

    // 验证流水线执行（至少处理了大部分帧）
    EXPECT_GE(detect_count.load(), 8); // 可能因为调度竞争少处理几帧
    EXPECT_GE(track_count.load(), 1); // Tracker 可能合并一些帧
    EXPECT_GE(solve_count.load(), 1);
    EXPECT_GE(output_count.load(), 1);
}

TEST(SchedulerTest, ConfigHotUpdate) {
    TestStrategy strategy;
    Scheduler<TestStrategy> sched;
    RuntimeConfig config;
    config.detector_mode = 0;
    sched.init(config, &strategy);

    EXPECT_EQ(sched.get_config().detector_mode, 0);

    config.detector_mode = 4;
    sched.update_config(config);

    EXPECT_EQ(sched.get_config().detector_mode, 4);
}

TEST(SchedulerTest, StatsReset) {
    TestStrategy strategy;
    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);
    sched.start();

    Frame frame;
    sched.push_frame(std::move(frame));

    std::this_thread::sleep_for(50ms);

    EXPECT_GT(sched.stats().frame_in.load(), 0);

    sched.reset_stats();

    EXPECT_EQ(sched.stats().frame_in.load(), 0);

    sched.stop();
}

// ============ LatencyHistogram Tests ============

TEST(LatencyHistogramTest, BasicRecord) {
    LatencyHistogram hist;
    hist.record(1000); // 1us
    hist.record(2000); // 2us
    hist.record(3000); // 3us

    EXPECT_EQ(hist.count(), 3);
    EXPECT_NEAR(hist.mean(), 2000.0, 1.0);
    EXPECT_EQ(hist.min(), 1000);
    EXPECT_EQ(hist.max(), 3000);
}

TEST(LatencyHistogramTest, Percentiles) {
    LatencyHistogram hist;

    // 记录 100 个样本: 1us, 2us, ..., 100us
    for (auto i = 1; i <= 100; ++i) {
        hist.record(i * 1000); // i us
    }

    EXPECT_EQ(hist.count(), 100);

    // p50 应该在 ~50us 附近
    EXPECT_GE(hist.p50(), 20000);
    EXPECT_LE(hist.p50(), 100000);

    // p99 应该在 ~99us 附近
    EXPECT_GE(hist.p99(), 50000);
}

TEST(LatencyHistogramTest, Reset) {
    LatencyHistogram hist;
    hist.record(1000);
    hist.record(2000);

    EXPECT_EQ(hist.count(), 2);

    hist.reset();

    EXPECT_EQ(hist.count(), 0);
    EXPECT_EQ(hist.min(), 0);
    EXPECT_EQ(hist.max(), 0);
}

TEST(LatencyHistogramTest, ConcurrentRecord) {
    LatencyHistogram hist;
    constexpr int kThreads = 4;
    constexpr int kSamplesPerThread = 10000;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&hist, t] {
            std::mt19937 gen(t);
            std::uniform_int_distribution<uint64_t> dist(100, 100000);
            for (int i = 0; i < kSamplesPerThread; ++i) {
                hist.record(dist(gen));
            }
        });
    }

    for (auto &th: threads) {
        th.join();
    }

    EXPECT_EQ(hist.count(), kThreads * kSamplesPerThread);
}

// ============ Preprocessor Tests ============

TEST(SchedulerTest, PreprocessorCallback) {
    TestStrategy strategy;
    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);

    std::atomic preprocess_count{0};

    strategy.set_preprocess([&](SharedFeature &sf) {
        preprocess_count.fetch_add(1, std::memory_order_relaxed);
        // 模拟预处理: 可以用 TBB parallel_for
        sf.preprocessed = ImageBuffer::create(100, 100, CV_8UC1);
    });

    strategy.set_detect([](const SharedFeature &sf) {
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    sched.start();

    for (int i = 0; i < 5; ++i) {
        Frame frame;
        frame.seq = i;
        sched.push_frame(std::move(frame));
    }

    std::this_thread::sleep_for(300ms);
    sched.stop();

    EXPECT_GE(preprocess_count.load(), 4);
}

// ============ Stress Tests ============

TEST(SchedulerTest, HighFrequencyFrames) {
    TestStrategy strategy;

    std::atomic detect_count{0};

    strategy.set_detect([&](const SharedFeature &sf) {
        detect_count.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        det.timestamp_ns = sf.timestamp_ns;
        return det;
    });

    strategy.set_track([](const Detection &d, const SharedFeature &) {
        TrackResult tr;
        tr.frame_seq = d.frame_seq;
        tr.timestamp_ns = d.timestamp_ns;
        return tr;
    });

    strategy.set_solve([](const TrackResult &tr) {
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        out.timestamp_ns = tr.timestamp_ns;
        return out;
    });
    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);

    sched.start();

    // 模拟 160Hz 输入，持续 500ms
    constexpr int kFrameCount = 80; // 160Hz * 0.5s
    for (int i = 0; i < kFrameCount; ++i) {
        Frame frame;
        frame.seq = i;
        frame.timestamp_ns = i * 6250000; // 160Hz = 6.25ms
        sched.push_frame(std::move(frame));
        std::this_thread::sleep_for(std::chrono::microseconds(6250));
    }

    std::this_thread::sleep_for(200ms);
    sched.stop();

    // 验证至少处理了大部分帧
    EXPECT_GE(detect_count.load(), kFrameCount * 0.8);
}

TEST(SchedulerTest, LatencyHistogramStats) {
    TestStrategy strategy;

    strategy.set_detect([](const SharedFeature &sf) {
        // 模拟一些处理延迟
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    strategy.set_track([](const Detection &det, const SharedFeature &) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        return tr;
    });

    strategy.set_solve([](const TrackResult &tr) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        return out;
    });
    Scheduler<TestStrategy> sched;
    constexpr RuntimeConfig config;
    sched.init(config, &strategy);
    sched.start();

    for (int i = 0; i < 20; ++i) {
        Frame frame;
        frame.seq = i;
        sched.push_frame(std::move(frame));
        std::this_thread::sleep_for(10ms);
    }

    std::this_thread::sleep_for(300ms);
    sched.stop();

    // 验证直方图有数据
    const auto &stats = sched.stats();
    EXPECT_GT(stats.detect_hist.count(), 0);
    EXPECT_GT(stats.detect_hist.p50(), 0);

    // 验证延迟在合理范围 (100us = 100000ns 左右)
    EXPECT_GE(stats.detect_hist.min(), 50000); // >= 50us
    EXPECT_LE(stats.detect_hist.max(), 10000000); // <= 10ms
}
