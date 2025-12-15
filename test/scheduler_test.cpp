#include <gtest/gtest.h>
#include "scheduler/scheduler.hpp"
#include "scheduler/primitives/latest_value.hpp"
#include "scheduler/primitives/event_flag.hpp"

#include <thread>
#include <chrono>
#include <atomic>

using namespace scheduler;
using namespace std::chrono_literals;

// ============ LatestValue Tests ============

TEST(LatestValueTest, BasicWriteRead) {
    LatestValue<int> lv;
    lv.write(42);
    EXPECT_EQ(lv.read(), 42);
}

TEST(LatestValueTest, VersionIncrement) {
    LatestValue<int> lv;
    EXPECT_EQ(lv.version(), 0);
    lv.write(1);
    EXPECT_EQ(lv.version(), 1);
    lv.write(2);
    EXPECT_EQ(lv.version(), 2);
}

TEST(LatestValueTest, TryReadNewData) {
    LatestValue<int> lv;
    uint64_t version = 0;
    int value = 0;

    // 没有新数据
    EXPECT_FALSE(lv.try_read(value, version));

    // 写入新数据
    lv.write(42);
    EXPECT_TRUE(lv.try_read(value, version));
    EXPECT_EQ(value, 42);
    EXPECT_EQ(version, 1);

    // 再次读取，没有新数据
    EXPECT_FALSE(lv.try_read(value, version));

    // 写入更多数据
    lv.write(100);
    EXPECT_TRUE(lv.try_read(value, version));
    EXPECT_EQ(value, 100);
}

TEST(LatestValueTest, ConcurrentReadWrite) {
    LatestValue<int> lv;
    std::atomic<bool> running{true};
    std::atomic<int> read_count{0};

    // Writer thread
    std::thread writer([&] {
        for (int i = 0; i < 10000 && running; ++i) {
            lv.write(i);
        }
        running = false;
    });

    // Reader thread
    std::thread reader([&] {
        uint64_t version = 0;
        int value = 0;
        while (running || version < lv.version()) {
            if (lv.try_read(value, version)) {
                read_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    writer.join();
    reader.join();

    EXPECT_GT(read_count.load(), 0);
}

// ============ EventFlag Tests ============

TEST(EventFlagTest, NotifyWait) {
    EventFlag flag;

    std::thread waiter([&] {
        flag.wait();
    });

    std::this_thread::sleep_for(10ms);
    flag.notify();
    waiter.join();
}

TEST(EventFlagTest, TryWait) {
    EventFlag flag;

    EXPECT_FALSE(flag.try_wait());
    flag.notify();
    EXPECT_TRUE(flag.try_wait());
    EXPECT_FALSE(flag.try_wait());
}

TEST(EventFlagTest, WaitForTimeout) {
    EventFlag flag;

    auto start = std::chrono::steady_clock::now();
    bool result = flag.wait_for(50ms);
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result);
    EXPECT_GE(elapsed, 50ms);
}

TEST(EventFlagTest, WaitForSignaled) {
    EventFlag flag;

    std::thread notifier([&] {
        std::this_thread::sleep_for(10ms);
        flag.notify();
    });

    auto start = std::chrono::steady_clock::now();
    bool result = flag.wait_for(1s);
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_TRUE(result);
    EXPECT_LT(elapsed, 500ms);

    notifier.join();
}

// ============ Scheduler Tests ============

TEST(SchedulerTest, InitStartStop) {
    Scheduler sched;
    RuntimeConfig config;
    config.detector_threads = 2;

    sched.init(config);
    EXPECT_FALSE(sched.is_running());

    sched.start();
    EXPECT_TRUE(sched.is_running());

    sched.stop();
    EXPECT_FALSE(sched.is_running());
}

TEST(SchedulerTest, PushFrame) {
    Scheduler sched;
    RuntimeConfig config;
    sched.init(config);
    sched.start();

    Frame frame;
    frame.seq = 1;
    frame.timestamp_ns = 123456789;

    EXPECT_TRUE(sched.push_frame(std::move(frame)));
    EXPECT_EQ(sched.stats().frame_in.load(), 1);

    sched.stop();
}

TEST(SchedulerTest, FullPipeline) {
    Scheduler sched;
    RuntimeConfig config;
    sched.init(config);

    std::atomic<int> detect_count{0};
    std::atomic<int> track_count{0};
    std::atomic<int> solve_count{0};
    std::atomic<int> output_count{0};

    sched.set_detector([&](const SharedFeature &sf) {
        detect_count.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        det.timestamp_ns = sf.timestamp_ns;
        return det;
    });

    sched.set_tracker([&](const Detection &det, const SharedFeature &) {
        track_count.fetch_add(1, std::memory_order_relaxed);
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        tr.timestamp_ns = det.timestamp_ns;
        return tr;
    });

    sched.set_solver([&](const TrackResult &tr) {
        solve_count.fetch_add(1, std::memory_order_relaxed);
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        out.timestamp_ns = tr.timestamp_ns;
        return out;
    });

    sched.set_output([&](const SolverOutput &) {
        output_count.fetch_add(1, std::memory_order_relaxed);
    });

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
    Scheduler sched;
    RuntimeConfig config;
    config.detector_threads = 2;
    sched.init(config);

    EXPECT_EQ(sched.get_config().detector_threads, 2);

    config.detector_threads = 4;
    sched.update_config(config);

    EXPECT_EQ(sched.get_config().detector_threads, 4);
}

TEST(SchedulerTest, StatsReset) {
    Scheduler sched;
    RuntimeConfig config;
    sched.init(config);
    sched.start();

    Frame frame;
    sched.push_frame(std::move(frame));

    std::this_thread::sleep_for(50ms);

    EXPECT_GT(sched.stats().frame_in.load(), 0);

    sched.reset_stats();

    EXPECT_EQ(sched.stats().frame_in.load(), 0);

    sched.stop();
}
