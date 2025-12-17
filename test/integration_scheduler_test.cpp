/**
 * @file integration_scheduler_test.cpp
 * @brief Scheduler 全面集成测试
 *
 * 测试内容:
 * 1. 0-copy 验证 - 确保图像数据无拷贝传递
 * 2. 完整流水线 - Detector -> Tracker -> Solver 全链路
 * 3. 高负载压力测试 - 160Hz 持续输入
 * 4. 延迟分布验证 - 确保满足 <1us 调度开销
 */

#include <gtest/gtest.h>
#include <oneapi/tbb/parallel_for.h>
#include "scheduler/scheduler.hpp"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

#include "test_strategy.hpp"

using namespace scheduler;
using namespace scheduler::test;
using namespace std::chrono_literals;

class SchedulerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.detector_threads = 4;
        sched_.init(config_, &strategy_);
    }

    void TearDown() override {
        if (sched_.is_running()) {
            sched_.stop();
        }
    }

    TestStrategy strategy_;
    Scheduler<TestStrategy> sched_;
    RuntimeConfig config_;
};

// ============ 0-Copy 验证测试 ============

TEST_F(SchedulerIntegrationTest, ZeroCopyVerification) {
    // 创建一个带有已知数据的图像缓冲
    constexpr int kRows = 480;
    constexpr int kCols = 640;
    std::vector<uint8_t> camera_buffer(kRows * kCols, 0xAB); // 模拟相机缓冲
    void *original_ptr = camera_buffer.data();

    std::atomic<void *> detected_ptr{nullptr};
    std::atomic<void *> tracked_ptr{nullptr};

    strategy_.set_detect([&](const SharedFeature &sf) {
        if (sf.image) {
            detected_ptr.store(sf.image->data(), std::memory_order_release);
        }
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    strategy_.set_track([&](const Detection &det, const SharedFeature &sf) {
        if (sf.image) {
            tracked_ptr.store(sf.image->data(), std::memory_order_release);
        }
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        return tr;
    });

    sched_.start();

    // 使用 wrap 创建 0-copy Frame
    auto frame = Frame::wrap(0, 1000000, original_ptr, kRows, kCols, CV_8UC1);
    sched_.push_frame(std::move(frame));

    std::this_thread::sleep_for(200ms);
    sched_.stop();

    // 验证指针相同 (0-copy)
    EXPECT_EQ(detected_ptr.load(), original_ptr) << "Detector should see original buffer (0-copy)";
    // Tracker 可能看到相同指针，取决于 TripleBuffer 状态
    // EXPECT_EQ(tracked_ptr.load(), original_ptr);
}

TEST_F(SchedulerIntegrationTest, ZeroCopyFromMat) {
    // 测试从 cv::Mat 创建的 Frame 也是 0-copy
    cv::Mat mat(480, 640, CV_8UC3);
    mat.setTo(cv::Scalar(100, 150, 200));
    void *original_ptr = mat.data;

    std::atomic<void *> detected_ptr{nullptr};

    strategy_.set_detect([&](const SharedFeature &sf) {
        if (sf.image) {
            detected_ptr.store(sf.image->data(), std::memory_order_release);
        }
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    sched_.start();

    auto frame = Frame::from_mat(0, 1000000, std::move(mat));
    void *frame_ptr = frame.image->data();
    sched_.push_frame(std::move(frame));

    std::this_thread::sleep_for(200ms);
    sched_.stop();

    // 验证 frame_ptr 和 detected_ptr 相同
    EXPECT_EQ(detected_ptr.load(), frame_ptr) << "Should be 0-copy from cv::Mat";
}

// ============ 完整流水线测试 ============

TEST_F(SchedulerIntegrationTest, FullPipelineWithRealImage) {
    constexpr int kFrameCount = 100;

    std::atomic<int> preprocess_count{0};
    std::atomic<int> detect_count{0};
    std::atomic<int> track_count{0};
    std::atomic<int> solve_count{0};
    std::atomic<int> output_count{0};

    std::vector<uint64_t> e2e_latencies;
    std::mutex latency_mutex;

    // 预处理: 使用 TBB 并行处理
    strategy_.set_preprocess([&](SharedFeature &sf) {
        preprocess_count.fetch_add(1, std::memory_order_relaxed);
        if (sf.image && sf.image->rows() > 0) {
            // 创建预处理结果
            sf.preprocessed = ImageBuffer::create(sf.image->rows() / 2, sf.image->cols() / 2, CV_8UC1);
            // 模拟并行预处理
            auto &src = sf.image->mat();
            auto &dst = sf.preprocessed->mat();
            tbb::parallel_for(0, dst.rows, [&](int y) {
                for (int x = 0; x < dst.cols; ++x) {
                    dst.at<uint8_t>(y, x) = src.at<cv::Vec3b>(y * 2, x * 2)[0];
                }
            });
        }
    });

    strategy_.set_detect([&](const SharedFeature &sf) {
        detect_count.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        det.timestamp_ns = sf.timestamp_ns;
        // 模拟检测结果
        det.boxes.emplace_back(100, 100, 50, 50);
        det.scores.push_back(0.95f);
        return det;
    });

    strategy_.set_track([&](const Detection &det, const SharedFeature &sf) {
        track_count.fetch_add(1, std::memory_order_relaxed);
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        tr.timestamp_ns = sf.timestamp_ns;
        return tr;
    });

    strategy_.set_solve([&](const TrackResult &tr) {
        solve_count.fetch_add(1, std::memory_order_relaxed);
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        out.timestamp_ns = tr.timestamp_ns;
        return out;
    });

    strategy_.set_output([&](const SolverOutput &out) {
        output_count.fetch_add(1, std::memory_order_relaxed);
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto latency = now - out.timestamp_ns;
        std::lock_guard<std::mutex> lock(latency_mutex);
        e2e_latencies.push_back(latency);
    });

    sched_.start();

    // 推送帧
    for (int i = 0; i < kFrameCount; ++i) {
        cv::Mat img(480, 640, CV_8UC3, cv::Scalar(i % 256, 128, 64));
        auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        auto frame = Frame::from_mat(i, ts, std::move(img));
        sched_.push_frame(std::move(frame));
        std::this_thread::sleep_for(6ms); // ~160 Hz
    }

    std::this_thread::sleep_for(500ms);
    sched_.stop();

    // 验证所有阶段都被调用
    EXPECT_GE(preprocess_count.load(), kFrameCount * 0.9);
    EXPECT_GE(detect_count.load(), kFrameCount * 0.9);
    EXPECT_GE(track_count.load(), kFrameCount * 0.5); // Tracker 可能合并
    EXPECT_GE(solve_count.load(), kFrameCount * 0.5);
    EXPECT_GE(output_count.load(), kFrameCount * 0.5);

    // 验证延迟分布
    if (!e2e_latencies.empty()) {
        std::sort(e2e_latencies.begin(), e2e_latencies.end());
        auto p50 = e2e_latencies[e2e_latencies.size() / 2];
        auto p99 = e2e_latencies[e2e_latencies.size() * 99 / 100];

        // 端到端延迟应该在合理范围内
        EXPECT_LT(p50, 50000000); // < 50ms
        EXPECT_LT(p99, 100000000); // < 100ms

        std::cout << "[E2E Latency] p50: " << p50 / 1000000.0 << "ms, "
                  << "p99: " << p99 / 1000000.0 << "ms" << std::endl;
    }
}

// ============ 高负载压力测试 ============

TEST_F(SchedulerIntegrationTest, HighLoadStressTest) {
    constexpr int kDurationMs = 2000;
    constexpr int kFrameIntervalUs = 6250; // 160 Hz

    std::atomic<uint64_t> frame_sent{0};
    std::atomic<uint64_t> frame_processed{0};

    strategy_.set_detect([&](const SharedFeature &sf) {
        frame_processed.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    sched_.start();

    auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::milliseconds(kDurationMs);

    while (std::chrono::steady_clock::now() < deadline) {
        cv::Mat img(480, 640, CV_8UC1);
        auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        auto frame = Frame::from_mat(frame_sent.load(), ts, std::move(img));

        if (sched_.push_frame(std::move(frame))) {
            frame_sent.fetch_add(1, std::memory_order_relaxed);
        }

        std::this_thread::sleep_for(std::chrono::microseconds(kFrameIntervalUs));
    }

    std::this_thread::sleep_for(500ms);
    sched_.stop();

    auto sent = frame_sent.load();
    auto processed = frame_processed.load();
    auto dropped = sched_.stats().frame_drop.load();

    std::cout << "[Stress Test] Sent: " << sent << ", Processed: " << processed << ", Dropped: " << dropped
              << ", Rate: " << (processed * 100.0 / sent) << "%" << std::endl;

    // 验证处理率
    EXPECT_GE(processed, sent * 0.8) << "Should process at least 80% of frames";
    EXPECT_LE(dropped, sent * 0.2) << "Should drop at most 20% of frames";
}

// ============ 调度开销验证 ============

TEST_F(SchedulerIntegrationTest, SchedulingOverheadVerification) {
    // 空回调，只测量调度开销
    strategy_.set_detect([](const SharedFeature &sf) {
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    sched_.start();

    std::vector<uint64_t> push_latencies;
    push_latencies.reserve(1000);

    for (int i = 0; i < 1000; ++i) {
        Frame frame;
        frame.seq = i;
        frame.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();

        auto t0 = std::chrono::steady_clock::now();
        sched_.push_frame(std::move(frame));
        auto t1 = std::chrono::steady_clock::now();

        push_latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    std::this_thread::sleep_for(200ms);
    sched_.stop();

    // 计算统计
    std::sort(push_latencies.begin(), push_latencies.end());
    auto p50 = push_latencies[push_latencies.size() / 2];
    auto p99 = push_latencies[push_latencies.size() * 99 / 100];
    auto max_lat = push_latencies.back();

    std::cout << "[Scheduling Overhead] p50: " << p50 << "ns, "
              << "p99: " << p99 << "ns, "
              << "max: " << max_lat << "ns" << std::endl;

    // 调度开销应该 < 1us (1000ns)
    EXPECT_LT(p50, 500) << "p50 scheduling overhead should be < 500ns";
    EXPECT_LT(p99, 1000) << "p99 scheduling overhead should be < 1us";
}

// ============ 延迟直方图验证 ============

TEST_F(SchedulerIntegrationTest, LatencyHistogramVerification) {
    // 设置有一定延迟的回调
    strategy_.set_detect([](const SharedFeature &sf) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    strategy_.set_track([](const Detection &det, const SharedFeature &) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        return tr;
    });

    strategy_.set_solve([](const TrackResult &tr) {
        std::this_thread::sleep_for(std::chrono::microseconds(30));
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        return out;
    });

    sched_.start();

    for (int i = 0; i < 100; ++i) {
        Frame frame;
        frame.seq = i;
        sched_.push_frame(std::move(frame));
        std::this_thread::sleep_for(10ms);
    }

    std::this_thread::sleep_for(500ms);
    sched_.stop();

    const auto &stats = sched_.stats();

    // 验证直方图有数据
    EXPECT_GT(stats.detect_hist.count(), 80);
    EXPECT_GT(stats.track_hist.count(), 50);
    EXPECT_GT(stats.solve_hist.count(), 50);

    // 打印延迟统计
    std::cout << "[Detect Latency] count: " << stats.detect_hist.count()
              << ", p50: " << stats.detect_hist.p50() / 1000.0 << "us"
              << ", p99: " << stats.detect_hist.p99() / 1000.0 << "us"
              << ", max: " << stats.detect_hist.max() / 1000.0 << "us" << std::endl;

    std::cout << "[Track Latency] count: " << stats.track_hist.count() << ", p50: " << stats.track_hist.p50() / 1000.0
              << "us"
              << ", p99: " << stats.track_hist.p99() / 1000.0 << "us"
              << ", max: " << stats.track_hist.max() / 1000.0 << "us" << std::endl;

    std::cout << "[Solve Latency] count: " << stats.solve_hist.count() << ", p50: " << stats.solve_hist.p50() / 1000.0
              << "us"
              << ", p99: " << stats.solve_hist.p99() / 1000.0 << "us"
              << ", max: " << stats.solve_hist.max() / 1000.0 << "us" << std::endl;

    // 验证延迟在预期范围
    EXPECT_GE(stats.detect_hist.p50(), 50000); // >= 50us (设置了100us sleep)
    EXPECT_GE(stats.track_hist.p50(), 20000); // >= 20us (设置了50us sleep)
    EXPECT_GE(stats.solve_hist.p50(), 10000); // >= 10us (设置了30us sleep)
}

// ============ 配置热更新测试 ============

TEST_F(SchedulerIntegrationTest, ConfigHotUpdateUnderLoad) {
    std::atomic<int> mode_at_detect{-1};

    strategy_.set_detect([&](const SharedFeature &sf) {
        auto cfg = sched_.get_config();
        mode_at_detect.store(cfg.detector_mode, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    sched_.start();

    // 初始模式
    EXPECT_EQ(config_.detector_mode, 0);

    // 推送一些帧
    for (int i = 0; i < 10; ++i) {
        Frame frame;
        frame.seq = i;
        sched_.push_frame(std::move(frame));
    }
    std::this_thread::sleep_for(100ms);

    EXPECT_EQ(mode_at_detect.load(), 0);

    // 热更新配置
    config_.detector_mode = 2;
    sched_.update_config(config_);

    // 继续推送帧
    for (int i = 10; i < 20; ++i) {
        Frame frame;
        frame.seq = i;
        sched_.push_frame(std::move(frame));
    }
    std::this_thread::sleep_for(100ms);

    EXPECT_EQ(mode_at_detect.load(), 2);

    sched_.stop();
}

// ============ 多生产者测试 ============

TEST_F(SchedulerIntegrationTest, MultipleProducers) {
    constexpr int kProducers = 4;
    constexpr int kFramesPerProducer = 50;

    std::atomic<int> detect_count{0};

    strategy_.set_detect([&](const SharedFeature &sf) {
        detect_count.fetch_add(1, std::memory_order_relaxed);
        Detection det;
        det.frame_seq = sf.frame_seq;
        return det;
    });

    sched_.start();

    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kFramesPerProducer; ++i) {
                Frame frame;
                frame.seq = p * kFramesPerProducer + i;
                sched_.push_frame(std::move(frame));
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    for (auto &th: producers) {
        th.join();
    }

    std::this_thread::sleep_for(500ms);
    sched_.stop();

    auto total_sent = kProducers * kFramesPerProducer;
    auto processed = detect_count.load();
    auto dropped = sched_.stats().frame_drop.load();

    std::cout << "[Multi-Producer] Sent: " << total_sent << ", Processed: " << processed << ", Dropped: " << dropped
              << std::endl;

    EXPECT_GE(processed + dropped, total_sent * 0.9);
}
