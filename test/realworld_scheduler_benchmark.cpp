/**
 * @file realworld_scheduler_benchmark.cpp
 * @brief 真实世界 Scheduler 性能测试
 *
 * 从摄像头读取数据，复用 buffer，做基本运算
 * 测量真实场景下的端到端性能
 */

#include "scheduler/scheduler.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace scheduler;
using namespace std::chrono_literals;

// ============ Buffer Pool ============

class BufferPool {
public:
    explicit BufferPool(size_t count, int rows, int cols, int type)
        : rows_(rows), cols_(cols), type_(type) {
        for (size_t i = 0; i < count; ++i) {
            pool_.push_back(ImageBuffer::create(rows, cols, type));
        }
        available_.resize(count, true);
    }

    ImagePtr acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < available_.size(); ++i) {
            if (available_[i]) {
                available_[i] = false;
                return pool_[i];
            }
        }
        // Pool exhausted, create new (will be GC'd later)
        return ImageBuffer::create(rows_, cols_, type_);
    }

    void release(ImagePtr& ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < pool_.size(); ++i) {
            if (pool_[i].get() == ptr.get()) {
                available_[i] = true;
                return;
            }
        }
        // Not from pool, let it go
    }

private:
    std::vector<ImagePtr> pool_;
    std::vector<bool> available_;
    std::mutex mutex_;
    int rows_, cols_, type_;
};

// ============ 统计工具 ============

struct LatencyStats {
    std::vector<uint64_t> samples;
    std::mutex mutex;

    void record(uint64_t ns) {
        std::lock_guard<std::mutex> lock(mutex);
        samples.push_back(ns);
    }

    void print(const std::string& name) {
        if (samples.empty()) {
            std::cout << "[" << name << "] No samples\n";
            return;
        }

        std::sort(samples.begin(), samples.end());
        auto sum = std::accumulate(samples.begin(), samples.end(), 0ULL);
        auto avg = sum / samples.size();
        auto p50 = samples[samples.size() / 2];
        auto p90 = samples[samples.size() * 90 / 100];
        auto p99 = samples[samples.size() * 99 / 100];
        auto max_val = samples.back();

        std::cout << "[" << name << "] "
                  << "n=" << samples.size()
                  << ", avg=" << avg / 1000.0 << "us"
                  << ", p50=" << p50 / 1000.0 << "us"
                  << ", p90=" << p90 / 1000.0 << "us"
                  << ", p99=" << p99 / 1000.0 << "us"
                  << ", max=" << max_val / 1000.0 << "us\n";
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex);
        samples.clear();
    }
};

// ============ 主测试 ============

int main(int argc, char** argv) {
    // 参数
    int camera_id = 0;
    int duration_sec = 5;
    bool use_camera = true;

    if (argc > 1) {
        camera_id = std::atoi(argv[1]);
    }
    if (argc > 2) {
        duration_sec = std::atoi(argv[2]);
    }
    if (argc > 3 && std::string(argv[3]) == "--no-camera") {
        use_camera = false;
    }

    std::cout << "=== RealWorld Scheduler Benchmark ===\n";
    std::cout << "Camera ID: " << camera_id << "\n";
    std::cout << "Duration: " << duration_sec << "s\n";
    std::cout << "Use Camera: " << (use_camera ? "yes" : "no (synthetic)") << "\n\n";

    // 打开摄像头或使用合成数据
    cv::VideoCapture cap;
    int frame_width = 640;
    int frame_height = 480;
    double target_fps = 60.0;

    if (use_camera) {
        cap.open(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera " << camera_id << "\n";
            std::cerr << "Try: " << argv[0] << " 0 5 --no-camera\n";
            return 1;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
        cap.set(cv::CAP_PROP_FPS, target_fps);

        frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        target_fps = cap.get(cv::CAP_PROP_FPS);

        std::cout << "Camera: " << frame_width << "x" << frame_height
                  << " @ " << target_fps << " FPS\n\n";
    } else {
        std::cout << "Synthetic: " << frame_width << "x" << frame_height
                  << " @ " << target_fps << " FPS\n\n";
    }

    // Buffer pool (复用 buffer)
    BufferPool frame_pool(8, frame_height, frame_width, CV_8UC3);
    BufferPool gray_pool(8, frame_height, frame_width, CV_8UC1);
    BufferPool small_pool(8, frame_height / 4, frame_width / 4, CV_8UC1);

    // 统计
    LatencyStats capture_latency;
    LatencyStats preprocess_latency;
    LatencyStats detect_latency;
    LatencyStats track_latency;
    LatencyStats e2e_latency;

    std::atomic<uint64_t> frame_count{0};
    std::atomic<uint64_t> processed_count{0};
    std::atomic<uint64_t> output_count{0};

    // 调度器
    Scheduler sched;
    RuntimeConfig config;
    config.detector_threads = 4;
    sched.init(config);

    // 预处理: RGB -> Gray -> Resize
    sched.set_preprocessor([&](SharedFeature& sf) {
        auto t0 = std::chrono::steady_clock::now();

        if (sf.image && sf.image->rows() > 0) {
            // 获取复用 buffer
            auto gray = gray_pool.acquire();
            auto small = small_pool.acquire();

            // 灰度转换
            cv::cvtColor(sf.image->mat(), gray->mat(), cv::COLOR_BGR2GRAY);

            // 下采样
            cv::resize(gray->mat(), small->mat(), small->mat().size());

            sf.preprocessed = small;

            // 释放中间 buffer
            gray_pool.release(gray);
        }

        auto t1 = std::chrono::steady_clock::now();
        preprocess_latency.record(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    });

    // 检测: 简单阈值 + 轮廓检测
    sched.set_detector([&](const SharedFeature& sf) {
        auto t0 = std::chrono::steady_clock::now();

        Detection det;
        det.frame_seq = sf.frame_seq;
        det.timestamp_ns = sf.timestamp_ns;

        if (sf.preprocessed && sf.preprocessed->rows() > 0) {
            cv::Mat binary;
            cv::threshold(sf.preprocessed->mat(), binary, 128, 255, cv::THRESH_BINARY);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                auto rect = cv::boundingRect(contour);
                if (rect.area() > 100) {  // 过滤噪声
                    // 缩放回原图尺寸
                    det.boxes.emplace_back(rect.x * 4, rect.y * 4, rect.width * 4, rect.height * 4);
                    det.scores.push_back(static_cast<float>(rect.area()) / 1000.0f);
                }
            }
        }

        processed_count.fetch_add(1, std::memory_order_relaxed);

        auto t1 = std::chrono::steady_clock::now();
        detect_latency.record(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        return det;
    });

    // 跟踪: 简单 IOU 匹配 (模拟)
    sched.set_tracker([&](const Detection& det, const SharedFeature&) {
        auto t0 = std::chrono::steady_clock::now();

        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        tr.timestamp_ns = det.timestamp_ns;

        // 模拟跟踪计算
        volatile int sum = 0;
        for (const auto& box : det.boxes) {
            sum += box.x + box.y + box.width + box.height;
        }
        (void)sum;

        auto t1 = std::chrono::steady_clock::now();
        track_latency.record(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        return tr;
    });

    // 解算: 模拟 PnP 解算
    sched.set_solver([](const TrackResult& tr) {
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        out.timestamp_ns = tr.timestamp_ns;

        // 模拟解算计算
        volatile double result = 0;
        for (int i = 0; i < 100; ++i) {
            result += std::sin(i * 0.1) * std::cos(i * 0.2);
        }
        (void)result;

        return out;
    });

    // 输出
    sched.set_output([&](const SolverOutput& out) {
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        e2e_latency.record(now - out.timestamp_ns);
        output_count.fetch_add(1, std::memory_order_relaxed);
    });

    // 启动
    sched.start();

    // 帧获取线程
    std::atomic<bool> running{true};
    cv::Mat synthetic_frame(frame_height, frame_width, CV_8UC3);

    std::thread capture_thread([&] {
        while (running.load(std::memory_order_relaxed)) {
            auto t0 = std::chrono::steady_clock::now();

            // 获取复用 buffer
            auto buffer = frame_pool.acquire();

            if (use_camera) {
                // 从摄像头读取
                cap.read(buffer->mat());
                if (buffer->mat().empty()) {
                    frame_pool.release(buffer);
                    continue;
                }
            } else {
                // 合成数据
                auto seq = frame_count.load();
                synthetic_frame.setTo(cv::Scalar(seq % 256, (seq * 2) % 256, (seq * 3) % 256));
                synthetic_frame.copyTo(buffer->mat());
            }

            auto t1 = std::chrono::steady_clock::now();
            capture_latency.record(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

            // 创建 Frame (0-copy, 使用 pool buffer)
            Frame frame;
            frame.seq = frame_count.fetch_add(1, std::memory_order_relaxed);
            frame.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            frame.image = buffer;

            // 推送到调度器
            if (!sched.push_frame(std::move(frame))) {
                frame_pool.release(buffer);
            }

            // 控制帧率
            if (!use_camera) {
                auto frame_time = std::chrono::microseconds(static_cast<int>(1000000.0 / target_fps));
                auto elapsed = std::chrono::steady_clock::now() - t0;
                if (elapsed < frame_time) {
                    std::this_thread::sleep_for(frame_time - elapsed);
                }
            }
        }
    });

    // 运行指定时间
    std::cout << "Running for " << duration_sec << " seconds...\n\n";

    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_sec);

    while (std::chrono::steady_clock::now() < end_time) {
        std::this_thread::sleep_for(100ms);

        // 实时统计
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        auto frames = frame_count.load();
        auto processed = processed_count.load();
        auto outputs = output_count.load();
        auto dropped = sched.stats().frame_drop.load();

        std::cout << "\r[" << elapsed << "s] "
                  << "Frames: " << frames
                  << ", Processed: " << processed
                  << ", Outputs: " << outputs
                  << ", Dropped: " << dropped
                  << ", FPS: " << std::fixed << std::setprecision(1)
                  << (frames * 1.0 / (elapsed + 1))
                  << "   " << std::flush;
    }

    // 停止
    running = false;
    capture_thread.join();
    sched.stop();

    // 最终统计
    std::cout << "\n\n=== Final Statistics ===\n";

    auto total_frames = frame_count.load();
    auto total_processed = processed_count.load();
    auto total_outputs = output_count.load();
    auto total_dropped = sched.stats().frame_drop.load();

    std::cout << "Total Frames:    " << total_frames << "\n";
    std::cout << "Total Processed: " << total_processed << "\n";
    std::cout << "Total Outputs:   " << total_outputs << "\n";
    std::cout << "Total Dropped:   " << total_dropped << "\n";
    std::cout << "Process Rate:    " << std::fixed << std::setprecision(1)
              << (total_processed * 100.0 / total_frames) << "%\n";
    std::cout << "Avg FPS:         " << std::fixed << std::setprecision(1)
              << (total_frames * 1.0 / duration_sec) << "\n\n";

    std::cout << "=== Latency Distribution ===\n";
    capture_latency.print("Capture");
    preprocess_latency.print("Preprocess");
    detect_latency.print("Detect");
    track_latency.print("Track");
    e2e_latency.print("E2E");

    std::cout << "\n=== Scheduler Stats ===\n";
    const auto& stats = sched.stats();
    std::cout << "Detect Hist: p50=" << stats.detect_hist.p50() / 1000.0 << "us"
              << ", p99=" << stats.detect_hist.p99() / 1000.0 << "us\n";
    std::cout << "Track Hist:  p50=" << stats.track_hist.p50() / 1000.0 << "us"
              << ", p99=" << stats.track_hist.p99() / 1000.0 << "us\n";
    std::cout << "Solve Hist:  p50=" << stats.solve_hist.p50() / 1000.0 << "us"
              << ", p99=" << stats.solve_hist.p99() / 1000.0 << "us\n";

    if (use_camera) {
        cap.release();
    }

    return 0;
}
