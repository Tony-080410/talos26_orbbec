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

#include "spdlog_hook.hpp"
#include "toml++/toml.hpp"
#include "scheduler/primitives/buffer_pool.hpp"
#include "scheduler/primitives/latency_stats.hpp"
#include "auto_aim/detector.hpp"
#include "sp_vision/config.hpp"

using namespace scheduler;
using namespace std::chrono_literals;

class RealworldStrategy {
public:
    void preprocess(SharedFeature &sf) {
        const auto t0 = std::chrono::steady_clock::now();

        if (sf.image && sf.image->rows() > 0) {
            const auto gray = gray_pool.acquire();

            cv::cvtColor(sf.image->mat(), gray->mat(), cv::COLOR_BGR2GRAY);
            cv::threshold(gray->mat(), gray->mat(), config.detector.threshold, 255, cv::THRESH_BINARY);

            sf.preprocessed = gray;
        }

        const auto t1 = std::chrono::steady_clock::now();
        preprocess_latency.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    Detection detect(const SharedFeature &sf) {
        const auto t0 = std::chrono::steady_clock::now();

        Detection det;
        det.frame_seq = sf.frame_seq;
        det.timestamp_ns = sf.timestamp_ns;

        if (sf.image && sf.image->rows() > 0) {
            // 使用传统检测器
            auto armor_slots = detector_.detect(
                sf.image->mat(),sf.preprocessed->mat(), sf.frame_seq, sf.timestamp_ns);


            // 转换 ArmorSlots 到 Detection 格式
            for (int i = 0; i < 10; i++) {
                if (armor_slots.is_valid(i)) {
                    const auto& armor = armor_slots.armors[i];
                    // 计算包围盒 (从四个角点)
                    float min_x = std::min({armor.pts[0], armor.pts[2], armor.pts[4], armor.pts[6]});
                    float min_y = std::min({armor.pts[1], armor.pts[3], armor.pts[5], armor.pts[7]});
                    float max_x = std::max({armor.pts[0], armor.pts[2], armor.pts[4], armor.pts[6]});
                    float max_y = std::max({armor.pts[1], armor.pts[3], armor.pts[5], armor.pts[7]});

                    det.boxes.emplace_back(
                        static_cast<int>(min_x),
                        static_cast<int>(min_y),
                        static_cast<int>(max_x - min_x),
                        static_cast<int>(max_y - min_y));
                    det.scores.push_back(armor.confidence);
                }
            }
        }

        processed_count.fetch_add(1, std::memory_order_relaxed);

        const auto t1 = std::chrono::steady_clock::now();
        detect_latency.record(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        return det;
    }

    TrackResult track(const Detection &det, [[maybe_unused]] const SharedFeature &sf) {
        const auto t0 = std::chrono::steady_clock::now();

        TrackResult tr;
        tr.frame_seq = det.frame_seq;
        tr.timestamp_ns = det.timestamp_ns;

        // 模拟跟踪计算
        volatile int sum = 0;
        for (const auto &box: det.boxes) {
            sum += box.x + box.y + box.width + box.height;
        }
        (void) sum;

        const auto t1 = std::chrono::steady_clock::now();
        track_latency.record(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

        return tr;
    }

    SolverOutput solve(const TrackResult &tr) {
        SolverOutput out;
        out.frame_seq = tr.frame_seq;
        out.timestamp_ns = tr.timestamp_ns;

        // 模拟解算计算
        volatile double result = 0;
        for (int i = 0; i < 100; ++i) {
            result += std::sin(i * 0.1) * std::cos(i * 0.2);
        }
        (void) result;

        return out;
    }

    void output(const SolverOutput &out) {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        e2e_latency.record(now - out.timestamp_ns);
        output_count.fetch_add(1, std::memory_order_relaxed);
    }

    RealworldStrategy(const SpVisionConfig cfg, const size_t count, const int rows,
                      const int cols) noexcept : frame_pool(count, rows, cols,
                                                            CV_8UC3),
                                                 gray_pool(count, rows, cols,CV_8UC1),
                                                 config(std::move(cfg)),
                                                 detector_(auto_aim::DetectorConfig{
                                                     config.detector.threshold,
                                                     config.detector.max_angle_error,
                                                     config.detector.min_lightbar_ratio,
                                                     config.detector.max_lightbar_ratio,
                                                     config.detector.min_lightbar_length,
                                                     config.detector.min_armor_ratio,
                                                     config.detector.max_armor_ratio,
                                                     config.detector.max_side_ratio,
                                                     config.detector.min_confidence,
                                                     config.detector.max_rectangular_error
                                                 }) {
    }

public:

    // Buffer pool (复用 buffer)
    BufferPool frame_pool;
    BufferPool gray_pool;
    SpVisionConfig config;
    auto_aim::TraditionalDetector detector_;

    // 统计
    LatencyStats capture_latency;
    LatencyStats preprocess_latency;
    LatencyStats detect_latency;
    LatencyStats track_latency;
    LatencyStats e2e_latency;

    std::atomic<uint64_t> frame_count{0};
    std::atomic<uint64_t> processed_count{0};
    std::atomic<uint64_t> output_count{0};
};

// ============ 主测试 ============

int main(const int argc, char **argv) {
    hook_cstream();
    SpVisionConfig config;
    if (auto result = config.read_from(toml::parse_file("sp_vision.toml")); !result) {
        std::cerr << "read sp_vision.toml: " + result.error();
        return 1;
    }
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

    // 调度器
    RealworldStrategy strategy(config, 8, frame_height, frame_width);

    Scheduler<RealworldStrategy> sched;

    sched.init(config, &strategy);

    // 启动
    sched.start();

    // 帧获取线程
    std::atomic running{true};
    cv::Mat synthetic_frame(frame_height, frame_width, CV_8UC3);

    std::thread capture_thread([&] {
        while (running.load(std::memory_order_relaxed)) {
            auto t0 = std::chrono::steady_clock::now();

            // 获取复用 buffer
            auto buffer = strategy.frame_pool.acquire();

            if (use_camera) {
                // 从摄像头读取
                cap.read(buffer->mat());
                if (buffer->mat().empty()) {
                    strategy.frame_pool.release(buffer);
                    continue;
                }
            } else {
                // 合成数据
                const auto seq = strategy.frame_count.load();
                synthetic_frame.setTo(cv::Scalar(seq % 256, (seq * 2) % 256, (seq * 3) % 256));
                synthetic_frame.copyTo(buffer->mat());
            }

            auto t1 = std::chrono::steady_clock::now();
            strategy.capture_latency.record(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

            // 创建 Frame (0-copy, 使用 pool buffer)
            Frame frame;
            frame.seq = strategy.frame_count.fetch_add(1, std::memory_order_relaxed);
            frame.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            frame.image = buffer;

            // 推送到调度器
            if (!sched.push_frame(std::move(frame))) {
                strategy.frame_pool.release(buffer);
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

    const auto start_time = std::chrono::steady_clock::now();
    const auto end_time = start_time + std::chrono::seconds(duration_sec);

    while (std::chrono::steady_clock::now() < end_time) {
        std::this_thread::sleep_for(100ms);

        // 实时统计
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        const auto frames = strategy.frame_count.load();
        const auto processed = strategy.processed_count.load();
        const auto outputs = strategy.output_count.load();
        const auto dropped = sched.stats().frame_drop.load();

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

    const auto total_frames = strategy.frame_count.load();
    const auto total_processed = strategy.processed_count.load();
    const auto total_outputs = strategy.output_count.load();
    const auto total_dropped = sched.stats().frame_drop.load();

    std::cout << "Total Frames:    " << total_frames << "\n";
    std::cout << "Total Processed: " << total_processed << "\n";
    std::cout << "Total Outputs:   " << total_outputs << "\n";
    std::cout << "Total Dropped:   " << total_dropped << "\n";
    std::cout << "Process Rate:    " << std::fixed << std::setprecision(1)
            << (total_processed * 100.0 / total_frames) << "%\n";
    std::cout << "Avg FPS:         " << std::fixed << std::setprecision(1)
            << (total_frames * 1.0 / duration_sec) << "\n\n";

    std::cout << "=== Latency Distribution ===\n";
    strategy.capture_latency.print("Capture");
    strategy.preprocess_latency.print("Preprocess");
    strategy.detect_latency.print("Detect");
    strategy.track_latency.print("Track");
    strategy.e2e_latency.print("E2E");

    std::cout << "\n=== Scheduler Stats ===\n";
    const auto &stats = sched.stats();
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
