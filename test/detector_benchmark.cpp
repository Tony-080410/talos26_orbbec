/**
 * @file detector_benchmark.cpp
 * @brief Full auto-aim pipeline benchmark: detector → solver → tracker
 *
 * Usage: ./detector_benchmark <dataset_path> [options]
 * Options:
 *   --iou value         IoU threshold for matching (default: 0.3)
 *   --debug             Show detection visualization
 *   --enemy red|blue    Enemy color (default: red)
 */

#include <chrono>
#include <fmt/format.h>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>

#include "auto_aim/config.hpp"
#include "auto_aim/dataset_loader.hpp"
#include "auto_aim/detector.hpp"
#include "auto_aim/solver.hpp"
#include "auto_aim/tracker.hpp"

using namespace auto_aim;
using namespace std::chrono;

/**
 * @brief Pipeline metrics for detector + solver + tracker
 */
struct PipelineMetrics {
    // Detection metrics
    AccuracyMetrics detection;

    // Solver metrics
    int solver_success{0};
    int solver_total{0};
    double total_distance{0};

    // Tracker metrics
    int tracking_frames{0}; // Frames in TRACKING state
    int detecting_frames{0}; // Frames in DETECTING state
    int temp_lost_frames{0}; // Frames in TEMP_LOST state
    int idle_frames{0}; // Frames in IDLE state
    int total_frames{0};

    // Latency
    std::vector<double> detect_latencies_ms;
    std::vector<double> solve_latencies_ms;
    std::vector<double> track_latencies_ms;
    std::vector<double> total_latencies_ms;

    void print() const {
        spdlog::info("\n=== Detection Metrics ===");
        detection.print();

        spdlog::info("\n=== Solver Metrics ===");
        spdlog::info("Success Rate: {}/{} ({:.1f}%)", solver_success, solver_total,
                     solver_total > 0 ? 100.0 * solver_success / solver_total : 0.0);
        spdlog::info("Avg Distance: {:.3f} m", solver_success > 0 ? total_distance / solver_success : 0.0);

        spdlog::info("\n=== Tracker State Distribution ===");
        spdlog::info("IDLE:      {} ({:.1f}%)", idle_frames, 100.0 * idle_frames / total_frames);
        spdlog::info("DETECTING: {} ({:.1f}%)", detecting_frames, 100.0 * detecting_frames / total_frames);
        spdlog::info("TRACKING:  {} ({:.1f}%)", tracking_frames, 100.0 * tracking_frames / total_frames);
        spdlog::info("TEMP_LOST: {} ({:.1f}%)", temp_lost_frames, 100.0 * temp_lost_frames / total_frames);

        auto print_latency_stats = [](const std::string &name, const std::vector<double> &latencies) {
            if (latencies.empty())
                return;
            std::vector<double> sorted = latencies;
            std::ranges::sort(sorted);
            const double p50 = sorted[sorted.size() / 2];
            const double p99 = sorted[static_cast<size_t>(sorted.size() * 0.99)];
            const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
            spdlog::info("{}: mean={:.2f}ms p50={:.2f}ms p99={:.2f}ms", name, mean, p50, p99);
        };

        spdlog::info("\n=== Latency Statistics ===");
        print_latency_stats("Detect", detect_latencies_ms);
        print_latency_stats("Solve ", solve_latencies_ms);
        print_latency_stats("Track ", track_latencies_ms);
        print_latency_stats("Total ", total_latencies_ms);
    }
};

/**
 * @brief Match detections with ground truth using Hungarian algorithm (greedy)
 */
AccuracyMetrics evaluate_frame(const talos::ArmorSlots &detections, const std::vector<GroundTruth> &ground_truths,
                               const double iou_threshold = 0.3) {
    AccuracyMetrics metrics;
    metrics.total_gt = static_cast<int>(ground_truths.size());

    // Collect valid detections
    std::vector<std::pair<int, cv::Rect2f>> det_boxes;
    for (int i = 0; i < 10; i++) {
        if (detections.is_valid(i)) {
            const auto &armor = detections.armors[i];
            const float min_x = std::min({armor.pts[0], armor.pts[2], armor.pts[4], armor.pts[6]});
            const float min_y = std::min({armor.pts[1], armor.pts[3], armor.pts[5], armor.pts[7]});
            const float max_x = std::max({armor.pts[0], armor.pts[2], armor.pts[4], armor.pts[6]});
            const float max_y = std::max({armor.pts[1], armor.pts[3], armor.pts[5], armor.pts[7]});
            det_boxes.emplace_back(i, cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y));
        }
    }
    metrics.total_det = static_cast<int>(det_boxes.size());

    // Greedy matching based on IoU
    std::vector gt_matched(ground_truths.size(), false);
    std::vector det_matched(det_boxes.size(), false);

    // Build IoU matrix and sort by IoU descending
    std::vector<std::tuple<double, int, int>> iou_pairs; // (iou, det_idx, gt_idx)
    for (size_t d = 0; d < det_boxes.size(); d++) {
        for (size_t g = 0; g < ground_truths.size(); g++) {
            if (double iou = calculate_iou(det_boxes[d].second, ground_truths[g].bounding_box());
                iou >= iou_threshold) {
                iou_pairs.emplace_back(iou, d, g);
            }
        }
    }

    // Sort by IoU descending
    std::ranges::sort(iou_pairs, [](const auto &a, const auto &b) { return std::get<0>(a) > std::get<0>(b); });

    // Greedy assignment
    for (const auto &[iou, d_idx, g_idx]: iou_pairs) {
        if (det_matched[d_idx] || gt_matched[g_idx])
            continue;

        det_matched[d_idx] = true;
        gt_matched[g_idx] = true;

        metrics.true_positives++;
        metrics.iou_sum += iou;
        metrics.iou_count++;

        // Check color accuracy
        const auto &det = detections.armors[det_boxes[d_idx].first];
        const auto &gt = ground_truths[g_idx];

        metrics.color_total++;
        if (const uint8_t expected_det_color = gt.color == 0 ? 1 : 0; det.color == expected_det_color) {
            metrics.color_correct++;
        }

        // Check type accuracy
        metrics.type_total++;
        if (const uint8_t expected_det_type = gt.type == 0 ? 1 : 0; det.type == expected_det_type) {
            metrics.type_correct++;
        }
    }

    // Count false positives and false negatives
    for (const bool matched: det_matched) {
        if (!matched)
            metrics.false_positives++;
    }
    for (const bool matched: gt_matched) {
        if (!matched)
            metrics.false_negatives++;
    }

    return metrics;
}

/**
 * @brief Visualize full pipeline results
 */
void visualize_frame(const cv::Mat &image, const talos::ArmorSlots &detections,
                     const std::array<ArmorPose3D, 10> &poses3d, const Tracker &tracker,
                     const std::vector<GroundTruth> &ground_truths, const std::string &window_name = "Pipeline") {
    cv::Mat display = image.clone();

    // Draw ground truth (green)
    for (const auto &gt: ground_truths) {
        std::vector<cv::Point> pts;
        for (const auto &p: gt.points) {
            pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        }
        cv::polylines(display, pts, true, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, fmt::format("GT c{} t{}", gt.color, gt.type), pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 255, 0), 1);
    }

    // Draw detections with 3D info
    for (int i = 0; i < 10; i++) {
        if (detections.is_valid(i)) {
            const auto &armor = detections.armors[i];
            const auto &pose = poses3d[i];

            std::vector<cv::Point> pts;
            for (int j = 0; j < 4; j++) {
                pts.emplace_back(static_cast<int>(armor.pts[j * 2]), static_cast<int>(armor.pts[j * 2 + 1]));
            }

            cv::Scalar color = armor.color == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            cv::polylines(display, pts, true, color, 1);
            cv::putText(display,
                        fmt::format("D c{} n{} t{} {:.2f}", talos::COLORS[armor.color], talos::ARMOR_NAMES[armor.name],
                                    talos::ARMOR_TYPES[armor.type], armor.confidence),
                        cv::Point(pts[0].x, pts[0].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);

            // Show 3D distance if solved
            if (pose.valid) {
                cv::putText(display, fmt::format("{:.2f}m", pose.ypd_in_world[2]), cv::Point(pts[0].x, pts[0].y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }
        }
    }

    // Draw tracker state
    std::string state_str = tracker.state_string();
    cv::Scalar state_color;
    switch (tracker.state()) {
        case talos::TrackerState::IDLE:
            state_color = cv::Scalar(128, 128, 128);
            break;
        case talos::TrackerState::DETECTING:
            state_color = cv::Scalar(0, 255, 255);
            break;
        case talos::TrackerState::TRACKING:
            state_color = cv::Scalar(0, 255, 0);
            break;
        case talos::TrackerState::TEMP_LOST:
            state_color = cv::Scalar(0, 165, 255);
            break;
    }
    cv::putText(display, fmt::format("State: {}", state_str), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                state_color, 2);

    // Draw tracked target prediction if available
    if (tracker.state() != talos::TrackerState::IDLE) {
        const auto &target = tracker.target();
        auto ekf_x = target.ekf_x();

        // Draw predicted armor positions
        const auto armor_list = tracker.armor_xyza_list();
        for (size_t i = 0; i < armor_list.size(); i++) {
            const auto &xyza = armor_list[i];
            // Simple projection (assuming fixed camera parameters)
            cv::putText(display, fmt::format("T{}: ({:.2f},{:.2f},{:.2f})", i, xyza[0], xyza[1], xyza[2]),
                        cv::Point(10, 60 + static_cast<int>(i) * 20), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(255, 255, 0), 1);
        }

        // Show target info
        cv::putText(display,
                    fmt::format("Target: {} v_yaw={:.2f}", talos::ARMOR_NAMES[static_cast<int>(target.name)], ekf_x[7]),
                    cv::Point(10, display.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }

    cv::imshow(window_name, display);
}

int main(int argc, char **argv) {
    hook_cstream();

    // Default parameters
    std::string dataset_path = "./datasets";
    double iou_threshold = 0.3;
    bool debug = false;
    talos::Color enemy_color = talos::Color::red;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (std::string arg = argv[i]; arg == "--iou" && i + 1 < argc) {
            iou_threshold = std::stod(argv[++i]);
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--enemy" && i + 1 < argc) {
            std::string color_str = argv[++i];
            enemy_color = color_str == "blue" ? talos::Color::blue : talos::Color::red;
        } else if (arg[0] != '-') {
            dataset_path = arg;
        }
    }

    spdlog::info("=== Full Pipeline Benchmark ===");
    spdlog::info("Dataset: {}", dataset_path);
    spdlog::info("IoU Threshold: {}", iou_threshold);
    spdlog::info("Enemy Color: {}", enemy_color == talos::Color::red ? "red" : "blue");
    spdlog::info("Debug: {}", debug);

    // Load config
    SpVisionConfig config;
    if (auto result = config.read_from(toml::parse_file("sp_vision.toml")); !result) {
        spdlog::critical("read sp_vision.toml: " + result.error());
        return 1;
    }

    // Override enemy color from command line
    config.tracker.enemy_color = enemy_color;

    // Load dataset
    DatasetLoader loader(dataset_path);
    if (loader.empty()) {
        spdlog::critical("no valid frames found in dataset");
        return 1;
    }
    const auto size = loader.size();
    spdlog::info("Loaded {} frames", size);

    // Initialize pipeline components
    TraditionalDetector detector(config, false);
    Solver solver(config.solver);
    Tracker tracker(config.tracker);

    // Set default gimbal orientation (identity for benchmark)
    solver.set_R_gimbal2world(Eigen::Quaterniond::Identity());

    // Metrics
    PipelineMetrics metrics;
    std::array<ArmorPose3D, 10> poses3d{};

    for (size_t i = 0; i < size; i++) {
        auto frame_opt = loader.load_frame(i);
        if (!frame_opt) {
            spdlog::error("load frame {}", i);
            continue;
        }

        const auto &frame = *frame_opt;
        metrics.total_frames++;

        // Generate binary image
        cv::Mat gray_img, binary_img;
        cv::cvtColor(frame.image, gray_img, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_img, binary_img, config.detector.threshold, 255, cv::THRESH_BINARY);

        // === Stage 1: Detection ===
        auto t0 = steady_clock::now();
        auto detections = detector.detect(frame.image, binary_img, frame.frame_id, 0);
        auto t1 = steady_clock::now();

        // === Stage 3: Solver ===
        solver.solve_all(detections, poses3d);
        auto t2 = steady_clock::now();

        // Count solver success
        for (int j = 0; j < 10; j++) {
            if (detections.is_valid(j)) {
                metrics.solver_total++;
                if (poses3d[j].valid) {
                    metrics.solver_success++;
                    metrics.total_distance += poses3d[j].ypd_in_world[2];
                }
            }
        }

        // === Stage 2: Tracker ===
        auto timestamp = steady_clock::now();
        tracker.track(detections, poses3d, timestamp);
        auto t3 = steady_clock::now();

        // Record tracker state
        switch (tracker.state()) {
            case talos::TrackerState::IDLE:
                metrics.idle_frames++;
                break;
            case talos::TrackerState::DETECTING:
                metrics.detecting_frames++;
                break;
            case talos::TrackerState::TRACKING:
                metrics.tracking_frames++;
                break;
            case talos::TrackerState::TEMP_LOST:
                metrics.temp_lost_frames++;
                break;
        }

        // Record latencies
        const double detect_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        const double solve_ms = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
        const double track_ms = duration_cast<microseconds>(t3 - t2).count() / 1000.0;
        const double total_ms = duration_cast<microseconds>(t3 - t0).count() / 1000.0;

        metrics.detect_latencies_ms.push_back(detect_ms);
        metrics.solve_latencies_ms.push_back(solve_ms);
        metrics.track_latencies_ms.push_back(track_ms);
        metrics.total_latencies_ms.push_back(total_ms);

        // Evaluate detection accuracy
        auto frame_metrics = evaluate_frame(detections, frame.ground_truths, iou_threshold);
        metrics.detection += frame_metrics;

        // Progress logging
        if ((i + 1) % 2000 == 0) {
            spdlog::info("Progress: {}/{} ({:.1f}%)", i + 1, size, 100.0 * (i + 1) / size);
            break;
        }

        // Debug visualization
        if (debug) {
            visualize_frame(frame.image, detections, poses3d, tracker, frame.ground_truths);
            visualize_frame(gray_img, detections, poses3d, tracker, frame.ground_truths, "Gray");
            visualize_frame(binary_img, detections, poses3d, tracker, frame.ground_truths, "Binary");
            if (auto key = cv::waitKey(0); key == 'q' || key == 27) { // q or ESC to quit
                break;
            }
        }
    }

    // Print final metrics
    metrics.print();

    return 0;
}
