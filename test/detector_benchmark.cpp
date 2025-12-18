/**
 * @file detector_benchmark.cpp
 * @brief Armor detector accuracy benchmark against ground truth dataset
 *
 * Usage: ./detector_benchmark <dataset_path> [options]
 * Options:
 *   --threshold value   Binary threshold (default: 150)
 *   --iou value         IoU threshold for matching (default: 0.3)
 *   --debug               Show detection visualization
 */

#include <chrono>
#include <fmt/format.h>
#include <numeric>
#include <oneapi/tbb/parallel_for.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <tbb/concurrent_vector.h>
#include "auto_aim/dataset_loader.hpp"
#include "auto_aim/detector.hpp"

#include "sp_vision/config.hpp"

using namespace auto_aim;
using namespace std::chrono;

/**
 * @brief Match detections with ground truth using Hungarian algorithm (greedy)
 */
AccuracyMetrics evaluate_frame(const spike::ArmorSlots &detections, const std::vector<GroundTruth> &ground_truths,
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
        // Map: GT color 0=Blue, 1=Red -> Detection color 0=red, 1=blue
        // Note: detector_state.hpp has: enum class Color { red = 0, blue = 1, ... }
        // Dataset has: 0=Blue, 1=Red
        if (const uint8_t expected_det_color = gt.color == 0 ? 1 : 0; det.color == expected_det_color) {
            metrics.color_correct++;
        }

        // Check type accuracy
        metrics.type_total++;
        // GT type: 0=Small, 1=Large -> Detection type: 0=big, 1=small
        // detector_state.hpp: enum class ArmorType { big = 0, small = 1 }
        // Dataset: 0=Small, 1=Large
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
 * @brief Visualize detection results
 */
void visualize_frame(const cv::Mat &image, const spike::ArmorSlots &detections,
                     const std::vector<GroundTruth> &ground_truths, const std::string &window_name = "Detection") {
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

    // Draw detections (red for red armor, blue for blue armor)
    for (int i = 0; i < 10; i++) {
        if (detections.is_valid(i)) {
            const auto &armor = detections.armors[i];
            std::vector<cv::Point> pts;
            for (int j = 0; j < 4; j++) {
                pts.emplace_back(static_cast<int>(armor.pts[j * 2]), static_cast<int>(armor.pts[j * 2 + 1]));
            }

            cv::Scalar color = armor.color == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            cv::polylines(display, pts, true, color, 1);

            cv::putText(display,
                        fmt::format("D c{} n{} t{} {:.2f}", spike::COLORS[armor.color], spike::ARMOR_NAMES[armor.name],
                                    spike::ARMOR_TYPES[armor.type], armor.confidence),
                        cv::Point(pts[0].x, pts[0].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
        }
    }

    cv::imshow(window_name, display);
}

int main(int argc, char **argv) {
    hook_cstream();
    // Default parameters
    std::string dataset_path = "./datasets";
    double iou_threshold = 0.3;
    bool debug = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (std::string arg = argv[i]; arg == "--iou" && i + 1 < argc) {
            iou_threshold = std::stod(argv[++i]);
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg[0] != '-') {
            dataset_path = arg;
        }
    }

    spdlog::info("=== Armor Detector Benchmark ===\n");
    spdlog::info("Dataset: {}\n", dataset_path);
    spdlog::info("IoU Threshold: {}\n", iou_threshold);
    spdlog::info("Debug: {}\n\n", debug);


    SpVisionConfig config;
    if (auto result = config.read_from(toml::parse_file("sp_vision.toml")); !result) {
        spdlog::critical("read sp_vision.toml: " + result.error());
        return 1;
    }

    // Load dataset
    DatasetLoader loader(dataset_path);
    if (loader.empty()) {
        spdlog::critical("no valid frames found in dataset");
        return 1;
    }
    const auto size = loader.size();
    spdlog::info("Loaded {} frames", size);

    // Metrics
    std::mutex metrics_mutex;
    AccuracyMetrics total_metrics;
    std::atomic<uint64_t> num_frames{0};
    tbb::concurrent_vector<double> latencies;
    std::mutex show_mutex;
    TraditionalDetector detector(config, false);

    for (size_t i = 0; i < size; i++) {
        auto frame_opt = loader.load_frame(i);
        if (!frame_opt) {
            spdlog::error("load frame {}", i);
            continue;
        }

        const auto &frame = *frame_opt;

        // Generate binary image
        cv::Mat gray_img, binary_img;
        cv::cvtColor(frame.image, gray_img, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_img, binary_img, config.detector.threshold, 255, cv::THRESH_BINARY);

        // Time the detection
        auto t0 = steady_clock::now();
        auto detections = detector.detect(frame.image, binary_img, frame.frame_id, 0);
        auto t1 = steady_clock::now();

        double latency_ms = static_cast<double>(duration_cast<microseconds>(t1 - t0).count()) / 1000.0;
        latencies.push_back(latency_ms);

        // Evaluate
        auto frame_metrics = evaluate_frame(detections, frame.ground_truths, iou_threshold);
        metrics_mutex.lock();
        total_metrics += frame_metrics;
        metrics_mutex.unlock();

        spdlog::debug("GT:{} Det:{} TP:{} Latency:{:.2f}ms", frame_metrics.total_gt, frame_metrics.total_det,
                      frame_metrics.true_positives, latency_ms);
        ++num_frames;
        if (num_frames % 2000 == 0) {
            spdlog::info("working: {}/{}", num_frames.load(), size);
        }
        // Debug visualization
        if (debug) {
            show_mutex.lock();
            visualize_frame(frame.image, detections, frame.ground_truths);
            visualize_frame(gray_img, detections, frame.ground_truths, "Gray");
            visualize_frame(binary_img, detections, frame.ground_truths, "Binary");
            cv::waitKey(0);
            show_mutex.unlock();
        }
    }


    // Calculate latency statistics
    if (!latencies.empty()) {
        std::ranges::sort(latencies);
        double p50 = latencies[latencies.size() / 2];
        double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
        double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

        spdlog::info("\n=== Latency Statistics ===\n");
        spdlog::info("Mean:  {:.2f} ms\n", mean);
        spdlog::info("P50:   {:.2f} ms\n", p50);
        spdlog::info("P99:   {:.2f} ms\n", p99);
    }

    // Print final metrics
    total_metrics.print();

    return 0;
}
