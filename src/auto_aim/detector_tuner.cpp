#include "detector_tuner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace auto_aim {

// Forward declaration
AccuracyMetrics evaluate_frame(
    const talos::ArmorSlots& detections, const std::vector<GroundTruth>& ground_truths,
    double iou_threshold = 0.3);

// === TuningConfig Implementation ===

std::expected<void, std::string> TuningConfig::read_from(const toml::table& tbl) {
    auto read_range = [](const toml::table& t, const std::string& name,
                         ParameterRange& range) -> std::expected<void, std::string> {
        const auto section = t[name].as_table();
        if (!section) {
            return std::unexpected(fmt::format("Missing section: {}", name));
        }

        if (const auto v = (*section)["min"].value<double>()) {
            range.min = *v;
        } else {
            return std::unexpected(fmt::format("{}.min is required", name));
        }

        if (const auto v = (*section)["max"].value<double>()) {
            range.max = *v;
        } else {
            return std::unexpected(fmt::format("{}.max is required", name));
        }

        if (const auto v = (*section)["initial_points"].value<int64_t>()) {
            range.initial_points = static_cast<size_t>(*v);
        } else {
            range.initial_points = 3; // Default
        }

        return {};
    };

    // Read tuning section
    if (const auto tuning = tbl["tuning"].as_table()) {
        if (const auto v = (*tuning)["max_iterations"].value<int64_t>()) {
            max_iterations = static_cast<size_t>(*v);
        }
        if (const auto v = (*tuning)["convergence_threshold"].value<double>()) {
            convergence_threshold = *v;
        }
        if (const auto v = (*tuning)["num_clusters"].value<int64_t>()) {
            num_clusters = static_cast<size_t>(*v);
        }
        if (const auto v = (*tuning)["top_percentile"].value<double>()) {
            top_percentile = *v;
        }
        if (const auto v = (*tuning)["max_pareto_solutions"].value<int64_t>()) {
            max_pareto_solutions = static_cast<size_t>(*v);
        }
        if (const auto v = (*tuning)["quick_mode"].value<bool>()) {
            quick_mode = *v;
        }
        if (const auto v = (*tuning)["sample_ratio"].value<double>()) {
            sample_ratio = *v;
        }
    }

    // Read search_space section
    const auto space = tbl["search_space"].as_table();
    if (!space) {
        return std::unexpected("Missing [search_space] section");
    }

    if (auto r = read_range(*space, "threshold", threshold); !r)
        return r;
    if (auto r = read_range(*space, "max_angle_error", max_angle_error); !r)
        return r;
    if (auto r = read_range(*space, "min_lightbar_ratio", min_lightbar_ratio); !r)
        return r;
    if (auto r = read_range(*space, "max_lightbar_ratio", max_lightbar_ratio); !r)
        return r;
    if (auto r = read_range(*space, "min_lightbar_length", min_lightbar_length); !r)
        return r;
    if (auto r = read_range(*space, "min_armor_ratio", min_armor_ratio); !r)
        return r;
    if (auto r = read_range(*space, "max_armor_ratio", max_armor_ratio); !r)
        return r;
    if (auto r = read_range(*space, "max_side_ratio", max_side_ratio); !r)
        return r;
    if (auto r = read_range(*space, "min_confidence", min_confidence); !r)
        return r;
    if (auto r = read_range(*space, "max_rectangular_error", max_rectangular_error); !r)
        return r;

    return {};
}

TuningConfig TuningConfig::default_config() {
    TuningConfig cfg;
    cfg.threshold             = {50.0, 200.0, 4};
    cfg.max_angle_error       = {30.0, 60.0, 3};
    cfg.min_lightbar_ratio    = {1.0, 3.0, 3};
    cfg.max_lightbar_ratio    = {15.0, 25.0, 3};
    cfg.min_lightbar_length   = {5.0, 12.0, 3};
    cfg.min_armor_ratio       = {0.8, 1.5, 3};
    cfg.max_armor_ratio       = {3.0, 7.0, 3};
    cfg.max_side_ratio        = {1.2, 2.0, 3};
    cfg.min_confidence        = {0.6, 0.95, 4};
    cfg.max_rectangular_error = {15.0, 35.0, 3};
    return cfg;
}

// === SearchRegion Implementation ===

std::vector<DetectorConfig> SearchRegion::generate_grid(size_t resolution) const {
    std::vector<DetectorConfig> configs;

    // Generate points within region: center +/- radius
    auto gen_range = [resolution](double center_val, double radius) -> std::vector<double> {
        std::vector<double> vals;
        vals.reserve(resolution);
        const double lo = center_val - radius;
        const double hi = center_val + radius;
        if (resolution == 1) {
            vals.push_back(center_val);
        } else {
            const double step = (hi - lo) / static_cast<double>(resolution - 1);
            for (size_t i = 0; i < resolution; ++i) {
                vals.push_back(lo + step * static_cast<double>(i));
            }
        }
        return vals;
    };

    const auto v0 = gen_range(center.threshold, radii[0]);
    const auto v1 = gen_range(center.max_angle_error, radii[1]);
    const auto v2 = gen_range(center.min_lightbar_ratio, radii[2]);
    const auto v3 = gen_range(center.max_lightbar_ratio, radii[3]);
    const auto v4 = gen_range(center.min_lightbar_length, radii[4]);
    const auto v5 = gen_range(center.min_armor_ratio, radii[5]);
    const auto v6 = gen_range(center.max_armor_ratio, radii[6]);
    const auto v7 = gen_range(center.max_side_ratio, radii[7]);
    const auto v8 = gen_range(center.min_confidence, radii[8]);
    const auto v9 = gen_range(center.max_rectangular_error, radii[9]);

    // Generate all combinations
    for (double d0 : v0) {
        for (double d1 : v1) {
            for (double d2 : v2) {
                for (double d3 : v3) {
                    for (double d4 : v4) {
                        for (double d5 : v5) {
                            for (double d6 : v6) {
                                for (double d7 : v7) {
                                    for (double d8 : v8) {
                                        for (double d9 : v9) {
                                            DetectorConfig cfg;
                                            cfg.threshold             = d0;
                                            cfg.max_angle_error       = d1;
                                            cfg.min_lightbar_ratio    = d2;
                                            cfg.max_lightbar_ratio    = d3;
                                            cfg.min_lightbar_length   = d4;
                                            cfg.min_armor_ratio       = d5;
                                            cfg.max_armor_ratio       = d6;
                                            cfg.max_side_ratio        = d7;
                                            cfg.min_confidence        = d8;
                                            cfg.max_rectangular_error = d9;
                                            configs.push_back(cfg);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return configs;
}

// === TuningResult Implementation ===

void TuningResult::save_to_toml(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        spdlog::error("Failed to open {} for writing", path);
        return;
    }

    out << "# Optimized detector parameters\n";
    out << "# Generated by detector_tuner\n\n";

    for (size_t i = 0; i < pareto_front.size(); ++i) {
        const auto& c = pareto_front[i];
        out << fmt::format(
            "# Solution {}: F1={:.4f}, IoU={:.4f}, Latency={:.2f}ms\n", i + 1,
            c.objectives.f1_score, c.objectives.mean_iou, c.objectives.latency_ms);
        out << fmt::format("[solution_{}]\n", i + 1);
        out << fmt::format("threshold = {:.1f}\n", c.config.threshold);
        out << fmt::format("max_angle_error = {:.1f}\n", c.config.max_angle_error);
        out << fmt::format("min_lightbar_ratio = {:.2f}\n", c.config.min_lightbar_ratio);
        out << fmt::format("max_lightbar_ratio = {:.1f}\n", c.config.max_lightbar_ratio);
        out << fmt::format("min_lightbar_length = {:.1f}\n", c.config.min_lightbar_length);
        out << fmt::format("min_armor_ratio = {:.2f}\n", c.config.min_armor_ratio);
        out << fmt::format("max_armor_ratio = {:.1f}\n", c.config.max_armor_ratio);
        out << fmt::format("max_side_ratio = {:.2f}\n", c.config.max_side_ratio);
        out << fmt::format("min_confidence = {:.2f}\n", c.config.min_confidence);
        out << fmt::format("max_rectangular_error = {:.1f}\n\n", c.config.max_rectangular_error);
    }

    out << "# Statistics\n";
    out << fmt::format("# Total evaluations: {}\n", total_evaluations);
    out << fmt::format("# Pareto solutions: {}\n", pareto_front.size());
}

void TuningResult::visualize(const std::string& output_dir) const {
    std::filesystem::create_directories(output_dir);

    plot_pareto_front(all_evaluated, pareto_front, output_dir + "/pareto_front.png");
    plot_convergence(iteration_best_f1, output_dir + "/convergence.png");
}

// === DetectorTuner Implementation ===

DetectorTuner::DetectorTuner(
    TuningConfig tuning_config, const DatasetLoader& dataset, SpVisionConfig base_config)
    : tuning_config_(std::move(tuning_config))
    , dataset_(dataset)
    , base_config_(std::move(base_config)) {}

std::expected<TuningResult, std::string> DetectorTuner::run() {
    TuningResult result;

    if (dataset_.empty()) {
        return std::unexpected("Dataset is empty");
    }

    // Step 1: Generate initial coarse grid
    auto initial_grid = generate_initial_grid();
    spdlog::info("Generated {} initial configurations", initial_grid.size());

    // Step 2: Evaluate all in parallel
    auto candidates          = evaluate_parallel(initial_grid);
    result.all_evaluated     = candidates;
    result.total_evaluations = candidates.size();

    // Track best F1 for convergence
    double best_f1 = 0.0;
    for (const auto& c : candidates) {
        best_f1 = std::max(best_f1, c.objectives.f1_score);
    }
    result.iteration_best_f1.push_back(best_f1);
    spdlog::info("Initial best F1: {:.4f}", best_f1);

    // Iterative refinement loop
    for (size_t iter = 0; iter < tuning_config_.max_iterations; ++iter) {
        spdlog::info("=== Iteration {} ===", iter + 1);

        // Step 3: Select top candidates
        const size_t top_k = std::max(
            size_t(1), static_cast<size_t>(candidates.size() * tuning_config_.top_percentile));

        std::partial_sort(
            candidates.begin(), candidates.begin() + static_cast<ptrdiff_t>(top_k),
            candidates.end(), [](const auto& a, const auto& b) {
                return a.objectives.f1_score > b.objectives.f1_score;
            });

        std::vector<ParameterCandidate> top_candidates(
            candidates.begin(), candidates.begin() + static_cast<ptrdiff_t>(top_k));

        spdlog::info("Selected top {} candidates", top_k);

        // Step 4: Cluster into regions
        auto regions = cluster_top_candidates(top_candidates);
        spdlog::info("Clustered into {} regions", regions.size());

        // Step 5: Generate refined grid (2x resolution)
        std::vector<DetectorConfig> refined_grid;
        for (const auto& region : regions) {
            auto local_grid = region.generate_grid(3); // 3 points per dimension in refined search
            refined_grid.insert(refined_grid.end(), local_grid.begin(), local_grid.end());
        }

        if (refined_grid.empty()) {
            spdlog::info("No refined candidates generated, stopping");
            break;
        }

        spdlog::info("Generated {} refined configurations", refined_grid.size());

        // Step 6: Evaluate refined grid
        auto refined_candidates = evaluate_parallel(refined_grid);
        result.all_evaluated.insert(
            result.all_evaluated.end(), refined_candidates.begin(), refined_candidates.end());
        result.total_evaluations += refined_candidates.size();

        // Step 7: Check convergence
        double new_best_f1 = best_f1;
        for (const auto& c : refined_candidates) {
            new_best_f1 = std::max(new_best_f1, c.objectives.f1_score);
        }

        const double improvement = (new_best_f1 - best_f1) / (best_f1 + 1e-10);
        result.iteration_best_f1.push_back(new_best_f1);

        spdlog::info(
            "Best F1: {:.4f} -> {:.4f} (improvement: {:.2f}%)", best_f1, new_best_f1,
            improvement * 100);

        if (improvement < tuning_config_.convergence_threshold) {
            spdlog::info("Converged after {} iterations", iter + 1);
            break;
        }

        best_f1 = new_best_f1;

        // Merge candidates for next iteration
        candidates.insert(candidates.end(), refined_candidates.begin(), refined_candidates.end());
    }

    // Step 8: Extract final Pareto front
    auto pareto         = extract_pareto_front(result.all_evaluated);
    result.pareto_front = select_diverse_pareto(pareto, tuning_config_.max_pareto_solutions);

    spdlog::info("Final Pareto front: {} solutions", result.pareto_front.size());

    return result;
}

std::vector<DetectorConfig> DetectorTuner::generate_initial_grid() const {
    std::vector<DetectorConfig> configs;

    const auto v0 = tuning_config_.threshold.sample_grid(tuning_config_.threshold.initial_points);
    const auto v1 =
        tuning_config_.max_angle_error.sample_grid(tuning_config_.max_angle_error.initial_points);
    const auto v2 = tuning_config_.min_lightbar_ratio.sample_grid(
        tuning_config_.min_lightbar_ratio.initial_points);
    const auto v3 = tuning_config_.max_lightbar_ratio.sample_grid(
        tuning_config_.max_lightbar_ratio.initial_points);
    const auto v4 = tuning_config_.min_lightbar_length.sample_grid(
        tuning_config_.min_lightbar_length.initial_points);
    const auto v5 =
        tuning_config_.min_armor_ratio.sample_grid(tuning_config_.min_armor_ratio.initial_points);
    const auto v6 =
        tuning_config_.max_armor_ratio.sample_grid(tuning_config_.max_armor_ratio.initial_points);
    const auto v7 =
        tuning_config_.max_side_ratio.sample_grid(tuning_config_.max_side_ratio.initial_points);
    const auto v8 =
        tuning_config_.min_confidence.sample_grid(tuning_config_.min_confidence.initial_points);
    const auto v9 = tuning_config_.max_rectangular_error.sample_grid(
        tuning_config_.max_rectangular_error.initial_points);

    // Calculate total combinations
    const size_t total = v0.size() * v1.size() * v2.size() * v3.size() * v4.size() * v5.size()
                       * v6.size() * v7.size() * v8.size() * v9.size();
    configs.reserve(total);

    // Generate all combinations
    for (double d0 : v0) {
        for (double d1 : v1) {
            for (double d2 : v2) {
                for (double d3 : v3) {
                    for (double d4 : v4) {
                        for (double d5 : v5) {
                            for (double d6 : v6) {
                                for (double d7 : v7) {
                                    for (double d8 : v8) {
                                        for (double d9 : v9) {
                                            DetectorConfig cfg;
                                            cfg.threshold             = d0;
                                            cfg.max_angle_error       = d1;
                                            cfg.min_lightbar_ratio    = d2;
                                            cfg.max_lightbar_ratio    = d3;
                                            cfg.min_lightbar_length   = d4;
                                            cfg.min_armor_ratio       = d5;
                                            cfg.max_armor_ratio       = d6;
                                            cfg.max_side_ratio        = d7;
                                            cfg.min_confidence        = d8;
                                            cfg.max_rectangular_error = d9;
                                            configs.push_back(cfg);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return configs;
}

std::vector<ParameterCandidate>
    DetectorTuner::evaluate_parallel(const std::vector<DetectorConfig>& configs) const {
    tbb::concurrent_vector<ParameterCandidate> results;
    results.reserve(configs.size());

    std::atomic<size_t> progress{0};
    const size_t total    = configs.size();
    const auto start_time = std::chrono::steady_clock::now();

    // Progress display interval based on total count
    const size_t report_interval = std::max(size_t(1), total / 100); // Report every 1%

    tbb::parallel_for(size_t(0), configs.size(), [&](size_t i) {
        auto candidate = evaluate_single(configs[i]);
        results.push_back(candidate);

        const size_t done = ++progress;
        if (done % report_interval == 0 || done == total) {
            const auto now           = std::chrono::steady_clock::now();
            const double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
            const double rate        = static_cast<double>(done) / elapsed_sec;
            const double eta_sec     = (total - done) / rate;

            // Build progress bar
            const int bar_width = 30;
            const double pct    = static_cast<double>(done) / static_cast<double>(total);
            const int filled    = static_cast<int>(pct * bar_width);

            std::string bar(bar_width, ' ');
            for (int j = 0; j < filled; ++j)
                bar[j] = '=';
            if (filled < bar_width)
                bar[filled] = '>';

            spdlog::info(
                "[{}] {:5.1f}% ({}/{}) | {:.0f}/s | ETA: {:.0f}s", bar, pct * 100, done, total,
                rate, eta_sec);
        }
    });

    return std::vector<ParameterCandidate>(results.begin(), results.end());
}
static auto classifier = Classifier(SpVisionClassifierConfig{"tiny_resnet.onnx", false, Gpu});

ParameterCandidate DetectorTuner::evaluate_single(const DetectorConfig& config) const {
    // Create test config
    SpVisionConfig test_config                 = base_config_;
    test_config.detector.threshold             = config.threshold;
    test_config.detector.max_angle_error       = config.max_angle_error;
    test_config.detector.min_lightbar_ratio    = config.min_lightbar_ratio;
    test_config.detector.max_lightbar_ratio    = config.max_lightbar_ratio;
    test_config.detector.min_lightbar_length   = config.min_lightbar_length;
    test_config.detector.min_armor_ratio       = config.min_armor_ratio;
    test_config.detector.max_armor_ratio       = config.max_armor_ratio;
    test_config.detector.max_side_ratio        = config.max_side_ratio;
    test_config.detector.min_confidence        = config.min_confidence;
    test_config.detector.max_rectangular_error = config.max_rectangular_error;

    TraditionalDetector detector(test_config, classifier);
    DetectorWorkBuffers buffers;

    AccuracyMetrics total_metrics;
    std::vector<double> latencies;

    // Calculate frame step based on sample_ratio
    const size_t total_frames = 1000;
    const size_t sample_count =
        std::max(size_t(1), static_cast<size_t>(total_frames * tuning_config_.sample_ratio));
    const size_t step = std::max(size_t(1), total_frames / sample_count);

    latencies.reserve(sample_count);

    // Evaluate on sampled frames
    for (size_t i = 0; i < total_frames; i += step) {
        auto frame_opt = dataset_.load_frame(i);
        if (!frame_opt) {
            continue;
        }

        const auto& frame = *frame_opt;

        // Prepare binary image
        cv::Mat gray_img;
        cv::Mat binary_img;
        cv::cvtColor(frame.image, gray_img, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_img, binary_img, config.threshold, 255, cv::THRESH_BINARY);

        // Measure latency
        const auto t0   = std::chrono::steady_clock::now();
        auto detections = detector.detect(buffers, frame.image, binary_img, i, 0);
        const auto t1   = std::chrono::steady_clock::now();

        const double latency_ms =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count())
            / 1000.0;
        latencies.push_back(latency_ms);

        // Evaluate accuracy
        auto frame_metrics = evaluate_frame(detections, frame.ground_truths);
        total_metrics += frame_metrics;
    }

    // Build result
    ParameterCandidate result;
    result.config              = config;
    result.objectives.f1_score = total_metrics.f1_score();
    result.objectives.mean_iou = total_metrics.mean_iou();
    result.precision           = total_metrics.precision();
    result.recall              = total_metrics.recall();

    // Use median latency (robust to outliers)
    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        result.objectives.latency_ms = latencies[latencies.size() / 2];
    }

    return result;
}

std::vector<SearchRegion>
    DetectorTuner::cluster_top_candidates(const std::vector<ParameterCandidate>& candidates) const {
    if (candidates.empty()) {
        return {};
    }

    // K-means clustering
    auto clusters = kmeans_cluster(candidates, tuning_config_.num_clusters);

    std::vector<SearchRegion> regions;
    regions.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        if (cluster.empty()) {
            continue;
        }

        SearchRegion region;

        // Compute cluster center (mean)
        double sum_threshold = 0, sum_angle = 0, sum_min_lb = 0, sum_max_lb = 0, sum_min_len = 0;
        double sum_min_armor = 0, sum_max_armor = 0, sum_side = 0, sum_conf = 0, sum_rect = 0;

        for (const auto& c : cluster) {
            sum_threshold += c.config.threshold;
            sum_angle += c.config.max_angle_error;
            sum_min_lb += c.config.min_lightbar_ratio;
            sum_max_lb += c.config.max_lightbar_ratio;
            sum_min_len += c.config.min_lightbar_length;
            sum_min_armor += c.config.min_armor_ratio;
            sum_max_armor += c.config.max_armor_ratio;
            sum_side += c.config.max_side_ratio;
            sum_conf += c.config.min_confidence;
            sum_rect += c.config.max_rectangular_error;
        }

        const auto n                        = static_cast<double>(cluster.size());
        region.center.threshold             = sum_threshold / n;
        region.center.max_angle_error       = sum_angle / n;
        region.center.min_lightbar_ratio    = sum_min_lb / n;
        region.center.max_lightbar_ratio    = sum_max_lb / n;
        region.center.min_lightbar_length   = sum_min_len / n;
        region.center.min_armor_ratio       = sum_min_armor / n;
        region.center.max_armor_ratio       = sum_max_armor / n;
        region.center.max_side_ratio        = sum_side / n;
        region.center.min_confidence        = sum_conf / n;
        region.center.max_rectangular_error = sum_rect / n;

        // Compute search radii (1.5x std deviation)
        double var_threshold = 0, var_angle = 0, var_min_lb = 0, var_max_lb = 0, var_min_len = 0;
        double var_min_armor = 0, var_max_armor = 0, var_side = 0, var_conf = 0, var_rect = 0;

        for (const auto& c : cluster) {
            var_threshold += std::pow(c.config.threshold - region.center.threshold, 2);
            var_angle += std::pow(c.config.max_angle_error - region.center.max_angle_error, 2);
            var_min_lb +=
                std::pow(c.config.min_lightbar_ratio - region.center.min_lightbar_ratio, 2);
            var_max_lb +=
                std::pow(c.config.max_lightbar_ratio - region.center.max_lightbar_ratio, 2);
            var_min_len +=
                std::pow(c.config.min_lightbar_length - region.center.min_lightbar_length, 2);
            var_min_armor += std::pow(c.config.min_armor_ratio - region.center.min_armor_ratio, 2);
            var_max_armor += std::pow(c.config.max_armor_ratio - region.center.max_armor_ratio, 2);
            var_side += std::pow(c.config.max_side_ratio - region.center.max_side_ratio, 2);
            var_conf += std::pow(c.config.min_confidence - region.center.min_confidence, 2);
            var_rect +=
                std::pow(c.config.max_rectangular_error - region.center.max_rectangular_error, 2);
        }

        // Minimum radius to ensure exploration
        constexpr double min_radius_factor = 0.1;
        region.radii[0]                    = std::max(
            1.5 * std::sqrt(var_threshold / n),
            (tuning_config_.threshold.max - tuning_config_.threshold.min) * min_radius_factor);
        region.radii[1] = std::max(
            1.5 * std::sqrt(var_angle / n),
            (tuning_config_.max_angle_error.max - tuning_config_.max_angle_error.min)
                * min_radius_factor);
        region.radii[2] = std::max(
            1.5 * std::sqrt(var_min_lb / n),
            (tuning_config_.min_lightbar_ratio.max - tuning_config_.min_lightbar_ratio.min)
                * min_radius_factor);
        region.radii[3] = std::max(
            1.5 * std::sqrt(var_max_lb / n),
            (tuning_config_.max_lightbar_ratio.max - tuning_config_.max_lightbar_ratio.min)
                * min_radius_factor);
        region.radii[4] = std::max(
            1.5 * std::sqrt(var_min_len / n),
            (tuning_config_.min_lightbar_length.max - tuning_config_.min_lightbar_length.min)
                * min_radius_factor);
        region.radii[5] = std::max(
            1.5 * std::sqrt(var_min_armor / n),
            (tuning_config_.min_armor_ratio.max - tuning_config_.min_armor_ratio.min)
                * min_radius_factor);
        region.radii[6] = std::max(
            1.5 * std::sqrt(var_max_armor / n),
            (tuning_config_.max_armor_ratio.max - tuning_config_.max_armor_ratio.min)
                * min_radius_factor);
        region.radii[7] = std::max(
            1.5 * std::sqrt(var_side / n),
            (tuning_config_.max_side_ratio.max - tuning_config_.max_side_ratio.min)
                * min_radius_factor);
        region.radii[8] = std::max(
            1.5 * std::sqrt(var_conf / n),
            (tuning_config_.min_confidence.max - tuning_config_.min_confidence.min)
                * min_radius_factor);
        region.radii[9] = std::max(
            1.5 * std::sqrt(var_rect / n),
            (tuning_config_.max_rectangular_error.max - tuning_config_.max_rectangular_error.min)
                * min_radius_factor);

        regions.push_back(region);
    }

    return regions;
}

std::vector<ParameterCandidate>
    DetectorTuner::extract_pareto_front(const std::vector<ParameterCandidate>& candidates) const {
    std::vector<ParameterCandidate> pareto_front;

    for (const auto& candidate : candidates) {
        bool is_dominated = false;

        for (const auto& other : candidates) {
            if (&candidate != &other && other.objectives.dominates(candidate.objectives)) {
                is_dominated = true;
                break;
            }
        }

        if (!is_dominated) {
            pareto_front.push_back(candidate);
        }
    }

    return pareto_front;
}

std::vector<std::vector<ParameterCandidate>> DetectorTuner::kmeans_cluster(
    const std::vector<ParameterCandidate>& candidates, size_t k) const {
    if (candidates.size() <= k) {
        // Each candidate is its own cluster
        std::vector<std::vector<ParameterCandidate>> result;
        result.reserve(candidates.size());
        for (const auto& c : candidates) {
            result.push_back({c});
        }
        return result;
    }

    // Initialize centroids using k-means++
    std::vector<std::array<double, 10>> centroids;
    centroids.reserve(k);

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);

    // First centroid: random
    centroids.push_back(candidates[dist(rng)].normalized_params());

    // Remaining centroids: k-means++ initialization
    for (size_t i = 1; i < k; ++i) {
        std::vector<double> min_distances(candidates.size());

        for (size_t j = 0; j < candidates.size(); ++j) {
            const auto params = candidates[j].normalized_params();
            double min_dist   = std::numeric_limits<double>::max();

            for (const auto& centroid : centroids) {
                double dist_sq = 0.0;
                for (size_t d = 0; d < 10; ++d) {
                    dist_sq += std::pow(params[d] - centroid[d], 2);
                }
                min_dist = std::min(min_dist, dist_sq);
            }
            min_distances[j] = min_dist;
        }

        // Sample proportional to squared distance
        std::discrete_distribution<size_t> weighted_dist(
            min_distances.begin(), min_distances.end());
        centroids.push_back(candidates[weighted_dist(rng)].normalized_params());
    }

    // K-means iterations
    std::vector<size_t> assignments(candidates.size());
    constexpr size_t max_iterations = 100;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        bool changed = false;

        // Assign points to nearest centroid
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto params   = candidates[i].normalized_params();
            double min_dist     = std::numeric_limits<double>::max();
            size_t best_cluster = 0;

            for (size_t c = 0; c < k; ++c) {
                double dist_sq = 0.0;
                for (size_t d = 0; d < 10; ++d) {
                    dist_sq += std::pow(params[d] - centroids[c][d], 2);
                }
                if (dist_sq < min_dist) {
                    min_dist     = dist_sq;
                    best_cluster = c;
                }
            }

            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed        = true;
            }
        }

        if (!changed) {
            break;
        }

        // Update centroids
        std::vector<std::array<double, 10>> new_centroids(k, std::array<double, 10>{});
        std::vector<size_t> cluster_sizes(k, 0);

        for (size_t i = 0; i < candidates.size(); ++i) {
            const size_t c    = assignments[i];
            const auto params = candidates[i].normalized_params();
            for (size_t d = 0; d < 10; ++d) {
                new_centroids[c][d] += params[d];
            }
            cluster_sizes[c]++;
        }

        for (size_t c = 0; c < k; ++c) {
            if (cluster_sizes[c] > 0) {
                for (size_t d = 0; d < 10; ++d) {
                    new_centroids[c][d] /= static_cast<double>(cluster_sizes[c]);
                }
                centroids[c] = new_centroids[c];
            }
        }
    }

    // Build result clusters
    std::vector<std::vector<ParameterCandidate>> clusters(k);
    for (size_t i = 0; i < candidates.size(); ++i) {
        clusters[assignments[i]].push_back(candidates[i]);
    }

    // Remove empty clusters
    clusters.erase(
        std::remove_if(clusters.begin(), clusters.end(), [](const auto& c) { return c.empty(); }),
        clusters.end());

    return clusters;
}

std::vector<ParameterCandidate> DetectorTuner::select_diverse_pareto(
    const std::vector<ParameterCandidate>& pareto_front, size_t max_count) const {
    if (pareto_front.size() <= max_count) {
        return pareto_front;
    }

    std::vector<ParameterCandidate> selected;
    selected.reserve(max_count);

    // Always include: best F1, best latency
    auto best_f1 = std::max_element(
        pareto_front.begin(), pareto_front.end(),
        [](const auto& a, const auto& b) { return a.objectives.f1_score < b.objectives.f1_score; });
    auto best_latency = std::min_element(
        pareto_front.begin(), pareto_front.end(), [](const auto& a, const auto& b) {
            return a.objectives.latency_ms < b.objectives.latency_ms;
        });

    selected.push_back(*best_f1);
    if (best_latency != best_f1) {
        selected.push_back(*best_latency);
    }

    // If we need more, add the most balanced (closest to ideal point)
    if (selected.size() < max_count) {
        // Find ideal point (best of each objective)
        double best_f1_val  = best_f1->objectives.f1_score;
        double best_iou_val = 0.0;
        double best_lat_val = best_latency->objectives.latency_ms;

        for (const auto& c : pareto_front) {
            best_iou_val = std::max(best_iou_val, c.objectives.mean_iou);
        }

        // Find point closest to ideal
        double min_dist = std::numeric_limits<double>::max();
        std::optional<size_t> best_balanced;

        for (size_t i = 0; i < pareto_front.size(); ++i) {
            const auto& c = pareto_front[i];

            // Skip if already selected
            bool already_selected = false;
            for (const auto& s : selected) {
                if (&s.config == &c.config) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected)
                continue;

            // Normalized distance to ideal point
            const double d_f1  = (best_f1_val - c.objectives.f1_score) / (best_f1_val + 1e-10);
            const double d_iou = (best_iou_val - c.objectives.mean_iou) / (best_iou_val + 1e-10);
            const double d_lat =
                (c.objectives.latency_ms - best_lat_val) / (c.objectives.latency_ms + 1e-10);

            const double dist = std::sqrt(d_f1 * d_f1 + d_iou * d_iou + d_lat * d_lat);

            if (dist < min_dist) {
                min_dist      = dist;
                best_balanced = i;
            }
        }

        if (best_balanced.has_value()) {
            selected.push_back(pareto_front[*best_balanced]);
        }
    }

    return selected;
}

// === Visualization Functions ===

void plot_pareto_front(
    const std::vector<ParameterCandidate>& all_candidates,
    const std::vector<ParameterCandidate>& pareto_front, const std::string& output_path) {
    constexpr int width       = 1200;
    constexpr int height      = 400;
    constexpr int margin      = 50;
    constexpr int panel_width = 350;

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    auto draw_scatter = [&](int panel_idx, auto get_x, auto get_y, const std::string& x_label,
                            const std::string& y_label, bool invert_y = false) {
        const int x_offset = panel_idx * 400 + margin;
        const int y_offset = margin;

        // Find data range
        double x_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        double y_min = std::numeric_limits<double>::max();
        double y_max = std::numeric_limits<double>::lowest();

        for (const auto& c : all_candidates) {
            x_min = std::min(x_min, get_x(c));
            x_max = std::max(x_max, get_x(c));
            y_min = std::min(y_min, get_y(c));
            y_max = std::max(y_max, get_y(c));
        }

        // Add padding
        const double x_pad = (x_max - x_min) * 0.1;
        const double y_pad = (y_max - y_min) * 0.1;
        x_min -= x_pad;
        x_max += x_pad;
        y_min -= y_pad;
        y_max += y_pad;

        auto scale_x = [&](double x) {
            return x_offset + static_cast<int>((x - x_min) / (x_max - x_min) * panel_width);
        };

        auto scale_y = [&](double y) {
            const double normalized = (y - y_min) / (y_max - y_min);
            const double flipped    = invert_y ? normalized : (1.0 - normalized);
            return y_offset + static_cast<int>(flipped * (height - 2 * margin));
        };

        // Draw axes
        cv::line(
            canvas, cv::Point(x_offset, height - margin),
            cv::Point(x_offset + panel_width, height - margin), cv::Scalar(0, 0, 0), 1);
        cv::line(
            canvas, cv::Point(x_offset, y_offset), cv::Point(x_offset, height - margin),
            cv::Scalar(0, 0, 0), 1);

        // Draw all points (gray)
        for (const auto& c : all_candidates) {
            cv::circle(
                canvas, cv::Point(scale_x(get_x(c)), scale_y(get_y(c))), 2,
                cv::Scalar(200, 200, 200), -1);
        }

        // Draw Pareto front (red, larger)
        for (const auto& c : pareto_front) {
            cv::circle(
                canvas, cv::Point(scale_x(get_x(c)), scale_y(get_y(c))), 5, cv::Scalar(0, 0, 255),
                -1);
        }

        // Labels
        cv::putText(
            canvas, x_label, cv::Point(x_offset + panel_width / 2 - 30, height - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        cv::putText(
            canvas, y_label, cv::Point(x_offset - 40, y_offset + (height - 2 * margin) / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    };

    // Panel 1: F1 vs Latency
    draw_scatter(
        0, [](const auto& c) { return c.objectives.f1_score; },
        [](const auto& c) { return c.objectives.latency_ms; }, "F1 Score", "Latency", true);

    // Panel 2: IoU vs Latency
    draw_scatter(
        1, [](const auto& c) { return c.objectives.mean_iou; },
        [](const auto& c) { return c.objectives.latency_ms; }, "Mean IoU", "Latency", true);

    // Panel 3: F1 vs IoU
    draw_scatter(
        2, [](const auto& c) { return c.objectives.f1_score; },
        [](const auto& c) { return c.objectives.mean_iou; }, "F1 Score", "Mean IoU", false);

    cv::imwrite(output_path, canvas);
    spdlog::info("Saved Pareto front visualization to {}", output_path);
}

void plot_convergence(const std::vector<double>& iteration_scores, const std::string& output_path) {
    if (iteration_scores.empty()) {
        return;
    }

    constexpr int width  = 600;
    constexpr int height = 400;
    constexpr int margin = 50;

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Find range
    double y_min = *std::min_element(iteration_scores.begin(), iteration_scores.end());
    double y_max = *std::max_element(iteration_scores.begin(), iteration_scores.end());

    const double y_pad = (y_max - y_min) * 0.1;
    y_min -= y_pad;
    y_max += y_pad;

    const auto n = static_cast<int>(iteration_scores.size());

    auto scale_x = [&](int i) { return margin + i * (width - 2 * margin) / std::max(1, n - 1); };

    auto scale_y = [&](double y) {
        return height - margin
             - static_cast<int>((y - y_min) / (y_max - y_min) * (height - 2 * margin));
    };

    // Draw axes
    cv::line(
        canvas, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin),
        cv::Scalar(0, 0, 0), 1);
    cv::line(
        canvas, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(0, 0, 0),
        1);

    // Draw line
    for (int i = 1; i < n; ++i) {
        cv::line(
            canvas, cv::Point(scale_x(i - 1), scale_y(iteration_scores[i - 1])),
            cv::Point(scale_x(i), scale_y(iteration_scores[i])), cv::Scalar(255, 0, 0), 2);
    }

    // Draw points
    for (int i = 0; i < n; ++i) {
        cv::circle(
            canvas, cv::Point(scale_x(i), scale_y(iteration_scores[i])), 4, cv::Scalar(0, 0, 255),
            -1);
    }

    // Labels
    cv::putText(
        canvas, "Iteration", cv::Point(width / 2 - 30, height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0, 0, 0), 1);
    cv::putText(
        canvas, "Best F1", cv::Point(5, height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0, 0, 0), 1);
    cv::putText(
        canvas, "Convergence", cv::Point(width / 2 - 40, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(0, 0, 0), 1);

    cv::imwrite(output_path, canvas);
    spdlog::info("Saved convergence plot to {}", output_path);
}

// Helper function for detector_benchmark.cpp compatibility
AccuracyMetrics evaluate_frame(
    const talos::ArmorSlots& detections, const std::vector<GroundTruth>& ground_truths,
    double iou_threshold) {
    AccuracyMetrics metrics;
    metrics.total_gt = static_cast<int>(ground_truths.size());

    // Collect valid detections
    std::vector<std::pair<int, cv::Rect2f>> det_boxes;
    for (int i = 0; i < 10; i++) {
        if (detections.is_valid(i)) {
            const auto& armor = detections.armors[i];
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
    std::vector<std::tuple<double, size_t, size_t>> iou_pairs; // (iou, det_idx, gt_idx)
    for (size_t d = 0; d < det_boxes.size(); d++) {
        for (size_t g = 0; g < ground_truths.size(); g++) {
            if (double iou = calculate_iou(det_boxes[d].second, ground_truths[g].bounding_box());
                iou >= iou_threshold) {
                iou_pairs.emplace_back(iou, d, g);
            }
        }
    }

    // Sort by IoU descending
    std::sort(iou_pairs.begin(), iou_pairs.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) > std::get<0>(b);
    });

    // Greedy assignment
    for (const auto& [iou, d_idx, g_idx] : iou_pairs) {
        if (det_matched[d_idx] || gt_matched[g_idx])
            continue;

        det_matched[d_idx] = true;
        gt_matched[g_idx]  = true;

        metrics.true_positives++;
        metrics.iou_sum += iou;
        metrics.iou_count++;

        // Check color accuracy
        const auto& det = detections.armors[det_boxes[d_idx].first];
        const auto& gt  = ground_truths[g_idx];

        metrics.color_total++;
        if (const uint8_t expected_det_color = gt.color == 0 ? 1 : 0;
            det.color == expected_det_color) {
            metrics.color_correct++;
        }

        // Check type accuracy
        metrics.type_total++;
        if (const uint8_t expected_det_type = gt.type == 0 ? 1 : 0; det.type == expected_det_type) {
            metrics.type_correct++;
        }
    }

    // Count false positives and false negatives
    for (const bool matched : det_matched) {
        if (!matched)
            metrics.false_positives++;
    }
    for (const bool matched : gt_matched) {
        if (!matched)
            metrics.false_negatives++;
    }

    return metrics;
}

} // namespace auto_aim
