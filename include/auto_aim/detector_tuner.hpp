#pragma once

#include <array>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <toml++/toml.hpp>

#include "config.hpp"
#include "dataset_loader.hpp"
#include "detector.hpp"

namespace auto_aim {

/**
 * @brief Parameter search range for a single parameter
 */
struct ParameterRange {
    double min{0.0};
    double max{1.0};
    size_t initial_points{3};

    /**
     * @brief Generate uniformly spaced sample points
     * @param resolution Number of points to generate
     * @return Vector of sample values
     */
    [[nodiscard]] std::vector<double> sample_grid(size_t resolution) const {
        std::vector<double> samples;
        samples.reserve(resolution);

        if (resolution == 1) {
            samples.push_back((min + max) / 2.0);
        } else {
            const double step = (max - min) / static_cast<double>(resolution - 1);
            for (size_t i = 0; i < resolution; ++i) {
                samples.push_back(min + step * static_cast<double>(i));
            }
        }
        return samples;
    }
};

/**
 * @brief Tuning search space configuration
 */
struct TuningConfig {
    // Parameter search ranges
    ParameterRange threshold;
    ParameterRange max_angle_error;
    ParameterRange min_lightbar_ratio;
    ParameterRange max_lightbar_ratio;
    ParameterRange min_lightbar_length;
    ParameterRange min_armor_ratio;
    ParameterRange max_armor_ratio;
    ParameterRange max_side_ratio;
    ParameterRange min_confidence;
    ParameterRange max_rectangular_error;

    // Algorithm parameters
    size_t max_iterations{5};
    double convergence_threshold{0.01}; // Stop if improvement < 1%
    size_t num_clusters{4};             // K-means clusters
    double top_percentile{0.2};         // Top 20% for clustering
    size_t max_pareto_solutions{3};     // Maximum Pareto solutions to return

    // Quick mode for local testing
    bool quick_mode{false};   // Enable quick mode
    double sample_ratio{1.0}; // Fraction of dataset to use (0.0-1.0)

    /**
     * @brief Read configuration from TOML table
     */
    [[nodiscard]] std::expected<void, std::string> read_from(const toml::table& tbl);

    /**
     * @brief Get default tuning configuration
     */
    [[nodiscard]] static TuningConfig default_config();
};

/**
 * @brief Multi-objective optimization targets
 * Maximize F1 and IoU, minimize latency
 */
struct OptimizationObjectives {
    double f1_score{0.0};   // Higher is better (0-1)
    double mean_iou{0.0};   // Higher is better (0-1)
    double latency_ms{0.0}; // Lower is better

    /**
     * @brief Check if this solution Pareto-dominates another
     * @param other The solution to compare against
     * @return true if this dominates other (better or equal in all, strictly better in at least
     * one)
     */
    [[nodiscard]] bool dominates(const OptimizationObjectives& other) const {
        // For F1 and IoU: higher is better
        // For latency: lower is better
        const bool ge_f1      = f1_score >= other.f1_score;
        const bool ge_iou     = mean_iou >= other.mean_iou;
        const bool le_latency = latency_ms <= other.latency_ms;

        const bool strictly_better = (f1_score > other.f1_score) || (mean_iou > other.mean_iou)
                                  || (latency_ms < other.latency_ms);

        return (ge_f1 && ge_iou && le_latency) && strictly_better;
    }

    /**
     * @brief Compute Euclidean distance in normalized objective space
     */
    [[nodiscard]] double distance_to(const OptimizationObjectives& other) const {
        // Normalize: F1 and IoU in [0,1], latency normalized to [0,1] assuming max 10ms
        const double d_f1  = f1_score - other.f1_score;
        const double d_iou = mean_iou - other.mean_iou;
        const double d_lat = (latency_ms - other.latency_ms) / 10.0; // Normalize by 10ms
        return std::sqrt(d_f1 * d_f1 + d_iou * d_iou + d_lat * d_lat);
    }
};

/**
 * @brief A parameter configuration candidate with its evaluation results
 */
struct ParameterCandidate {
    DetectorConfig config;
    OptimizationObjectives objectives;

    // Additional metrics for analysis
    double precision{0.0};
    double recall{0.0};

    /**
     * @brief Convert config to normalized parameter vector for clustering
     * @return 10-element array of normalized parameters [0,1]
     */
    [[nodiscard]] std::array<double, 10> normalized_params(const TuningConfig& tc) const {
        auto norm = [](double val, const ParameterRange& r) {
            // 防止除以 0，且保证结果在 0-1 之间
            double range = r.max - r.min;
            return (range > 1e-6) ? (val - r.min) / range : 0.5;
        };

        return {
            norm(config.threshold, tc.threshold),
            norm(config.max_angle_error, tc.max_angle_error),
            norm(config.min_lightbar_ratio, tc.min_lightbar_ratio),
            norm(config.max_lightbar_ratio, tc.max_lightbar_ratio),
            norm(config.min_lightbar_length, tc.min_lightbar_length),
            norm(config.min_armor_ratio, tc.min_armor_ratio),
            norm(config.max_armor_ratio, tc.max_armor_ratio),
            norm(config.max_side_ratio, tc.max_side_ratio),
            norm(config.min_confidence, tc.min_confidence),
            norm(config.max_rectangular_error, tc.max_rectangular_error)};
    }
};

/**
 * @brief Search region for iterative refinement
 */
struct SearchRegion {
    DetectorConfig center;
    std::array<double, 10> radii; // Search radius per parameter

    /**
     * @brief Generate grid of configurations within this region
     * @param resolution Points per dimension
     * @return Vector of configurations
     */
    [[nodiscard]] std::vector<DetectorConfig> generate_grid(size_t resolution) const;
};

/**
 * @brief Tuning result with Pareto-optimal configurations
 */
struct TuningResult {
    std::vector<ParameterCandidate> pareto_front;
    std::vector<ParameterCandidate> all_evaluated;
    size_t total_evaluations{0};
    std::vector<double> iteration_best_f1; // Convergence history

    /**
     * @brief Save best configurations to TOML file
     */
    void save_to_toml(const std::string& path) const;

    /**
     * @brief Generate visualization images
     */
    void visualize(const std::string& output_dir) const;
};

/**
 * @brief Main detector parameter tuning engine
 *
 * Implements multi-resolution grid search with K-means clustering
 * and Pareto multi-objective optimization.
 */
class DetectorTuner {
public:
    /**
     * @brief Construct tuner with configuration
     * @param tuning_config Search space and algorithm configuration
     * @param dataset Dataset for evaluation
     * @param base_config Base SpVisionConfig (non-detector params)
     */
    DetectorTuner(
        TuningConfig tuning_config, const DatasetLoader& dataset, SpVisionConfig base_config);

    /**
     * @brief Run the full tuning pipeline
     * @return Tuning result or error message
     */
    [[nodiscard]] std::expected<TuningResult, std::string> run();

private:
    TuningConfig tuning_config_;
    const DatasetLoader& dataset_;
    SpVisionConfig base_config_;

    // === Core Algorithm Steps ===

    /**
     * @brief Generate initial coarse grid covering entire search space
     */
    [[nodiscard]] std::vector<DetectorConfig> generate_initial_grid() const;

    /**
     * @brief Evaluate all configs in parallel using TBB
     */
    [[nodiscard]] std::vector<ParameterCandidate>
        evaluate_parallel(const std::vector<DetectorConfig>& configs) const;

    /**
     * @brief Evaluate single config on entire dataset
     */
    [[nodiscard]] ParameterCandidate evaluate_single(const DetectorConfig& config) const;

    /**
     * @brief Select top candidates and cluster into search regions
     */
    [[nodiscard]] std::vector<SearchRegion>
        cluster_top_candidates(const std::vector<ParameterCandidate>& candidates) const;

    /**
     * @brief Extract Pareto-optimal solutions
     */
    [[nodiscard]] std::vector<ParameterCandidate>
        extract_pareto_front(const std::vector<ParameterCandidate>& candidates) const;

    /**
     * @brief K-means clustering in normalized parameter space
     */
    [[nodiscard]] std::vector<std::vector<ParameterCandidate>>
        kmeans_cluster(const std::vector<ParameterCandidate>& candidates, size_t k) const;

    /**
     * @brief Select diverse subset from Pareto front
     */
    [[nodiscard]] std::vector<ParameterCandidate> select_diverse_pareto(
        const std::vector<ParameterCandidate>& pareto_front, size_t max_count) const;
};

// === Visualization Utilities ===

/**
 * @brief Plot Pareto front projections (F1 vs IoU, F1 vs Latency, IoU vs Latency)
 */
void plot_pareto_front(
    const std::vector<ParameterCandidate>& all_candidates,
    const std::vector<ParameterCandidate>& pareto_front, const std::string& output_path);

/**
 * @brief Plot convergence history
 */
void plot_convergence(const std::vector<double>& iteration_scores, const std::string& output_path);

} // namespace auto_aim
