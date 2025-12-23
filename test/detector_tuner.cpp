/**
 * @file detector_tuner.cpp
 * @brief Detector parameter tuning tool with multi-objective optimization
 *
 * Implements multi-resolution grid search with K-means clustering
 * and Pareto multi-objective optimization for detector parameters.
 *
 * Usage: ./detector_tuner <dataset_path> [options]
 * Options:
 *   --config <path>    Tuning config file (default: tuning.toml)
 *   --output <dir>     Output directory (default: tuning_results)
 *   --base <path>      Base detector config (default: sp_vision.toml)
 *   --quick            Quick mode: 1 iteration, 10% sample ratio
 */

#include <filesystem>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "auto_aim/config.hpp"
#include "auto_aim/dataset_loader.hpp"
#include "auto_aim/detector_tuner.hpp"
#include "spdlog_hook.hpp"

int main(int argc, char **argv) {
    hook_cstream();

    std::string dataset_path = "./datasets";
    std::string tuning_config_path = "tuning.toml";
    std::string output_dir = "tuning_results";
    std::string base_config_path = "sp_vision.toml";
    bool quick_mode = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            tuning_config_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--base" && i + 1 < argc) {
            base_config_path = argv[++i];
        } else if (arg == "--quick" || arg == "-q") {
            quick_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            fmt::print("Usage: {} <dataset_path> [options]\n", argv[0]);
            fmt::print("Options:\n");
            fmt::print("  --config <path>    Tuning config file (default: tuning.toml)\n");
            fmt::print("  --output <dir>     Output directory (default: tuning_results)\n");
            fmt::print("  --base <path>      Base detector config (default: sp_vision.toml)\n");
            fmt::print("  --quick, -q        Quick mode: 1 iteration, 10%% sample ratio\n");
            return 0;
        } else if (arg[0] != '-') {
            dataset_path = arg;
        }
    }

    spdlog::info("=== Detector Parameter Tuning ===");
    spdlog::info("Dataset: {}", dataset_path);
    spdlog::info("Tuning Config: {}", tuning_config_path);
    spdlog::info("Base Config: {}", base_config_path);
    spdlog::info("Output: {}", output_dir);
    if (quick_mode) {
        spdlog::info("Mode: QUICK (1 iteration, 10% samples)");
    }

    // Load dataset
    auto_aim::DatasetLoader loader(dataset_path);
    if (loader.empty()) {
        spdlog::critical("No valid frames found in dataset: {}", dataset_path);
        return 1;
    }

    spdlog::info("Loaded {} frames", loader.size());

    // Load base config
    SpVisionConfig base_config;
    if (!std::filesystem::exists(base_config_path)) {
        spdlog::warn("Base config {} not found, using defaults", base_config_path);
    } else {
        try {
            auto tbl = toml::parse_file(base_config_path);
            if (auto result = base_config.read_from(tbl); !result) {
                spdlog::critical("Failed to read base config: {}", result.error());
                return 1;
            }
        } catch (const toml::parse_error &e) {
            spdlog::critical("Failed to parse {}: {}", base_config_path, e.what());
            return 1;
        }
    }

    // Load tuning config
    auto_aim::TuningConfig tuning_config;
    if (!std::filesystem::exists(tuning_config_path)) {
        spdlog::warn("Tuning config {} not found, using defaults", tuning_config_path);
        tuning_config = auto_aim::TuningConfig::default_config();
    } else {
        try {
            auto tbl = toml::parse_file(tuning_config_path);
            if (auto result = tuning_config.read_from(tbl); !result) {
                spdlog::critical("Failed to read tuning config: {}", result.error());
                return 1;
            }
        } catch (const toml::parse_error &e) {
            spdlog::critical("Failed to parse {}: {}", tuning_config_path, e.what());
            return 1;
        }
    }

    // Apply quick mode overrides
    if (quick_mode) {
        tuning_config.quick_mode = true;
        tuning_config.max_iterations = 1;
        tuning_config.sample_ratio = 0.1;
        // Reduce initial points for faster search
        tuning_config.threshold.initial_points = 3;
        tuning_config.max_angle_error.initial_points = 2;
        tuning_config.min_lightbar_ratio.initial_points = 2;
        tuning_config.max_lightbar_ratio.initial_points = 2;
        tuning_config.min_lightbar_length.initial_points = 2;
        tuning_config.min_armor_ratio.initial_points = 2;
        tuning_config.max_armor_ratio.initial_points = 2;
        tuning_config.max_side_ratio.initial_points = 2;
        tuning_config.min_confidence.initial_points = 3;
        tuning_config.max_rectangular_error.initial_points = 2;
    }

    // Run tuning
    spdlog::info("Starting parameter tuning...");
    auto_aim::DetectorTuner tuner(tuning_config, loader, base_config);
    auto result = tuner.run();

    if (!result) {
        spdlog::critical("Tuning failed: {}", result.error());
        return 1;
    }

    const auto &tuning_result = *result;

    // Print summary
    spdlog::info("\n========================================");
    spdlog::info("          TUNING COMPLETE");
    spdlog::info("========================================");
    spdlog::info("Total evaluations: {}", tuning_result.total_evaluations);
    spdlog::info("Pareto-optimal solutions: {}", tuning_result.pareto_front.size());

    // Print Pareto front details
    for (size_t i = 0; i < tuning_result.pareto_front.size(); ++i) {
        const auto &c = tuning_result.pareto_front[i];
        spdlog::info("");
        spdlog::info("=== Solution {} ===", i + 1);
        spdlog::info("  F1 Score:  {:.4f}", c.objectives.f1_score);
        spdlog::info("  Precision: {:.4f}", c.precision);
        spdlog::info("  Recall:    {:.4f}", c.recall);
        spdlog::info("  Mean IoU:  {:.4f}", c.objectives.mean_iou);
        spdlog::info("  Latency:   {:.2f} ms", c.objectives.latency_ms);
        spdlog::info("  Parameters:");
        spdlog::info("    threshold:            {:.1f}", c.config.threshold);
        spdlog::info("    max_angle_error:      {:.1f}", c.config.max_angle_error);
        spdlog::info("    min_lightbar_ratio:   {:.2f}", c.config.min_lightbar_ratio);
        spdlog::info("    max_lightbar_ratio:   {:.1f}", c.config.max_lightbar_ratio);
        spdlog::info("    min_lightbar_length:  {:.1f}", c.config.min_lightbar_length);
        spdlog::info("    min_armor_ratio:      {:.2f}", c.config.min_armor_ratio);
        spdlog::info("    max_armor_ratio:      {:.1f}", c.config.max_armor_ratio);
        spdlog::info("    max_side_ratio:       {:.2f}", c.config.max_side_ratio);
        spdlog::info("    min_confidence:       {:.2f}", c.config.min_confidence);
        spdlog::info("    max_rectangular_error:{:.1f}", c.config.max_rectangular_error);
    }

    // Create output directory and save results
    std::filesystem::create_directories(output_dir);

    const std::string toml_path = output_dir + "/optimized_params.toml";
    tuning_result.save_to_toml(toml_path);
    spdlog::info("\nSaved optimized parameters to: {}", toml_path);

    // Generate visualizations
    tuning_result.visualize(output_dir);

    spdlog::info("\nAll results saved to: {}/", output_dir);

    return 0;
}
