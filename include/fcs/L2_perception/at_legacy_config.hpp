#pragma once

#include "../core/types.hpp"

#include <array>
#include <expected>
#include <memory>
#include <string>

#include "toml_helper.hpp"
#include <toml++/toml.hpp>

namespace fcs {

// ============================================================================
// AT Legacy Backend Type
// ============================================================================

enum class AtLegacyBackendTypeConfig : int {
    Traditional      = 0,
    NeuralNetwork    = 1, // AT model (YOLO11-pose)
    NeuralNetworkTup = 2, // TUP model (YOLOX multi-scale)
};

// ============================================================================
// AT Legacy Traditional Backend Config
// ============================================================================

struct AtLegacyTraditionalConfig {
    int binary_threshold;

    // Light detection parameters
    double light_min_ratio;
    double light_max_ratio;
    double light_max_angle;
    int light_color_diff_thresh;

    // Armor matching parameters
    double armor_min_light_ratio;
    double armor_min_small_center_distance;
    double armor_max_small_center_distance;
    double armor_min_large_center_distance;
    double armor_max_large_center_distance;
    double armor_max_angle;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["at_legacy_traditional"].as_table()) {
            auto t = *_t;

            READ_KEY(t, binary_threshold);
            READ_KEY(t, light_min_ratio);
            READ_KEY(t, light_max_ratio);
            READ_KEY(t, light_max_angle);
            READ_KEY(t, light_color_diff_thresh);
            READ_KEY(t, armor_min_light_ratio);
            READ_KEY(t, armor_min_small_center_distance);
            READ_KEY(t, armor_max_small_center_distance);
            READ_KEY(t, armor_min_large_center_distance);
            READ_KEY(t, armor_max_large_center_distance);
            READ_KEY(t, armor_max_angle);

            return {};
        }
        return std::unexpected("at_legacy_traditional section not found");
    }
};

// ============================================================================
// AT Legacy NN Backend Config
// ============================================================================

struct AtLegacyNNConfig {
    std::string model_path;
    std::string device;
    std::string device_priorities;
    bool enable_profiling;
    bool enable_multi_thread;

    int input_width;
    int input_height;
    int num_classes;
    int num_colors;

    double confidence_threshold;
    double nms_threshold;
    int top_k;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["at_legacy_nn"].as_table()) {
            auto t = *_t;

            READ_KEY(t, model_path);
            READ_KEY(t, device);
            READ_KEY(t, device_priorities);
            READ_KEY(t, enable_profiling);
            READ_KEY(t, enable_multi_thread);
            READ_KEY(t, input_width);
            READ_KEY(t, input_height);
            READ_KEY(t, num_classes);
            READ_KEY(t, num_colors);
            READ_KEY(t, confidence_threshold);
            READ_KEY(t, nms_threshold);
            READ_KEY(t, top_k);

            return {};
        }
        return std::unexpected("at_legacy_nn section not found");
    }
};

// ============================================================================
// AT Legacy NN TUP Backend Config (YOLOX multi-scale)
// ============================================================================

struct AtLegacyNNTupConfig {
    std::string model_path;
    std::string device;
    std::string device_priorities;
    bool enable_profiling;
    bool enable_multi_thread;

    int input_width;
    int input_height;
    int num_classes;
    int num_colors;

    double confidence_threshold;
    double nms_threshold;
    int top_k;

    // TUP-specific: multi-scale strides
    std::array<int, 3> strides;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["at_legacy_nn_tup"].as_table()) {
            auto t = *_t;

            READ_KEY(t, model_path);
            READ_KEY(t, device);
            READ_KEY(t, device_priorities);
            READ_KEY(t, enable_profiling);
            READ_KEY(t, enable_multi_thread);
            READ_KEY(t, input_width);
            READ_KEY(t, input_height);
            READ_KEY(t, num_classes);
            READ_KEY(t, num_colors);
            READ_KEY(t, confidence_threshold);
            READ_KEY(t, nms_threshold);
            READ_KEY(t, top_k);

            // Read strides array
            if (auto arr = t["strides"].as_array()) {
                if (arr->size() >= 3) {
                    for (size_t i = 0; i < 3; ++i) {
                        if (auto v = (*arr)[i].value<int64_t>()) {
                            strides[i] = static_cast<int>(*v);
                        }
                    }
                }
            } else {
                // Default strides
                strides = {8, 16, 32};
            }

            return {};
        }
        return std::unexpected("at_legacy_nn_tup section not found");
    }
};

// ============================================================================
// AT Legacy Detector Config (combines Traditional + NN)
// ============================================================================

struct AtLegacyDetectorConfig {
    AtLegacyBackendTypeConfig backend_type = AtLegacyBackendTypeConfig::Traditional;
    int detect_color; // 0=BLUE, 1=RED

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["at_legacy_detector"].as_table()) {
            auto t = *_t;

            // Read backend type
            if (auto val = t["backend_type"].value<std::string>()) {
                if (*val == "traditional" || *val == "Traditional") {
                    backend_type = AtLegacyBackendTypeConfig::Traditional;
                } else if (*val == "nn" || *val == "neural_network" || *val == "NeuralNetwork") {
                    backend_type = AtLegacyBackendTypeConfig::NeuralNetwork;
                } else if (
                    *val == "nn_tup" || *val == "neural_network_tup"
                    || *val == "NeuralNetworkTup") {
                    backend_type = AtLegacyBackendTypeConfig::NeuralNetworkTup;
                } else {
                    return std::unexpected("invalid backend_type: " + *val);
                }
            } else {
                return std::unexpected("backend_type not set or invalid");
            }

            READ_KEY(t, detect_color);

            return {};
        }
        return std::unexpected("at_legacy_detector section not found");
    }
};

// ============================================================================
// AT Legacy Full Config
// ============================================================================

struct AtLegacyConfig {
    AtLegacyDetectorConfig detector{};
    AtLegacyTraditionalConfig traditional{};
    AtLegacyNNConfig nn{};
    AtLegacyNNTupConfig nn_tup{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const AtLegacyTraditionalConfig> traditional_ptr() const {
        return std::make_shared<const AtLegacyTraditionalConfig>(traditional);
    }

    [[nodiscard]] std::shared_ptr<const AtLegacyNNConfig> nn_ptr() const {
        return std::make_shared<const AtLegacyNNConfig>(nn);
    }

    [[nodiscard]] std::shared_ptr<const AtLegacyNNTupConfig> nn_tup_ptr() const {
        return std::make_shared<const AtLegacyNNTupConfig>(nn_tup);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_SUB(detector, tbl);

        // Read backend-specific config based on type
        switch (detector.backend_type) {
        case AtLegacyBackendTypeConfig::Traditional: READ_SUB(traditional, tbl); break;
        case AtLegacyBackendTypeConfig::NeuralNetwork: READ_SUB(nn, tbl); break;
        case AtLegacyBackendTypeConfig::NeuralNetworkTup: READ_SUB(nn_tup, tbl); break;
        }

        return {};
    }
};

} // namespace fcs
