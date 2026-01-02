#pragma once

#include "../core/types.hpp"

#include <array>
#include <expected>
#include <memory>
#include <string>

#include "toml_helper.hpp"
#include <toml++/toml.hpp>

namespace fcs {

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

    // Classifier parameters (optional)
    bool classifier_enabled = false;
    std::string classifier_model_path;
    double classifier_confidence_threshold = 0.5;
    bool classifier_enable_type_filtering  = true;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, binary_threshold);
        READ(tbl, light_min_ratio);
        READ(tbl, light_max_ratio);
        READ(tbl, light_max_angle);
        READ(tbl, light_color_diff_thresh);
        READ(tbl, armor_min_light_ratio);
        READ(tbl, armor_min_small_center_distance);
        READ(tbl, armor_max_small_center_distance);
        READ(tbl, armor_min_large_center_distance);
        READ(tbl, armor_max_large_center_distance);
        READ(tbl, armor_max_angle);

        // Optional classifier configuration
        READ_OPT(tbl, classifier_enabled);
        READ_OPT(tbl, classifier_model_path);
        READ_OPT(tbl, classifier_confidence_threshold);
        READ_OPT(tbl, classifier_enable_type_filtering);

        return {};
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
        READ(tbl, model_path);
        READ(tbl, device);
        READ(tbl, device_priorities);
        READ(tbl, enable_profiling);
        READ(tbl, enable_multi_thread);
        READ(tbl, input_width);
        READ(tbl, input_height);
        READ(tbl, num_classes);
        READ(tbl, num_colors);
        READ(tbl, confidence_threshold);
        READ(tbl, nms_threshold);
        READ(tbl, top_k);
        return {};
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

        READ(tbl, model_path);
        READ(tbl, device);
        READ(tbl, device_priorities);
        READ(tbl, enable_profiling);
        READ(tbl, enable_multi_thread);
        READ(tbl, input_width);
        READ(tbl, input_height);
        READ(tbl, num_classes);
        READ(tbl, num_colors);
        READ(tbl, confidence_threshold);
        READ(tbl, nms_threshold);
        READ(tbl, top_k);

        // Read strides array
        if (auto arr = tbl["strides"].as_array()) {
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
};

// ============================================================================
// AT Legacy Detector Config (combines Traditional + NN)
// ============================================================================

struct AtLegacyDetectorConfig {
    AtLegacyBackendTypeConfig backend_type = AtLegacyBackendTypeConfig::Traditional;
    ArmorColor detect_color; // 0=BLUE, 1=RED

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, backend_type);
        READ(tbl, detect_color);
        return {};
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
        READ(tbl, detector);
        // Read backend-specific config based on type
        switch (detector.backend_type) {
        case AtLegacyBackendTypeConfig::Traditional: READ(tbl, traditional); break;
        case AtLegacyBackendTypeConfig::NeuralNetwork: READ(tbl, nn); break;
        case AtLegacyBackendTypeConfig::NeuralNetworkTup: READ(tbl, nn_tup); break;
        }
        return {};
    }
};

} // namespace fcs
