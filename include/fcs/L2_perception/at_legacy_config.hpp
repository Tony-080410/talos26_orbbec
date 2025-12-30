#pragma once

#include "../core/types.hpp"
#include "at_legacy_detector_backend.hpp"
#include "at_legacy_nn_backend.hpp"
#include "at_legacy_traditional_backend.hpp"

#include <expected>
#include <string>

#include "toml_helper.hpp"
#include <toml++/toml.hpp>

namespace fcs {

// ============================================================================
// AT Legacy Backend Type
// ============================================================================

enum class AtLegacyBackendTypeConfig : int {
    Traditional   = 0,
    NeuralNetwork = 1,
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

    /// Convert to backend parameters
    [[nodiscard]] AtLegacyTraditionalBackend::LightParams light_params() const noexcept {
        return {
            .min_ratio         = light_min_ratio,
            .max_ratio         = light_max_ratio,
            .max_angle         = light_max_angle,
            .color_diff_thresh = light_color_diff_thresh,
        };
    }

    [[nodiscard]] AtLegacyTraditionalBackend::ArmorParams armor_params() const noexcept {
        return {
            .min_light_ratio           = armor_min_light_ratio,
            .min_small_center_distance = armor_min_small_center_distance,
            .max_small_center_distance = armor_max_small_center_distance,
            .min_large_center_distance = armor_min_large_center_distance,
            .max_large_center_distance = armor_max_large_center_distance,
            .max_angle                 = armor_max_angle,
        };
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

    /// Convert to backend config
    [[nodiscard]] AtLegacyNNBackendConfig to_backend_config() const noexcept {
        return {
            .model_path           = model_path,
            .device               = device,
            .device_priorities    = device_priorities,
            .enable_profiling     = enable_profiling,
            .enable_multi_thread  = enable_multi_thread,
            .input_width          = input_width,
            .input_height         = input_height,
            .num_classes          = num_classes,
            .num_colors           = num_colors,
            .confidence_threshold = static_cast<float>(confidence_threshold),
            .nms_threshold        = static_cast<float>(nms_threshold),
            .top_k                = top_k,
        };
    }
};

// ============================================================================
// AT Legacy Detector Config (combines Traditional + NN)
// ============================================================================

struct AtLegacyDetectorConfig {
    AtLegacyBackendTypeConfig backend_type = AtLegacyBackendTypeConfig::Traditional;
    int detect_color; // 0=BLUE, 1=RED

    AtLegacyTraditionalConfig traditional{};
    AtLegacyNNConfig nn{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["at_legacy_detector"].as_table()) {
            auto t = *_t;

            // Read backend type
            if (auto val = t["backend_type"].value<std::string>()) {
                if (*val == "traditional" || *val == "Traditional") {
                    backend_type = AtLegacyBackendTypeConfig::Traditional;
                } else if (*val == "nn" || *val == "neural_network" || *val == "NeuralNetwork") {
                    backend_type = AtLegacyBackendTypeConfig::NeuralNetwork;
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

    /// Create detector backend based on config
    [[nodiscard]] std::expected<AtLegacyDetectorBackend, DetectorError> create_backend() const {
        switch (backend_type) {
        case AtLegacyBackendTypeConfig::Traditional: {
            auto backend = std::make_unique<AtLegacyTraditionalBackend>(
                traditional.binary_threshold, traditional.light_params(),
                traditional.armor_params());
            backend->set_detect_color(detect_color == 0 ? ArmorColor::Blue : ArmorColor::Red);
            return AtLegacyDetectorBackend(std::move(backend));
        }
        case AtLegacyBackendTypeConfig::NeuralNetwork: {
            auto backend = std::make_unique<AtLegacyNNBackend>(nn.to_backend_config());
            if (auto result = backend->init(); !result) {
                return std::unexpected(result.error());
            }
            backend->set_detect_color(detect_color);
            return AtLegacyDetectorBackend(std::move(backend));
        }
        }
        return std::unexpected(DetectorError::NotInitialized);
    }
};

// ============================================================================
// AT Legacy Full Config
// ============================================================================

struct AtLegacyConfig {
    AtLegacyDetectorConfig detector{};
    AtLegacyTraditionalConfig traditional{};
    AtLegacyNNConfig nn{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_SUB(detector, tbl);

        // Read backend-specific config based on type
        if (detector.backend_type == AtLegacyBackendTypeConfig::Traditional) {
            READ_SUB(traditional, tbl);
        } else {
            READ_SUB(nn, tbl);
        }

        // Copy to detector
        detector.traditional = traditional;
        detector.nn          = nn;

        return {};
    }
};

} // namespace fcs
