#pragma once

#include "core/armor_types.hpp"

#include <array>
#include <expected>
#include <memory>
#include <string>
// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L2 {

enum class AtLegacyBackendTypeConfig : int {
    Traditional      = 0,
    NeuralNetwork    = 1, // AT model (YOLO11-pose)
    NeuralNetworkTup = 2, // TUP model (YOLOX multi-scale)
#if TALOS_HAS_TENSORRT
    TensorRT = 3,         // TensorRT backend (GPU-accelerated)
#endif
#if TALOS_HAS_AXERA
    Axera = 4,            // Axera AX650 NPU backend
#endif
};

// ============================================================================
// AT Legacy Traditional Backend Config
// ============================================================================

struct AtLegacyTraditionalConfig {
    bool advanced_binary = false;
    float dark_percentage{0.5};
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

    // Classifier parameters
    std::string classifier_model_path;
    double classifier_confidence_threshold = 0.5;
    bool classifier_enable_type_filtering  = true;
    bool classifier_use_softmax            = true; // Legacy networks output logits, need softmax

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
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

    double confidence_threshold;
    double nms_threshold;
    int top_k;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// Type alias for backward compatibility (NNTup uses same config as NN)
// TODO: Deprecate this alias
using AtLegacyNNTupConfig = AtLegacyNNConfig;

#if TALOS_HAS_TENSORRT
// ============================================================================
// AT Legacy TRT Backend Config (TensorRT)
// ============================================================================

struct AtLegacyTrtConfig {
    std::string engine_path;
    std::string model_path;
    std::string engine_cache_dir;

    // Device and concurrency
    int device_id                  = 0;
    int num_streams                = 2;
    std::string compute_capability = "86";

    // TensorRT options
    bool enable_fp16 = true;
    bool enable_dla  = false;
    int dla_core     = 0;

    // Detection parameters
    double confidence_threshold = 0.75;
    double nms_threshold        = 0.30;
    int top_k                   = 128;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};
#endif

#if TALOS_HAS_AXERA
// ============================================================================
// AT Legacy Axera Backend Config (AX650 NPU)
// ============================================================================

struct AtLegacyAxeraConfig {
    std::string model_path;

    // Detection parameters
    double confidence_threshold = 0.25;
    int top_k                   = 128;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};
#endif

// ============================================================================
// AT Legacy Detector Config (combines Traditional + NN)
// ============================================================================

struct AtLegacyDetectorConfig {
    AtLegacyBackendTypeConfig backend_type = AtLegacyBackendTypeConfig::Traditional;
    ArmorColor detect_color; // 0=BLUE, 1=RED

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// ============================================================================
// AT Legacy Full Config
// ============================================================================

struct AtLegacyConfig {
    AtLegacyDetectorConfig detector{};
    AtLegacyTraditionalConfig traditional{};
    AtLegacyNNConfig nn{};
    AtLegacyNNTupConfig nn_tup{};
#if TALOS_HAS_TENSORRT
    AtLegacyTrtConfig trt{};
#endif
#if TALOS_HAS_AXERA
    AtLegacyAxeraConfig axera{};
#endif

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

#if TALOS_HAS_TENSORRT
    [[nodiscard]] std::shared_ptr<const AtLegacyTrtConfig> trt_ptr() const {
        return std::make_shared<const AtLegacyTrtConfig>(trt);
    }
#endif

#if TALOS_HAS_AXERA
    [[nodiscard]] std::shared_ptr<const AtLegacyAxeraConfig> axera_ptr() const {
        return std::make_shared<const AtLegacyAxeraConfig>(axera);
    }
#endif

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L2
