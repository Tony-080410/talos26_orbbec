#pragma once

#include "L2_perception/armor/readback_roi.hpp"
#include "core/armor_types.hpp"

#include <string>
#include <vector>

namespace fcs::L2 {

enum class ArmorBackendType : int {
    Traditional = 0,
#if TALOS_HAS_OPENVINO
    OpenVINO    = 1, // AT model (YOLO11-pose)
    OpenVINOTup = 2, // TUP model (YOLOX multi-scale)
#endif
#if TALOS_HAS_TENSORRT
    TensorRT = 3,    // TensorRT backend (GPU-accelerated)
#endif
#if TALOS_HAS_AXERA
    Axera = 4,       // Axera AX650 NPU backend
#endif
};

// ============================================================================
// AT Legacy Traditional Backend Config
// ============================================================================

struct ArmorTraditionalConfig {
    bool advanced_binary = false;
    float dark_percentage{0.5};
    int binary_threshold{};

    // Light detection parameters
    double light_min_ratio{};
    double light_max_ratio{};
    double light_max_angle{};
    int light_color_diff_thresh{};

    // Armor matching parameters
    double armor_min_light_ratio{};
    double armor_min_small_center_distance{};
    double armor_max_small_center_distance{};
    double armor_min_large_center_distance{};
    double armor_max_large_center_distance{};
    double armor_max_angle{};

    // Classifier parameters
    std::string classifier_model_path{};
    double classifier_confidence_threshold = 0.5;
    bool classifier_enable_type_filtering  = true;
    bool classifier_use_softmax            = true; // Legacy networks output logits, need softmax
};

// ============================================================================
// AT Legacy NN Backend Config
// ============================================================================

struct ArmorOpenVinoConfig {
    std::string model_path{};
    std::string device{};
    std::string device_priorities{};
    bool enable_profiling{};
    bool enable_multi_thread{};

    double confidence_threshold{};
    double nms_threshold{};
    int top_k{};
};

// ============================================================================
// AT Legacy NN TUP Backend Config (YOLOX multi-scale)
// ============================================================================

struct ArmorOpenVinoTupConfig {
    std::string model_path{};
    std::string device{};
    std::string device_priorities{};
    bool enable_profiling{};
    bool enable_multi_thread{};

    double confidence_threshold{};
    double nms_threshold{};
    int top_k{};
};

// ============================================================================
// AT Legacy TRT Backend Config (TensorRT)
// ============================================================================

struct ArmorTensorRtConfig {
    std::string engine_path{};
    std::string model_path{};
    std::string engine_cache_dir{};

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
};

// ============================================================================
// AT Legacy Axera Backend Config (AX650 NPU)
// ============================================================================

struct ArmorAxeraConfig {
    std::string model_path{};

    // Model input dimensions
    int input_width  = 768;
    int input_height = 576;

    // Model output dimensions
    int num_colors           = 4;
    int num_kpts             = 4;
    int num_pairs            = 12;
    std::vector<int> strides = {8, 16, 32};

    // Detection parameters
    double confidence_threshold = 0.25;
    double nms_threshold        = 0.70;
    int top_k                   = 128;
};

// ============================================================================
// AT Legacy Full Config
// ============================================================================

struct ArmorDetectorConfig {
    ArmorBackendType backend_type;
    ArmorTraditionalConfig traditional;
    ArmorOpenVinoConfig openvino;
    ArmorOpenVinoTupConfig openvino_tup;
    ArmorTensorRtConfig tensor_rt;
    ArmorAxeraConfig axera;
    ArmorReadbackRoiConfig readback_roi;
};

} // namespace fcs::L2
