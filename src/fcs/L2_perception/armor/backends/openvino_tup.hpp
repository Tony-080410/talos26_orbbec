#pragma once

#include "../backend.hpp"
#include "../config.hpp"
#include "core/armor_types.hpp"
#include "core/types.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

// Forward declare OpenVINO types to avoid header pollution
namespace ov {
class Core;
class CompiledModel;
class InferRequest;
} // namespace ov

namespace fcs::L2 {

// ============================================================================
// Grid and Stride for YOLOX multi-scale detection
// ============================================================================

struct GridAndStride {
    int grid0{};
    int grid1{};
    int stride{};
};

// ============================================================================
// AT Legacy NN TUP Backend Implementation
// ============================================================================

/// Neural network detection backend using OpenVINO (TUP model)
class NNTupBackend : public DetectorBackendBase<NNTupBackend> {
public:
    using DetectionResult = std::expected<std::vector<ArmorDetection>, std::string>;
    using Config          = ArmorOpenVinoTupConfig;

    static constexpr int INPUT_W     = 416;
    static constexpr int INPUT_H     = 416;
    static constexpr int NUM_COLORS  = 4;
    static constexpr int NUM_CLASSES = 8;

    /// Factory: construct a fully-initialized backend (load model via OpenVINO).
    /// Construction IS initialization — no separate init() needed.
    [[nodiscard]] static std::expected<NNTupBackend, std::string> create(Config config) noexcept;

    ~NNTupBackend();

    // Move-only (OpenVINO resources)
    NNTupBackend(NNTupBackend&&) noexcept;
    NNTupBackend& operator=(NNTupBackend&&) noexcept;
    NNTupBackend(const NNTupBackend&)            = delete;
    NNTupBackend& operator=(const NNTupBackend&) = delete;

    /// Detect armors in image (synchronous)
    [[nodiscard]] DetectionResult detect_impl(const cv::Mat& image, ArmorColor color) noexcept;

    /// Get execution devices
    [[nodiscard]] std::vector<std::string> execution_devices() const noexcept;

    /// Get current config
    [[nodiscard]] const Config& get_config() const noexcept { return config_; }

private:
    /// Private constructor — use create() factory
    explicit NNTupBackend(Config config) noexcept;

    /// Preprocess context
    struct PreprocContext {
        cv::Mat preprocessed;
        Eigen::Matrix<float, 3, 3> T; // Transform matrix: network -> original
        std::vector<GridAndStride> grids;
        int rows        = 0;
        int cols        = 0;
        int num_classes = NUM_CLASSES;
    };

    /// Generate grids and strides for multi-scale detection
    static void generate_grids_and_strides(
        int width, int height, const std::array<int, 3>& strides,
        std::vector<GridAndStride>& grids) noexcept;

    /// Preprocess image (letterbox without normalization)
    [[nodiscard]] static PreprocContext
        preprocess(const cv::Mat& image, const ArmorOpenVinoTupConfig& cfg) noexcept;

    /// Run inference
    [[nodiscard]] bool infer(const PreprocContext& ctx) noexcept;

    /// Postprocess inference output
    [[nodiscard]] std::vector<ArmorDetection>
        postprocess(const PreprocContext& ctx, const ArmorOpenVinoTupConfig& cfg) const noexcept;

    /// Non-maximum suppression
    [[nodiscard]] static std::vector<ArmorDetection>
        nms(std::vector<ArmorDetection>& detections, const ArmorOpenVinoTupConfig& cfg) noexcept;

private:
    Config config_;

    // OpenVINO resources (PIMPL to avoid header pollution)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fcs::L2
