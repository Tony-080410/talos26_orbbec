#pragma once

#include "at_legacy_nn_backend_base.hpp"

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

namespace fcs::L2 {

struct AtLegacyNNConfig;

/// OpenVINO backend template
/// Single implementation for all input sizes via template parameterization
template <int INPUT_W, int INPUT_H>
class AtLegacyOpenVINOBackend final : public AtLegacyNNBackendBase {
public:
    using ConfigPtr = std::shared_ptr<const AtLegacyNNConfig>;

    explicit AtLegacyOpenVINOBackend(ConfigPtr config);
    ~AtLegacyOpenVINOBackend() override;

    // Non-copyable, movable
    AtLegacyOpenVINOBackend(const AtLegacyOpenVINOBackend&)                = delete;
    AtLegacyOpenVINOBackend& operator=(const AtLegacyOpenVINOBackend&)     = delete;
    AtLegacyOpenVINOBackend(AtLegacyOpenVINOBackend&&) noexcept            = default;
    AtLegacyOpenVINOBackend& operator=(AtLegacyOpenVINOBackend&&) noexcept = default;

    [[nodiscard]] std::expected<void, DetectorError> init();
    [[nodiscard]] std::vector<std::string> execution_devices() const;

    [[nodiscard]] std::string_view name() const noexcept override { return "openvino"; }

    [[nodiscard]] ConfigPtr get_config() const noexcept { return config_; }

protected:
    struct PreprocessContext {
        cv::Mat preprocessed;
        float scale_x = 1.0f;
        float scale_y = 1.0f;
        int pad_x     = 0;
        int pad_y     = 0;
        int orig_w    = 0;
        int orig_h    = 0;
    };

    [[nodiscard]] void* preprocess_impl(const cv::Mat& image) override;
    [[nodiscard]] bool infer_impl(void* ctx) override;
    [[nodiscard]] std::vector<AtLegacyArmorDetection> postprocess_impl(void* ctx) override;
    [[nodiscard]] std::vector<AtLegacyArmorDetection>
        nms_impl(std::vector<AtLegacyArmorDetection> detections) override;

private:
    ConfigPtr config_;
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest request_;
    bool model_loaded_ = false;
    std::unique_ptr<PreprocessContext> context_;
};

// Type aliases for factory compatibility
using AtLegacyNNBackend    = AtLegacyOpenVINOBackend<640, 640>;
using AtLegacyNNTupBackend = AtLegacyOpenVINOBackend<416, 416>;

} // namespace fcs::L2
