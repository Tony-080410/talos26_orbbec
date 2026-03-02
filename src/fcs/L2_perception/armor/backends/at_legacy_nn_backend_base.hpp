#pragma once

#include "at_legacy_detector_interface.hpp"

namespace fcs::L2 {

class AtLegacyNNBackendBase : public IDetectorBackend {
public:
    using DetectionResult = IDetectorBackend::DetectionResult;

    [[nodiscard]] DetectionResult detect(const cv::Mat& image) override {
        auto* ctx = preprocess_impl(image);
        if (!ctx) {
            return std::unexpected(DetectorError::PreprocessFailed);
        }

        if (!infer_impl(ctx)) {
            return std::unexpected(DetectorError::InferenceFailed);
        }

        auto detections = postprocess_impl(ctx);
        if (detections.empty()) {
            return detections;
        }

        return nms_impl(std::move(detections));
    }

    [[nodiscard]] virtual void* preprocess_impl(const cv::Mat& image)                     = 0;
    [[nodiscard]] virtual bool infer_impl(void* ctx)                                      = 0;
    [[nodiscard]] virtual std::vector<AtLegacyArmorDetection> postprocess_impl(void* ctx) = 0;

    [[nodiscard]] virtual std::vector<AtLegacyArmorDetection>
        nms_impl(std::vector<AtLegacyArmorDetection> detections) {
        return detections;
    }

protected:
    bool initialized_ = false;
};

} // namespace fcs::L2
