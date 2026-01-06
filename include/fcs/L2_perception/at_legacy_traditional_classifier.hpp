#pragma once

#include "at_legacy_detector_backend.hpp"
#include "core/types.hpp"

#include <expected>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace fcs {

// ============================================================================
// AT Legacy Traditional ONNX Classifier
// ============================================================================

/// OpenCV DNN-based ONNX classifier for armor number classification
/// Supports LeNet and MLP models with typical ONNX format
class AtLegacyTraditionalClassifier {
public:
    using ClassifyResult = std::expected<ArmorName, DetectorError>;

    /// Constructor: takes path to ONNX model file and softmax flag
    explicit AtLegacyTraditionalClassifier(
        std::string_view model_path, bool use_softmax = true) noexcept;

    /// Load and initialize the ONNX model
    [[nodiscard]] std::expected<void, DetectorError> init() noexcept;

    /// Classify a single armor image (preprocessed: 20x28 grayscale OTSU-binary, normalized [0,1])
    /// Returns the predicted ArmorName and updates confidence_out with confidence score
    [[nodiscard]] ClassifyResult
        classify(const cv::Mat& number_img, float& confidence_out) const noexcept;

    /// Check if classifier is initialized
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

private:
    std::string model_path_;
    mutable cv::dnn::Net net_; // mutable because OpenCV DNN operations are non-const
    bool initialized_ = false;
    bool use_softmax_ = true;  // Whether to apply softmax to network output (for legacy networks)

    // Helper methods
    [[nodiscard]] ArmorName
        post_process_output(const cv::Mat& output, float& confidence_out) const noexcept;

    /// Map network output index to ArmorName
    [[nodiscard]] static ArmorName index_to_armor_name(int idx) noexcept;
};

} // namespace fcs
