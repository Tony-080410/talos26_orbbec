#include "fcs/L2_perception/at_legacy_traditional_classifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace fcs {

// ============================================================================
// Constructor and Initialization
// ============================================================================

AtLegacyTraditionalClassifier::AtLegacyTraditionalClassifier(std::string_view model_path) noexcept
    : model_path_(model_path) {}

std::expected<void, DetectorError> AtLegacyTraditionalClassifier::init() noexcept {
    try {
        // Load ONNX model via OpenCV DNN
        net_ = cv::dnn::readNetFromONNX(model_path_);

        if (net_.empty()) {
            return std::unexpected(DetectorError::ModelLoadError);
        }

        // Set backend (CPU for now, can be extended to GPU)
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        initialized_ = true;
        return {};
    } catch (const std::exception& e) {
        // This can happen if ONNX model format is invalid or has incompatible layers
        return std::unexpected(DetectorError::ModelLoadError);
    } catch (...) {
        return std::unexpected(DetectorError::ModelLoadError);
    }
}

// ============================================================================
// Classification
// ============================================================================

AtLegacyTraditionalClassifier::ClassifyResult AtLegacyTraditionalClassifier::classify(
    const cv::Mat& number_img, float& confidence_out) const noexcept {
    if (!initialized_) {
        return std::unexpected(DetectorError::NotInitialized);
    }

    if (number_img.empty()) {
        return std::unexpected(DetectorError::InvalidImage);
    }

    try {
        // Create blob from preprocessed image
        cv::Mat blob = cv::dnn::blobFromImage(number_img, 1.0, cv::Size(20, 28));

        // Set input to network
        net_.setInput(blob);

        // Forward inference
        cv::Mat output = net_.forward();

        // Post-process and return result
        return post_process_output(output, confidence_out);
    } catch (...) {
        return std::unexpected(DetectorError::InferenceError);
    }
}

// ============================================================================
// Output Post-Processing
// ============================================================================

ArmorName AtLegacyTraditionalClassifier::post_process_output(
    const cv::Mat& output, float& confidence_out) const noexcept {
    // Output should be 1D array with class probabilities
    cv::Mat flat_output = output.reshape(0, 1);

    // Find class with maximum probability
    double max_confidence = 0.0;
    int max_idx           = -1;

    for (int i = 0; i < flat_output.cols; ++i) {
        float prob = flat_output.at<float>(0, i);
        if (prob > max_confidence) {
            max_confidence = prob;
            max_idx        = i;
        }
    }

    confidence_out = static_cast<float>(max_confidence);

    if (max_idx < 0) {
        return ArmorName::Invalid;
    }

    // Convert index to armor name
    ArmorName predicted_name = index_to_armor_name(max_idx);

    return predicted_name;
}

// ============================================================================
// Helper Methods
// ============================================================================

ArmorName AtLegacyTraditionalClassifier::index_to_armor_name(int idx) noexcept {
    switch (idx) {
    case 0: return ArmorName::One;
    case 1: return ArmorName::Two;
    case 2: return ArmorName::Three;
    case 3: return ArmorName::Four;
    case 4: return ArmorName::Five;
    case 5: return ArmorName::Outpost;
    case 6: return ArmorName::Sentry;
    case 7: return ArmorName::Base;
    case 8: return ArmorName::Invalid;
    }
}

} // namespace fcs
