#include "L2_perception/armor/backends/at_legacy_traditional_classifier.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// Constructor and Initialization
// ============================================================================

AtLegacyTraditionalClassifier::AtLegacyTraditionalClassifier(
    std::string_view model_path, bool use_softmax) noexcept
    : model_path_(model_path)
    , use_softmax_(use_softmax) {}

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
        // Normalize image to 0-1 range (matching atvision_ws behavior)
        cv::Mat normalized_img = number_img / 255.0;

        // Create blob from preprocessed image (input is already 28x28 from extract_number)
        cv::Mat blob;
        cv::dnn::blobFromImage(normalized_img, blob);

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
    // Output should be 1D array with class logits or probabilities
    cv::Mat flat_output = output.reshape(1, 1);

    double confidence = 0.0;
    int label_id      = -1;

    if (use_softmax_) {
        // Legacy networks output logits, need to apply softmax (matching atvision_ws)
        float max_val = *std::max_element(flat_output.begin<float>(), flat_output.end<float>());
        cv::Mat exp_scores;
        cv::exp(flat_output - max_val, exp_scores);
        float sum = static_cast<float>(cv::sum(exp_scores)[0]);
        exp_scores /= sum;

        cv::Point class_id_point;
        cv::minMaxLoc(exp_scores, nullptr, &confidence, nullptr, &class_id_point);
        label_id = class_id_point.x;
    } else {
        // New networks already output probabilities, directly take max
        cv::Point class_id_point;
        cv::minMaxLoc(flat_output, nullptr, &confidence, nullptr, &class_id_point);
        label_id = class_id_point.x;
    }

    confidence_out = static_cast<float>(confidence);

    if (label_id < 0) {
        return ArmorName::Invalid;
    }

    return index_to_armor_name(label_id);
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
    default: return ArmorName::Invalid;
    }
}

} // namespace fcs::L2
