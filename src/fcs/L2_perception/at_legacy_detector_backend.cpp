#include "fcs/L2_perception/at_legacy_detector_backend.hpp"
#include "fcs/L2_perception/at_legacy_nn_backend.hpp"
#include "fcs/L2_perception/at_legacy_traditional_backend.hpp"

namespace fcs {

// ============================================================================
// AtLegacyDetectorBackend Implementation
// ============================================================================

AtLegacyDetectorBackend::AtLegacyDetectorBackend()  = default;
AtLegacyDetectorBackend::~AtLegacyDetectorBackend() = default;

AtLegacyDetectorBackend::AtLegacyDetectorBackend(AtLegacyDetectorBackend&&) noexcept = default;
AtLegacyDetectorBackend&
    AtLegacyDetectorBackend::operator=(AtLegacyDetectorBackend&&) noexcept = default;

AtLegacyDetectorBackend::AtLegacyDetectorBackend(
    std::unique_ptr<AtLegacyTraditionalBackend> backend)
    : backend_(std::move(backend)) {}

AtLegacyDetectorBackend::AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNBackend> backend)
    : backend_(std::move(backend)) {}

AtLegacyDetectorBackend::DetectionResult AtLegacyDetectorBackend::detect(const cv::Mat& image) {
    return std::visit(
        [&image](auto& backend) -> DetectionResult {
            if (!backend) {
                return std::unexpected(DetectorError::NotInitialized);
            }
            return backend->detect(image);
        },
        backend_);
}

bool AtLegacyDetectorBackend::is_initialized() const noexcept {
    return std::visit(
        [](const auto& backend) -> bool { return backend && backend->is_initialized(); }, backend_);
}

std::string_view AtLegacyDetectorBackend::name() const noexcept {
    return std::visit(
        [](const auto& backend) -> std::string_view {
            if (!backend)
                return "None";
            return backend->name();
        },
        backend_);
}

void AtLegacyDetectorBackend::set_backend(std::unique_ptr<AtLegacyTraditionalBackend> backend) {
    backend_ = std::move(backend);
}

void AtLegacyDetectorBackend::set_backend(std::unique_ptr<AtLegacyNNBackend> backend) {
    backend_ = std::move(backend);
}

// ============================================================================
// Factory Function
// ============================================================================

[[nodiscard]] std::expected<AtLegacyDetectorBackend, DetectorError>
    create_detector_backend(const AtLegacyBackendConfig& config) {
    switch (config.type) {
    case AtLegacyBackendType::Traditional: {
        auto backend = std::make_unique<AtLegacyTraditionalBackend>(config.binary_threshold);
        return AtLegacyDetectorBackend(std::move(backend));
    }
    case AtLegacyBackendType::NeuralNetworkAt: {
        AtLegacyNNBackendConfig nn_config;
        nn_config.model_path           = config.model_path;
        nn_config.num_classes          = config.num_classes;
        nn_config.num_colors           = config.num_colors;
        nn_config.confidence_threshold = config.confidence_threshold;
        nn_config.nms_threshold        = config.nms_threshold;

        auto backend = std::make_unique<AtLegacyNNBackend>(nn_config);
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        return AtLegacyDetectorBackend(std::move(backend));
    }
    }
    return std::unexpected(DetectorError::NotInitialized);
}

} // namespace fcs
