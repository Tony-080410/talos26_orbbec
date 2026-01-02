#include "fcs/L2_perception/at_legacy_detector_backend.hpp"

#include "fcs/L2_perception/at_legacy_config.hpp"
#include "fcs/L2_perception/at_legacy_nn_backend.hpp"
#include "fcs/L2_perception/at_legacy_nn_tup_backend.hpp"
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

AtLegacyDetectorBackend::AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNTupBackend> backend)
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

void AtLegacyDetectorBackend::set_backend(std::unique_ptr<AtLegacyNNTupBackend> backend) {
    backend_ = std::move(backend);
}

// ============================================================================
// Factory Function
// ============================================================================

[[nodiscard]] std::expected<AtLegacyDetectorBackend, DetectorError>
    create_detector_backend(const AtLegacyConfig& config) {
    switch (config.detector.backend_type) {
    case AtLegacyBackendTypeConfig::Traditional: {
        auto cfg_ptr = config.traditional_ptr();
        auto backend = std::make_unique<AtLegacyTraditionalBackend>(std::move(cfg_ptr));
        backend->set_detect_color(config.detector.detect_color);
        return AtLegacyDetectorBackend(std::move(backend));
    }
    case AtLegacyBackendTypeConfig::NeuralNetwork: {
        auto cfg_ptr = config.nn_ptr();
        auto backend = std::make_unique<AtLegacyNNBackend>(std::move(cfg_ptr));
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        backend->set_detect_color(config.detector.detect_color);
        return AtLegacyDetectorBackend(std::move(backend));
    }
    case AtLegacyBackendTypeConfig::NeuralNetworkTup: {
        auto cfg_ptr = config.nn_tup_ptr();
        auto backend = std::make_unique<AtLegacyNNTupBackend>(std::move(cfg_ptr));
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        backend->set_detect_color(config.detector.detect_color);
        return AtLegacyDetectorBackend(std::move(backend));
    }
    }
    return std::unexpected(DetectorError::NotInitialized);
}

} // namespace fcs
