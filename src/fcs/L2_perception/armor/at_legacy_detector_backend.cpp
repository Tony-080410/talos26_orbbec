#include "L2_perception/armor/at_legacy_config.hpp"
#include "L2_perception/armor/backends/at_legacy_traditional_backend.hpp"
#if TALOS_HAS_OPENVINO
# include "L2_perception/armor/backends/at_legacy_nn_backend.hpp"
# include "L2_perception/armor/backends/at_legacy_nn_tup_backend.hpp"
#endif
#if TALOS_HAS_TENSORRT
# include "L2_perception/armor/backends/at_legacy_trt_backend.hpp"
#endif
#if TALOS_HAS_AXERA
# include "L2_perception/armor/backends/at_legacy_axera_backend.hpp"
#endif

#include <type_traits>

namespace fcs::L2 {

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

#if TALOS_HAS_OPENVINO
AtLegacyDetectorBackend::AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNBackend> backend)
    : backend_(std::move(backend)) {}

AtLegacyDetectorBackend::AtLegacyDetectorBackend(std::unique_ptr<AtLegacyNNTupBackend> backend)
    : backend_(std::move(backend)) {}
#endif

#if TALOS_HAS_TENSORRT
AtLegacyDetectorBackend::AtLegacyDetectorBackend(std::unique_ptr<AtLegacyTrtBackend> backend)
    : backend_(std::move(backend)) {}
#endif

#if TALOS_HAS_AXERA
AtLegacyDetectorBackend::AtLegacyDetectorBackend(std::unique_ptr<AtLegacyAxeraBackend> backend)
    : backend_(std::move(backend)) {}
#endif

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

#if TALOS_HAS_OPENVINO
void AtLegacyDetectorBackend::set_backend(std::unique_ptr<AtLegacyNNBackend> backend) {
    backend_ = std::move(backend);
}

void AtLegacyDetectorBackend::set_backend(std::unique_ptr<AtLegacyNNTupBackend> backend) {
    backend_ = std::move(backend);
}
#endif

#if TALOS_HAS_TENSORRT
void AtLegacyDetectorBackend::set_backend(std::unique_ptr<AtLegacyTrtBackend> backend) {
    backend_ = std::move(backend);
}
#endif

#if TALOS_HAS_AXERA
void AtLegacyDetectorBackend::set_backend(std::unique_ptr<AtLegacyAxeraBackend> backend) {
    backend_ = std::move(backend);
}
#endif

void AtLegacyDetectorBackend::set_detect_color(ArmorColor color) noexcept {
    std::visit(
        [color](auto& backend) {
            if (backend) {
                backend->set_detect_color(color);
            }
        },
        backend_);
}

void AtLegacyDetectorBackend::update_config(const AtLegacyConfig& config) noexcept {
    std::visit(
        [&config](auto& backend) {
            if (!backend) {
                return;
            }
            using BackendT = std::decay_t<decltype(*backend)>;
            if constexpr (std::is_same_v<BackendT, AtLegacyTraditionalBackend>) {
                backend->update_config(config.traditional_ptr());
            }
#if TALOS_HAS_OPENVINO
            else if constexpr (std::is_same_v<BackendT, AtLegacyNNBackend>) {
                backend->update_config(config.nn_ptr());
            } else if constexpr (std::is_same_v<BackendT, AtLegacyNNTupBackend>) {
                backend->update_config(config.nn_tup_ptr());
            }
#endif
#if TALOS_HAS_TENSORRT
            else if constexpr (std::is_same_v<BackendT, AtLegacyTrtBackend>) {
                backend->update_config(config.trt_ptr());
            }
#endif
#if TALOS_HAS_AXERA
            else if constexpr (std::is_same_v<BackendT, AtLegacyAxeraBackend>) {
                backend->update_config(config.axera_ptr());
            }
#endif
        },
        backend_);
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
#if TALOS_HAS_OPENVINO
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
#endif
#if TALOS_HAS_TENSORRT
    case AtLegacyBackendTypeConfig::TensorRT: {
        auto cfg_ptr = config.trt_ptr();
        auto backend = std::make_unique<AtLegacyTrtBackend>(std::move(cfg_ptr));
        if (!backend->is_initialized()) {
            return std::unexpected(DetectorError::InitFailed);
        }
        backend->set_detect_color(config.detector.detect_color);
        return AtLegacyDetectorBackend(std::move(backend));
    }
#endif
#if TALOS_HAS_AXERA
    case AtLegacyBackendTypeConfig::Axera: {
        auto cfg_ptr = config.axera_ptr();
        auto backend = std::make_unique<AtLegacyAxeraBackend>(std::move(cfg_ptr));
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        backend->set_detect_color(config.detector.detect_color);
        return AtLegacyDetectorBackend(std::move(backend));
    }
#endif
    default: return std::unexpected(DetectorError::NotInitialized);
    }
}

} // namespace fcs::L2
