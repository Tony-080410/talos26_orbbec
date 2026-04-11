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

AtLegacyDetectorBackend::DetectionResult
    AtLegacyDetectorBackend::detect(const cv::Mat& image) noexcept {
    return std::visit(
        [&image](auto& backend) -> DetectionResult {
            if (!backend) {
                return std::unexpected("No detector backend is active");
            }
            return backend->detect(image);
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

AtLegacyBackendInputResolution AtLegacyDetectorBackend::input_resolution() const noexcept {
    return std::visit(
        [](const auto& backend) -> AtLegacyBackendInputResolution {
            using BackendPtr = std::decay_t<decltype(backend)>;
            using Backend    = typename BackendPtr::element_type;
            if (!backend) {
                return {};
            }
            if constexpr (std::is_same_v<Backend, AtLegacyTraditionalBackend>) {
                return {.width = 480, .height = 270};
#if TALOS_HAS_AXERA
            } else if constexpr (std::is_same_v<Backend, AtLegacyAxeraBackend>) {
                return {
                    .width  = backend->get_config().input_width,
                    .height = backend->get_config().input_height};
#endif
            } else {
                return {.width = Backend::INPUT_W, .height = Backend::INPUT_H};
            }
        },
        backend_);
}

// ============================================================================
// Factory Function
// ============================================================================

[[nodiscard]] std::expected<AtLegacyDetectorBackend, std::string>
    create_detector_backend(const AtLegacyConfig& config) noexcept {
    switch (config.backend_type) {
    case AtLegacyBackendTypeConfig::Traditional: {
        auto result = AtLegacyTraditionalBackend::create(config.traditional);
        if (!result) {
            return std::unexpected(result.error());
        }
        result->set_detect_color(config.default_detect_color);
        return AtLegacyDetectorBackend(
            std::make_unique<AtLegacyTraditionalBackend>(std::move(*result)));
    }
#if TALOS_HAS_OPENVINO
    case AtLegacyBackendTypeConfig::NeuralNetwork: {
        auto result = AtLegacyNNBackend::create(config.nn);
        if (!result) {
            return std::unexpected(result.error());
        }
        result->set_detect_color(config.default_detect_color);
        return AtLegacyDetectorBackend(std::make_unique<AtLegacyNNBackend>(std::move(*result)));
    }
    case AtLegacyBackendTypeConfig::NeuralNetworkTup: {
        auto result = AtLegacyNNTupBackend::create(config.nn_tup);
        if (!result) {
            return std::unexpected(result.error());
        }
        result->set_detect_color(config.default_detect_color);
        return AtLegacyDetectorBackend(std::make_unique<AtLegacyNNTupBackend>(std::move(*result)));
    }
#endif
#if TALOS_HAS_TENSORRT
    case AtLegacyBackendTypeConfig::TensorRT: {
        auto result = AtLegacyTrtBackend::create(config.trt);
        if (!result) {
            return std::unexpected(result.error());
        }
        result->set_detect_color(config.default_detect_color);
        return AtLegacyDetectorBackend(std::make_unique<AtLegacyTrtBackend>(std::move(*result)));
    }
#endif
#if TALOS_HAS_AXERA
    case AtLegacyBackendTypeConfig::Axera: {
        auto result = AtLegacyAxeraBackend::create(config.axera);
        if (!result) {
            return std::unexpected(result.error());
        }
        result->set_detect_color(config.default_detect_color);
        return AtLegacyDetectorBackend(std::make_unique<AtLegacyAxeraBackend>(std::move(*result)));
    }
#endif
    }
    std::unreachable();
}

} // namespace fcs::L2
