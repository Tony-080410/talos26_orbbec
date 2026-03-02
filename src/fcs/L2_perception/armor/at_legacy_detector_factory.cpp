#include "L2_perception/armor/at_legacy_config.hpp"
#include "L2_perception/armor/backends/at_legacy_detector_interface.hpp"
#include "L2_perception/armor/backends/at_legacy_traditional_backend.hpp"
#if TALOS_HAS_OPENVINO
# include "L2_perception/armor/backends/at_legacy_openvino_backend.hpp"
#endif
#if TALOS_HAS_TENSORRT
# include "L2_perception/armor/backends/at_legacy_trt_backend.hpp"
#endif
#if TALOS_HAS_AXERA
# include "L2_perception/armor/backends/at_legacy_axera_backend.hpp"
#endif

namespace fcs::L2 {

[[nodiscard]] std::expected<std::unique_ptr<IDetectorBackend>, DetectorError>
    create_detector_backend(const AtLegacyConfig& config) {

    switch (config.detector.backend_type) {
    case AtLegacyBackendTypeConfig::Traditional: {
        auto cfg_ptr = config.traditional_ptr();
        auto backend = std::make_unique<AtLegacyTraditionalBackend>(std::move(cfg_ptr));
        backend->set_detect_color(config.detector.detect_color);
        return backend;
    }
#if TALOS_HAS_OPENVINO
    case AtLegacyBackendTypeConfig::NeuralNetwork: {
        auto cfg_ptr = config.nn_ptr();
        auto backend = std::make_unique<AtLegacyNNBackend>(std::move(cfg_ptr));
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        return backend;
    }
    case AtLegacyBackendTypeConfig::NeuralNetworkTup: {
        auto cfg_ptr = config.nn_tup_ptr();
        auto backend = std::make_unique<AtLegacyNNTupBackend>(std::move(cfg_ptr));
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        return backend;
    }
#endif
#if TALOS_HAS_TENSORRT
    case AtLegacyBackendTypeConfig::TensorRT: {
        auto cfg_ptr = config.trt_ptr();
        auto backend = std::make_unique<AtLegacyTrtBackend>(std::move(cfg_ptr));
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        return backend;
    }
#endif
#if TALOS_HAS_AXERA
    case AtLegacyBackendTypeConfig::Axera: {
        auto cfg_ptr = config.axera_ptr();
        auto backend = std::make_unique<AtLegacyAxeraBackend>(std::move(cfg_ptr));
        if (auto result = backend->init(); !result) {
            return std::unexpected(result.error());
        }
        return backend;
    }
#endif
    default: return std::unexpected(DetectorError::InitFailed);
    }
}

} // namespace fcs::L2
