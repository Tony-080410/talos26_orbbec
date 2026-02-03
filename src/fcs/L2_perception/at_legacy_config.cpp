#include "fcs/L2_perception/at_legacy_config.hpp"

#include "toml_helper.hpp"

namespace fcs::L2 {

std::expected<void, std::string>
    AtLegacyTraditionalConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, dark_percentage);
    READ(tbl, binary_threshold);
    READ(tbl, light_min_ratio);
    READ(tbl, light_max_ratio);
    READ(tbl, light_max_angle);
    READ(tbl, light_color_diff_thresh);
    READ(tbl, armor_min_light_ratio);
    READ(tbl, armor_min_small_center_distance);
    READ(tbl, armor_max_small_center_distance);
    READ(tbl, armor_min_large_center_distance);
    READ(tbl, armor_max_large_center_distance);
    READ(tbl, armor_max_angle);

    // classifier configuration
    READ_OPT(tbl, advanced_binary);
    READ_OPT(tbl, classifier_model_path);
    READ_OPT(tbl, classifier_confidence_threshold);
    READ_OPT(tbl, classifier_enable_type_filtering);
    READ_OPT(tbl, classifier_use_softmax);

    return {};
}

std::expected<void, std::string> AtLegacyNNConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, model_path);
    READ(tbl, device);
    READ(tbl, device_priorities);
    READ(tbl, enable_profiling);
    READ(tbl, enable_multi_thread);
    READ(tbl, confidence_threshold);
    READ(tbl, nms_threshold);
    READ(tbl, top_k);
    return {};
}

std::expected<void, std::string> AtLegacyNNTupConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, model_path);
    READ(tbl, device);
    READ(tbl, device_priorities);
    READ(tbl, enable_profiling);
    READ(tbl, enable_multi_thread);
    READ(tbl, confidence_threshold);
    READ(tbl, nms_threshold);
    READ(tbl, top_k);
    return {};
}

std::expected<void, std::string>
    AtLegacyDetectorConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, backend_type);
    READ(tbl, detect_color);
    return {};
}

std::expected<void, std::string> AtLegacyConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, detector);
    // Read backend-specific config based on type
    READ(tbl, traditional);
    READ(tbl, nn);
    READ(tbl, nn_tup);
#if TALOS_HAS_TENSORRT
    READ(tbl, trt);
#endif
    return {};
}

#if TALOS_HAS_TENSORRT
std::expected<void, std::string> AtLegacyTrtConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, engine_path);
    READ(tbl, model_path);
    READ(tbl, engine_cache_dir);
    READ(tbl, device_id);
    READ(tbl, num_streams);
    READ(tbl, compute_capability);
    READ_OPT(tbl, enable_fp16);
    READ_OPT(tbl, enable_dla);
    READ(tbl, dla_core);
    READ(tbl, confidence_threshold);
    READ(tbl, nms_threshold);
    READ(tbl, top_k);

    // Validate compute_capability format (e.g., "86", "75", "89")
    if (!compute_capability.empty()) {
        bool valid = true;
        for (char c : compute_capability) {
            if (!std::isdigit(c)) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            return std::unexpected(
                "Invalid compute_capability: must be numeric string (e.g., '86', '75', '89')");
        }
    }

    // Validate num_streams > 0
    if (num_streams <= 0) {
        return std::unexpected("num_streams must be greater than 0");
    }

    // Validate device_id >= 0
    if (device_id < 0) {
        return std::unexpected("device_id must be greater than or equal to 0");
    }

    return {};
}
#endif

} // namespace fcs::L2
