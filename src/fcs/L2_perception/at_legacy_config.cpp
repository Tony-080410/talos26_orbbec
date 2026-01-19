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
    return {};
}

} // namespace fcs::L2
