#include "fcs/L3_estimation/tracker/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L3 {

template <>
std::expected<void, std::string>
    TargetConfig<RobotModelParams>::read_from(const toml::table& tbl) noexcept {
    READ(tbl, lost_threshold);
    READ(tbl, tracking_threshold);
    READ(tbl, matcher_gate);
    READ(tbl, model);
    return {};
}

template <>
std::expected<void, std::string>
    TargetConfig<OutpostModelParams>::read_from(const toml::table& tbl) noexcept {
    READ(tbl, lost_threshold);
    READ(tbl, tracking_threshold);
    READ(tbl, matcher_gate);
    READ(tbl, model);
    return {};
}

std::expected<void, std::string> TrackerConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, robot);
    READ(tbl, outpost);
    READ(tbl, inekf_enabled);
    READ(tbl, robot_inekf);
    READ(tbl, outpost_inekf);
    READ(tbl, jpda_matcher);
    return {};
}

std::expected<void, std::string> JpdaMatcherConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, enable_jpda_matcher);
    READ(tbl, phase_ambiguous_threshold);
    READ(tbl, phase_switch_lambda);
    READ(tbl, semantic_reweight_lambda);
    READ(tbl, enable_semantic);
    READ(tbl, enable_viterbi);
    // Geometry consistency parameters
    READ(tbl, enable_geometry);
    READ(tbl, geometry_weight);
    READ(tbl, sigma_radius);
    READ(tbl, sigma_height);
    return {};
}

std::expected<void, std::string> L3Config::read_from(const toml::table& tbl) noexcept {
    READ(tbl, tracker);
    return {};
}

} // namespace fcs::L3
