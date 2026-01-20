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
    READ_OPT(tbl, jpda_matcher);
    return {};
}

std::expected<void, std::string> JpdaMatcherConfig::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, enable_jpda_matcher);
    READ_OPT(tbl, phase_ambiguous_threshold);
    READ_OPT(tbl, phase_switch_lambda);
    READ_OPT(tbl, semantic_reweight_lambda);
    READ_OPT(tbl, enable_semantic);
    READ_OPT(tbl, enable_viterbi);
    // Geometry consistency parameters
    READ_OPT(tbl, enable_geometry);
    READ_OPT(tbl, geometry_weight);
    READ_OPT(tbl, sigma_radius);
    READ_OPT(tbl, sigma_height);
    return {};
}

std::expected<void, std::string> L3Config::read_from(const toml::table& tbl) noexcept {
    READ(tbl, tracker);
    return {};
}

} // namespace fcs::L3
