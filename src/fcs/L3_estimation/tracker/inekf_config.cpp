#include "fcs/L3_estimation/tracker/inekf_config.hpp"

#include "toml_helper.hpp"

namespace fcs::L3 {

std::expected<void, std::string> RobotInEKFConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, sigma_a_xy);
    READ(tbl, sigma_a_z);
    READ(tbl, sigma_yaw);
    READ(tbl, sigma_r);
    READ(tbl, sigma_h);
    READ(tbl, sigma_m);
    READ(tbl, sigma_w);
    READ(tbl, use_dynamic_noise);
    READ(tbl, dynamic_noise_k_radial);
    READ(tbl, dynamic_noise_k_tangent);
    READ(tbl, radius0);
    READ(tbl, radius1);
    READ(tbl, height);
    return {};
}

std::expected<void, std::string> OutpostInEKFConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, sigma_a_xy);
    READ(tbl, sigma_a_z);
    READ(tbl, sigma_yaw);
    READ(tbl, sigma_h);
    READ(tbl, sigma_m);
    READ(tbl, sigma_w);
    READ(tbl, use_dynamic_noise);
    READ(tbl, dynamic_noise_k_radial);
    READ(tbl, dynamic_noise_k_tangent);
    return {};
}

} // namespace fcs::L3
