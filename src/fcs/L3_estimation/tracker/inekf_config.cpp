#include "fcs/L3_estimation/tracker/inekf_config.hpp"

#include "toml_helper.hpp"

namespace fcs::L3 {

namespace {
template <typename T>
std::expected<void, std::string>
    discard_opt(const toml::table& tbl, std::string_view key) noexcept {
    auto res = toml_helper::take_optional<T>(tbl, key);
    if (!res) {
        return std::unexpected(res.error());
    }
    return {};
}
} // namespace

std::expected<void, std::string> RobotInEKFConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, sigma_a_xy);
    READ(tbl, sigma_a_z);
    READ(tbl, sigma_yaw);
    READ(tbl, use_coordinated_turn);
    READ(tbl, enable_slip_gated_turn);
    READ(tbl, enable_adaptive_process_noise);
    READ(tbl, enable_pseudo_velocity_update);
    READ(tbl, enable_robust_update);
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
    // Spherical observation mode (optional for backward compatibility)
    READ(tbl, use_spherical_observation);
    READ(tbl, sigma_yaw_obs);
    READ(tbl, sigma_pitch_obs);
    READ(tbl, sigma_armor_yaw);
    // Innovation gating
    READ(tbl, max_angle_innovation);
    READ(tbl, max_dist_innovation);
    return {};
}

std::expected<void, std::string> OutpostInEKFConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, sigma_a_xy);
    READ(tbl, sigma_a_z);
    READ(tbl, sigma_yaw);
    READ(tbl, use_coordinated_turn);
    READ(tbl, enable_slip_gated_turn);
    READ(tbl, enable_adaptive_process_noise);
    READ(tbl, enable_pseudo_velocity_update);
    READ(tbl, enable_robust_update);
    READ(tbl, sigma_h);
    READ(tbl, sigma_m);
    READ(tbl, sigma_w);
    READ(tbl, use_dynamic_noise);
    READ(tbl, dynamic_noise_k_radial);
    READ(tbl, dynamic_noise_k_tangent);
    // Spherical observation mode (optional for backward compatibility)
    READ(tbl, use_spherical_observation);
    READ(tbl, sigma_yaw_obs);
    READ(tbl, sigma_pitch_obs);
    READ(tbl, sigma_armor_yaw);
    // Innovation gating
    READ(tbl, max_angle_innovation);
    READ(tbl, max_dist_innovation);
    return {};
}

} // namespace fcs::L3
