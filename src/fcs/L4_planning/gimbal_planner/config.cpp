#include "fcs/L4_planning/gimbal_planner/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L4 {

std::expected<void, std::string> TrajectoryConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, iteration_times);
    READ(tbl, bullet_speed);
    READ(tbl, gravity);
    READ(tbl, resistance);
    READ_OPT(tbl, cd);
    READ_OPT(tbl, rho);
    READ_OPT(tbl, diameter);
    READ_OPT(tbl, mass);
    READ_OPT(tbl, rk4_dt);
    READ_OPT(tbl, rk4_min_steps);
    READ_OPT(tbl, rk4_max_steps);
    return {};
}

std::expected<void, std::string> GimbalPlannerConfig::read_from(const toml::table& tbl) noexcept {
    READ(tbl, shooting_range_w);
    READ(tbl, shooting_range_h);
    READ(tbl, max_tracking_v_yaw);
    READ(tbl, plan_to_muzzle_delay);
    READ(tbl, side_angle);
    READ(tbl, min_switching_v_yaw);
    READ(tbl, transfer_thresh);
    READ(tbl, trajectory);
    READ(tbl, compensator_type);
    return {};
}

std::expected<void, std::string> L4Config::read_from(const toml::table& tbl) noexcept {
    READ(tbl, planner);
    return {};
}

} // namespace fcs::L4
