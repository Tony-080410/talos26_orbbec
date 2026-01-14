#include "fcs/L4_planning/gimbal_planner/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L4 {

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
    READ_OPT(tbl, mpc_planner);
    return {};
}

} // namespace fcs::L4
