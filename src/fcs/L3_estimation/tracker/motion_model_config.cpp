#include "fcs/L3_estimation/tracker/motion_model.hpp"

#include "toml_helper.hpp"

namespace fcs::L3 {

std::expected<void, std::string> RobotModel::Params::read_from(const toml::table& tbl) noexcept {
    READ(tbl, sigma_a_xy);
    READ(tbl, sigma_a_z);
    READ(tbl, sigma_a_yaw);
    READ(tbl, sigma_r0);
    READ(tbl, sigma_r1);
    READ(tbl, sigma_h);
    return {};
}

std::expected<void, std::string> OutpostModel::Params::read_from(const toml::table& tbl) noexcept {
    READ(tbl, sigma_q_xy);
    READ(tbl, sigma_q_z);
    READ(tbl, sigma_a_yaw);
    return {};
}

} // namespace fcs::L3
