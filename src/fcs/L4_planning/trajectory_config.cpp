#include "fcs/L4_planning/trajectory_config.hpp"

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
    READ_OPT(tbl, enable_lookup_table);
    READ_OPT(tbl, table_range_max);
    READ_OPT(tbl, table_range_step);
    READ_OPT(tbl, table_pitch_min);
    READ_OPT(tbl, table_pitch_max);
    READ_OPT(tbl, table_pitch_step);
    return {};
}

} // namespace fcs::L4
