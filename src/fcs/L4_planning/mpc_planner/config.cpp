#include "fcs/L4_planning/mpc_planner/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L4 {

std::expected<void, std::string> MpcPlannerConfig::read_from(const toml::table& tbl) noexcept {
    // MPC cost weights
    READ_OPT(tbl, q_yaw_position);
    READ_OPT(tbl, q_yaw_velocity);
    READ_OPT(tbl, r_yaw);
    READ_OPT(tbl, max_yaw_acc);
    READ_OPT(tbl, q_pitch_position);
    READ_OPT(tbl, q_pitch_velocity);
    READ_OPT(tbl, r_pitch);
    READ_OPT(tbl, max_pitch_acc);

    // Fire decision parameters
    READ_OPT(tbl, fire_thresh);
    READ_OPT(tbl, fire_lookahead);

    // Solver parameters
    READ_OPT(tbl, max_iterations);
    READ_OPT(tbl, abs_tol);
    READ_OPT(tbl, rho);
    READ_OPT(tbl, dt);
    READ_OPT(tbl, horizon);
    READ_OPT(tbl, half_horizon);

    // Trajectory configuration (nested table)
    READ_OPT(tbl, trajectory);
    READ_OPT(tbl, compensator_type);

    // Operational parameters
    READ_OPT(tbl, plan_to_muzzle_delay);

    return {};
}

} // namespace fcs::L4
