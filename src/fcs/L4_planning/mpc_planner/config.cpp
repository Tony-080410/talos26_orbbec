#include "fcs/L4_planning/mpc_planner/config.hpp"

#include "toml_helper.hpp"

namespace fcs::L4 {

std::expected<void, std::string> MpcPlannerConfig::read_from(const toml::table& tbl) noexcept {
    // MPC cost weights
    READ(tbl, q_yaw_position);
    READ(tbl, q_yaw_velocity);
    READ(tbl, r_yaw);
    READ(tbl, max_yaw_acc);
    READ(tbl, q_pitch_position);
    READ(tbl, q_pitch_velocity);
    READ(tbl, r_pitch);
    READ(tbl, max_pitch_acc);

    // Fire decision parameters
    READ(tbl, fire_thresh);
    READ(tbl, fire_lookahead);

    // Solver parameters
    READ(tbl, max_iterations);
    READ(tbl, abs_tol);
    READ(tbl, rho);
    READ(tbl, dt);
    READ(tbl, horizon);
    READ(tbl, half_horizon);

    // Trajectory configuration (nested table)
    READ(tbl, trajectory);
    READ(tbl, compensator_type);

    // Adaptive delay compensation
    READ(tbl, decision_speed);
    READ(tbl, high_speed_delay);
    READ(tbl, low_speed_delay);

    return {};
}

} // namespace fcs::L4
