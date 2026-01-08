#pragma once

#include "toml_helper.hpp"
#include <expected>
#include <toml++/toml.hpp>

namespace fcs::L4 {

// ============================================================================
// Trajectory Configuration (L4 Layer - Master)
// ============================================================================

enum class TrajectoryCompensatorType {
    Ideal,
    Resistance,
    RK4,
};

struct TrajectoryConfig {
    /// Number of iterations for trajectory compensation
    int iteration_times{20};

    /// Bullet velocity (m/s)
    double bullet_speed{25.0};

    /// Gravity acceleration (m/s^2)
    double gravity{9.8};

    /// Air resistance coefficient
    double resistance{0.001};

    /// Drag coefficient Cd (dimensionless, needs calibration)
    double cd{0.47};

    /// Air density rho (kg/m^3)
    double rho{1.225};

    /// Bullet diameter (meters), used to compute cross-sectional area A
    double diameter{0.017};

    /// Bullet mass (kg)
    double mass{0.0032};

    /// Desired RK4 integration dt (seconds)
    double rk4_dt{0.001};

    /// Minimum RK4 steps per integration
    int rk4_min_steps{50};

    /// Maximum RK4 steps per integration
    int rk4_max_steps{2000};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
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
};

// ============================================================================
// Gimbal Planner Configuration (L4 Layer)
// ============================================================================

/// Gimbala planning configuration
struct GimbalPlannerConfig {
    /// Shooting range width (meters)
    double shooting_range_w{0.135};

    /// Shooting range height (meters)
    double shooting_range_h{0.135};

    /// Maximum angular velocity for armor tracking mode
    double max_tracking_v_yaw{6.0};

    /// Delay from plan output to bullet leaving muzzle (seconds)
    double plan_to_muzzle_delay{0.0};

    /// Side angle threshold for armor switching (degrees)
    double side_angle{10.0};

    /// Minimum angular velocity for armor switching (rad/s)
    double min_switching_v_yaw{1.0};

    /// State transfer counter threshold
    int transfer_thresh{5};

    /// Trajectory compensator configuration
    TrajectoryConfig trajectory;

    /// Compensator type: "ideal" or "resistance"
    TrajectoryCompensatorType compensator_type = TrajectoryCompensatorType::Resistance;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
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
};

// ============================================================================
// L4 Planning Full Config
// ============================================================================

struct L4Config {
    GimbalPlannerConfig planner{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const GimbalPlannerConfig> planner_ptr() const {
        return std::make_shared<const GimbalPlannerConfig>(planner);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, planner);
        return {};
    }
};

} // namespace fcs::L4
