#pragma once

#include "motion_model.hpp"
#include "toml_helper.hpp"
#include "trajectory_compensator.hpp"
#include <expected>
#include <memory>
#include <string>
#include <toml++/toml.hpp>

namespace fcs::L3 {

// ============================================================================
// Trajectory Compensator Configuration
// ============================================================================

struct TrajectoryConfig {
    /// Number of iterations for trajectory compensation
    int iteration_times{20};

    /// Bullet velocity (m/s)
    double bullet_speed{25.0};

    /// Gravity acceleration (m/s^2)
    double gravity{9.8};

    /// Air resistance coefficient
    double resistance{0.001};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, iteration_times);
        READ(tbl, bullet_speed);
        READ(tbl, gravity);
        READ(tbl, resistance);
        return {};
    }
};

// ============================================================================
// Tracker Configuration
// ============================================================================

template <typename T>
struct TargetConfig {
    /// Timeout before returning to IDLE from TEMP_LOST (seconds)
    double lost_threshold;
    /// Number of consecutive frames required to transition DETECTING -> TRACKING
    uint32_t tracking_threshold;
    /// Mahalanobis distance gate for target matching
    double matcher_gate{10.0};
    /// motion model parameters
    T model;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, lost_threshold);
        READ(tbl, tracking_threshold);
        READ(tbl, matcher_gate);
        READ(tbl, model);
        return {};
    }
};

struct TrackerConfig {
    /// Robot target parameters
    TargetConfig<RobotModel::Params> robot;
    /// Outpost target parameters
    TargetConfig<OutpostModel::Params> outpost;
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, robot);
        READ(tbl, outpost);
        return {};
    }
};

// ============================================================================
// Solver Configuration
// ============================================================================

struct SolverConfig {
    /// Shooting range width (meters)
    double shooting_range_w{0.135};

    /// Shooting range height (meters)
    double shooting_range_h{0.135};

    /// Maximum angular velocity for armor tracking mode
    double max_tracking_v_yaw{6.0};

    /// Additional prediction delay (seconds)
    double prediction_delay{0.0};

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
        READ(tbl, prediction_delay);
        READ(tbl, side_angle);
        READ(tbl, min_switching_v_yaw);
        READ(tbl, transfer_thresh);
        READ(tbl, compensator_type);
        return {};
    }
};

// ============================================================================
// L3 Estimation Full Config
// ============================================================================

struct L3Config {
    TrackerConfig tracker{};
    SolverConfig solver{};
    TrajectoryConfig trajectory{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const TrackerConfig> tracker_ptr() const {
        return std::make_shared<const TrackerConfig>(tracker);
    }

    [[nodiscard]] std::shared_ptr<const SolverConfig> solver_ptr() const {
        return std::make_shared<const SolverConfig>(solver);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, tracker);
        READ(tbl, solver);
        READ(tbl, trajectory);
        // Keep backward compatibility with existing configs:
        // [l3.trajectory] drives the solver's ballistic parameters (matches atvision_ws layout).
        solver.trajectory = trajectory;
        return {};
    }
};

} // namespace fcs::L3
