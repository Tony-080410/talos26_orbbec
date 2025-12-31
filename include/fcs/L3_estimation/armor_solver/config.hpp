#pragma once

#include "motion_model.hpp"

#include "toml_helper.hpp"
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
        if (const auto _t = tbl["trajectory"].as_table()) {
            auto t = *_t;
            READ_KEY(t, iteration_times);
            READ_KEY(t, bullet_speed);
            READ_KEY(t, gravity);
            READ_KEY(t, resistance);
            return {};
        }
        // Optional section - use defaults
        return {};
    }
};

// ============================================================================
// Tracker Configuration
// ============================================================================

struct TrackerConfig {
    /// Timeout before returning to IDLE from TEMP_LOST (seconds)
    double lost_thres{1.0};

    /// Number of consecutive frames required to transition DETECTING -> TRACKING
    int tracking_thres{3};

    /// Mahalanobis distance gate for robot target matching
    double matcher_gate{25.0};

    /// Mahalanobis distance gate for outpost target matching
    double outpost_matcher_gate{50.0};

    /// Robot motion model parameters
    RobotModel::Params robot_params;

    /// Outpost motion model parameters
    OutpostModel::Params outpost_params;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["tracker"].as_table()) {
            auto t = *_t;
            READ_KEY(t, lost_thres);
            READ_KEY(t, tracking_thres);
            READ_KEY(t, matcher_gate);
            READ_KEY(t, outpost_matcher_gate);

            // Robot model noise parameters
            if (const auto _r = t["robot_model"].as_table()) {
                auto r = *_r;
                READ_KEY(r, robot_params.sigma_a_xy);
                READ_KEY(r, robot_params.sigma_a_z);
                READ_KEY(r, robot_params.sigma_a_yaw);
                READ_KEY(r, robot_params.sigma_r0);
                READ_KEY(r, robot_params.sigma_h);
                READ_KEY(r, robot_params.meas_dist_k);
                READ_KEY(r, robot_params.yaw_log_k);
            }

            // Outpost model noise parameters
            if (const auto _o = t["outpost_model"].as_table()) {
                auto o = *_o;
                READ_KEY(o, outpost_params.sigma_q_xy);
                READ_KEY(o, outpost_params.sigma_q_z);
                READ_KEY(o, outpost_params.sigma_a_yaw);
                READ_KEY(o, outpost_params.meas_dist_k);
                READ_KEY(o, outpost_params.yaw_log_k);
            }

            return {};
        }
        // Optional section - use defaults
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
    double side_angle{15.0};

    /// Minimum angular velocity for armor switching (rad/s)
    double min_switching_v_yaw{1.0};

    /// State transfer counter threshold
    int transfer_thresh{5};

    /// Trajectory compensator configuration
    TrajectoryConfig trajectory;

    /// Compensator type: "ideal" or "resistance"
    std::string compensator_type{"resistance"};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["armor_solver"].as_table()) {
            auto t = *_t;
            READ_KEY(t, shooting_range_w);
            READ_KEY(t, shooting_range_h);
            READ_KEY(t, max_tracking_v_yaw);
            READ_KEY(t, prediction_delay);
            READ_KEY(t, side_angle);
            READ_KEY(t, min_switching_v_yaw);
            READ_KEY(t, transfer_thresh);
            READ_KEY(t, compensator_type);
            return {};
        }
        // Optional section - use defaults
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
        // All sections are optional - use defaults if not present
        if (auto result = tracker.read_from(tbl); !result) {
            // Tracker config is optional, only fail on parse error
            if (result.error() != "tracker section not found") {
                return result;
            }
        }

        if (auto result = solver.read_from(tbl); !result) {
            if (result.error() != "armor_solver section not found") {
                return result;
            }
        }

        if (auto result = trajectory.read_from(tbl); !result) {
            if (result.error() != "trajectory section not found") {
                return result;
            }
        }

        // Copy trajectory config to solver if solver didn't have its own
        solver.trajectory = trajectory;

        return {};
    }
};

} // namespace fcs::L3
