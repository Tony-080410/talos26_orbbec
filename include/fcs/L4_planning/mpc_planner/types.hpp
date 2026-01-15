#pragma once

#include <Eigen/Core>
#include <optional>
#include <vector>

namespace fcs::L4 {

/// MPC planner output structure
///
/// Contains the planned gimbal control and fire decision from MPC optimization.
struct MpcPlannerOutput {
    /// Timestamp of this plan (nanoseconds)
    uint64_t timestamp_ns{0};

    /// Target gimbal yaw angle (radians, in odom frame)
    double yaw{0.0};

    /// Target gimbal pitch angle (radians, in odom frame)
    double pitch{0.0};

    /// Planned yaw angular velocity (rad/s) - feedforward term
    double yaw_vel{0.0};

    /// Planned pitch angular velocity (rad/s) - feedforward term
    double pitch_vel{0.0};

    /// Planned yaw angular acceleration (rad/s^2) - feedforward term
    double yaw_acc{0.0};

    /// Planned pitch angular acceleration (rad/s^2) - feedforward term
    double pitch_acc{0.0};

    /// Fire advice based on trajectory tracking error
    bool fire_advice{false};

    /// Tracking error at the lookahead step (for diagnostics)
    double tracking_error{0.0};

    /// Distance to target (meters)
    double distance{0.0};

    /// Predicted time when bullet arrives (nanoseconds, absolute)
    uint64_t predicted_future{0};

    /// Predicted armor/center position at bullet arrival time
    std::optional<Eigen::Vector3d> predicted_position;

    /// Planned ballistic trajectory points for visualization (x, z in muzzle frame, meters)
    std::vector<std::pair<double, double>> planned_trajectory;

    // =========================================================================
    // MPC Debug Data (for Foxglove visualization)
    // =========================================================================
    // Time axis: t[k] = k * dt, relative to current time
    // Reference trajectory (input): what MPC is trying to track
    // Optimized trajectory (output): what MPC produced
    // Control sequence: feedforward accelerations

    /// MPC time step (seconds)
    double mpc_dt{0.0};

    /// Index of current time in horizon (half_horizon)
    int mpc_current_idx{0};

    /// Reference yaw trajectory (horizon points): target for MPC
    std::vector<double> ref_yaw;

    /// Reference yaw velocity trajectory
    std::vector<double> ref_yaw_vel;

    /// Reference pitch trajectory (horizon points): target for MPC
    std::vector<double> ref_pitch;

    /// Reference pitch velocity trajectory
    std::vector<double> ref_pitch_vel;

    /// Optimized yaw trajectory (MPC output)
    std::vector<double> opt_yaw;

    /// Optimized yaw velocity trajectory (MPC output)
    std::vector<double> opt_yaw_vel;

    /// Optimized pitch trajectory (MPC output)
    std::vector<double> opt_pitch;

    /// Optimized pitch velocity trajectory (MPC output)
    std::vector<double> opt_pitch_vel;

    /// Yaw acceleration control sequence (feedforward)
    std::vector<double> ctrl_yaw_acc;

    /// Pitch acceleration control sequence (feedforward)
    std::vector<double> ctrl_pitch_acc;
};

} // namespace fcs::L4
