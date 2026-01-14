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

    /// Whether tracking center (vs individual armor)
    bool track_center{false};

    /// Predicted time when bullet arrives (nanoseconds, absolute)
    uint64_t predicted_future{0};

    /// Predicted armor/center position at bullet arrival time
    std::optional<Eigen::Vector3d> predicted_position;

    /// Planned trajectory points for visualization (yaw, pitch pairs)
    std::vector<std::pair<double, double>> planned_trajectory;
};

} // namespace fcs::L4
