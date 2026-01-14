#pragma once

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/core/channel_tags.hpp"

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <vector>

namespace fcs::L4 {

// ============================================================================
// L4 Planning Output Types
// ============================================================================

/// @brief Gimbal plan containing targeting information
/// This struct represents the strategic decision output from the L4 planner
struct GimbalPlan {
    uint64_t timestamp_ns{0};

    // Target gimbal angles (radians, in odom frame)
    double yaw{0};
    double pitch{0};

    // Angular differences from current pose
    double yaw_diff{0};
    double pitch_diff{0};

    // Feedforward terms (from MPC planner)
    double yaw_vel{0};   // Planned yaw angular velocity (rad/s)
    double pitch_vel{0}; // Planned pitch angular velocity (rad/s)
    double yaw_acc{0};   // Planned yaw angular acceleration (rad/s^2)
    double pitch_acc{0}; // Planned pitch angular acceleration (rad/s^2)

    // Target info
    double distance{0};
    bool fire_advice{false};
    double confidence{0.0};     // Confidence level in the fire decision
    double tracking_error{0.0}; // MPC tracking error (radians)
    bool track_center;

    // Predicted armor position (for visualization)
    uint64_t predicted_future;
    std::optional<Eigen::Vector3d> predicted_position;

    // Trajectory points (x, z) in muzzle_link frame (for visualization)
    std::vector<std::pair<double, double>> trajectory;

    // =========================================================================
    // MPC Debug Data (for Foxglove visualization)
    // =========================================================================
    // Time axis: t[k] = (k - mpc_current_idx) * mpc_dt, relative to current time

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

// ============================================================================
// L4 Channel Tags (for ECS)
// ============================================================================

// Re-export L3 types for convenience
using fcs::L3::TrackerOutput;
using fcs::L3::TrackerOutputChannelTag;

} // namespace fcs::L4
