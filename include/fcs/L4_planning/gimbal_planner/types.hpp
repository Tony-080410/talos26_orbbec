#pragma once

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/core/channel_topics.hpp"

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <vector>

namespace fcs::L4 {

// ============================================================================
// MPC Debug Data (Optional, for Visualization)
// ============================================================================

/// MPC debug data (optional, for Foxglove visualization)
///
/// Contains detailed MPC solver information for debugging and visualization.
/// This data is expensive to compute and should only be populated when
/// debugging is enabled.
struct MpcDebugData {
    // Time axis
    double mpc_dt{0.0};
    int mpc_current_idx{0};

    // Reference trajectory
    std::vector<double> ref_yaw;
    std::vector<double> ref_yaw_vel;
    std::vector<double> ref_pitch;
    std::vector<double> ref_pitch_vel;

    // Optimized trajectory
    std::vector<double> opt_yaw;
    std::vector<double> opt_yaw_vel;
    std::vector<double> opt_pitch;
    std::vector<double> opt_pitch_vel;

    // Control sequence
    std::vector<double> ctrl_yaw_acc;
    std::vector<double> ctrl_pitch_acc;

    // Solver info
    int mpc_iterations{0};
    double mpc_solve_time_us{0.0};
    bool mpc_converged{false};

    // Cost breakdown
    std::vector<double> cost_yaw_pos;
    std::vector<double> cost_yaw_vel;
    std::vector<double> cost_pitch_pos;
    std::vector<double> cost_pitch_vel;
    double total_cost{0.0};

    // Constraint margins
    std::vector<double> constraint_yaw_acc_margin;
    std::vector<double> constraint_pitch_acc_margin;
};

// ============================================================================
// Visualization Debug Data (Optional)
// ============================================================================

/// Visualization debug data (optional)
///
/// Contains data for visualization purposes (e.g., Foxglove plots).
/// This data is optional and should only be populated when visualization
/// is enabled.
struct VisualizationDebugData {
    std::optional<Eigen::Vector3d> predicted_position; ///< For visualization
    uint64_t predicted_future{0};                      ///< Predicted arrival time
    std::vector<std::pair<double, double>> trajectory; ///< Ballistic trajectory (x, z)
};

// ============================================================================
// Gimbal Plan - Final Output from L4 Planner
// ============================================================================

/// Gimbal plan - final output from L4 planner
///
/// This is the final output of the L4 planning pipeline, containing
/// optimized control outputs with optional debug data.
///
/// The planner takes TargetPrediction from the Aimer, optimizes the
/// trajectory using MPC, and produces this GimbalPlan with feedforward
/// terms (velocities, accelerations) for smooth control.
struct GimbalPlan {
    // ========================================================================
    // Control Output (always available)
    // ========================================================================
    uint64_t timestamp_ns{0};

    double yaw{0.0};            ///< Optimized yaw angle
    double pitch{0.0};          ///< Optimized pitch angle
    double yaw_vel{0.0};        ///< Feedforward yaw velocity
    double pitch_vel{0.0};      ///< Feedforward pitch velocity
    double yaw_acc{0.0};        ///< Feedforward yaw acceleration
    double pitch_acc{0.0};      ///< Feedforward pitch acceleration

    bool fire_advice{false};    ///< Final fire advice (may override aimer's)
    double tracking_error{0.0}; ///< MPC tracking error

    // ========================================================================
    // Target Information (for reference)
    // ========================================================================
    double distance{0.0};
    std::optional<Eigen::Vector3d> predicted_position;
    uint64_t predicted_future{0};

    // ========================================================================
    // Aimer's Original Advice (for debugging/comparison)
    // ========================================================================
    bool aimer_fire_advice{false}; ///< Aimer's fire advice
    double aim_yaw{0.0};           ///< Aimer's aim angles
    double aim_pitch{0.0};

    // ========================================================================
    // Optional Debug Data (only filled when debugging)
    // ========================================================================
    std::optional<VisualizationDebugData> viz_debug;
    std::optional<MpcDebugData> mpc_debug;
};

// ============================================================================
// L4 Channel Topics (for ECS)
// ============================================================================

// Re-export L3 types for convenience
using fcs::L3::TrackerOutput;
using fcs::L3::TrackerOutputChannelTopic;

} // namespace fcs::L4
