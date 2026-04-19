#pragma once

#include "core/armor_types.hpp"

#include <Eigen/Core>

#include <vector>

namespace fcs::core::trajectory {

/// Reference trajectory for weapon control.
///
/// Contains the state matrix [yaw, yaw_rate, pitch, pitch_rate] × horizon,
/// plus metadata needed for fire gate computation and debug visualization.
///
/// Built by L4 planning layer, consumed by L5 weapon layer.
struct ReferenceTrajectory {
    using StateMatrix = Eigen::Matrix<double, 4, Eigen::Dynamic>;

    /// State matrix: 4 rows × horizon columns.
    /// Row 0: yaw (relative to yaw_origin)
    /// Row 1: yaw_rate (rad/s)
    /// Row 2: pitch (rad)
    /// Row 3: pitch_rate (rad/s)
    StateMatrix state;

    /// Distance to target at each horizon step (meters).
    std::vector<double> distances;

    /// Projectile time-of-flight at each step (seconds).
    std::vector<double> time_of_flights;

    /// Predicted world-frame positions at each step (odom frame).
    std::vector<Eigen::Vector3d> positions;

    /// Selected armor ID at each horizon step.
    std::vector<int> selected_armor_ids;

    /// Rough (pre-refinement) selected armor ID at each step.
    std::vector<int> rough_selected_armor_ids;

    /// Target armor type for fire-window shaping.
    ArmorType armor_type{ArmorType::Invalid};

    /// Yaw angle origin for normalization (yaw at center horizon step).
    double yaw_origin{0.0};

    /// Total horizon length.
    [[nodiscard]] int horizon() const noexcept { return static_cast<int>(state.cols()); }
};

} // namespace fcs::core::trajectory
