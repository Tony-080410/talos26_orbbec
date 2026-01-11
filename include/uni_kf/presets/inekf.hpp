/// @file inekf.hpp
/// @brief Invariant Extended Kalman Filter presets
///
/// Provides type aliases for InEKF on SE(3), SE₂(3), and augmented state spaces.

#pragma once

#include "../core/filter.hpp"

namespace uni_kf {

// ============================================================================
// InEKF on SE(3) - 6-DOF pose
// ============================================================================

/// @brief Right-invariant EKF on SE(3)
using RInEKF_SE3 = LieEKF<group::SEn3<double, 1>, LieSide::Right>;

/// @brief Left-invariant EKF on SE(3)
using LInEKF_SE3 = LieEKF<group::SEn3<double, 1>, LieSide::Left>;

/// @brief Default InEKF on SE(3) (right-invariant)
using InEKF_SE3 = RInEKF_SE3;

// ============================================================================
// InEKF on SE₂(3) - Extended pose (rotation, velocity, position)
// ============================================================================

/// @brief Right-invariant EKF on SE₂(3)
using RInEKF_SE23 = InEKF_SE23<LieSide::Right>;

/// @brief Left-invariant EKF on SE₂(3)
using LInEKF_SE23 = InEKF_SE23<LieSide::Left>;

// ============================================================================
// InEKF on SE₂(3) × ℝⁿ - Augmented state (matching existing InEKF)
// ============================================================================

/// @brief InCKF for robot tracking: SE₂(3) × ℝ⁴ [v_yaw, r0, r1, h]
using InCKF_Robot = InCKF_SE23_Augmented<4, LieSide::Right>;

/// @brief InCKF for outpost tracking: SE₂(3) × ℝ³ [v_yaw, dz1, dz2]
using InCKF_Outpost = InCKF_SE23_Augmented<3, LieSide::Right>;

// ============================================================================
// InEKF on SO(3) - Orientation only
// ============================================================================

/// @brief Right-invariant EKF on SO(3)
using RInEKF_SO3 = LieEKF<group::SO3<double>, LieSide::Right>;

/// @brief Left-invariant EKF on SO(3)
using LInEKF_SO3 = LieEKF<group::SO3<double>, LieSide::Left>;

/// @brief Default InEKF on SO(3) (right-invariant)
using InEKF_SO3 = RInEKF_SO3;

// ============================================================================
// Factory helpers
// ============================================================================

/// @brief Create InCKF_Robot initialized at given pose
/// @param position Initial position [x, y, z]
/// @param yaw Initial yaw angle (radians)
/// @param pos_std Position uncertainty std dev
/// @param vel_std Velocity uncertainty std dev
/// @param yaw_std Yaw uncertainty std dev
[[nodiscard]] inline auto make_robot_inekf(
    const Eigen::Vector3d& position, double yaw, double pos_std = 0.1, double vel_std = 0.1,
    double yaw_std = 0.1) noexcept -> InCKF_Robot {
    using Space = typename InCKF_Robot::Space;
    using Point = typename Space::Point;
    using Cov   = typename InCKF_Robot::Covariance;

    // SE₂(3) part
    group::SEn3<double, 2> se23;
    typename group::SEn3<double, 2>::IsometriesType t;
    t[0]              = Eigen::Vector3d::Zero(); // velocity
    t[1]              = position;
    Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    se23              = group::SEn3<double, 2>(R, t);

    // Extra state part: [v_yaw, r0, r1, h]
    VecN<4, double> extra;
    extra.data << 0.0, 0.22, 0.22, 0.0; // Default geometry

    // Combined point
    Point state{se23, extra};

    // Covariance
    Cov P = Cov::Zero();
    // Rotation (yaw only)
    P(2, 2) = yaw_std * yaw_std;
    // Velocity
    for (int i = 3; i < 6; ++i) {
        P(i, i) = vel_std * vel_std;
    }
    // Position
    for (int i = 6; i < 9; ++i) {
        P(i, i) = pos_std * pos_std;
    }
    // Extra states
    P(9, 9)   = 1.0;  // v_yaw
    P(10, 10) = 0.01; // r0
    P(11, 11) = 0.01; // r1
    P(12, 12) = 0.01; // h

    return InCKF_Robot{state, P};
}

} // namespace uni_kf
