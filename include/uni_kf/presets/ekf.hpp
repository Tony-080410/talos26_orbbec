/// @file ekf.hpp
/// @brief Extended Kalman Filter presets
///
/// Provides convenient type aliases for common EKF configurations.

#pragma once

#include "../core/filter.hpp"
#include "../spaces/euclidean.hpp"

namespace uni_kf {

// ============================================================================
// EKF Presets (Euclidean state spaces)
// ============================================================================

/// @brief 1D position-velocity EKF
using EKF_1D = EKF<2>;

/// @brief 2D position-velocity EKF
using EKF_2D = EKF<4>;

/// @brief 3D position-velocity EKF
using EKF_3D = EKF<6>;

/// @brief 3D position-velocity-acceleration EKF
using EKF_3D_Accel = EKF<9>;

/// @brief 6-DOF pose EKF (position + orientation as Euler angles)
/// Note: For proper orientation handling, prefer LieEKF<SE3> instead
using EKF_6DOF = EKF<12>;

// ============================================================================
// Factory functions
// ============================================================================

/// @brief Create 3D constant-velocity EKF
/// @param initial_pos Initial position [x, y, z]
/// @param initial_vel Initial velocity [vx, vy, vz]
/// @param pos_std Initial position standard deviation
/// @param vel_std Initial velocity standard deviation
[[nodiscard]] inline auto make_cv3d_ekf(
    const Eigen::Vector3d& initial_pos, const Eigen::Vector3d& initial_vel, double pos_std,
    double vel_std) noexcept -> EKF_3D {
    Vec6 mean;
    mean.data << initial_pos, initial_vel;

    SymmetricMatrix<6, double> cov = SymmetricMatrix<6, double>::Zero();
    for (int i = 0; i < 3; ++i) {
        cov(i, i)         = pos_std * pos_std;
        cov(i + 3, i + 3) = vel_std * vel_std;
    }

    return EKF_3D{mean, cov};
}

} // namespace uni_kf
