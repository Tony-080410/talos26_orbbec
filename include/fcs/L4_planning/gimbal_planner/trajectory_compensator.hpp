#pragma once

// Public interface header (thin).
// Concrete implementations live in `trajectory_compensator_impl.hpp` and are built into the `fcs`
// library. Prefer including this header from other modules.

#include "fcs/L4_planning/trajectory_config.hpp"

#include <Eigen/Core>

#include <cmath>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

namespace fcs::L4 {

/// Abstract trajectory compensator for ballistic compensation
class TrajectoryCompensator {
public:
    virtual ~TrajectoryCompensator() = default;

    /// Hook called after parameters are updated (e.g. rebuild caches).
    virtual void on_params_updated() noexcept {}

    /// Compensate the trajectory of the bullet, output compensated yaw/pitch (rad).
    /// Default implementation compensates pitch only (yaw stays line-of-sight).
    [[nodiscard]] virtual bool compensate(
        const Eigen::Vector3d& target_position, double& yaw, double& pitch) const noexcept {
        yaw = std::atan2(target_position.y(), target_position.x());
        return compensate_pitch(target_position, pitch, velocity);
    }

    /// Compensate with specified bullet velocity (const-safe, no mutation)
    [[nodiscard]] virtual bool compensate(
        const Eigen::Vector3d& target_position, double& yaw, double& pitch,
        double bullet_velocity) const noexcept {
        yaw = std::atan2(target_position.y(), target_position.x());
        return compensate_pitch(target_position, pitch, bullet_velocity);
    }

    /// Compensate the trajectory of the bullet, output compensated pitch (rad).
    /// This implementation matches rm_utils/math/trajectory_compensator.cpp (atvision_ws).
    [[nodiscard]] bool
        compensate(const Eigen::Vector3d& target_position, double& pitch) const noexcept {
        return compensate_pitch(target_position, pitch, velocity);
    }

    /// Compensate pitch with specified bullet velocity (const-safe, core implementation)
    [[nodiscard]] virtual bool compensate_pitch(
        const Eigen::Vector3d& target_position, double& pitch,
        double bullet_velocity) const noexcept {
        constexpr double kDhThreshold = 0.01;                   ///< Convergence threshold (m)
        constexpr double kMaxAngle    = std::numbers::pi / 2.5; ///< Max valid pitch (rad)

        const double target_height = target_position.z();
        double iterative_height    = target_height;
        double impact_height       = 0.0;
        const double distance      = std::hypot(target_position.x(), target_position.y());
        double angle               = std::atan2(target_height, distance);
        double dh                  = 0.0;

        for (int i = 0; i < iteration_times; ++i) {
            angle = std::atan2(iterative_height, distance);
            if (std::abs(angle) > kMaxAngle) {
                break;
            }

            impact_height = calculate_trajectory(distance, angle, bullet_velocity);
            dh            = target_height - impact_height;
            if (std::abs(dh) < kDhThreshold) {
                break;
            }
            iterative_height += dh;
        }

        if (std::abs(dh) > kDhThreshold || std::abs(angle) > kMaxAngle) {
            return false;
        }

        pitch = angle;
        return true;
    }

    /// Get flying time to target
    [[nodiscard]] virtual double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept = 0;

    /// Get trajectory points for visualization
    [[nodiscard]] virtual std::vector<std::pair<double, double>>
        get_trajectory(double max_distance, double pitch) const noexcept {
        constexpr double kTrajectoryStep = 0.03; ///< Trajectory visualization step (m)

        if (max_distance < 0) {
            return {};
        }

        const size_t num_points = static_cast<size_t>(max_distance / kTrajectoryStep) + 1;
        std::vector<std::pair<double, double>> trajectory;
        trajectory.reserve(num_points);

        for (double x = 0; x < max_distance; x += kTrajectoryStep) {
            trajectory.emplace_back(x, calculate_trajectory(x, pitch));
        }

        return trajectory;
    }

    // Configuration
    double velocity{15.0};
    int iteration_times{20};
    double gravity{9.8};
    double resistance{0.01};

    // RK4 (quadratic drag) physical parameters
    double cd{0.47};
    double rho{1.225};
    double diameter{0.017};
    double mass{0.0032};
    double rk4_dt{0.001};
    int rk4_min_steps{50};
    int rk4_max_steps{2000};

protected:
    /// Calculate trajectory z position at horizontal distance x
    [[nodiscard]] virtual double calculate_trajectory(double x, double pitch) const noexcept {
        return calculate_trajectory(x, pitch, velocity);
    }

    /// Calculate trajectory z position at horizontal distance x with specified velocity
    [[nodiscard]] virtual double
        calculate_trajectory(double x, double pitch, double bullet_velocity) const noexcept = 0;
};

/// Factory function to create compensator (implemented in `trajectory_compensator.cpp`).
[[nodiscard]] std::unique_ptr<TrajectoryCompensator>
    create_compensator(TrajectoryCompensatorType type) noexcept;

} // namespace fcs::L4
