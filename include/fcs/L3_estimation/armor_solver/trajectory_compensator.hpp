#pragma once

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <numbers>
#include <string_view>
#include <vector>

namespace fcs::L3 {

enum class TrajectoryCompensatorType {
    Ideal,
    Resistance,
};

/// Abstract trajectory compensator for ballistic compensation
class TrajectoryCompensator {
public:
    virtual ~TrajectoryCompensator() = default;

    /// Compensate the trajectory of the bullet, output compensated pitch (rad).
    /// This implementation matches rm_utils/math/trajectory_compensator.cpp (atvision_ws).
    [[nodiscard]] bool
        compensate(const Eigen::Vector3d& target_position, double& pitch) const noexcept {
        const double target_height = target_position.z();
        double iterative_height    = target_height;
        double impact_height       = 0.0;
        const double distance      = std::sqrt(
            target_position.x() * target_position.x() + target_position.y() * target_position.y());
        double angle                  = std::atan2(target_height, distance);
        double dh                     = 0.0;
        constexpr double kDhThreshold = 0.01;
        constexpr double kMaxAngle    = std::numbers::pi / 2.5;

        for (int i = 0; i < iteration_times; ++i) {
            angle = std::atan2(iterative_height, distance);
            if (std::abs(angle) > kMaxAngle) {
                break;
            }

            impact_height = calculate_trajectory(distance, angle);
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
    [[nodiscard]] std::vector<std::pair<double, double>>
        get_trajectory(double max_distance, double pitch) const noexcept {
        std::vector<std::pair<double, double>> trajectory;

        if (max_distance < 0) {
            return trajectory;
        }

        for (double x = 0; x < max_distance; x += 0.03) {
            trajectory.emplace_back(x, calculate_trajectory(x, pitch));
        }

        return trajectory;
    }

    // Configuration
    double velocity{15.0};
    int iteration_times{20};
    double gravity{9.8};
    double resistance{0.01};

protected:
    /// Calculate trajectory z position at horizontal distance x
    [[nodiscard]] virtual double calculate_trajectory(double x, double pitch) const noexcept = 0;
};

/// Ideal trajectory compensator (no air resistance)
class IdealCompensator final : public TrajectoryCompensator {
public:
    [[nodiscard]] double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept override {
        const double distance = std::sqrt(
            target_position.x() * target_position.x() + target_position.y() * target_position.y());
        const double angle = std::atan2(target_position.z(), distance);
        const double t     = distance / (velocity * std::cos(angle));
        return t;
    }

protected:
    [[nodiscard]] double calculate_trajectory(double x, double pitch) const noexcept override {
        const double t = x / (velocity * std::cos(pitch));
        const double y = velocity * std::sin(pitch) * t - 0.5 * gravity * t * t;
        return y;
    }
};

/// Resistance trajectory compensator (with air resistance)
class ResistanceCompensator final : public TrajectoryCompensator {
public:
    [[nodiscard]] double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept override {
        const double r        = resistance < 1e-4 ? 1e-4 : resistance;
        const double distance = std::sqrt(
            target_position.x() * target_position.x() + target_position.y() * target_position.y());
        const double angle = std::atan2(target_position.z(), distance);
        const double t     = (std::exp(r * distance) - 1.0) / (r * velocity * std::cos(angle));
        return t;
    }

protected:
    [[nodiscard]] double calculate_trajectory(double x, double pitch) const noexcept override {
        const double r = resistance < 1e-4 ? 1e-4 : resistance;
        const double t = (std::exp(r * x) - 1.0) / (r * velocity * std::cos(pitch));
        const double y = velocity * std::sin(pitch) * t - 0.5 * gravity * t * t;
        return y;
    }
};

/// Factory function to create compensator
[[nodiscard]] inline std::unique_ptr<TrajectoryCompensator>
    create_compensator(TrajectoryCompensatorType type) noexcept {
    if (type == TrajectoryCompensatorType::Ideal) {
        return std::make_unique<IdealCompensator>();
    }
    return std::make_unique<ResistanceCompensator>();
}

} // namespace fcs::L3
