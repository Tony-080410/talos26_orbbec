#pragma once

#include <Eigen/Core>
#include <cmath>
#include <memory>
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

    /// Compensate pitch for gravity drop
    /// @param target_position Target position (muzzle-relative)
    /// @param pitch [in/out] Input: initial pitch, Output: compensated pitch
    /// @return true if compensation successful
    [[nodiscard]] bool
        compensate(const Eigen::Vector3d& target_position, double& pitch) const noexcept {
        const double target_height = target_position.z();
        const double distance_xy   = std::sqrt(
            target_position.x() * target_position.x() + target_position.y() * target_position.y());

        if (distance_xy < 0.01) {
            return false; // Too close
        }

        double compensated_pitch = pitch;

        for (int i = 0; i < iteration_times; ++i) {
            const double z     = calculate_trajectory(distance_xy, compensated_pitch);
            const double error = target_height - z;

            if (std::abs(error) < 0.001) {
                break;    // Converged
            }

            // Adjust pitch
            compensated_pitch += std::atan2(error, distance_xy);

            // Clamp to reasonable range
            compensated_pitch = std::clamp(compensated_pitch, -1.0, 1.0);
        }

        pitch = compensated_pitch;
        return true;
    }

    /// Get flying time to target
    [[nodiscard]] virtual double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept = 0;

    /// Get trajectory points for visualization
    [[nodiscard]] std::vector<std::pair<double, double>>
        get_trajectory(double max_distance, double pitch) const noexcept {
        std::vector<std::pair<double, double>> trajectory;
        const double step = max_distance / 50.0;

        for (double x = 0; x <= max_distance; x += step) {
            const double z = calculate_trajectory(x, pitch);
            trajectory.emplace_back(x, z);
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
        return distance / velocity;
    }

protected:
    [[nodiscard]] double calculate_trajectory(double x, double pitch) const noexcept override {
        // Ideal projectile motion: z = x*tan(pitch) - g*x^2 / (2*v^2*cos^2(pitch))
        const double cos_pitch = std::cos(pitch);
        const double t         = x / (velocity * cos_pitch);
        const double z         = x * std::tan(pitch) - 0.5 * gravity * t * t;
        return z;
    }
};

/// Resistance trajectory compensator (with air resistance)
class ResistanceCompensator final : public TrajectoryCompensator {
public:
    [[nodiscard]] double
        get_flying_time(const Eigen::Vector3d& target_position) const noexcept override {
        const double distance = std::sqrt(
            target_position.x() * target_position.x() + target_position.y() * target_position.y());

        // Approximate time with resistance
        // Using simplified exponential decay model
        const double k = resistance;
        if (k < 1e-9) {
            return distance / velocity;
        }

        // t = (1/k) * ln(1 + k*d/v)
        return (1.0 / k) * std::log(1.0 + k * distance / velocity);
    }

protected:
    [[nodiscard]] double calculate_trajectory(double x, double pitch) const noexcept override {
        // Numerical integration with air resistance
        // v' = -k*v, z' = vz, x' = vx

        const double dt    = 0.0001;   // Time step
        const double k     = resistance;
        const double cos_p = std::cos(pitch);
        const double sin_p = std::sin(pitch);

        double vx = velocity * cos_p;
        double vz = velocity * sin_p;
        double px = 0.0;
        double pz = 0.0;

        while (px < x && px < 100.0) { // Max 100m range
            // Apply resistance and gravity
            const double v_mag = std::sqrt(vx * vx + vz * vz);
            if (v_mag < 0.1)
                break;

            const double drag = k * v_mag;
            vx -= drag * vx / v_mag * dt;
            vz -= (gravity + drag * vz / v_mag) * dt;

            px += vx * dt;
            pz += vz * dt;
        }

        return pz;
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
