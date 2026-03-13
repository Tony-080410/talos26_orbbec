#include "core/trajectory/solver/trajectory_solver.hpp"

#include <cmath>

namespace fcs::core::trajectory::solver {

// ============================================================================
// DirectSolver Implementation
// ============================================================================

AimSolution DirectSolver::solve(const Eigen::Vector3d& target_pos, double v0) const noexcept {
    const double target_height = target_pos.z();
    const double distance      = std::hypot(target_pos.x(), target_pos.y());

    if (distance < 1e-6) {
        return {
            .yaw            = std::atan2(target_pos.y(), target_pos.x()),
            .pitch          = 0.0,
            .time_of_flight = 0.0,
            .converged      = false,
            .iterations     = 0};
    }

    // For DirectSolver, we use iterative refinement with the model's compute_impact
    // This works for IdealModel and LinearDragModel which have fast compute_impact
    constexpr int kMaxIterations      = 20;
    constexpr double kHeightTolerance = 0.01;
    constexpr double kMaxPitch        = std::numbers::pi / 2.5;

    double iterative_height = target_height;
    double angle            = std::atan2(target_height, distance);
    double impact_height    = 0.0;
    double dh               = 0.0;
    double tof              = 0.0;
    int iterations          = 0;

    for (int i = 0; i < kMaxIterations; ++i) {
        iterations = i + 1;
        angle      = std::atan2(iterative_height, distance);

        if (std::abs(angle) > kMaxPitch) {
            return {
                .yaw            = std::atan2(target_pos.y(), target_pos.x()),
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        const auto impact = model_->compute_impact(distance, angle, v0);
        if (!impact) {
            return {
                .yaw            = std::atan2(target_pos.y(), target_pos.x()),
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        impact_height = impact->z;
        tof           = impact->tof;
        dh            = target_height - impact_height;

        if (std::abs(dh) < kHeightTolerance) {
            return {
                .yaw            = std::atan2(target_pos.y(), target_pos.x()),
                .pitch          = angle,
                .time_of_flight = tof,
                .converged      = true,
                .iterations     = iterations};
        }

        iterative_height += dh;
    }

    return {
        .yaw            = std::atan2(target_pos.y(), target_pos.x()),
        .pitch          = angle,
        .time_of_flight = tof,
        .converged      = std::abs(dh) < kHeightTolerance,
        .iterations     = iterations};
}

std::vector<std::pair<double, double>>
    DirectSolver::generate_trajectory(double pitch, double v0, double max_distance) const noexcept {
    constexpr double kTrajectoryStep = 0.03;

    if (max_distance < 0.0) {
        return {};
    }

    const size_t num_points = static_cast<size_t>(max_distance / kTrajectoryStep) + 1;
    std::vector<std::pair<double, double>> trajectory;
    trajectory.reserve(num_points);

    for (double x = 0; x < max_distance; x += kTrajectoryStep) {
        const auto impact = model_->compute_impact(x, pitch, v0);
        const double z    = impact ? impact->z : 0.0;
        trajectory.emplace_back(x, z);
    }

    return trajectory;
}

// ============================================================================
// IterativeSolver Implementation
// ============================================================================

AimSolution IterativeSolver::solve(const Eigen::Vector3d& target_pos, double v0) const noexcept {
    return detail::iterative_solve_pitch(
        *model_, target_pos, v0, static_cast<int>(config_.max_iterations), config_.height_tolerance,
        config_.max_pitch);
}

std::vector<std::pair<double, double>> IterativeSolver::generate_trajectory(
    double pitch, double v0, double max_distance) const noexcept {
    constexpr double kTrajectoryStep = 0.03;

    if (max_distance < 0.0) {
        return {};
    }

    const size_t num_points = static_cast<size_t>(max_distance / kTrajectoryStep) + 1;
    std::vector<std::pair<double, double>> trajectory;
    trajectory.reserve(num_points);

    for (double x = 0; x < max_distance; x += kTrajectoryStep) {
        const auto impact = model_->compute_impact(x, pitch, v0);
        const double z    = impact ? impact->z : 0.0;
        trajectory.emplace_back(x, z);
    }

    return trajectory;
}

// ============================================================================
// Utility Functions
// ============================================================================

namespace detail {

AimSolution iterative_solve_pitch(
    const model::BallisticModel& model, const Eigen::Vector3d& target_pos, double v0,
    int max_iterations, double height_tolerance, double max_pitch) noexcept {

    const double target_height = target_pos.z();
    const double distance      = std::hypot(target_pos.x(), target_pos.y());
    const double yaw           = std::atan2(target_pos.y(), target_pos.x());

    if (distance < 1e-6) {
        return {
            .yaw = yaw, .pitch = 0.0, .time_of_flight = 0.0, .converged = false, .iterations = 0};
    }

    double iterative_height = target_height;
    double angle            = std::atan2(target_height, distance);
    double impact_height    = 0.0;
    double dh               = 0.0;
    double tof              = 0.0;
    int iterations          = 0;

    for (int i = 0; i < max_iterations; ++i) {
        iterations = i + 1;
        angle      = std::atan2(iterative_height, distance);

        if (std::abs(angle) > max_pitch) {
            return {
                .yaw            = yaw,
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        const auto impact = model.compute_impact(distance, angle, v0);
        if (!impact) {
            return {
                .yaw            = yaw,
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        impact_height = impact->z;
        tof           = impact->tof;
        dh            = target_height - impact_height;

        if (std::abs(dh) < height_tolerance) {
            return {
                .yaw            = yaw,
                .pitch          = angle,
                .time_of_flight = tof,
                .converged      = true,
                .iterations     = iterations};
        }

        iterative_height += dh;
    }

    return {
        .yaw            = yaw,
        .pitch          = angle,
        .time_of_flight = tof,
        .converged      = std::abs(dh) < height_tolerance,
        .iterations     = iterations};
}

} // namespace detail

} // namespace fcs::core::trajectory::solver
