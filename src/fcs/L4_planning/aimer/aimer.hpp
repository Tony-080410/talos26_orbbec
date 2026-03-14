#pragma once

#include <Eigen/Core>
#include <expected>
#include <string>

#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/aimer/aimer_utils.hpp"
#include "L4_planning/aimer/types.hpp"
#include "L4_planning/config.hpp"
#include "core/trajectory/solver/solver_interfaces.hpp"

namespace fcs::L4 {

/// Aimer - predicts aim point for all target types (Robot/Outpost/Rune)
///
/// Single class with overloaded aim() methods for different target types.
/// Uses composition with shared utility functions from aimer_utils.hpp.
class Aimer {
public:
    using GimbalTransform =
        fast_tf::TransformMatrixd<fast_tf::gimbal_pitch_fuxk_frame, fast_tf::odom_fuxk_frame>;
    using MuzzleTransform =
        fast_tf::TransformMatrixd<fast_tf::muzzle_link_fuxk_frame, fast_tf::odom_fuxk_frame>;

    /// Constructor
    explicit Aimer(const AimerConfig& config) noexcept
        : config_(config) {}

    /// Compute aim point for robot target (Infantry/Hero/Sentry)
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const L3::RobotTargetState& target, const GimbalTransform& gimbal,
            const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
            double bullet_speed,
            const core::trajectory::solver::TrajectorySolver& solver) const noexcept;

    /// Compute aim point for robot target with extra delay on top of AimerConfig::delay.
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const L3::RobotTargetState& target, const GimbalTransform& gimbal,
            const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
            double extra_prediction_delay, double bullet_speed,
            const core::trajectory::solver::TrajectorySolver& solver) const noexcept;

    /// Compute aim point for outpost target
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const L3::OutpostTargetState& target, const GimbalTransform& gimbal,
            const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
            double bullet_speed,
            const core::trajectory::solver::TrajectorySolver& solver) const noexcept;

    /// Compute aim point for outpost target with extra delay on top of AimerConfig::delay.
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const L3::OutpostTargetState& target, const GimbalTransform& gimbal,
            const MuzzleTransform& muzzle, uint64_t measurement_ns, uint64_t current_ns,
            double extra_prediction_delay, double bullet_speed,
            const core::trajectory::solver::TrajectorySolver& solver) const noexcept;

    /// Compute aim point for rune target (EnergyMeterState)
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const energy_meter::EnergyMeterState& state, const GimbalTransform& gimbal,
            const MuzzleTransform& muzzle, uint64_t current_ns, double bullet_speed,
            const core::trajectory::solver::TrajectorySolver& solver) const noexcept;

private:
    // ========================================================================
    // Target type utility
    // ========================================================================

    /// Get target type for a given target state type
    template <class TargetType>
    [[nodiscard]] static constexpr AimerTargetType get_target_type() noexcept;

    // ========================================================================
    // Common aim angle computation (shared by all target types)
    // ========================================================================

    /// Common gimbal-based aim angle computation for all target types
    [[nodiscard]] static std::pair<double, double> compute_gimbal_aim_angles(
        const Eigen::Vector3d& predicted_pos, const GimbalTransform& gimbal,
        const MuzzleTransform& muzzle, const core::trajectory::solver::TrajectorySolver& solver,
        double bullet_speed) noexcept {
        const Eigen::Vector2d aim_angles =
            compute_aim_point(solver, predicted_pos - muzzle.translation(), bullet_speed);
        const auto desired_muzzle = math_fuxk::rpy(0.0, aim_angles.y(), aim_angles.x()).so3();
        const auto muzzle_delta   = muzzle.so3().inv() * desired_muzzle;
        const auto desired_gimbal = gimbal.so3() * muzzle_delta;
        const auto [r, p, y]      = math_fuxk::rpy(desired_gimbal).rpy();
        return {y, p};
    }

    // ========================================================================
    // Generic aim() - Template method for common aiming logic
    // ========================================================================

    /// Generic aim implementation for targets with prediction (Robot/Outpost)
    template <class TargetType>
    [[nodiscard]] std::expected<TargetPrediction, std::string> aim_generic(
        const TargetType& target, const GimbalTransform& gimbal, const MuzzleTransform& muzzle,
        uint64_t measurement_ns, uint64_t current_ns, double prediction_delay, double bullet_speed,
        const core::trajectory::solver::TrajectorySolver& solver) const noexcept;

    /// Generic aim implementation for targets without prediction (Rune)
    template <class TargetType>
    [[nodiscard]] std::expected<TargetPrediction, std::string> aim_generic_no_prediction(
        const TargetType& target, const GimbalTransform& gimbal, const MuzzleTransform& muzzle,
        uint64_t current_ns, double bullet_speed,
        const core::trajectory::solver::TrajectorySolver& solver) const noexcept;

    /// Predict robot target position at future time dt
    [[nodiscard]] static Eigen::Vector3d
        predict_position(const L3::RobotTargetState& target, double dt) noexcept;

    /// Predict outpost target position at future time dt
    [[nodiscard]] static Eigen::Vector3d
        predict_position(const L3::OutpostTargetState& target, double dt) noexcept;

    AimerConfig config_;
};

// ========================================================================
// get_target_type specializations
// ========================================================================

template <>
inline constexpr AimerTargetType Aimer::get_target_type<L3::RobotTargetState>() noexcept {
    return AimerTargetType::Robot;
}

template <>
inline constexpr AimerTargetType Aimer::get_target_type<L3::OutpostTargetState>() noexcept {
    return AimerTargetType::Outpost;
}

template <>
inline constexpr AimerTargetType Aimer::get_target_type<energy_meter::EnergyMeterState>() noexcept {
    return AimerTargetType::Rune;
}

} // namespace fcs::L4
