#pragma once

#include <Eigen/Core>
#include <expected>
#include <string>

#include "fcs/L3_estimation/energy_meter/types.hpp"
#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/aimer/aimer_utils.hpp"
#include "fcs/L4_planning/aimer/types.hpp"
#include "fcs/L4_planning/config.hpp"
#include "fcs/L4_planning/trajectory/solver_interfaces.hpp"

namespace fcs::L4 {

/// Aimer - predicts aim point for all target types (Robot/Outpost/Rune)
///
/// Single class with overloaded aim() methods for different target types.
/// Uses composition with shared utility functions from aimer_utils.hpp.
class Aimer {
public:
    /// Constructor
    explicit Aimer(
        const AimerConfig& config, std::unique_ptr<trajectory::IBallisticSolver> solver) noexcept
        : config_(config)
        , solver_(std::move(solver)) {}

    /// Compute aim point for robot target (Infantry/Hero/Sentry)
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const L3::RobotTargetState& target, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
            uint64_t measurement_ns, uint64_t current_ns, double bullet_speed) const noexcept;

    /// Compute aim point for outpost target
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const L3::OutpostTargetState& target, fast_tf::FuckOdom gimbal,
            fast_tf::FuckOdom muzzle, uint64_t measurement_ns, uint64_t current_ns,
            double bullet_speed) const noexcept;

    /// Compute aim point for rune target (EnergyMeterState)
    [[nodiscard]] std::expected<TargetPrediction, std::string>
        aim(const energy_meter::EnergyMeterState& state, fast_tf::FuckOdom gimbal,
            fast_tf::FuckOdom muzzle, uint64_t current_ns, double bullet_speed) const noexcept;

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
        const Eigen::Vector3d& predicted_pos, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
        const trajectory::IBallisticSolver& solver, double bullet_speed) noexcept {
        const Eigen::Vector2d aim_angles =
            compute_aim_point(solver, predicted_pos - muzzle->translation(), bullet_speed);
        const auto muzzle_des = math_fuxk::rpy(0.0, aim_angles.y(), aim_angles.x()).so3();
        const auto idk        = muzzle->so3().inv() * muzzle_des;
        const auto x          = gimbal->so3() * idk;
        const auto [r, p, y]  = math_fuxk::rpy(x).rpy();
        return {y, p};
    }

    // ========================================================================
    // Generic aim() - Template method for common aiming logic
    // ========================================================================

    /// Generic aim implementation for targets with prediction (Robot/Outpost)
    template <class TargetType>
    [[nodiscard]] std::expected<TargetPrediction, std::string> aim_generic(
        const TargetType& target, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
        uint64_t measurement_ns, uint64_t current_ns, double bullet_speed) const noexcept;

    /// Generic aim implementation for targets without prediction (Rune)
    template <class TargetType>
    [[nodiscard]] std::expected<TargetPrediction, std::string> aim_generic_no_prediction(
        const TargetType& target, fast_tf::FuckOdom gimbal, fast_tf::FuckOdom muzzle,
        uint64_t current_ns, double bullet_speed) const noexcept;

    /// Predict robot target position at future time dt
    [[nodiscard]] static Eigen::Vector3d
        predict_position(const L3::RobotTargetState& target, double dt) noexcept;

    /// Predict outpost target position at future time dt
    [[nodiscard]] static Eigen::Vector3d
        predict_position(const L3::OutpostTargetState& target, double dt) noexcept;

    AimerConfig config_;
    std::unique_ptr<trajectory::IBallisticSolver> solver_;
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
