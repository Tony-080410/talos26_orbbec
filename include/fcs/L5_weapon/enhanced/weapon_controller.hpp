#pragma once

#include "L5_weapon/config.hpp"
#include "L5_weapon/fire_control.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"

#include <memory>

namespace fcs::L5 {

// ============================================================================
// Enhanced Weapon Control with Bullet Compensation
// ============================================================================

/// @brief Enhanced weapon control with sophisticated firing logic
class EnhancedWeaponController {
public:
    explicit EnhancedWeaponController(
        const WeaponControllerConfig& config,
        std::unique_ptr<fcs::L4::TrajectoryCompensator> compensator) noexcept
        : config_(config)
        , compensator_(std::move(compensator)) {
        // Trajectory compensator parameters are now owned and managed by L4
    }

    /// @brief Generate enhanced weapon command from L4 planning output
    /// @param plan Gimbal plan from L4 planner
    /// @param target_distance Target distance for bullet timing
    /// @return Enhanced weapon command with bullet compensation
    [[nodiscard]] WeaponCommand
        generate_command(const fcs::L4::GimbalPlan& plan, double target_distance) noexcept;

    /// @brief Update trajectory compensator parameters
    void
        update_compensator_params(double bullet_speed, double gravity, double resistance) noexcept {
        if (compensator_) {
            compensator_->velocity   = bullet_speed;
            compensator_->gravity    = gravity;
            compensator_->resistance = resistance;
            compensator_->on_params_updated();
        }
    }

    /// @brief Update trajectory compensator parameters from L4 plan
    void update_compensator_from_plan([[maybe_unused]] const fcs::L4::GimbalPlan& plan) noexcept {
        // Note: L5 doesn't own trajectory parameters anymore
        // This method can be used if we need to extract trajectory info from L4 plan
        // Currently, we let L4 handle all trajectory calculations
    }

private:
    WeaponControllerConfig config_;
    std::unique_ptr<fcs::L4::TrajectoryCompensator> compensator_;

    /// @brief Calculate bullet time of flight for target distance
    [[nodiscard]] double calculate_time_of_flight(double distance) const noexcept;

    /// @brief Calculate angular rates for predictive control
    [[nodiscard]] std::pair<double, double>
        calculate_angular_rates(const fcs::L4::GimbalPlan& plan, double dt) const noexcept;
};

} // namespace fcs::L5
