#pragma once

#include "L5_weapon/config.hpp"
#include "L5_weapon/fire_control.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"

#include <memory>
#include <spdlog/spdlog.h>

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
        // Note: compensator type tracking is done at system level
    }

    /// @brief Generate enhanced weapon command from L4 planning output
    /// @param plan Gimbal plan from L4 planner
    /// @param target_distance Target distance for bullet timing
    /// @return Enhanced weapon command with bullet compensation
    [[nodiscard]] WeaponCommand
        generate_command(const fcs::L4::GimbalPlan& plan, double target_distance) noexcept;

    /// Set configuration (lightweight: only stores config)
    void set_config(const WeaponControllerConfig& config) noexcept { config_ = config; }

    /// Sync configuration (heavyweight: applies config like updating compensator)
    void sync_config(std::unique_ptr<fcs::L4::TrajectoryCompensator> compensator) noexcept {
        compensator_ = std::move(compensator);
    }

private:
    WeaponControllerConfig config_;
    std::unique_ptr<fcs::L4::TrajectoryCompensator> compensator_;

    /// Log compensator type switch (called from sync_config)
    static void log_compensator_switch(
        fcs::L4::TrajectoryCompensatorType old_type,
        fcs::L4::TrajectoryCompensatorType new_type) noexcept {
        SPDLOG_INFO(
            "EnhancedWeaponController: switching compensator type from {} to {}",
            magic_enum::enum_name(old_type).data(),
            magic_enum::enum_name(new_type).data());
    }

    /// @brief Calculate bullet time of flight for target distance
    [[nodiscard]] double calculate_time_of_flight(double distance) const noexcept;
};

} // namespace fcs::L5
