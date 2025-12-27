#include "shooter.hpp"

#include <cmath>

namespace auto_aim {

Shooter::Shooter(const ShooterConfig& config) noexcept
    : config_(config) {}

bool Shooter::should_shoot(
    const GimbalCommand& command, const AimPoint& aim_point, const bool target_converged,
    const double gimbal_yaw) noexcept {
    // 检查基本条件
    if (!command.valid || !config_.auto_fire || !aim_point.valid || !target_converged) {
        last_command_ = command;
        return false;
    }

    // 从 aim_point 计算水平距离，选择容差
    const double target_x        = aim_point.xyza.x();
    const double target_y        = aim_point.xyza.y();
    const double target_distance = std::sqrt(target_x * target_x + target_y * target_y);
    const double tolerance = target_distance > config_.judge_distance ? config_.second_tolerance
                                                                      : config_.first_tolerance;

    // 判断条件:
    // 1. command 未突变 (与上次 yaw 差 < tolerance * 2)
    // 2. 云台已对准 (gimbal_yaw 与上次 command.yaw 差 < tolerance)
    const bool command_stable = std::abs(last_command_.yaw - command.yaw) < tolerance * 2;
    const bool gimbal_aligned = std::abs(gimbal_yaw - last_command_.yaw) < tolerance;

    // 更新 last_command_
    last_command_ = command;

    return command_stable && gimbal_aligned;
}

} // namespace auto_aim
