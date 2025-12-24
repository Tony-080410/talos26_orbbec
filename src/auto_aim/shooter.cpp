#include "shooter.hpp"

#include <cmath>

namespace auto_aim {

Shooter::Shooter(const ShooterConfig& config) noexcept
    : config_(config) {}

bool Shooter::should_shoot(
    const GimbalCommand& command, const Aimer& aimer, const Target& target,
    const double gimbal_yaw) noexcept {
    // 检查基本条件
    if (!command.valid || !config_.auto_fire) {
        last_command_ = command;
        return false;
    }

    // 计算目标水平距离，选择容差
    // ekf_x: [x, vx, y, vy, z, vz, a, w, r, l, h]
    const Eigen::VectorXd ekf_x  = target.ekf_x();
    const double target_x        = ekf_x[0];
    const double target_y        = ekf_x[2];
    const double target_distance = std::sqrt(target_x * target_x + target_y * target_y);
    const double tolerance = target_distance > config_.judge_distance ? config_.second_tolerance
                                                                      : config_.first_tolerance;

    // 判断条件:
    // 1. command 未突变 (与上次 yaw 差 < tolerance * 2)
    // 2. 云台已对准 (gimbal_yaw 与上次 command.yaw 差 < tolerance)
    // 3. aimer.debug_aim_point 有效
    const bool command_stable  = std::abs(last_command_.yaw - command.yaw) < tolerance * 2;
    const bool gimbal_aligned  = std::abs(gimbal_yaw - last_command_.yaw) < tolerance;
    const bool aim_point_valid = aimer.debug_aim_point().valid;

    // 更新 last_command_
    last_command_ = command;

    if (command_stable && gimbal_aligned && aim_point_valid) {
        return true;
    }

    return false;
}

} // namespace auto_aim
