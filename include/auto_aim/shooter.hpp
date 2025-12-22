#pragma once

#include "aimer.hpp"
#include "target.hpp"

namespace auto_aim {

/**
 * @brief Shooter 配置
 */
struct ShooterConfig {
    double judge_distance{5.0};     // 距离判断阈值 (米)
    double first_tolerance{0.087};  // 近距离容差 (rad, 约5度)
    double second_tolerance{0.052}; // 远距离容差 (rad, 约3度)
    bool auto_fire{true};           // 是否启用自动发射
};

/**
 * @brief 发射控制器
 *
 * 移植自 sp_vision_25/tasks/auto_aim/shooter
 * 功能: 基于云台对准情况判断是否应该发射
 *
 * 判断条件:
 * 1. command 有效
 * 2. auto_fire 开关开启
 * 3. command 未突变 (与上次比较)
 * 4. 云台已对准 (gimbal_yaw 与 command.yaw 比较)
 * 5. aimer.debug_aim_point 有效
 */
class Shooter {
public:
    explicit Shooter(const ShooterConfig& config) noexcept;

    /**
     * @brief 判断是否应该发射
     *
     * @param command 当前瞄准命令
     * @param aimer 瞄准器 (用于获取 debug_aim_point)
     * @param target 跟踪目标
     * @param gimbal_yaw 当前云台 yaw 角 (弧度)
     * @return 是否应该发射
     */
    [[nodiscard]] bool should_shoot(
        const GimbalCommand& command, const Aimer& aimer, const Target& target,
        double gimbal_yaw) noexcept;

private:
    ShooterConfig config_;
    GimbalCommand last_command_{false, false, 0.0, 0.0, 0.0};
};

} // namespace auto_aim
