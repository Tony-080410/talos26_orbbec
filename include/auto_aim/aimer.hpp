#pragma once

#include <Eigen/Dense>
#include <optional>

#include "target.hpp"

namespace auto_aim {

/**
 * @brief 瞄准点数据
 */
struct AimPoint {
    bool valid{false};
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()}; // [x, y, z, angle]
};

/**
 * @brief 云台控制命令
 */
struct GimbalCommand {
    bool valid{false};       // 是否有效
    bool fire_advice{false}; // 发射建议
    double yaw{0.0};         // 目标 yaw (弧度)
    double pitch{0.0};       // 目标 pitch (弧度)
    double distance{0.0};    // 目标距离 (米)
};

/**
 * @brief Aimer 配置
 */
struct AimerConfig {
    double yaw_offset{0.0};                   // yaw 偏移 (rad)
    double pitch_offset{0.0};                 // pitch 偏移 (rad)
    double comming_angle{55.0 / 57.3};        // 小陀螺来袭角度阈值 (rad)
    double leaving_angle{20.0 / 57.3};        // 小陀螺离开角度阈值 (rad)
    double high_speed_delay_time{0.0};        // 高速延时 (s)
    double low_speed_delay_time{0.0};         // 低速延时 (s)
    double decision_speed{7.0};               // 速度阈值 (rad/s)
    std::optional<double> left_yaw_offset{};  // 左枪管偏移 (rad)
    std::optional<double> right_yaw_offset{}; // 右枪管偏移 (rad)
};

/**
 * @brief 瞄准控制器
 *
 * 移植自 sp_vision_25/tasks/auto_aim/aimer
 * 功能:
 * 1. 弹道迭代求解 (最多10次, 收敛阈值0.001s)
 * 2. 装甲板选择策略 (小陀螺/非小陀螺/锁定防抖)
 * 3. 延时补偿
 */
class Aimer {
public:
    explicit Aimer(const AimerConfig& config) noexcept;

    /**
     * @brief 计算瞄准命令
     *
     * @param target 跟踪目标 (不会被修改)
     * @param timestamp_ns 当前时间戳
     * @param bullet_speed 弹速 (m/s)
     * @param to_now 是否补偿到当前时刻
     * @return 云台控制命令
     */
    [[nodiscard]] GimbalCommand
        aim(const Target& target, uint64_t timestamp_ns, double bullet_speed,
            bool to_now = true) noexcept;

    /**
     * @brief 获取调试瞄准点 (用于 Shooter 判断)
     */
    [[nodiscard]] const AimPoint& debug_aim_point() const noexcept { return debug_aim_point_; }

private:
    AimerConfig config_;
    AimPoint debug_aim_point_;
    int lock_id_{-1}; // 锁定装甲板 ID (-1 表示未锁定)

    /**
     * @brief 选择瞄准点 (小陀螺/非小陀螺策略)
     */
    [[nodiscard]] AimPoint choose_aim_point(const Target& target) noexcept;

    /**
     * @brief 选择瞄准点 (从预测状态)
     */
    [[nodiscard]] AimPoint choose_aim_point(const PredictedState& predicted) noexcept;
};

} // namespace auto_aim
