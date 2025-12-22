#pragma once

#include <expected>
#include <string_view>

namespace auto_aim {

/**
 * @brief 弹道求解结果
 */
struct TrajectoryResult {
    double fly_time; // 飞行时间 (秒)
    double pitch;    // 发射俯仰角 (弧度, 抬头为正)
};

/**
 * @brief 弹道求解错误类型
 */
enum class TrajectoryError {
    UnsolvableTarget, // 目标超出射程 (delta < 0)
    InvalidInput      // 输入参数非法 (v0 <= 0 或 d <= 0)
};

/**
 * @brief 获取弹道错误的描述信息
 */
[[nodiscard]] constexpr std::string_view trajectory_error_message(TrajectoryError err) noexcept {
    switch (err) {
        case TrajectoryError::UnsolvableTarget:
            return "Target out of range";
        case TrajectoryError::InvalidInput:
            return "Invalid input parameters (v0 <= 0 or d <= 0)";
    }
    return "Unknown error";
}

/**
 * @brief 抛物线弹道计算 (不考虑空阻)
 *
 * 移植自 sp_vision_25/tools/trajectory
 * 使用二次方程求解发射角:
 *   x = v0*cos(theta)*t
 *   z = v0*sin(theta)*t - 0.5*g*t^2
 * 消去 t 得 tan(theta) 的二次方程
 *
 * @param v0 子弹初速度 (m/s)
 * @param d 目标水平距离 (m)
 * @param h 目标竖直高度 (m)
 * @return 弹道结果或错误
 */
[[nodiscard]] std::expected<TrajectoryResult, TrajectoryError> solve_trajectory(double v0, double d,
                                                                                 double h) noexcept;

} // namespace auto_aim
