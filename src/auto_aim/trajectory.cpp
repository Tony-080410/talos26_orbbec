#include "trajectory.hpp"

#include <cmath>

namespace auto_aim {

namespace {
constexpr double g = 9.7833; // 重力加速度 m/s^2
} // namespace

std::expected<TrajectoryResult, TrajectoryError>
    solve_trajectory(const double v0, const double d, const double h) noexcept {
    // 参数校验
    if (v0 <= 0 || d <= 0) {
        return std::unexpected(TrajectoryError::InvalidInput);
    }

    // 二次方程系数
    // a*tan^2(theta) + b*tan(theta) + c = 0
    // 其中: a = g*d^2 / (2*v0^2), b = -d, c = a + h
    const auto a     = g * d * d / (2 * v0 * v0);
    const auto b     = -d;
    const auto c     = a + h;
    const auto delta = b * b - 4 * a * c;

    // 判别式 < 0 表示目标超出射程
    if (delta < 0) {
        return std::unexpected(TrajectoryError::UnsolvableTarget);
    }

    // 求解两个发射角
    const auto sqrt_delta  = std::sqrt(delta);
    const auto tan_pitch_1 = (-b + sqrt_delta) / (2 * a);
    const auto tan_pitch_2 = (-b - sqrt_delta) / (2 * a);
    const auto pitch_1     = std::atan(tan_pitch_1);
    const auto pitch_2     = std::atan(tan_pitch_2);

    // 计算对应的飞行时间
    const auto t_1 = d / (v0 * std::cos(pitch_1));
    const auto t_2 = d / (v0 * std::cos(pitch_2));

    // 选择飞行时间较短的解 (低弹道)
    if (t_1 < t_2) {
        return TrajectoryResult{.fly_time = t_1, .pitch = pitch_1};
    }
    return TrajectoryResult{.fly_time = t_2, .pitch = pitch_2};
}

} // namespace auto_aim
