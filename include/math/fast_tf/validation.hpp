#pragma once

#include "matrix.hpp"
#include "types.hpp"
#include <expected>
#include <fmt/format.h>
#include <limits>

namespace fast_tf {

/// @brief 验证变换矩阵的数值有效性
/// @return 成功返回 void，失败返回详细错误消息
[[nodiscard]] std::expected<void, std::string>
    validate_transform(const TransformMatrix4d& transform) noexcept;

/// @brief 检查四元数是否归一化
/// @param q 四元数
/// @param tolerance 容差（默认 0.01，参考 tf2）
[[nodiscard]] bool
    is_quaternion_normalized(const Eigen::Quaterniond& q, double tolerance = 0.01) noexcept;

/// @brief 检查数值是否包含 NaN 或 Inf
[[nodiscard]] bool has_nan_or_inf(double value) noexcept;
[[nodiscard]] bool has_nan_or_inf(const Eigen::Vector3d& v) noexcept;
[[nodiscard]] bool has_nan_or_inf(const Eigen::Matrix3d& m) noexcept;
[[nodiscard]] bool has_nan_or_inf(const Eigen::Quaterniond& q) noexcept;

/// @brief 格式化变换的 RPY 和平移分量（用于错误消息）
/// @return 格式化的字符串，如 "roll=0.000°, pitch=0.024°, yaw=1.350°, x=0.044m, y=0.010m,
/// z=-0.053m"
[[nodiscard]] std::string format_transform_values(const TransformMatrix4d& transform) noexcept;

} // namespace fast_tf
