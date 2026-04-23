#pragma once

#include "matrix.hpp"
#include <cmath>
#include <concepts>
#include <expected>
#include <string>

namespace fast_tf {

/// @brief 验证变换矩阵的数值有效性
/// @return 成功返回 void，失败返回详细错误消息
template <std::floating_point T, typename From, typename To>
[[nodiscard]] std::expected<void, std::string>
    validate_transform(const TransformMatrix<T, From, To>& transform) noexcept;

/// @brief 检查四元数是否归一化
/// @param q 四元数
/// @param tolerance 容差（默认 0.01，参考 tf2）
template <std::floating_point T>
[[nodiscard]] bool is_quaternion_normalized(
    const Eigen::Quaternion<T>& q, T tolerance = static_cast<T>(0.01)) noexcept;

/// @brief 检查数值是否包含 NaN 或 Inf
template <std::floating_point T>
[[nodiscard]] bool has_nan_or_inf(T value) noexcept;

template <std::floating_point T>
[[nodiscard]] bool has_nan_or_inf(const Eigen::Matrix<T, 3, 1>& v) noexcept;

template <std::floating_point T>
[[nodiscard]] bool has_nan_or_inf(const Eigen::Matrix<T, 3, 3>& m) noexcept;

template <std::floating_point T>
[[nodiscard]] bool has_nan_or_inf(const Eigen::Quaternion<T>& q) noexcept;

/// @brief 格式化变换的 RPY 和平移分量（用于错误消息）
template <std::floating_point T, typename From, typename To>
[[nodiscard]] std::string
    format_transform_values(const TransformMatrix<T, From, To>& transform) noexcept;

} // namespace fast_tf
