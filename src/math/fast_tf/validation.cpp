#include "validation.hpp"

#include <cmath>

namespace fast_tf {

// ============================================================================
// NaN/Inf 检测
// ============================================================================

bool has_nan_or_inf(double value) noexcept { return std::isnan(value) || std::isinf(value); }

bool has_nan_or_inf(const Eigen::Vector3d& v) noexcept {
    for (int i = 0; i < 3; ++i) {
        if (has_nan_or_inf(v(i))) {
            return true;
        }
    }
    return false;
}

bool has_nan_or_inf(const Eigen::Matrix3d& m) noexcept {
    for (int i = 0; i < 9; ++i) {
        if (has_nan_or_inf(m(i))) {
            return true;
        }
    }
    return false;
}

bool has_nan_or_inf(const Eigen::Quaterniond& q) noexcept {
    return has_nan_or_inf(q.x()) || has_nan_or_inf(q.y()) || has_nan_or_inf(q.z())
        || has_nan_or_inf(q.w());
}

// ============================================================================
// 四元数归一化检查
// ============================================================================

bool is_quaternion_normalized(const Eigen::Quaterniond& q, double tolerance) noexcept {
    const double norm = q.norm();
    return std::abs(norm - 1.0) <= tolerance;
}

// ============================================================================
// 格式化变换值
// ============================================================================

std::string format_transform_values(const TransformMatrix4d& transform) noexcept {
    const auto euler = transform.euler_rot();
    const auto trans = transform.translation();

    // 角度转换为度数
    constexpr double RAD_TO_DEG = 180.0 / M_PI;

    return fmt::format(
        "roll={:.3f}°, pitch={:.3f}°, yaw={:.3f}°, x={:.3f}m, y={:.3f}m, z={:.3f}m",
        euler.roll * RAD_TO_DEG, euler.pitch * RAD_TO_DEG, euler.yaw * RAD_TO_DEG, trans.x(),
        trans.y(), trans.z());
}

// ============================================================================
// 变换验证
// ============================================================================

std::expected<void, std::string> validate_transform(const TransformMatrix4d& transform) noexcept {
    const auto translation = transform.translation();
    const auto rotation    = transform.rotation();
    const auto quaternion  = transform.quaternion();

    // 检查平移分量
    if (has_nan_or_inf(translation.x())) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [translation.x is NaN or Inf]. Full values: {}",
                format_transform_values(transform)));
    }
    if (has_nan_or_inf(translation.y())) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [translation.y is NaN or Inf]. Full values: {}",
                format_transform_values(transform)));
    }
    if (has_nan_or_inf(translation.z())) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [translation.z is NaN or Inf]. Full values: {}",
                format_transform_values(transform)));
    }

    // 检查旋转矩阵
    if (has_nan_or_inf(rotation)) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [rotation matrix contains NaN or Inf]. Full values: "
                "{}",
                format_transform_values(transform)));
    }

    // 检查四元数
    if (has_nan_or_inf(quaternion)) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [quaternion contains NaN or Inf]. Full values: {}",
                format_transform_values(transform)));
    }

    // 检查四元数归一化（参考 tf2 的 0.01 容差）
    if (!is_quaternion_normalized(quaternion, 0.01)) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [quaternion not normalized, norm={:.6f}]. Full "
                "values: {}",
                quaternion.norm(), format_transform_values(transform)));
    }

    return {};
}

} // namespace fast_tf
