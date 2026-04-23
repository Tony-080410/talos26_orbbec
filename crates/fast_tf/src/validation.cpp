#include "validation.hpp"
#include "frame.hpp"

#include <fmt/format.h>

namespace fast_tf {

// ============================================================================
// Definitions (previously inline in header)
// ============================================================================

template <std::floating_point T>
bool has_nan_or_inf(T value) noexcept {
    return std::isnan(value) || std::isinf(value);
}

template <std::floating_point T>
bool has_nan_or_inf(const Eigen::Matrix<T, 3, 1>& v) noexcept {
    for (int i = 0; i < 3; ++i) {
        if (has_nan_or_inf(v(i))) {
            return true;
        }
    }
    return false;
}

template <std::floating_point T>
bool has_nan_or_inf(const Eigen::Matrix<T, 3, 3>& m) noexcept {
    for (int i = 0; i < 9; ++i) {
        if (has_nan_or_inf(m(i))) {
            return true;
        }
    }
    return false;
}

template <std::floating_point T>
bool has_nan_or_inf(const Eigen::Quaternion<T>& q) noexcept {
    return has_nan_or_inf(q.x()) || has_nan_or_inf(q.y()) || has_nan_or_inf(q.z())
        || has_nan_or_inf(q.w());
}

template <std::floating_point T>
bool is_quaternion_normalized(const Eigen::Quaternion<T>& q, T tolerance) noexcept {
    const T norm = q.norm();
    return std::abs(norm - static_cast<T>(1)) <= tolerance;
}

template <std::floating_point T, typename From, typename To>
std::string format_transform_values(const TransformMatrix<T, From, To>& transform) noexcept {
    const auto euler = transform.euler_rot();
    const auto trans = transform.translation();

    constexpr T rad_to_deg = static_cast<T>(180.0 / M_PI);

    return fmt::format(
        "roll={:.3f}\xC2\xB0, pitch={:.3f}\xC2\xB0, yaw={:.3f}\xC2\xB0, "
        "x={:.3f}m, y={:.3f}m, z={:.3f}m",
        euler.roll * rad_to_deg, euler.pitch * rad_to_deg, euler.yaw * rad_to_deg, trans.x(),
        trans.y(), trans.z());
}

template <std::floating_point T, typename From, typename To>
std::expected<void, std::string>
    validate_transform(const TransformMatrix<T, From, To>& transform) noexcept {
    const auto translation = transform.translation();
    const auto rotation    = transform.rotation();
    const auto quaternion  = transform.quaternion();

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

    if (has_nan_or_inf(rotation)) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [rotation matrix contains NaN or Inf]. Full values: "
                "{}",
                format_transform_values(transform)));
    }

    if (has_nan_or_inf(quaternion)) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [quaternion contains NaN or Inf]. Full values: {}",
                format_transform_values(transform)));
    }

    if (!is_quaternion_normalized(quaternion, static_cast<T>(0.01))) {
        return std::unexpected(
            fmt::format(
                "Transform validation failed: [quaternion not normalized, norm={:.6f}]. Full "
                "values: {}",
                quaternion.norm(), format_transform_values(transform)));
    }

    return {};
}

// ============================================================================
// Explicit instantiations for all edge transforms in the coordinate system
// ============================================================================

// world -> odom
template std::expected<void, std::string>
    validate_transform(const TransformMatrix<double, world_fuxk_frame, odom_fuxk_frame>&) noexcept;
template std::string format_transform_values(
    const TransformMatrix<double, world_fuxk_frame, odom_fuxk_frame>&) noexcept;

// odom -> gimbal_yaw
template std::expected<void, std::string> validate_transform(
    const TransformMatrix<double, odom_fuxk_frame, gimbal_yaw_fuxk_frame>&) noexcept;
template std::string format_transform_values(
    const TransformMatrix<double, odom_fuxk_frame, gimbal_yaw_fuxk_frame>&) noexcept;

// gimbal_yaw -> gimbal_pitch
template std::expected<void, std::string> validate_transform(
    const TransformMatrix<double, gimbal_yaw_fuxk_frame, gimbal_pitch_fuxk_frame>&) noexcept;
template std::string format_transform_values(
    const TransformMatrix<double, gimbal_yaw_fuxk_frame, gimbal_pitch_fuxk_frame>&) noexcept;

// gimbal_pitch -> camera_link
template std::expected<void, std::string> validate_transform(
    const TransformMatrix<double, gimbal_pitch_fuxk_frame, camera_link_fuxk_frame>&) noexcept;
template std::string format_transform_values(
    const TransformMatrix<double, gimbal_pitch_fuxk_frame, camera_link_fuxk_frame>&) noexcept;

// camera_link -> camera_optical
template std::expected<void, std::string> validate_transform(
    const TransformMatrix<double, camera_link_fuxk_frame, camera_optical_fuxk_frame>&) noexcept;
template std::string format_transform_values(
    const TransformMatrix<double, camera_link_fuxk_frame, camera_optical_fuxk_frame>&) noexcept;

// gimbal_pitch -> muzzle_link
template std::expected<void, std::string> validate_transform(
    const TransformMatrix<double, gimbal_pitch_fuxk_frame, muzzle_link_fuxk_frame>&) noexcept;
template std::string format_transform_values(
    const TransformMatrix<double, gimbal_pitch_fuxk_frame, muzzle_link_fuxk_frame>&) noexcept;

} // namespace fast_tf
