#pragma once

#include "concepts.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include <cmath>

namespace fast_tf {

// ============================================================================
// RPY Transform Data
// ============================================================================

/// Transform data in Roll-Pitch-Yaw format
struct RPYTransformData {
    fp_t roll  = 0.0; // Rotation around X axis (radians)
    fp_t pitch = 0.0; // Rotation around Y axis (radians)
    fp_t yaw   = 0.0; // Rotation around Z axis (radians)
    fp_t x     = 0.0; // Translation X
    fp_t y     = 0.0; // Translation Y
    fp_t z     = 0.0; // Translation Z

    RPYTransformData() = default;

    RPYTransformData(fp_t r, fp_t p, fp_t y_val, fp_t tx, fp_t ty, fp_t tz)
        : roll(r)
        , pitch(p)
        , yaw(y_val)
        , x(tx)
        , y(ty)
        , z(tz) {}

    [[nodiscard]] bool is_valid() const {
        // In -ffinite-math-only mode, all values are considered finite
        return true;
    }
};

// ============================================================================
// Quaternion Transform Data
// ============================================================================

/// Transform data in Quaternion format
struct QuaternionTransformData {
    fp_t w  = 1.0; // Real part (default: identity quaternion)
    fp_t x  = 0.0; // Imaginary i
    fp_t y  = 0.0; // Imaginary j
    fp_t z  = 0.0; // Imaginary k
    fp_t tx = 0.0; // Translation X
    fp_t ty = 0.0; // Translation Y
    fp_t tz = 0.0; // Translation Z

    QuaternionTransformData() = default;

    QuaternionTransformData(
        fp_t w_val, fp_t x_val, fp_t y_val, fp_t z_val, fp_t tx_val, fp_t ty_val, fp_t tz_val)
        : w(w_val)
        , x(x_val)
        , y(y_val)
        , z(z_val)
        , tx(tx_val)
        , ty(ty_val)
        , tz(tz_val) {}

    /// Check if quaternion is normalized (unit quaternion)
    [[nodiscard]] bool is_valid() const {
        fp_t norm = w * w + x * x + y * y + z * z;
        return std::abs(norm - 1.0) < 1e-6;
    }

    /// Normalize the quaternion in place
    void normalize() {
        fp_t norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (norm > 1e-10) {
            w /= norm;
            x /= norm;
            y /= norm;
            z /= norm;
        }
    }
};

// ============================================================================
// Into Specializations
// ============================================================================

/// Convert RPYTransformData to TransformMatrix4d
template <>
inline TransformMatrix4d into<TransformMatrix4d, RPYTransformData>(RPYTransformData f) {
    return TransformMatrix4d::from_rpy(f.roll, f.pitch, f.yaw, f.x, f.y, f.z);
}

/// Convert QuaternionTransformData to TransformMatrix4d
template <>
inline TransformMatrix4d
    into<TransformMatrix4d, QuaternionTransformData>(QuaternionTransformData f) {
    return TransformMatrix4d::from_quaternion(f.w, f.x, f.y, f.z, f.tx, f.ty, f.tz);
}

/// Generic conversion from any_transform concept to TransformMatrix4d
template <any_transform T>
TransformMatrix4d transform_into_matrix(const T& t) {
    if constexpr (rpy_transform<T>) {
        return TransformMatrix4d::from_rpy(t.roll, t.pitch, t.yaw, t.x, t.y, t.z);
    } else if constexpr (quat_transform<T>) {
        return TransformMatrix4d::from_quaternion(t.w, t.x, t.y, t.z, t.tx, t.ty, t.tz);
    }
}

} // namespace fast_tf
