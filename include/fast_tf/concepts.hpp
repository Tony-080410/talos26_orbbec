#pragma once

#include "types.hpp"
#include <concepts>
#include <cstdint>
#include <string_view>

namespace fast_tf {

// ============================================================================
// Transform Input Concepts
// ============================================================================

/// RPY (Roll-Pitch-Yaw) transform input
template <typename T>
concept rpy_transform = requires(T t) {
    { t.roll } -> std::convertible_to<fp_t>;
    { t.pitch } -> std::convertible_to<fp_t>;
    { t.yaw } -> std::convertible_to<fp_t>;
    { t.x } -> std::convertible_to<fp_t>;
    { t.y } -> std::convertible_to<fp_t>;
    { t.z } -> std::convertible_to<fp_t>;
};

/// Quaternion transform input
template <typename T>
concept quat_transform = requires(T t) {
    { t.w } -> std::convertible_to<fp_t>;
    { t.x } -> std::convertible_to<fp_t>;
    { t.y } -> std::convertible_to<fp_t>;
    { t.z } -> std::convertible_to<fp_t>;
    { t.tx } -> std::convertible_to<fp_t>;
    { t.ty } -> std::convertible_to<fp_t>;
    { t.tz } -> std::convertible_to<fp_t>;
};

/// Any transform input (RPY or Quaternion)
template <typename T>
concept any_transform = rpy_transform<T> || quat_transform<T>;

// ============================================================================
// Frame Identification Concept
// ============================================================================

/// Compile-time known coordinate frame
template <typename T>
concept known_coordinate = requires {
    { T::frame_id } -> std::convertible_to<uint64_t>;
    { T::frame_name } -> std::convertible_to<std::string_view>;
};

// ============================================================================
// Transform Matrix Concept
// ============================================================================

/// A type that behaves like a transform matrix
template <typename T>
concept TransformMatrixLike = requires(T m) {
    { m * m } -> std::convertible_to<T>;
    { m.inverse() } -> std::convertible_to<T>;
};

} // namespace fast_tf
