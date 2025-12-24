#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <concepts>

namespace fast_tf {

// Forward declarations
class TransformMatrix4d;
struct RPYTransformData;
struct QuaternionTransformData;

// ============================================================================
// Core Conversion Template (Rust-style Into trait)
// ============================================================================

/// Primary template - deleted for unsupported conversions
template <typename To, typename From>
constexpr To into(From f) = delete;

/// Standard conversion for implicitly convertible types
template <typename To, typename From>
requires std::convertible_to<From, To> constexpr To into(From f) {
    return static_cast<To>(f);
}

// ============================================================================
// Into Concept
// ============================================================================

/// Check if a type F can be converted Into type T
template <typename From, typename To>
concept Into = requires(From f) {
    { into<To>(f) } -> std::same_as<To>;
};

// ============================================================================
// Specializations: Rpy -> Eigen::Matrix4d
// ============================================================================

template <>
constexpr Eigen::Matrix4d into<Eigen::Matrix4d, Rpy>(Rpy f) {
    const auto [rpy, xyz]         = f;
    const auto [roll, pitch, yaw] = rpy;
    const auto [x, y, z]          = xyz;

    const auto rx = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    const auto ry = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
    const auto rz = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    const Eigen::Quaterniond q = rz * ry * rx;

    Eigen::Matrix4d m   = Eigen::Matrix4d::Identity();
    m.block<3, 3>(0, 0) = q.toRotationMatrix();
    m(0, 3)             = x;
    m(1, 3)             = y;
    m(2, 3)             = z;

    return m;
}

// ============================================================================
// Generic Transform Conversion (using concepts)
// ============================================================================

// Note: Specializations for RPYTransformData and QuaternionTransformData
// will be defined in matrix.hpp after TransformMatrix4d is defined

} // namespace fast_tf
