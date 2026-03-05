#pragma once

#include "frame.hpp"

#include <cstdint>
#include <expected>
#include <fmt/format.h>

namespace fcs::L4 {

// ============================================================================
// Transform Lookup Result
// ============================================================================

/// Result of gimbal/muzzle transform lookup
///
/// Both transforms are in odom frame (FuckOdom = FuckTransform<odom_fuxk_frame>).
/// This is consistent with the coordinate lookup return type.
struct TransformLookupResult {
    fast_tf::FuckOdom gimbal; ///< Odom -> Gimbal transform (in odom frame)
    fast_tf::FuckOdom muzzle; ///< Odom -> Muzzle transform (in odom frame)
    uint64_t timestamp_ns;    ///< Query timestamp (ns)
};

// ============================================================================
// Transform Lookup Utilities
// ============================================================================

/// Lookup both gimbal and muzzle transforms from tf buffer
///
/// @param tf_buffer Coordinate system buffer (shared_ptr)
/// @param current_ns Query timestamp in nanoseconds
/// @return TransformLookupResult on success, error message on failure
///
/// @note Uses fast_tf::lookup_clamped which clamps to buffer time range
/// @note Error messages include detailed diagnostic information
/// @note Both transforms are returned as FuckOdom (in odom frame)
[[nodiscard]] inline std::expected<TransformLookupResult, std::string>
    lookup_gimbal_muzzle_transforms(
        const std::shared_ptr<fast_tf::CoordinateSystem<1024>>& tf_buffer,
        uint64_t current_ns) noexcept {
    using namespace fast_tf;

    // Lookup muzzle transform (odom -> muzzle)
    auto muzzle_tf = lookup_clamped<odom_fuxk_frame, muzzle_link_fuxk_frame>(
        *tf_buffer, FuckTransform<muzzle_link_fuxk_frame>{}, current_ns);
    if (!muzzle_tf) {
        return std::unexpected(
            fmt::format(
                "Transform lookup failed: odom -> muzzle. Error: {}. Query timestamp: {} ns",
                muzzle_tf.error(), current_ns));
    }

    // Lookup gimbal transform (odom -> gimbal)
    auto gimbal_tf = lookup_clamped<odom_fuxk_frame, fast_tf::gimbal>(
        *tf_buffer, FuckTransform<fast_tf::gimbal>{}, current_ns);
    if (!gimbal_tf) {
        return std::unexpected(
            fmt::format(
                "Transform lookup failed: odom -> gimbal. Error: {}. Query timestamp: {} ns",
                gimbal_tf.error(), current_ns));
    }

    return TransformLookupResult{
        .gimbal       = *gimbal_tf,
        .muzzle       = *muzzle_tf,
        .timestamp_ns = current_ns,
    };
}

/// Lookup only gimbal transform from tf buffer
///
/// @param tf_buffer Coordinate system buffer (shared_ptr)
/// @param current_ns Query timestamp in nanoseconds
/// @return FuckOdom (gimbal transform in odom frame) on success, error message on failure
///
/// @note Used by Planner which only requires gimbal transform
/// @note Error messages include detailed diagnostic information
[[nodiscard]] inline std::expected<fast_tf::FuckOdom, std::string> lookup_gimbal_transform(
    const std::shared_ptr<fast_tf::CoordinateSystem<1024>>& tf_buffer,
    uint64_t current_ns) noexcept {
    using namespace fast_tf;

    auto gimbal_tf = lookup_clamped<odom_fuxk_frame, fast_tf::gimbal>(
        *tf_buffer, FuckTransform<fast_tf::gimbal>{}, current_ns);
    if (!gimbal_tf) {
        return std::unexpected(
            fmt::format(
                "Transform lookup failed: odom -> gimbal. Error: {}. Query timestamp: {} ns",
                gimbal_tf.error(), current_ns));
    }

    return *gimbal_tf;
}

} // namespace fcs::L4
