#pragma once

#include <cstdint>
#include <tuple>

namespace fast_tf {

// ============================================================================
// Basic Type Aliases
// ============================================================================

/// Timestamp in nanoseconds
using timestamp_ns_t = std::uint64_t;

/// Floating point type used throughout fast_tf
using fp_t = double;

/// RPY + XYZ transform representation as nested tuples
/// Format: ((roll, pitch, yaw), (x, y, z))
using Rpy = std::tuple<std::tuple<fp_t, fp_t, fp_t>, std::tuple<fp_t, fp_t, fp_t>>;

} // namespace fast_tf
