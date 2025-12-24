#pragma once

#include "matrix.hpp"
#include "types.hpp"

namespace fast_tf {

// ============================================================================
// Transform Query Result
// ============================================================================

/// Result of a successful transform query
struct TransformResult {
    TransformMatrix4d transform;
    timestamp_ns_t timestamp = 0;
    bool is_interpolated     = false;

    TransformResult() = default;

    TransformResult(const TransformMatrix4d& tf, timestamp_ns_t ts, bool interpolated = false)
        : transform(tf)
        , timestamp(ts)
        , is_interpolated(interpolated) {}
};

} // namespace fast_tf
