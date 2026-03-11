#pragma once

#include <cstdint>
#include <string_view>

#include <fmt/core.h>
#include <magic_enum.hpp>

namespace fast_tf {

// ============================================================================
// Transform Error Types
// ============================================================================

/// Error codes for transform queries
enum class TransformError : uint8_t {
    FrameNotFound,               // Frame not registered in tree
    NoDataAvailable,             // Buffer is empty
    ExactTimestampNotFound,      // Exact timestamp match required but missing
    PastExtrapolationRequired,   // Timestamp outside buffer range
    FutureExtrapolationRequired, // Timestamp outside buffer range
    Timeout,                     // Query timed out
    PathNotFound,                // No path between frames
};

} // namespace fast_tf

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<fast_tf::TransformError> : formatter<std::string_view> {
    auto format(const fast_tf::TransformError e, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(e), ctx);
    }
};

} // namespace fmt
