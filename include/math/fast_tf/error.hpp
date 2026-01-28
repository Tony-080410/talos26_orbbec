#pragma once

#include <cstdint>
#include <string_view>

#include <fmt/core.h>

namespace fast_tf {

// ============================================================================
// Transform Error Types
// ============================================================================

/// Error codes for transform queries
enum class TransformError : uint8_t {
    FrameNotFound,               // Frame not registered in tree
    NoDataAvailable,             // Buffer is empty
    PastExtrapolationRequired,   // Timestamp outside buffer range
    FutureExtrapolationRequired, // Timestamp outside buffer range
    Timeout,                     // Query timed out
    PathNotFound,                // No path between frames
};

/// Convert error to string (constexpr for compile-time use)
constexpr std::string_view to_string(TransformError e) noexcept {
    switch (e) {
    case TransformError::FrameNotFound: return "frame not found";
    case TransformError::NoDataAvailable: return "no data available";
    case TransformError::PastExtrapolationRequired: return "past extrapolation required";
    case TransformError::FutureExtrapolationRequired: return "future extrapolation required";
    case TransformError::Timeout: return "timeout";
    case TransformError::PathNotFound: return "path not found";
    }
    return "unknown error";
}

} // namespace fast_tf

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<fast_tf::TransformError> : formatter<std::string_view> {
    auto format(const fast_tf::TransformError e, format_context& ctx) const {
        return formatter<std::string_view>::format(fast_tf::to_string(e), ctx);
    }
};

} // namespace fmt
