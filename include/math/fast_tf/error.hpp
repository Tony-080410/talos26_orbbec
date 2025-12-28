#pragma once

#include <cstdint>
#include <string_view>

namespace fast_tf {

// ============================================================================
// Transform Error Types
// ============================================================================

/// Error codes for transform queries
enum class TransformError : uint8_t {
    FrameNotFound,         // Frame not registered in tree
    NoDataAvailable,       // Buffer is empty
    ExtrapolationRequired, // Timestamp outside buffer range
    Timeout,               // Query timed out
    PathNotFound,          // No path between frames
};

/// Convert error to string (constexpr for compile-time use)
constexpr std::string_view to_string(TransformError e) noexcept {
    switch (e) {
    case TransformError::FrameNotFound: return "frame not found";
    case TransformError::NoDataAvailable: return "no data available";
    case TransformError::ExtrapolationRequired: return "extrapolation required";
    case TransformError::Timeout: return "timeout";
    case TransformError::PathNotFound: return "path not found";
    }
    return "unknown error";
}

} // namespace fast_tf
