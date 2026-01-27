#pragma once

#include <cstdint>
#include <string_view>
#include <variant>

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
    TimestampMismatch,           // Timestamps in frame chain are inconsistent
};

/// Detailed information about timestamp inconsistency detected during lookup
struct TimestampMismatchInfo {
    std::string_view frame_id; // Name of the frame that caused the mismatch
    uint64_t max_delta_ns;     // Maximum timestamp difference found (ns)
    uint64_t query_ns;         // Original query timestamp
    uint64_t actual_ns;        // Actual available timestamp
};

/// Variant error type for transform lookups - can be either a simple error code or detailed
/// mismatch info
using TransformResult = std::variant<TransformError, TimestampMismatchInfo>;

/// Convert error to string (constexpr for compile-time use)
constexpr std::string_view to_string(TransformError e) noexcept {
    switch (e) {
    case TransformError::FrameNotFound: return "frame not found";
    case TransformError::NoDataAvailable: return "no data available";
    case TransformError::PastExtrapolationRequired: return "past extrapolation required";
    case TransformError::FutureExtrapolationRequired: return "future extrapolation required";
    case TransformError::Timeout: return "timeout";
    case TransformError::PathNotFound: return "path not found";
    case TransformError::TimestampMismatch: return "timestamp mismatch";
    }
    return "unknown error";
}

/// Helper to get string representation of TransformResult for logging
inline std::string_view to_string_view(const TransformResult& result) noexcept {
    return std::visit(
        [](auto&& arg) -> std::string_view {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, TransformError>) {
                return to_string(arg);
            } else if constexpr (std::is_same_v<T, TimestampMismatchInfo>) {
                (void)arg; // Suppress unused warning
                return "timestamp mismatch";
            }
        },
        result);
}

} // namespace fast_tf

// fmt formatter for TransformResult (for spdlog logging)
#include <spdlog/fmt/fmt.h>

template <>
struct fmt::formatter<fast_tf::TransformResult> : fmt::formatter<std::string_view> {
    auto format(const fast_tf::TransformResult& result, fmt::format_context& ctx) const {
        return std::visit(
            [&](auto&& arg) -> decltype(ctx.out()) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, fast_tf::TransformError>) {
                    return fmt::format_to(ctx.out(), "{}", fast_tf::to_string(arg));
                } else if constexpr (std::is_same_v<T, fast_tf::TimestampMismatchInfo>) {
                    // Format detailed mismatch info with frame name
                    return fmt::format_to(
                        ctx.out(),
                        "timestamp mismatch in frame '{}': max_delta={}µs, query={}, actual={}",
                        arg.frame_id,
                        arg.max_delta_ns / 1000, // Convert to µs for readability
                        arg.query_ns, arg.actual_ns);
                }
            },
            result);
    }
};
