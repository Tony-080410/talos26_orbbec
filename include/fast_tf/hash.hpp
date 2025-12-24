#pragma once

#include <cstdint>
#include <string_view>

namespace fast_tf {

// ============================================================================
// FNV-1a Hash Function
// ============================================================================

/// Compile-time FNV-1a hash for string_view
/// Used for frame ID generation from string names
constexpr uint64_t fnv1a_hash(std::string_view str) noexcept {
    uint64_t hash = 14695981039346656037ULL;
    for (const char c : str) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace fast_tf
