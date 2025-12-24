#pragma once

#include <cstdint>
#include <string_view>

namespace fast_tf {

// ============================================================================
// Frame Type System - Compile-Time Frame Identification
// ============================================================================

/**
 * Base frame type for compile-time frame identification.
 * Each frame in the hierarchy is a distinct type, enabling type-safe queries.
 *
 * Usage:
 *   DECLARE_FRAME(world);            // Root frame: world_Frame type
 *   DECLARE_FRAME(base_link, world); // Child frame: base_link_Frame type
 *
 * Each frame type has:
 * - frame_name: std::string_view with the frame name
 * - frame_id: uint64_t compile-time hash of the name
 */

// FNV-1a hash for compile-time string hashing
constexpr auto compile_time_hash(std::string_view str, uint64_t hash = 14695981039346656037ULL)
    -> uint64_t {
    for (char c : str) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ============================================================================
// Frame Hierarchy Type Traits
// ============================================================================

/**
 * Type trait to extract parent frame type.
 * Specialized for each frame to provide parent information.
 */
template <typename Frame>
struct parent_frame {
    using parent_type = void; // Default: no parent (root frame)
};

/**
 * Type trait to check if a frame is the same as another.
 */
template <typename A, typename B>
struct is_same_frame : std::false_type {};

template <typename T>
struct is_same_frame<T, T> : std::true_type {};

// ============================================================================
// Frame Hierarchy Validation
// ============================================================================

/**
 * Check if frame B is an ancestor of frame A in the hierarchy.
 * Used for compile-time path validation.
 */
template <typename Current, typename Target>
struct is_ancestor_of {
    static constexpr bool value =
        is_same_frame<Current, Target>::value
        || is_ancestor_of<typename parent_frame<Current>::parent_type, Target>::value;
};

// Specialization for void (end of hierarchy)
template <typename Target>
struct is_ancestor_of<void, Target> {
    static constexpr bool value = false;
};

// ============================================================================
// Frame Utility Functions
// ============================================================================

/**
 * Get the root frame by walking up the parent chain.
 */
template <typename Frame>
struct find_root {
    using parent_type = Frame;
};

template <typename Frame>
requires(!std::is_same_v<typename parent_frame<Frame>::parent_type, void>) struct find_root<Frame> {
    using parent_type = find_root<typename parent_frame<Frame>::parent_type>::parent_type;
};

template <typename Frame>
using find_root_t = find_root<Frame>::parent_type;

/**
 * Check if two frame types are in the same hierarchy.
 * Used to validate template query parameters at compile-time.
 */
template <typename From, typename To>
struct frames_in_same_tree {
    static constexpr bool value = std::is_same_v<find_root_t<From>, find_root_t<To>>;
};

// ============================================================================
// Macro for Frame Declaration
// ============================================================================

/**
 * Declare a root frame (no parent) or child frame with parent.
 *
 * Usage:
 *   DECLARE_FRAME(world);              // Root frame
 *   DECLARE_FRAME(base_link, world);   // Child frame with parent
 *
 * Creates types named name##_frame
 */

/**
 * Single-argument version: root frame
 */
#define DECLARE_FRAME_1(name)                                                             \
    struct name##_frame {                                                                 \
        static constexpr std::string_view frame_name = #name;                             \
        static constexpr uint64_t frame_id           = fast_tf::compile_time_hash(#name); \
    };                                                                                    \
    template <>                                                                           \
    struct parent_frame<name##_frame> {                                                   \
        using parent_type = void;                                                         \
    }

/**
 * Two-argument version: child frame with parent
 */
#define DECLARE_FRAME_2(name, parent)                                                     \
    struct name##_frame {                                                                 \
        static constexpr std::string_view frame_name = #name;                             \
        static constexpr uint64_t frame_id           = fast_tf::compile_time_hash(#name); \
    };                                                                                    \
    template <>                                                                           \
    struct parent_frame<name##_frame> {                                                   \
        using parent_type = parent##_frame;                                               \
    }

// ============================================================================
// Macro Overloading for DECLARE_FRAME
// ============================================================================

#define GET_MACRO(_1, _2, NAME, ...) NAME

/**
 * Overloaded DECLARE_FRAME macro that accepts 1 or 2 arguments:
 * - DECLARE_FRAME(world) - root frame
 * - DECLARE_FRAME(base_link, world) - child frame
 */
#define DECLARE_FRAME(...) GET_MACRO(__VA_ARGS__, DECLARE_FRAME_2, DECLARE_FRAME_1)(__VA_ARGS__)

} // namespace fast_tf
