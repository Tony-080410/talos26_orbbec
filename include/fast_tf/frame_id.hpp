#pragma once

#include "hash.hpp"
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace fast_tf {

// ============================================================================
// Compile-Time Frame Definition Macros
// ============================================================================

/// Type trait to extract parent frame type
template <typename Frame>
struct parent_frame {
    using type = void; // Default: no parent (root frame)
};

/// Type trait to check if two frames are the same
template <typename A, typename B>
struct is_same_frame : std::false_type {};

template <typename T>
struct is_same_frame<T, T> : std::true_type {};

// ============================================================================
// Ancestor Checking
// ============================================================================

/// Check if Target is an ancestor of Current in the hierarchy
template <typename Current, typename Target>
struct is_ancestor_of {
    static constexpr bool value =
        is_same_frame<Current, Target>::value
        || is_ancestor_of<typename parent_frame<Current>::type, Target>::value;
};

/// Specialization for void (end of hierarchy)
template <typename Target>
struct is_ancestor_of<void, Target> {
    static constexpr bool value = false;
};

// ============================================================================
// Root Frame Finding
// ============================================================================

/// Find the root frame by walking up the parent chain
template <typename Frame>
struct find_root {
    using type = Frame;
};

template <typename Frame>
requires(!std::is_same_v<typename parent_frame<Frame>::type, void>) struct find_root<Frame> {
    using type = typename find_root<typename parent_frame<Frame>::type>::type;
};

template <typename Frame>
using find_root_t = typename find_root<Frame>::type;

// ============================================================================
// Tree Membership Checking
// ============================================================================

/// Check if two frame types are in the same hierarchy
template <typename From, typename To>
struct frames_in_same_tree {
    static constexpr bool value = std::is_same_v<find_root_t<From>, find_root_t<To>>;
};

// ============================================================================
// Frame Declaration Macros
// ============================================================================

/// Declare a root frame (no parent)
#define DECLARE_FRAME_ROOT(name)                                                     \
    struct name##_frame {                                                            \
        static constexpr std::string_view frame_name = #name;                        \
        static constexpr uint64_t frame_id           = ::fast_tf::fnv1a_hash(#name); \
    };                                                                               \
    template <>                                                                      \
    struct parent_frame<name##_frame> {                                              \
        using type = void;                                                           \
    }

/// Declare a child frame with parent
#define DECLARE_FRAME_CHILD(name, parent)                                            \
    struct name##_frame {                                                            \
        static constexpr std::string_view frame_name = #name;                        \
        static constexpr uint64_t frame_id           = ::fast_tf::fnv1a_hash(#name); \
    };                                                                               \
    template <>                                                                      \
    struct parent_frame<name##_frame> {                                              \
        using type = parent##_frame;                                                 \
    }

// Overloaded DECLARE_FRAME macro
#define GET_MACRO(_1, _2, NAME, ...) NAME

/// Unified DECLARE_FRAME macro:
/// - DECLARE_FRAME(world)           -> root frame
/// - DECLARE_FRAME(base_link, world) -> child frame
#define DECLARE_FRAME(...) \
    GET_MACRO(__VA_ARGS__, DECLARE_FRAME_CHILD, DECLARE_FRAME_ROOT)(__VA_ARGS__)

} // namespace fast_tf
