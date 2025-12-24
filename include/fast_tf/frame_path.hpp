#pragma once

#include "frame_id.hpp"
#include <string_view>
#include <type_traits>

namespace fast_tf {

// ============================================================================
// Lowest Common Ancestor (LCA) Computation
// ============================================================================

/// Find the Lowest Common Ancestor of two frames
template <typename From, typename To>
struct find_lca {
    static_assert(
        frames_in_same_tree<From, To>::value,
        "Frames From and To must be in the same hierarchy tree");
    using type = find_root_t<From>;
};

/// If From is ancestor of To, LCA is From
template <typename From, typename To>
requires is_ancestor_of<To, From>::value struct find_lca<From, To> {
    using type = From;
};

/// If To is ancestor of From, LCA is To
template <typename From, typename To>
requires is_ancestor_of<From, To>::value && (!is_ancestor_of<To, From>::value)
struct find_lca<From, To> {
    using type = To;
};

/// Recursive case: neither is ancestor, go up From's parent
template <typename From, typename To>
requires(
    !is_ancestor_of<From, To>::value && !is_ancestor_of<To, From>::value
    && !std::is_same_v<typename parent_frame<From>::type, void>) struct find_lca<From, To> {
    using type = typename find_lca<typename parent_frame<From>::type, To>::type;
};

template <typename From, typename To>
using find_lca_t = typename find_lca<From, To>::type;

// ============================================================================
// Frame Validation
// ============================================================================

/// Compile-time validation that two frames are in the same tree
template <typename From, typename To>
struct validate_frames_same_tree {
    static_assert(
        frames_in_same_tree<From, To>::value, "Frames must be in the same hierarchy tree. "
                                              "Use DECLARE_FRAME to define frame hierarchies.");
    static constexpr bool value = true;
};

// ============================================================================
// Frame Path Type
// ============================================================================

/// Represents a valid path between two frames (compile-time verified)
template <typename From, typename To>
requires validate_frames_same_tree<From, To>::value struct FramePath {
    using from_frame = From;
    using to_frame   = To;
    using lca        = find_lca_t<From, To>;

    static constexpr bool from_is_ancestor = is_ancestor_of<To, From>::value;
    static constexpr bool to_is_ancestor   = is_ancestor_of<From, To>::value;
    static constexpr bool same_frame       = std::is_same_v<From, To>;

    static constexpr std::string_view from_name = From::frame_name;
    static constexpr std::string_view to_name   = To::frame_name;
};

} // namespace fast_tf
