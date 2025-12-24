#pragma once

#include "frame_id.hpp"
#include <string>
#include <type_traits>
#include <unordered_map>

namespace fast_tf {

// ============================================================================
// Compile-Time Frame Hierarchy Analysis
// ============================================================================

/**
 * Find the Lowest Common Ancestor (LCA) of two frames.
 * Helper for path computation between frames in the hierarchy.
 */

template <typename From, typename To>
struct find_lca {
    // Verify frames are in same tree
    static_assert(
        frames_in_same_tree<From, To>::value,
        "Frames From and To must be in the same hierarchy tree");
    using type = find_root_t<From>;
};

// If From is ancestor of To, LCA is From
template <typename Child, typename Parent>
requires is_ancestor_of<Child, Parent>::value struct find_lca<Child, Parent> {
    using type = Child;
};

// If To is ancestor of From, LCA is To
template <typename From, typename To>
requires is_ancestor_of<To, From>::value && (!is_ancestor_of<From, To>::value)
struct find_lca<From, To> {
    using type = To;
};

// Otherwise, recurse with From's parent
template <typename From, typename To>
requires(
    !is_ancestor_of<From, To>::value && !is_ancestor_of<To, From>::value
    && !std::is_same_v<typename parent_frame<From>::type, void>) struct find_lca<From, To> {
    using type = typename find_lca<typename parent_frame<From>::type, To>::type;
};

template <typename From, typename To>
using find_lca_t = typename find_lca<From, To>::type;

// ============================================================================
// Frame Type Hierarchy Validation
// ============================================================================

/**
 * Compile-time verification that two frames are in the same tree.
 * If not, this will produce a compile error.
 *
 * This is the key validation mechanism that provides type safety
 * for template-based queries like lookupTransform<From, To>.
 */
template <typename From, typename To>
struct validate_frames_same_tree {
    static_assert(
        frames_in_same_tree<From, To>::value, "Frames must be in the same hierarchy tree. "
                                              "Use DECLARE_FRAME to define frame hierarchies.");
    static constexpr bool value = true;
};

/**
 * Type that represents a valid path between two frames.
 * Used for safe dispatch in transform queries.
 */
template <typename From, typename To>
requires validate_frames_same_tree<From, To>::value struct FramePath {
    using from_frame = From;
    using to_frame   = To;
    using lca        = find_lca_t<From, To>;

    // Helper: Is From ancestor of To?
    static constexpr bool from_is_ancestor = is_ancestor_of<From, To>::value;

    // Helper: Is To ancestor of From?
    static constexpr bool to_is_ancestor = is_ancestor_of<To, From>::value;

    // Helper: Are they the same frame?
    static constexpr bool same_frame = std::is_same_v<From, To>;

    // For debugging
    static constexpr std::string_view from_name = From::frame_name;
    static constexpr std::string_view to_name   = To::frame_name;
};

// ============================================================================
// Frame Registry for Dynamic Frame Management
// ============================================================================

/**
 * Runtime registry of frames and their parents.
 * Needed for dynamic frame creation at runtime.
 */
class FrameHierarchyRegistry {
public:
    /**
     * Register a frame and its parent relationship.
     * Used for dynamic frame creation at runtime.
     */
    static void
        register_parent_relationship(std::string_view frame_name, std::string_view parent_name) {
        parents[std::string(frame_name)] = std::string(parent_name);
    }

    /**
     * Get parent frame name.
     * Returns empty string if frame is root or not registered.
     */
    static std::string_view get_parent(std::string_view frame_name) {
        auto it = parents.find(std::string(frame_name));
        if (it != parents.end()) {
            return it->second;
        }
        return "";
    }

    /**
     * Check if frame is in same tree as root.
     */
    static bool in_same_tree(std::string_view frame, std::string_view root) {
        // Walk up from frame to root
        std::string current(frame);
        std::string root_str(root);

        while (!current.empty()) {
            if (current == root_str) {
                return true;
            }
            current = std::string(get_parent(current));
        }
        return false;
    }

    /**
     * Clear all registered relationships (for testing).
     */
    static void clear() { parents.clear(); }

private:
    static std::unordered_map<std::string, std::string> parents;
};

// Inline definition for header-only library
inline std::unordered_map<std::string, std::string> FrameHierarchyRegistry::parents;

} // namespace fast_tf
