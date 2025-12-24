#pragma once

#include "hash.hpp"
#include <string>
#include <string_view>
#include <unordered_map>

namespace fast_tf {

// ============================================================================
// Unified Frame Registry
// ============================================================================

/// Runtime registry for frame names, IDs, and parent relationships
/// Combines functionality from former FrameRegistry and FrameHierarchyRegistry
class FrameRegistry {
public:
    // ========================================================================
    // Frame Registration
    // ========================================================================

    /// Register a frame with its name and ID
    static void register_frame(std::string_view name, uint64_t frame_id) {
        auto name_str = std::string(name);
        if (id_to_name_.find(frame_id) == id_to_name_.end()) {
            id_to_name_[frame_id] = name_str;
            name_to_id_[name_str] = frame_id;
        }
    }

    /// Register a frame (ID computed from name)
    static void register_frame(std::string_view name) { register_frame(name, fnv1a_hash(name)); }

    /// Register parent-child relationship
    static void register_parent(std::string_view child, std::string_view parent) {
        parents_[std::string(child)] = std::string(parent);
    }

    // ========================================================================
    // Alias Management
    // ========================================================================

    /// Register an alias for a canonical frame name
    static void register_alias(std::string_view alias, std::string_view canonical) {
        auto alias_str = std::string(alias);
        if (aliases_.find(alias_str) == aliases_.end()) {
            aliases_[alias_str] = std::string(canonical);
        }
    }

    /// Resolve alias to canonical name (returns input if not an alias)
    static std::string_view resolve(std::string_view name) {
        auto it = aliases_.find(std::string(name));
        if (it != aliases_.end()) {
            return it->second;
        }
        return name;
    }

    // ========================================================================
    // Queries
    // ========================================================================

    /// Get frame ID from name
    static uint64_t get_id(std::string_view name) {
        std::string canonical(resolve(name));
        auto it = name_to_id_.find(canonical);
        if (it != name_to_id_.end()) {
            return it->second;
        }
        // If not registered, compute hash
        return fnv1a_hash(name);
    }

    /// Get frame name from ID
    static std::string_view get_name(uint64_t frame_id) {
        auto it = id_to_name_.find(frame_id);
        if (it != id_to_name_.end()) {
            return it->second;
        }
        return "unknown_frame";
    }

    /// Get parent frame name
    static std::string_view get_parent(std::string_view name) {
        auto it = parents_.find(std::string(name));
        if (it != parents_.end()) {
            return it->second;
        }
        return "";
    }

    /// Check if frame exists in registry
    static bool has_frame(std::string_view name) {
        return name_to_id_.find(std::string(resolve(name))) != name_to_id_.end();
    }

    /// Check if two frames are in the same tree
    static bool in_same_tree(std::string_view frame, std::string_view root) {
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

    // ========================================================================
    // Management
    // ========================================================================

    /// Clear all registrations (useful for testing)
    static void clear() {
        id_to_name_.clear();
        name_to_id_.clear();
        parents_.clear();
        aliases_.clear();
    }

private:
    static inline std::unordered_map<uint64_t, std::string> id_to_name_;
    static inline std::unordered_map<std::string, uint64_t> name_to_id_;
    static inline std::unordered_map<std::string, std::string> parents_;
    static inline std::unordered_map<std::string, std::string> aliases_;
};

} // namespace fast_tf
