#pragma once

#include "frame_hierarchy.hpp"
#include "frame_types.hpp" // For TransformMatrix4d
#include <memory>
#include <string>
#include <unordered_map>

namespace fast_tf {

// ============================================================================
// Static Transform Storage
// ============================================================================

/**
 * Storage for static (unchanging) transforms between frame pairs.
 * Static transforms represent fixed relationships like sensor calibrations.
 *
 * Unlike dynamic transforms, static transforms:
 * - Don't require a timestamp parameter
 * - Are queried directly without interpolation
 * - Reduce memory footprint significantly
 * - Are typically computed once at initialization
 *
 * Example use cases:
 * - Camera offset from robot base
 * - IMU mounting orientation
 * - Lidar position relative to chassis
 */
class StaticTransformStore {
public:
    /**
     * Register a static transform between two frames.
     * Can be called multiple times for different frame pairs.
     *
     * Usage:
     *   TransformMatrix4d camera_offset = TransformMatrix4d::from_rpy(...);
     *   StaticTransformStore::register_transform<camera_Frame, base_link_Frame>(
     *       camera_offset);
     */
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    static void register_transform(const TransformMatrix4d& transform) {
        std::string key = make_key<FromFrame, ToFrame>();
        transforms[key] = std::make_unique<TransformMatrix4d>(transform);
    }

    /**
     * Look up a static transform between two frames.
     * Returns nullptr if no static transform is registered.
     */
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    static const TransformMatrix4d* get_transform() {
        std::string key = make_key<FromFrame, ToFrame>();
        auto it         = transforms.find(key);
        if (it != transforms.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    /**
     * Look up a static transform by string names (runtime lookup).
     * Returns nullptr if not found.
     */
    static const TransformMatrix4d*
        get_transform(std::string_view from_name, std::string_view to_name) {
        std::string key = make_key_str(from_name, to_name);
        auto it         = transforms.find(key);
        if (it != transforms.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    /**
     * Check if a static transform exists between two frames.
     */
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value static bool has_transform() {
        return get_transform<FromFrame, ToFrame>() != nullptr;
    }

    /**
     * Check if a static transform exists (runtime check).
     */
    static bool has_transform(std::string_view from_name, std::string_view to_name) {
        return get_transform(from_name, to_name) != nullptr;
    }

    /**
     * Clear all registered static transforms (useful for testing).
     */
    static void clear() { transforms.clear(); }

    /**
     * Get the number of registered static transforms (for debugging).
     */
    static std::size_t count() { return transforms.size(); }

private:
    // Storage: key is "from_frame_name->to_frame_name"
    static std::unordered_map<std::string, std::unique_ptr<TransformMatrix4d>> transforms;

    /**
     * Create a unique key for a frame pair.
     */
    template <typename FromFrame, typename ToFrame>
    static std::string make_key() {
        return make_key_str(FromFrame::frame_name, ToFrame::frame_name);
    }

    /**
     * Create a unique key from string names.
     */
    static std::string make_key_str(std::string_view from, std::string_view to) {
        return std::string(from) + "->" + std::string(to);
    }
};

// ============================================================================
// Static Transform Query Helper
// ============================================================================

/**
 * Result of querying for a static transform.
 * Can hold either the transform value or indicate no static transform.
 */
class StaticTransformQuery {
public:
    StaticTransformQuery()
        : transform(nullptr) {}

    explicit StaticTransformQuery(const TransformMatrix4d* tf)
        : transform(tf) {}

    /**
     * Check if a static transform was found.
     */
    bool has_value() const { return transform != nullptr; }

    /**
     * Get the transform (only valid if has_value() is true).
     */
    const TransformMatrix4d& value() const { return *transform; }

    /**
     * Get the transform or nullptr if not found.
     */
    const TransformMatrix4d* get() const { return transform; }

private:
    const TransformMatrix4d* transform;
};

// ============================================================================
// Composition of Static and Dynamic Transforms
// ============================================================================

/**
 * Helper to compose static and dynamic transforms in a path.
 *
 * When a path from Frame A to Frame B contains:
 * - A → C (static)
 * - C → B (dynamic)
 *
 * Result = static(A→C) × dynamic(C→B)
 *
 * This is used internally by TransformTree when computing chain transforms.
 */
struct TransformComposition {
    /**
     * Compose two transforms: left × right
     * Used when we have both static and dynamic transforms in a path.
     */
    static TransformMatrix4d
        compose(const TransformMatrix4d& left, const TransformMatrix4d& right) {
        return left * right;
    }

    /**
     * Check if a path contains any static transforms and compose accordingly.
     */
    static TransformMatrix4d compose_path(
        const TransformMatrix4d& base_transform, const TransformMatrix4d& remaining_transform) {
        // Check if there's a static transform in the path
        // For now, simple composition: base * remaining
        return base_transform * remaining_transform;
    }
};

} // namespace fast_tf

// ============================================================================
// Implementation of static members
// ============================================================================

namespace fast_tf {
// Inline definition for header-only library
inline std::unordered_map<std::string, std::unique_ptr<TransformMatrix4d>>
    StaticTransformStore::transforms;
} // namespace fast_tf
