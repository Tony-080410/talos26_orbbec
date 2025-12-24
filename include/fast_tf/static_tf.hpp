#pragma once

#include "frame_path.hpp"
#include "matrix.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace fast_tf {

// ============================================================================
// Static Transform Storage
// ============================================================================

/// Storage for static (unchanging) transforms between frame pairs
/// Static transforms represent fixed relationships like sensor calibrations
class StaticTransformStore {
public:
    /// Register a static transform between two compile-time frames
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    static void register_transform(const TransformMatrix4d& transform) {
        std::string key  = make_key<FromFrame, ToFrame>();
        transforms_[key] = std::make_unique<TransformMatrix4d>(transform);
    }

    /// Look up a static transform between two compile-time frames
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    static const TransformMatrix4d* get_transform() {
        std::string key = make_key<FromFrame, ToFrame>();
        auto it         = transforms_.find(key);
        if (it != transforms_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    /// Look up a static transform by string names (runtime lookup)
    static const TransformMatrix4d*
        get_transform(std::string_view from_name, std::string_view to_name) {
        std::string key = make_key_str(from_name, to_name);
        auto it         = transforms_.find(key);
        if (it != transforms_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    /// Check if a static transform exists between two compile-time frames
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value static bool has_transform() {
        return get_transform<FromFrame, ToFrame>() != nullptr;
    }

    /// Check if a static transform exists (runtime check)
    static bool has_transform(std::string_view from_name, std::string_view to_name) {
        return get_transform(from_name, to_name) != nullptr;
    }

    /// Clear all registered static transforms
    static void clear() { transforms_.clear(); }

    /// Get the number of registered static transforms
    static std::size_t count() { return transforms_.size(); }

private:
    static inline std::unordered_map<std::string, std::unique_ptr<TransformMatrix4d>> transforms_;

    template <typename FromFrame, typename ToFrame>
    static std::string make_key() {
        return make_key_str(FromFrame::frame_name, ToFrame::frame_name);
    }

    static std::string make_key_str(std::string_view from, std::string_view to) {
        return std::string(from) + "->" + std::string(to);
    }
};

// ============================================================================
// Static Transform Query Helper
// ============================================================================

/// Result of querying for a static transform
class StaticTransformQuery {
public:
    StaticTransformQuery()
        : transform_(nullptr) {}

    explicit StaticTransformQuery(const TransformMatrix4d* tf)
        : transform_(tf) {}

    bool has_value() const { return transform_ != nullptr; }
    const TransformMatrix4d& value() const { return *transform_; }
    const TransformMatrix4d* get() const { return transform_; }

private:
    const TransformMatrix4d* transform_;
};

// ============================================================================
// Transform Composition
// ============================================================================

/// Helper to compose static and dynamic transforms in a path
struct TransformComposition {
    static TransformMatrix4d
        compose(const TransformMatrix4d& left, const TransformMatrix4d& right) {
        return left * right;
    }

    static TransformMatrix4d compose_path(
        const TransformMatrix4d& base_transform, const TransformMatrix4d& remaining_transform) {
        return base_transform * remaining_transform;
    }
};

} // namespace fast_tf
