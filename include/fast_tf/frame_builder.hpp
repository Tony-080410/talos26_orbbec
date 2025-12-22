#pragma once

#include <string_view>
#include <unordered_map>
#include <string>

namespace fast_tf {

// ============================================================================
// Frame Buffer Configuration
// ============================================================================

/**
 * Register buffer size for a frame type at runtime.
 * Use this to specify how many transform entries each frame stores.
 */
class FrameBufferConfig {
public:
    /**
     * Register buffer size for a frame type.
     * Should be called during initialization.
     *
     * Usage:
     *   FrameBufferConfig::set_buffer_size<world_Frame>(100);
     *   FrameBufferConfig::set_buffer_size<camera_Frame>(50);
     */
    template<typename FrameType>
    static void set_buffer_size(std::size_t capacity) {
        buffer_sizes[std::string(FrameType::frame_name)] = capacity;
    }

    /**
     * Get buffer size for a frame type.
     * Returns default (100) if not registered.
     */
    static std::size_t get_buffer_size(std::string_view frame_name) {
        auto it = buffer_sizes.find(std::string(frame_name));
        if (it != buffer_sizes.end()) {
            return it->second;
        }
        return 100;  // Default buffer size
    }

    /**
     * Get buffer size for a frame type using template parameter.
     */
    template<typename FrameType>
    static std::size_t get_buffer_size() {
        return get_buffer_size(FrameType::frame_name);
    }

private:
    static std::unordered_map<std::string, std::size_t> buffer_sizes;
};

// ============================================================================
// Frame Metadata and Registration
// ============================================================================

/**
 * Register a frame in the system.
 * This allows runtime lookup of frame names and hierarchy.
 *
 * Usage:
 *   register_frame<world_Frame>();
 *   register_frame<base_link_Frame>();
 */
template<typename FrameType>
void register_frame() {
    // Store frame name in registry for runtime lookup
    FrameBufferConfig::set_buffer_size<FrameType>(100);  // Default size
}

} // namespace fast_tf

// ============================================================================
// Implementation of static members
// ============================================================================

// Must be defined in a .cpp file, but declare here for now
namespace fast_tf {
// Inline definition for header-only library
inline std::unordered_map<std::string, std::size_t> FrameBufferConfig::buffer_sizes;
}

// ============================================================================
// Predefined Frame Names (for convenience)
// ============================================================================

namespace fast_tf::predef {

// Common frame name string literals
constexpr std::string_view WORLD = "world";
constexpr std::string_view ODOM = "odom";
constexpr std::string_view BASE_LINK = "base_link";
constexpr std::string_view BASE_FOOTPRINT = "base_footprint";
constexpr std::string_view MAP = "map";
constexpr std::string_view CAMERA = "camera";
constexpr std::string_view LIDAR = "lidar";
constexpr std::string_view IMU = "imu";

} // namespace fast_tf::predef
