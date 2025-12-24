#pragma once

/**
 * Generic Robot TF Configuration Framework
 *
 * This framework allows defining TF trees for different robot types with:
 * - Compile-time frame hierarchy validation using fast_tf
 * - Direct Foxglove SDK integration for visualization
 * - Pluggable robot configurations
 *
 * Usage:
 *   1. Define frame hierarchy using DECLARE_FRAME (in fast_tf namespace)
 *   2. Implement RobotTFConfig interface
 *   3. Create RobotTFManager instance
 */

#include <chrono>
#include <cstdint>
#include <foxglove/foxglove.hpp>
#include <foxglove/schemas.hpp>
#include <memory>
#include <optional>
#include <string_view>

#include "frame_hierarchy.hpp"
#include "frame_types.hpp"
#include "static_transforms.hpp"

#include <spdlog/spdlog.h>

namespace fast_tf {

// ============================================================================
// Robot TF Configuration Interface
// ============================================================================

/**
 * Abstract base class for robot-specific TF configurations.
 * Each robot type (Infantry, Sentry, Hero, etc.) should implement this interface.
 */
class RobotTFConfig {
public:
    virtual ~RobotTFConfig() = default;

    /**
     * Get robot type name (e.g., "Infantry", "Sentry", "Hero")
     */
    virtual std::string_view robot_type() const = 0;

    /**
     * Initialize all static transforms for this robot.
     * This includes sensor calibrations, mechanical offsets, etc.
     */
    virtual void initialize_static_transforms() = 0;

    /**
     * Get the root frame for this robot's TF tree.
     * Returns the frame name (e.g., "world", "odom")
     */
    virtual std::string_view get_root_frame() const = 0;

    /**
     * Get list of all frame names in this robot's TF tree.
     * Used for validation and debugging.
     */
    virtual std::vector<std::string> get_all_frame_names() const = 0;
};

// ============================================================================
// Generic Robot TF Publisher - Foxglove Integration
// ============================================================================

/**
 * Bridge between fast_tf and Foxglove SDK.
 * Publishes any robot's TF tree to Foxglove for visualization.
 */
class RobotTFPublisher {
public:
    explicit RobotTFPublisher(std::optional<foxglove::schemas::FrameTransformsChannel> channel)
        : channel_(std::move(channel)) {}

    /**
     * Publish a generic transform by frame type.
     */
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    void publish_transform(const TransformMatrix4d& transform, uint64_t timestamp_ns = 0) {

        publish_transform(transform, timestamp_ns, FromFrame::frame_name, ToFrame::frame_name);
    }

    /**
     * Publish a transform by frame names (runtime lookup).
     */
    void publish_transform(
        const TransformMatrix4d& transform, uint64_t timestamp_ns, std::string_view from_name,
        std::string_view to_name) {

        if (!channel_) {
            return;
        }

        // Extract translation
        Eigen::Vector3d translation = transform.translation();

        // Extract quaternion from rotation matrix
        Eigen::Quaterniond quat = transform.quaternion();

        // Create Foxglove FrameTransform
        foxglove::schemas::FrameTransform frame_transform;
        frame_transform.parent_frame_id = std::string(to_name);
        frame_transform.child_frame_id  = std::string(from_name);
        frame_transform.timestamp       = {
            static_cast<uint32_t>(timestamp_ns / 1'000'000'000),
            static_cast<uint32_t>(timestamp_ns % 1'000'000'000)};
        frame_transform.translation = {translation.x(), translation.y(), translation.z()};
        frame_transform.rotation    = {quat.x(), quat.y(), quat.z(), quat.w()};

        // Publish
        foxglove::schemas::FrameTransforms transforms_msg;
        transforms_msg.transforms.push_back(frame_transform);
        auto error = channel_->log(transforms_msg);
        (void)error;
    }

    /**
     * Publish all static transforms.
     */
    static void publish_static_transforms(uint64_t timestamp_ns = 0) {
        spdlog::debug("Publishing static transforms to Foxglove");
    }

    /**
     * Check if publisher is connected.
     */
    bool is_connected() const { return channel_.has_value(); }

private:
    std::optional<foxglove::schemas::FrameTransformsChannel> channel_;
};

// ============================================================================
// Generic Robot TF Manager
// ============================================================================

/**
 * Generic TF manager that works with any RobotTFConfig implementation.
 * Combines compile-time safety with runtime flexibility.
 */
class RobotTFManager {
public:
    /**
     * Create a TF manager for a specific robot configuration.
     */
    explicit RobotTFManager(
        std::unique_ptr<RobotTFConfig> config,
        std::optional<foxglove::schemas::FrameTransformsChannel> foxglove_channel = std::nullopt)
        : config_(std::move(config))
        , publisher_(std::move(foxglove_channel)) {

        // Initialize robot-specific configuration
        config_->initialize_static_transforms();

        // Publish static transforms if Foxglove is available
        if (foxglove_channel) {
            publisher_->publish_static_transforms();
        }

        spdlog::info("RobotTFManager initialized for {} robot", config_->robot_type());
    }

    /**
     * Get the robot configuration.
     */
    RobotTFConfig* get_config() const { return config_.get(); }

    /**
     * Get the Foxglove publisher.
     */
    RobotTFPublisher& get_publisher() { return *publisher_; }

    /**
     * Get a static transform using template-based query.
     */
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    const TransformMatrix4d* get_static_transform() {
        return StaticTransformStore::get_transform<FromFrame, ToFrame>();
    }

    /**
     * Check if a static transform exists.
     */
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    bool has_static_transform() const {
        return StaticTransformStore::has_transform<FromFrame, ToFrame>();
    }

    /**
     * Manually set a static transform (for calibration/testing).
     */
    template <typename FromFrame, typename ToFrame>
    requires validate_frames_same_tree<FromFrame, ToFrame>::value
    void set_static_transform(const TransformMatrix4d& transform) {
        StaticTransformStore::register_transform<FromFrame, ToFrame>(transform);

        // Also publish to Foxglove if available
        if (publisher_->is_connected()) {
            uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count();
            publisher_->publish_transform<FromFrame, ToFrame>(transform, timestamp_ns);
        }
    }

private:
    std::unique_ptr<RobotTFConfig> config_;
    std::optional<RobotTFPublisher> publisher_;
};

} // namespace fast_tf
