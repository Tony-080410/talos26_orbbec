#pragma once

/**
 * Infantry Robot TF Configuration
 *
 * Concrete implementation of RobotTFConfig for Infantry robot.
 * Defines the complete kinematic tree with sensor calibrations.
 *
 * Tree Structure (ROS2 URDF-compatible):
 *   world
 *   └── odom
 *       └── gimbal_link
 *           ├── camera_link
 *           │   └── camera_optical_frame
 *           └── muzzle_link
 */

#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "frame_id.hpp"
#include "frame_types.hpp"
#include "robot_tf_config.hpp"
#include "static_transforms.hpp"

#include <spdlog/spdlog.h>

// ============================================================================
// Infantry Robot Frame Declarations
// ============================================================================

namespace fast_tf {

/**
 * Define the complete frame hierarchy for Infantry robot.
 * Using DECLARE_FRAME macro for compile-time type safety.
 */

// Root frame
DECLARE_FRAME(world);

// Odometry frame
DECLARE_FRAME(odom, world);

// Gimbal frame
DECLARE_FRAME(gimbal_link, odom);

// Camera frame
DECLARE_FRAME(camera_link, gimbal_link);

// Camera optical frame
DECLARE_FRAME(camera_optical_frame, camera_link);

// Muzzle frame
DECLARE_FRAME(muzzle_link, gimbal_link);

// ============================================================================
// Infantry Robot TF Configuration
// ============================================================================

namespace infantry {

/**
 * Infantry robot TF configuration.
 * Implements RobotTFConfig interface for Infantry platform.
 */
class InfantryTFConfig : public RobotTFConfig {
public:
    InfantryTFConfig() = default;

    std::string_view robot_type() const override { return "Infantry"; }

    std::string_view get_root_frame() const override { return "world"; }

    std::vector<std::string> get_all_frame_names() const override {
        return {"world",      "odom", "gimbal_link", "camera_link", "camera_optical_frame",
                "muzzle_link"};
    }

    void initialize_static_transforms() override {
        // Camera offset from gimbal
        // Values from rm_gimbal.urdf.xacro
        TransformMatrix4d camera_to_gimbal = TransformMatrix4d::from_rpy(
            0.0,     // Roll
            0.0,     // Pitch
            0.0,     // Yaw
            0.04444, // X offset (meters)
            0.0,     // Y offset
            0.0      // Z offset
        );
        StaticTransformStore::register_transform<camera_link_Frame, gimbal_link_Frame>(
            camera_to_gimbal);

        // Camera optical frame rotation
        // Standard ROS camera optical frame: -90° around X, -90° around Z
        const double pi_2                   = M_PI / 2.0;
        TransformMatrix4d optical_to_camera = TransformMatrix4d::from_rpy(
            -pi_2, // Roll: -90°
            0.0,   // Pitch
            -pi_2, // Yaw: -90°
            0.0, 0.0, 0.0);
        StaticTransformStore::register_transform<camera_optical_frame_Frame, camera_link_Frame>(
            optical_to_camera);

        // Muzzle offset from gimbal
        // Values from rm_gimbal.urdf.xacro
        TransformMatrix4d muzzle_to_gimbal = TransformMatrix4d::from_rpy(
            0.0,    // Roll
            0.0,    // Pitch
            0.0,    // Yaw
            0.0874, // X offset (meters)
            0.0,    // Y offset
            -0.0874 // Z offset
        );
        StaticTransformStore::register_transform<muzzle_link_Frame, gimbal_link_Frame>(
            muzzle_to_gimbal);

        // World to odom (identity initially)
        TransformMatrix4d world_to_odom = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        StaticTransformStore::register_transform<odom_Frame, world_Frame>(world_to_odom);

        spdlog::info("Infantry robot TF configuration initialized");
    }

    // Infantry-specific accessors
    static const TransformMatrix4d* get_camera_offset()  {
        return StaticTransformStore::get_transform<camera_link_Frame, gimbal_link_Frame>();
    }

    static const TransformMatrix4d* get_muzzle_offset()  {
        return StaticTransformStore::get_transform<muzzle_link_Frame, gimbal_link_Frame>();
    }

    static const TransformMatrix4d* get_optical_frame_transform()  {
        return StaticTransformStore::get_transform<camera_optical_frame_Frame, camera_link_Frame>();
    }
};

/**
 * Infantry-specific TF Manager with dynamic gimbal and odometry support.
 * Extends the generic RobotTFManager with Infantry-specific update methods.
 * Thread-safe for concurrent publish operations from multiple threads.
 */
class InfantryTFManager : public RobotTFManager {
private:
    mutable std::shared_mutex publish_mutex_;

public:
    explicit InfantryTFManager(
        std::optional<foxglove::schemas::FrameTransformsChannel> foxglove_channel)
        : RobotTFManager(std::make_unique<InfantryTFConfig>(), std::move(foxglove_channel)) {}

    /**
     * Update gimbal orientation (yaw and pitch angles).
     * Creates a rotation transform from odom to gimbal_link.
     * Thread-safe: can be called concurrently from multiple threads.
     */
    void update_gimbal(double yaw, double pitch) {
        // Use from_rpy to create gimbal transform
        // Roll is typically 0 for gimbal, pitch and yaw are the active angles
        TransformMatrix4d gimbal_transform = TransformMatrix4d::from_rpy(
            0.0,   // Roll
            pitch, // Pitch
            yaw,   // Yaw
            0.0,   // X translation
            0.0,   // Y translation
            0.0    // Z translation
        );

        // Publish the gimbal transform with thread safety
        uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();

        {
            std::unique_lock<std::shared_mutex> lock(publish_mutex_);
            get_publisher().publish_transform<gimbal_link_Frame, odom_Frame>(
                gimbal_transform, timestamp_ns);
        }
    }

    /**
     * Update robot odometry (position and heading).
     * Creates a transform from world to odom based on x, y position and theta heading.
     * Thread-safe: can be called concurrently from multiple threads.
     */
    void update_odometry(double x, double y, double theta) {
        // Create odometry transform with heading angle and position
        TransformMatrix4d odom_transform = TransformMatrix4d::from_rpy(
            0.0,   // Roll
            0.0,   // Pitch
            theta, // Yaw (heading)
            x,     // X translation
            y,     // Y translation
            0.0    // Z translation
        );

        // Publish the odometry transform with thread safety
        uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();

        {
            std::unique_lock<std::shared_mutex> lock(publish_mutex_);
            get_publisher().publish_transform<odom_Frame, world_Frame>(
                odom_transform, timestamp_ns);
        }
    }
};

} // namespace infantry

} // namespace fast_tf

// ============================================================================
// Example Usage - How to create TF managers for different robots
// ============================================================================

/**
 * EXAMPLE 1: Infantry Robot
 *
 *   auto infantry_config = std::make_unique<fast_tf::infantry::InfantryTFConfig>();
 *   auto tf_manager = std::make_unique<fast_tf::RobotTFManager>(
 *       std::move(infantry_config), foxglove_tf_channel);
 *
 *   // Update gimbal angle
 *   tf_manager->get_publisher().publish_transform<gimbal_link_Frame, odom_Frame>(
 *       gimbal_transform, timestamp_ns);
 *
 *
 * EXAMPLE 2: Adding a new robot type (e.g., Sentry)
 *
 *   // 1. Declare frames in sentry_tf.hpp
 *   namespace fast_tf {
 *       DECLARE_FRAME(sentry_world);
 *       DECLARE_FRAME(sentry_base, sentry_world);
 *       DECLARE_FRAME(sentry_turret, sentry_base);
 *       // ... more frames
 *   }
 *
 *   // 2. Implement RobotTFConfig for Sentry
 *   class SentryTFConfig : public RobotTFConfig {
 *   public:
 *       std::string_view robot_type() const override { return "Sentry"; }
 *       void initialize_static_transforms() override {
 *           // Register Sentry-specific transforms
 *       }
 *       // ... implement other methods
 *   };
 *
 *   // 3. Use the same RobotTFManager interface
 *   auto sentry_config = std::make_unique<SentryTFConfig>();
 *   auto sentry_tf = std::make_unique<fast_tf::RobotTFManager>(
 *       std::move(sentry_config), foxglove_channel);
 */
