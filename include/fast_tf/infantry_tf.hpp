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
 *           ├── imu_link
 *           ├── camera_link
 *           │   └── camera_optical_frame
 *           │       └── object (dynamic, PnP result)
 *           └── muzzle_link
 */

#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

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

// IMU frame (for gimbal-to-IMU calibration)
DECLARE_FRAME(imu_link, gimbal_link);

// Object frame (dynamic, populated by PnP solver)
DECLARE_FRAME(object, camera_optical_frame);

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
        return {"world",       "odom",        "gimbal_link",
                "imu_link",    "camera_link", "camera_optical_frame",
                "muzzle_link", "object"};
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
        StaticTransformStore::register_transform<camera_link_frame, gimbal_link_frame>(
            camera_to_gimbal);

        // Camera optical frame rotation
        // Standard ROS camera optical frame: -90° around X, -90° around Z
        const double pi_2                   = M_PI / 2.0;
        TransformMatrix4d optical_to_camera = TransformMatrix4d::from_rpy(
            -pi_2, // Roll: -90°
            0.0,   // Pitch
            -pi_2, // Yaw: -90°
            0.0, 0.0, 0.0);
        StaticTransformStore::register_transform<camera_optical_frame_frame, camera_link_frame>(
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
        StaticTransformStore::register_transform<muzzle_link_frame, gimbal_link_frame>(
            muzzle_to_gimbal);

        // World to odom (identity initially)
        TransformMatrix4d world_to_odom = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        StaticTransformStore::register_transform<odom_frame, world_frame>(world_to_odom);

        spdlog::info("Infantry robot TF configuration initialized");
    }

    // Infantry-specific accessors
    static const TransformMatrix4d* get_camera_offset() {
        return StaticTransformStore::get_transform<camera_link_frame, gimbal_link_frame>();
    }

    static const TransformMatrix4d* get_muzzle_offset() {
        return StaticTransformStore::get_transform<muzzle_link_frame, gimbal_link_frame>();
    }

    static const TransformMatrix4d* get_optical_frame_transform() {
        return StaticTransformStore::get_transform<camera_optical_frame_frame, camera_link_frame>();
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

    // 存储动态变换（用于查询），默认初始化为identity
    TransformMatrix4d gimbal_to_odom_{};
    TransformMatrix4d odom_to_world_{};

public:
    explicit InfantryTFManager(
        std::optional<foxglove::schemas::FrameTransformsChannel> foxglove_channel)
        : RobotTFManager(std::make_unique<InfantryTFConfig>(), std::move(foxglove_channel)) {}

    /**
     * Update camera optical orientation (rpy).
     * Creates a rotation transform from gimbal_link to camera optical.
     * Thread-safe: can be called concurrently from multiple threads.
     */
    void update_camera_optical(const double roll, const double pitch, const double yaw) noexcept {
        const auto gimbal_transform = TransformMatrix4d::from_rpy(
            roll,  // Roll
            pitch, // Pitch
            yaw,   // Yaw
            0.0,   // X translation
            0.0,   // Y translation
            0.0    // Z translation
        );

        const auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();

        std::unique_lock lock(publish_mutex_);
        gimbal_to_odom_ = gimbal_transform;
        get_publisher().publish_transform<camera_link_frame, gimbal_link_frame>(
            gimbal_transform, timestamp_ns);
        get_publisher().publish_transform<camera_optical_frame_frame, camera_link_frame>(
            gimbal_transform, timestamp_ns);
    }

    /**
     * Update gimbal orientation (rpy).
     * Creates a rotation transform from odom to gimbal_link.
     * Thread-safe: can be called concurrently from multiple threads.
     */
    void update_gimbal(const double roll, const double pitch, const double yaw) noexcept {
        const auto gimbal_transform = TransformMatrix4d::from_rpy(
            roll,  // Roll
            pitch, // Pitch
            yaw,   // Yaw
            0.0,   // X translation
            0.0,   // Y translation
            0.0    // Z translation
        );

        const auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();

        std::unique_lock lock(publish_mutex_);
        gimbal_to_odom_ = gimbal_transform;
        get_publisher().publish_transform<gimbal_link_frame, odom_frame>(
            gimbal_transform, timestamp_ns);
    }

    /**
     * Update robot odometry (position).
     * Thread-safe: can be called concurrently from multiple threads.
     */
    void update_odometry(const double x, const double y, const double z) {
        const auto odom_transform = TransformMatrix4d::from_rpy(
            0.0, // Roll
            0.0, // Pitch
            0.0, // Yaw
            x,   // X translation
            y,   // Y translation
            z    // Z translation
        );
        const auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();
        std::unique_lock lock(publish_mutex_);
        odom_to_world_ = odom_transform;
        get_publisher().publish_transform<odom_frame, world_frame>(odom_transform, timestamp_ns);
    }

    /**
     * Get the current gimbal_link → odom transform.
     * Thread-safe for concurrent reads.
     */
    [[nodiscard]] TransformMatrix4d get_gimbal_to_odom() const {
        std::shared_lock<std::shared_mutex> lock(publish_mutex_);
        return gimbal_to_odom_;
    }

    /**
     * Get the current odom → world transform.
     * Thread-safe for concurrent reads.
     */
    [[nodiscard]] TransformMatrix4d get_odom_to_world() const {
        std::shared_lock<std::shared_mutex> lock(publish_mutex_);
        return odom_to_world_;
    }

    /**
     * Get the current gimbal_link → world transform.
     * Composes gimbal_to_odom * odom_to_world.
     * Thread-safe for concurrent reads.
     */
    [[nodiscard]] TransformMatrix4d get_gimbal_to_world() const {
        std::shared_lock<std::shared_mutex> lock(publish_mutex_);
        return gimbal_to_odom_ * odom_to_world_;
    }

    /**
     * Get the full camera_optical_frame → gimbal_link transform.
     * Composes camera_optical→camera_link→gimbal_link (all static).
     * Returns identity if static transforms are not registered.
     */
    [[nodiscard]] TransformMatrix4d get_camera_optical_to_gimbal() const {
        // Get static transforms
        const auto* optical_to_camera =
            StaticTransformStore::get_transform<camera_optical_frame_frame, camera_link_frame>();
        const auto* camera_to_gimbal =
            StaticTransformStore::get_transform<camera_link_frame, gimbal_link_frame>();

        if (!optical_to_camera || !camera_to_gimbal) {
            spdlog::warn(
                "[InfantryTFManager] Static transforms not registered, returning identity");
            return TransformMatrix4d{};
        }

        return (*optical_to_camera) * (*camera_to_gimbal);
    }

    /**
     * Get the full camera_optical_frame → world transform.
     * Composes camera_optical→camera_link→gimbal_link→odom→world.
     * Thread-safe for concurrent reads.
     */
    [[nodiscard]] TransformMatrix4d get_camera_optical_to_world() const {
        TransformMatrix4d optical_to_gimbal = get_camera_optical_to_gimbal();
        TransformMatrix4d gimbal_to_world   = get_gimbal_to_world();
        return optical_to_gimbal * gimbal_to_world;
    }

    /**
     * Publish object pose from PnP solver.
     * The transform represents object→camera_optical_frame.
     * Thread-safe: can be called concurrently from multiple threads.
     *
     * @param T_object_to_optical Transform from object frame to camera_optical_frame
     * @param timestamp_ns Timestamp in nanoseconds
     */
    void publish_object_pose(const TransformMatrix4d& T_object_to_optical, uint64_t timestamp_ns) {
        std::unique_lock<std::shared_mutex> lock(publish_mutex_);
        get_publisher().publish_transform<object_frame, camera_optical_frame_frame>(
            T_object_to_optical, timestamp_ns);
    }

    /**
     * Update IMU orientation.
     * Computes gimbal_link→imu_link transform from IMU quaternion.
     * Thread-safe: can be called concurrently from multiple threads.
     *
     * @param q_imu_body_to_abs Quaternion from IMU body frame to absolute frame
     * @param R_gimbal_to_imu_body Static calibration: gimbal_link to IMU body rotation
     * @param timestamp_ns Timestamp in nanoseconds
     */
    void update_imu(
        const Eigen::Quaterniond& q_imu_body_to_abs, const Eigen::Matrix3d& R_gimbal_to_imu_body,
        uint64_t timestamp_ns) {
        // Compute gimbal→world rotation via IMU
        // R_gimbal2world = R_gimbal2imubody^T * R_imubody2imuabs * R_gimbal2imubody
        Eigen::Matrix3d R_imu_body_to_abs = q_imu_body_to_abs.toRotationMatrix();
        Eigen::Matrix3d R_gimbal_to_world =
            R_gimbal_to_imu_body.transpose() * R_imu_body_to_abs * R_gimbal_to_imu_body;

        // Create transform matrix from rotation (no translation)
        Eigen::Matrix4d m   = Eigen::Matrix4d::Identity();
        m.block<3, 3>(0, 0) = R_gimbal_to_world;
        TransformMatrix4d imu_transform(m);

        {
            std::unique_lock<std::shared_mutex> lock(publish_mutex_);
            get_publisher().publish_transform<imu_link_frame, gimbal_link_frame>(
                imu_transform, timestamp_ns);
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
 *   tf_manager->get_publisher().publish_transform<gimbal_link_frame, odom_frame>(
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
