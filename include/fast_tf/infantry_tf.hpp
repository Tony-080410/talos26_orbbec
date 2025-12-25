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
 *
 * 设计说明:
 * - 静态变换(camera偏移等)支持从配置文件加载
 * - 动态变换(gimbal姿态)使用CircularTransformBuffer存储历史
 * - 支持带时间戳的插值查询 get_*_at(timestamp_ns)
 */

#include <chrono>
#include <expected>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "buffer.hpp"
#include "error.hpp"
#include "frame_id.hpp"
#include "matrix.hpp"
#include "robot_tf_config.hpp"
#include "static_tf.hpp"

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
 * Infantry robot extrinsic configuration.
 * Loaded from sp_vision.toml.
 */
struct InfantryExtrinsicConfig {
    Eigen::Matrix3d R_camera_gimbal{Eigen::Matrix3d::Identity()}; // 相机到云台旋转
    Eigen::Vector3d t_camera_gimbal{0, 0, 0};                     // 相机到云台平移
    Eigen::Matrix3d R_gimbal_imu{Eigen::Matrix3d::Identity()};    // 云台到IMU旋转
    Eigen::Vector3d t_muzzle_gimbal{0.0874, 0, -0.0874};          // 枪口到云台平移 (默认值)

    // 使用默认ROS光学坐标系旋转
    static constexpr double pi_2 = M_PI / 2.0;
};

/**
 * Infantry robot TF configuration.
 * Implements RobotTFConfig interface for Infantry platform.
 * Supports loading extrinsic parameters from config file.
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

    /**
     * Initialize static transforms with default values (hardcoded).
     * Use initialize_from_config() for configurable initialization.
     */
    void initialize_static_transforms() override {
        InfantryExtrinsicConfig default_config;
        default_config.t_camera_gimbal = Eigen::Vector3d(0.04444, 0, 0);
        initialize_static_transforms_impl(default_config);
    }

    /**
     * Initialize static transforms from external config.
     * @param config Extrinsic configuration loaded from sp_vision.toml
     */
    void initialize_from_config(const InfantryExtrinsicConfig& config) {
        initialize_static_transforms_impl(config);
    }

private:
    void initialize_static_transforms_impl(const InfantryExtrinsicConfig& config) {
        // Camera offset from gimbal (from config)
        Eigen::Matrix4d camera_to_gimbal_m   = Eigen::Matrix4d::Identity();
        camera_to_gimbal_m.block<3, 3>(0, 0) = config.R_camera_gimbal;
        camera_to_gimbal_m.block<3, 1>(0, 3) = config.t_camera_gimbal;
        TransformMatrix4d camera_to_gimbal(camera_to_gimbal_m);
        StaticTransformStore::register_transform<camera_link_frame, gimbal_link_frame>(
            camera_to_gimbal);

        // Camera optical frame rotation (ROS standard: -90° around X, -90° around Z)
        TransformMatrix4d optical_to_camera = TransformMatrix4d::from_rpy(
            -InfantryExtrinsicConfig::pi_2, // Roll: -90°
            0.0,                            // Pitch
            -InfantryExtrinsicConfig::pi_2, // Yaw: -90°
            0.0, 0.0, 0.0);
        StaticTransformStore::register_transform<camera_optical_frame_frame, camera_link_frame>(
            optical_to_camera);

        // Muzzle offset from gimbal (from config)
        TransformMatrix4d muzzle_to_gimbal = TransformMatrix4d::from_rpy(
            0.0, 0.0, 0.0, config.t_muzzle_gimbal.x(), config.t_muzzle_gimbal.y(),
            config.t_muzzle_gimbal.z());
        StaticTransformStore::register_transform<muzzle_link_frame, gimbal_link_frame>(
            muzzle_to_gimbal);

        // World to odom (identity initially)
        TransformMatrix4d world_to_odom = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        StaticTransformStore::register_transform<odom_frame, world_frame>(world_to_odom);

        // Store IMU calibration for later use
        R_gimbal_imu_ = config.R_gimbal_imu;

        spdlog::info("Infantry robot TF configuration initialized from config");
    }

public:
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

    [[nodiscard]] const Eigen::Matrix3d& get_gimbal_imu_rotation() const { return R_gimbal_imu_; }

private:
    Eigen::Matrix3d R_gimbal_imu_{Eigen::Matrix3d::Identity()};
};

/**
 * Infantry-specific TF Manager with dynamic gimbal and odometry support.
 * Extends the generic RobotTFManager with Infantry-specific update methods.
 * Thread-safe for concurrent publish operations from multiple threads.
 *
 * 新增功能:
 * - CircularTransformBuffer 存储gimbal变换历史 (支持时间插值查询)
 * - get_*_at(timestamp_ns) 系列方法用于历史查询
 * - transform_point_to_world() / transform_rotation_to_world_rpy() 便捷方法
 */
class InfantryTFManager : public RobotTFManager {
public:
    // 缓冲区容量：存储约100帧的gimbal变换历史 (~1秒 @ 100Hz)
    static constexpr std::size_t kGimbalBufferCapacity = 100;

private:
    mutable std::shared_mutex publish_mutex_;

    // 存储动态变换（用于查询），默认初始化为identity
    TransformMatrix4d gimbal_to_odom_{};
    TransformMatrix4d odom_to_world_{};

    // 历史变换缓冲区（支持时间插值查询）
    CircularTransformBuffer<kGimbalBufferCapacity> gimbal_buffer_;

    // 缓存的静态变换组合 (optical → gimbal)
    mutable std::optional<TransformMatrix4d> cached_optical_to_gimbal_;

public:
    explicit InfantryTFManager(
        std::optional<foxglove::schemas::FrameTransformsChannel> foxglove_channel)
        : RobotTFManager(std::make_unique<InfantryTFConfig>(), std::move(foxglove_channel)) {}

    // ========================================================================
    // 配置化初始化
    // ========================================================================

    /**
     * Initialize with config from sp_vision.toml.
     * Must be called before using any transform queries.
     */
    void initialize_from_config(const InfantryExtrinsicConfig& config) {
        auto* tf_config = dynamic_cast<InfantryTFConfig*>(get_config());
        if (tf_config) {
            tf_config->initialize_from_config(config);
            // 清除缓存，强制重新计算
            cached_optical_to_gimbal_.reset();
        }
    }

    // ========================================================================
    // 动态变换更新 (带时间戳存入历史缓冲区)
    // ========================================================================

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
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), timestamp_ns);
    }

    /**
     * Update gimbal orientation (rpy) with explicit timestamp.
     * Stores transform in history buffer for interpolated queries.
     * Thread-safe: can be called concurrently from multiple threads.
     */
    void update_gimbal(
        const double roll, const double pitch, const double yaw, uint64_t timestamp_ns) noexcept {
        const auto gimbal_transform = TransformMatrix4d::from_rpy(
            roll,  // Roll
            pitch, // Pitch
            yaw,   // Yaw
            0.0,   // X translation
            0.0,   // Y translation
            0.0    // Z translation
        );

        std::unique_lock lock(publish_mutex_);
        gimbal_to_odom_ = gimbal_transform;
        gimbal_buffer_.push(timestamp_ns, gimbal_transform);
        get_publisher().publish_transform<gimbal_link_frame, odom_frame>(
            gimbal_transform, timestamp_ns);
    }

    /**
     * Update gimbal orientation (rpy) using current time.
     * Convenience overload that generates timestamp automatically.
     */
    void update_gimbal(const double roll, const double pitch, const double yaw) noexcept {
        const auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();
        update_gimbal(roll, pitch, yaw, static_cast<uint64_t>(timestamp_ns));
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

    // ========================================================================
    // 最新变换查询 (无时间戳)
    // ========================================================================

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
     * Uses cached result for performance.
     */
    [[nodiscard]] TransformMatrix4d get_camera_optical_to_gimbal() const {
        std::shared_lock<std::shared_mutex> lock(publish_mutex_);
        if (cached_optical_to_gimbal_) {
            return *cached_optical_to_gimbal_;
        }

        // 计算并缓存
        const auto* optical_to_camera =
            StaticTransformStore::get_transform<camera_optical_frame_frame, camera_link_frame>();
        const auto* camera_to_gimbal =
            StaticTransformStore::get_transform<camera_link_frame, gimbal_link_frame>();

        if (!optical_to_camera || !camera_to_gimbal) {
            spdlog::warn(
                "[InfantryTFManager] Static transforms not registered, returning identity");
            return TransformMatrix4d{};
        }

        auto result = (*optical_to_camera) * (*camera_to_gimbal);

        // 升级锁来缓存结果
        lock.unlock();
        std::unique_lock write_lock(publish_mutex_);
        cached_optical_to_gimbal_ = result;

        return result;
    }

    /**
     * Get the full camera_optical_frame → world transform (latest).
     * Composes camera_optical→camera_link→gimbal_link→odom→world.
     */
    [[nodiscard]] TransformMatrix4d get_camera_optical_to_world() const {
        TransformMatrix4d optical_to_gimbal = get_camera_optical_to_gimbal();
        TransformMatrix4d gimbal_to_world   = get_gimbal_to_world();
        return optical_to_gimbal * gimbal_to_world;
    }

    // ========================================================================
    // 带时间戳的变换查询 (使用历史缓冲区插值)
    // ========================================================================

    /**
     * Get gimbal_link → world transform at specific timestamp.
     * Uses interpolation if exact timestamp not found.
     * @param timestamp_ns Timestamp in nanoseconds
     * @return Transform or error if timestamp out of range
     */
    [[nodiscard]] std::expected<TransformMatrix4d, TransformError>
        get_gimbal_to_world_at(uint64_t timestamp_ns, bool allow_extrapolation = true) const {
        std::shared_lock<std::shared_mutex> lock(publish_mutex_);

        auto gimbal_result =
            allow_extrapolation
                ? gimbal_buffer_.at_interpolated_clamped(static_cast<timestamp_ns_t>(timestamp_ns))
                : gimbal_buffer_.at_interpolated(static_cast<timestamp_ns_t>(timestamp_ns));
        if (!gimbal_result) {
            return std::unexpected(gimbal_result.error());
        }

        return gimbal_result->transform * odom_to_world_;
    }

    /**
     * Get camera_optical_frame → world transform at specific timestamp.
     * Composes static (optical→gimbal) with dynamic (gimbal→world at time).
     * @param timestamp_ns Timestamp in nanoseconds
     * @return Transform or error if timestamp out of range
     */
    [[nodiscard]] std::expected<TransformMatrix4d, TransformError>
        get_camera_optical_to_world_at(uint64_t timestamp_ns) const {
        TransformMatrix4d optical_to_gimbal = get_camera_optical_to_gimbal();

        auto gimbal_to_world_result = get_gimbal_to_world_at(timestamp_ns);
        if (!gimbal_to_world_result) {
            return std::unexpected(gimbal_to_world_result.error());
        }

        return optical_to_gimbal * (*gimbal_to_world_result);
    }

    // ========================================================================
    // 便捷变换方法 (用于Target/Tracker)
    // ========================================================================

    /**
     * Transform a point from camera_optical_frame to world frame.
     * @param p_camera Point in camera_optical_frame (meters)
     * @param timestamp_ns Timestamp for tf lookup
     * @return Point in world frame, or error
     */
    [[nodiscard]] std::expected<Eigen::Vector3d, TransformError>
        transform_point_to_world(const Eigen::Vector3d& p_camera, uint64_t timestamp_ns) const {
        auto T_result = get_camera_optical_to_world_at(timestamp_ns);
        if (!T_result) {
            return std::unexpected(T_result.error());
        }

        Eigen::Vector4d p_homogeneous(p_camera.x(), p_camera.y(), p_camera.z(), 1.0);
        Eigen::Vector4d p_world = T_result->matrix() * p_homogeneous;
        return p_world.head<3>();
    }

    /**
     * Transform a rotation matrix from camera_optical_frame to world frame,
     * and extract roll-pitch-yaw euler angles.
     * @param R_camera Rotation matrix in camera_optical_frame
     * @param timestamp_ns Timestamp for tf lookup
     * @return RPY euler angles in world frame (rad), or error
     */
    [[nodiscard]] std::expected<Eigen::Vector3d, TransformError> transform_rotation_to_world_rpy(
        const Eigen::Matrix3d& R_camera, uint64_t timestamp_ns) const {
        auto T_result = get_camera_optical_to_world_at(timestamp_ns);
        if (!T_result) {
            return std::unexpected(T_result.error());
        }

        Eigen::Matrix3d R_world = T_result->rotation() * R_camera;

        // Extract ZYX euler angles (yaw, pitch, roll)
        Eigen::Vector3d euler = R_world.eulerAngles(2, 1, 0); // ZYX order
        return Eigen::Vector3d(euler[2], euler[1], euler[0]); // roll, pitch, yaw
    }

    /**
     * Transform point from camera_optical_frame to world frame,
     * then compute spherical coordinates (yaw, pitch, distance).
     * @param p_camera Point in camera_optical_frame (meters)
     * @param timestamp_ns Timestamp for tf lookup
     * @return YPD (yaw, pitch, distance) in world frame, or error
     */
    [[nodiscard]] std::expected<Eigen::Vector3d, TransformError>
        transform_to_world_ypd(const Eigen::Vector3d& p_camera, uint64_t timestamp_ns) const {
        auto p_world_result = transform_point_to_world(p_camera, timestamp_ns);
        if (!p_world_result) {
            return std::unexpected(p_world_result.error());
        }

        const Eigen::Vector3d& p = *p_world_result;
        double yaw               = std::atan2(p.y(), p.x());
        double distance          = p.norm();
        double pitch             = std::asin(p.z() / distance);

        return Eigen::Vector3d(yaw, pitch, distance);
    }

    // ========================================================================
    // 其他功能
    // ========================================================================

    /**
     * Publish object pose from PnP solver.
     * The transform represents object→camera_optical_frame.
     * Thread-safe: can be called concurrently from multiple threads.
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

    /**
     * Get the time range covered by the gimbal buffer.
     * @return Pair of (min_timestamp, max_timestamp) or nullopt if empty
     */
    [[nodiscard]] std::optional<std::pair<uint64_t, uint64_t>> get_buffer_time_range() const {
        auto range = gimbal_buffer_.time_range();
        if (range) {
            return std::make_pair(
                static_cast<uint64_t>(range->first), static_cast<uint64_t>(range->second));
        }
        return std::nullopt;
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
