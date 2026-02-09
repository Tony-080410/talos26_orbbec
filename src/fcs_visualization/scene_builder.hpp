#pragma once

// This header provides a type-safe, composable builder pattern for creating
// Foxglove scene entities with semantic styling.
//
// Design Philosophy:
// - Fluent interface: chain methods for readable code
// - Semantic defaults: colors, sizes from tactical palette
// - Convenience patterns: common operations in single function calls
//
// Usage Example:
// ```cpp
// using namespace fcs::visualization::builder;
// using namespace tactical;
//
// auto entity = EntityBuilder::create("l3", "target")
//     .position(1.0, 2.0, 3.0)
//     .color(L3::TRACKING_LOCKED)
//     .size(L3::TARGET_SIZE)
//     .sphere()
//     .build();
// ```

#include "foxglove_server.hpp"
#include "foxglove_types.hpp"
#include "frame.hpp"
#include "tactical_palette.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <fmt/core.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fcs::visualization::builder {

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// @brief Create Vector3 from components
inline foxglove::schemas::Vector3 make_vector3(double x, double y, double z) noexcept {
    foxglove::schemas::Vector3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

/// @brief Create Vector3 from Eigen::Vector3d
inline foxglove::schemas::Vector3 make_vector3(const Eigen::Vector3d& vec) noexcept {
    return make_vector3(vec.x(), vec.y(), vec.z());
}

/// @brief Create Quaternion from components
inline foxglove::schemas::Quaternion
    make_quaternion(double x, double y, double z, double w) noexcept {
    foxglove::schemas::Quaternion q;
    q.x = x;
    q.y = y;
    q.z = z;
    q.w = w;
    return q;
}

/// @brief Create Quaternion from Eigen::Quaterniond
inline foxglove::schemas::Quaternion make_quaternion(const Eigen::Quaterniond& q) noexcept {
    return make_quaternion(q.x(), q.y(), q.z(), q.w());
}

/// @brief Create identity quaternion
inline foxglove::schemas::Quaternion identity_quaternion() noexcept {
    return make_quaternion(0.0, 0.0, 0.0, 1.0);
}

/// @brief Create Color from components
inline foxglove::schemas::Color make_color(double r, double g, double b, double a) noexcept {
    foxglove::schemas::Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    c.a = a;
    return c;
}

/// @brief Create Pose from position and orientation
inline foxglove::schemas::Pose make_pose(
    std::optional<foxglove::schemas::Vector3> position       = std::nullopt,
    std::optional<foxglove::schemas::Quaternion> orientation = std::nullopt) noexcept {
    foxglove::schemas::Pose pose;
    pose.position    = position;
    pose.orientation = orientation;
    return pose;
}

/// @brief Create Pose from position with identity orientation
inline foxglove::schemas::Pose make_pose(const foxglove::schemas::Vector3& position) noexcept {
    return make_pose(position, identity_quaternion());
}

// ============================================================================
// ENTITY BUILDER
// ============================================================================

/// @brief Fluent builder for Foxglove SceneEntity
///
/// Provides a type-safe, composable interface for building scene entities
/// with semantic styling from the tactical palette.
///
/// Thread Safety: Not thread-safe. Create and use builders within a single thread.
///
/// Design Notes:
/// - SceneEntity does not have a top-level pose; each primitive has its own pose
/// - This builder maintains a "current_pose" that gets applied to primitives
/// - Call position()/orientation() before adding primitives to set their pose
class EntityBuilder {
public:
    /// @brief Create a new EntityBuilder with frame template parameter
    template <fast_tf::frame T>
    [[nodiscard]] static EntityBuilder create(std::string layer, std::string entity_id) {
        EntityBuilder builder;
        builder.entity_.id       = fmt::format("{}.{}", layer, entity_id);
        builder.entity_.frame_id = T::frame_id;
        builder.layer_           = std::move(layer);
        return builder;
    }

    // ========================================================================
    // POSITION & ORIENTATION
    // ========================================================================

    /// @brief Set position from components
    EntityBuilder& position(double x, double y, double z) {
        current_pose_.position = make_vector3(x, y, z);
        return *this;
    }

    /// @brief Set position from Eigen::Vector3d
    EntityBuilder& position(const Eigen::Vector3d& pos) {
        return position(pos.x(), pos.y(), pos.z());
    }

    /// @brief Set position from Vector3
    EntityBuilder& position(const foxglove::schemas::Vector3& pos) {
        current_pose_.position = pos;
        return *this;
    }

    /// @brief Set orientation from quaternion components
    EntityBuilder& orientation(double qx, double qy, double qz, double qw) {
        current_pose_.orientation = make_quaternion(qx, qy, qz, qw);
        return *this;
    }

    /// @brief Set orientation from Eigen::Quaterniond
    EntityBuilder& orientation(const Eigen::Quaterniond& q) {
        return orientation(q.x(), q.y(), q.z(), q.w());
    }

    /// @brief Set orientation from Quaternion
    EntityBuilder& orientation(const foxglove::schemas::Quaternion& q) {
        current_pose_.orientation = q;
        return *this;
    }

    /// @brief Set orientation from Euler angles (yaw, pitch, roll)
    EntityBuilder& orientation_euler(double yaw, double pitch, double roll) {
        Eigen::Quaterniond q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
                             * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
                             * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
        return orientation(q);
    }

    /// @brief Reset position to origin (for primitives at origin in current frame)
    EntityBuilder& reset_position() {
        current_pose_.position = make_vector3(0, 0, 0);
        return *this;
    }

    /// @brief Reset orientation to identity
    EntityBuilder& reset_orientation() {
        current_pose_.orientation = identity_quaternion();
        return *this;
    }

    // ========================================================================
    // VISUAL PROPERTIES
    // ========================================================================

    /// @brief Set color from components
    EntityBuilder& color(double r, double g, double b, double a = 1.0) {
        current_color_ = make_color(r, g, b, a);
        return *this;
    }

    /// @brief Set color from Color
    EntityBuilder& color(const foxglove::schemas::Color& c) {
        current_color_ = c;
        return *this;
    }

    /// @brief Set size (uniform for all axes)
    EntityBuilder& size(double s) {
        current_size_ = {s, s, s};
        return *this;
    }

    /// @brief Set size from components
    EntityBuilder& size(double sx, double sy, double sz) {
        current_size_ = {sx, sy, sz};
        return *this;
    }

    /// @brief Set alpha (transparency) for current color
    EntityBuilder& alpha(double a) {
        if (current_color_.has_value()) {
            current_color_->a = static_cast<float>(a);
        }
        return *this;
    }

    // ========================================================================
    // PRIMITIVES
    // ========================================================================

    /// @brief Add a sphere primitive with current pose, size, and color
    EntityBuilder& sphere() {
        foxglove::schemas::SpherePrimitive prim;
        prim.pose  = current_pose_;
        prim.size  = current_size_.value_or(foxglove::schemas::Vector3{0.1, 0.1, 0.1});
        prim.color = current_color_.value_or(tactical::L3::TRACKING_LOCKED);
        entity_.spheres.push_back(prim);
        return *this;
    }

    /// @brief Add a cube primitive with specified dimensions
    EntityBuilder& cube(double width, double height, double depth) {
        foxglove::schemas::CubePrimitive prim;
        prim.pose  = current_pose_;
        prim.size  = {width, height, depth};
        prim.color = current_color_.value_or(tactical::L2::MEASUREMENT_GHOST);
        entity_.cubes.push_back(prim);
        return *this;
    }

    /// @brief Add text label
    EntityBuilder& text(std::string content, double font_size = tactical::Text::SIZE_MEDIUM) {
        foxglove::schemas::TextPrimitive prim;
        prim.text      = std::move(content);
        prim.pose      = current_pose_;
        prim.font_size = font_size;
        prim.color     = current_color_.value_or(tactical::Text::PRIMARY);
        prim.billboard = tactical::Text::BILLBOARD_ENABLED;
        entity_.texts.push_back(prim);
        return *this;
    }

    /// @brief Add text label at offset from current position
    EntityBuilder& text_with_offset(
        std::string content, double offset_x, double offset_y, double offset_z,
        double font_size = tactical::Text::SIZE_MEDIUM) {
        foxglove::schemas::TextPrimitive prim;
        prim.text = std::move(content);

        auto text_pose = current_pose_;
        if (text_pose.position.has_value()) {
            text_pose.position->x += offset_x;
            text_pose.position->y += offset_y;
            text_pose.position->z += offset_z;
        }
        prim.pose      = text_pose;
        prim.font_size = font_size;
        prim.color     = current_color_.value_or(tactical::Text::PRIMARY);
        prim.billboard = tactical::Text::BILLBOARD_ENABLED;
        entity_.texts.push_back(prim);
        return *this;
    }

    // ========================================================================
    // METADATA
    // ========================================================================

    /// @brief Add metadata key-value pair
    EntityBuilder& metadata(std::string key, std::string value) {
        foxglove::schemas::KeyValuePair kv;
        kv.key   = std::move(key);
        kv.value = std::move(value);
        entity_.metadata.push_back(kv);
        return *this;
    }

    /// @brief Set timestamp for the entity
    EntityBuilder& timestamp(uint64_t ns) {
        entity_.timestamp = ::fcs::timestamp_from_ns(ns);
        return *this;
    }

    /// @brief Set entity lifetime (nanoseconds)
    EntityBuilder& lifetime(uint64_t ns) {
        entity_.lifetime = foxglove::schemas::Duration{static_cast<int32_t>(ns)};
        return *this;
    }

    // ========================================================================
    // CLEAR & RETAIN
    // ========================================================================

    /// @brief Clear all accumulated primitives (spheres, cubes, texts, arrows, etc.)
    /// @return Reference to this builder for chaining
    ///
    /// Use this to reuse a builder instance in a loop:
    /// ```cpp
    /// auto builder = EntityBuilder::create<Frame>("l1", "dynamic");
    /// for (const auto& item : items) {
    ///     builder.clear()
    ///         .position(item.pos)
    ///         .sphere()
    ///         .publish();
    /// }
    /// ```
    EntityBuilder& clear() noexcept {
        entity_.spheres.clear();
        entity_.cubes.clear();
        entity_.texts.clear();
        entity_.arrows.clear();
        entity_.lines.clear();
        entity_.triangles.clear();
        entity_.cylinders.clear();
        entity_.models.clear();
        return *this;
    }

    /// @brief Clear primitives but retain metadata (timestamp, lifetime, custom metadata)
    /// @return Reference to this builder for chaining
    ///
    /// Similar to clear(), but preserves metadata that was set up before the loop:
    /// ```cpp
    /// auto builder = EntityBuilder::create<Frame>("l1", "batch")
    ///     .timestamp(now)
    ///     .lifetime(100ms);
    /// for (const auto& item : items) {
    ///     builder.retain()
    ///         .position(item.pos)
    ///         .sphere()
    ///         .publish();
    /// }
    /// ```
    EntityBuilder& retain() noexcept {
        // Clear only primitives, preserve metadata
        clear();
        return *this;
    }

    // ========================================================================
    // BUILD
    // ========================================================================

    /// @brief Build and return the SceneEntity
    [[nodiscard]] foxglove::schemas::SceneEntity build() && {
        // Set defaults if not explicitly set
        if (!entity_.timestamp.has_value()) {
            entity_.timestamp = ::fcs::timestamp_from_ns(0);
        }
        if (!entity_.lifetime.has_value()) {
            entity_.lifetime =
                foxglove::schemas::Duration{.nsec = tactical::Temporal::ENTITY_LIFETIME_NS};
        }

        return std::move(entity_);
    }

    /// @brief Build and return the SceneEntity (copy overload)
    [[nodiscard]] foxglove::schemas::SceneEntity build() const& {
        foxglove::schemas::SceneEntity result = entity_;
        if (!result.timestamp.has_value()) {
            result.timestamp = ::fcs::timestamp_from_ns(0);
        }
        if (!result.lifetime.has_value()) {
            result.lifetime =
                foxglove::schemas::Duration{.nsec = tactical::Temporal::ENTITY_LIFETIME_NS};
        }
        return result;
    }

private:
    EntityBuilder() = default;

    foxglove::schemas::SceneEntity entity_;
    std::string layer_;
    foxglove::schemas::Pose current_pose_{};
    std::optional<foxglove::schemas::Color> current_color_;
    std::optional<foxglove::schemas::Vector3> current_size_;
};

// ============================================================================
// CONVENIENCE PATTERNS
// ============================================================================

namespace patterns {

// ============================================================================
// L2: MEASUREMENT VISUALIZATION
// ============================================================================

/// @brief Create a ghost measurement sphere (L2 perception layer)
/// @param pos Position of the measurement
/// @param confidence Confidence score (0-1), affects opacity
/// @param timestamp_ns Entity timestamp for TF interpolation
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity measurement_sphere(
    const Eigen::Vector3d& pos, double confidence, uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    const auto color = (confidence > 0.5) ? L2::MEASUREMENT_CONFIDENCE : L2::MEASUREMENT_GHOST;

    return EntityBuilder::create<Frame>("l2", "measurement")
        .position(pos)
        .size(L2::ARMOR_SIZE)
        .color(color)
        .sphere()
        .metadata("confidence", fmt::format("{:.3f}", confidence))
        .timestamp(timestamp_ns)
        .build();
}

/// @brief Create a measurement cube with full pose information
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity measurement_cube(
    const Eigen::Vector3d& pos, const Eigen::Quaterniond& orientation, const std::string& label,
    double confidence = 0.0) noexcept {
    using namespace tactical;

    const auto color = L2::MEASUREMENT_GHOST;

    return EntityBuilder::create<Frame>("l2", "measurement")
        .position(pos)
        .orientation(orientation)
        .cube(Geometry::ARMOR_THICKNESS, Geometry::ARMOR_HEIGHT_SMALL, Geometry::ARMOR_WIDTH)
        .text_with_offset(label, 0, -Text::SIZE_MEDIUM, 0, L2::LABEL_FONT_SIZE)
        .metadata("confidence", fmt::format("{:.3f}", confidence))
        .build();
}

// ============================================================================
// L3: TRACKER VISUALIZATION
// ============================================================================

/// @brief Create tracker target sphere with status-based coloring
/// @param pos Target position
/// @param status Tracker status (determines color)
/// @param timestamp_ns Entity timestamp for TF interpolation
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    tracker_target(const Eigen::Vector3d& pos, int status_int, uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    return EntityBuilder::create<Frame>("l3", "target")
        .position(pos)
        .size(L3::TARGET_SIZE)
        .color(tracker_status_color(status_int))
        .sphere()
        .timestamp(timestamp_ns)
        .build();
}

/// @brief Create predicted armor sphere
/// @param pos Predicted armor position
/// @param armor_id Armor index (0-3 for robot, 0-2 for outpost)
/// @param timestamp_ns Entity timestamp for TF interpolation
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    predicted_armor(const Eigen::Vector3d& pos, int armor_id, uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    return EntityBuilder::create<Frame>("l3", fmt::format("armor_{}", armor_id))
        .position(pos)
        .size(L3::ARMOR_PLATE_SIZE)
        .color(L3::PREDICTION_AMBER)
        .sphere()
        .text_with_offset(
            fmt::format("pred_{}", armor_id), 0, 0, L3::LABEL_OFFSET_Z, Text::SIZE_MEDIUM)
        .timestamp(timestamp_ns)
        .build();
}

/// @brief Create robot center marker
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    robot_center(const Eigen::Vector3d& pos, int status_int) noexcept {
    using namespace tactical;

    return EntityBuilder::create<Frame>("l3", "robot_center")
        .position(pos)
        .size(L3::ROBOT_CENTER_SIZE)
        .color(tracker_status_color(status_int))
        .sphere()
        .build();
}

/// @brief Create outpost center marker
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    outpost_center(const Eigen::Vector3d& pos, int status_int) noexcept {
    using namespace tactical;

    return EntityBuilder::create<Frame>("l3", "outpost_center")
        .position(pos)
        .size(L3::OUTPOST_CENTER_SIZE)
        .color(tracker_status_color(status_int))
        .sphere()
        .build();
}

/// @brief Create armor cube with tilt
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    armor_cube(const Eigen::Vector3d& pos, double yaw, bool is_big_armor) noexcept {
    using namespace tactical;

    const Eigen::Quaterniond q =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(Geometry::ARMOR_TILT_ANGLE, Eigen::Vector3d::UnitY());

    return EntityBuilder::create<Frame>("l3", "armor_plate")
        .position(pos)
        .orientation(q)
        .cube(
            Geometry::ARMOR_THICKNESS,
            is_big_armor ? Geometry::ARMOR_HEIGHT_BIG : Geometry::ARMOR_HEIGHT_SMALL,
            Geometry::ARMOR_WIDTH)
        .build();
}

// ============================================================================
// L4: GIMBAL VISUALIZATION
// ============================================================================

/// @brief Create gimbal selection marker (fire decision aware)
/// @param pos Selection position (spherical coordinates converted to Cartesian)
/// @param can_fire Whether fire is permitted (affects color)
/// @param timestamp_ns Entity timestamp for TF interpolation
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    selection_marker(const Eigen::Vector3d& pos, bool can_fire, uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    const auto color = can_fire ? L4::GIMBAL_AIM_FIRE : L4::GIMBAL_AIM_HOLD;

    return EntityBuilder::create<Frame>("l4", "selection")
        .position(pos)
        .size(L4::SELECTION_SIZE)
        .color(color)
        .sphere()
        .metadata("fire_decision", can_fire ? "YES" : "NO")
        .timestamp(timestamp_ns)
        .build();
}

/// @brief Create predicted position marker
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    predicted_position(const Eigen::Vector3d& pos, uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    return EntityBuilder::create<Frame>("l4", "prediction")
        .position(pos)
        .size(L4::PREDICTION_SIZE)
        .color(L3::PREDICTION_AMBER)
        .timestamp(timestamp_ns)
        .sphere()
        .build();
}

/// @brief Create MPC trajectory point with temporal fade
/// @param pos Point position
/// @param temporal_distance Distance from present (0 = now)
/// @param is_reference True for reference trajectory, false for optimized
/// @param timestamp_ns Entity timestamp for TF interpolation
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity mpc_trajectory_point(
    const Eigen::Vector3d& pos, int temporal_distance, bool is_reference,
    uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    const auto base_color  = is_reference ? L4::MPC_REFERENCE : L4::MPC_PRESENT;
    const auto faded_color = temporal_fade(base_color, temporal_distance);

    const std::string type = is_reference ? "reference" : "optimized";

    return EntityBuilder::create<Frame>("l4", fmt::format("mpc_{}_{}", type, temporal_distance))
        .position(pos)
        .size(L4::TRAJECTORY_DOT)
        .color(faded_color)
        .sphere()
        .metadata("temporal_offset", std::to_string(temporal_distance))
        .metadata("type", type)
        .timestamp(timestamp_ns)
        .build();
}

/// @brief Create bullet trajectory (sequence of dots)
/// @param points Trajectory points
/// @param can_fire Fire decision affects trajectory color
/// @param timestamp_ns Entity timestamp for TF interpolation
inline std::vector<foxglove::schemas::SceneEntity> bullet_trajectory(
    const std::vector<Eigen::Vector3d>& points, bool can_fire, uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    const auto color = can_fire ? L4::TRAJECTORY_FIRE : L4::TRAJECTORY_HOLD;

    std::vector<foxglove::schemas::SceneEntity> entities;
    entities.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        entities.push_back(
            EntityBuilder::create<fast_tf::muzzle>("l4", fmt::format("bullet_{}", i))
                .position(points[i])
                .size(L4::TRAJECTORY_DOT)
                .color(color)
                .sphere()
                .timestamp(timestamp_ns)
                .build());
    }

    return entities;
}

// ============================================================================
// VELOCITY VISUALIZATION
// ============================================================================

namespace Vel {
constexpr double PARALLEL_THRESHOLD = 1e-6;

/// @brief Create linear velocity arrow
inline std::optional<foxglove::schemas::ArrowPrimitive> linear_arrow(
    const foxglove::schemas::Vector3& position, const Eigen::Vector3d& velocity) noexcept {
    using namespace tactical;

    const double length = velocity.norm();
    if (length <= Velocity::ARROW_MIN_LENGTH) {
        return std::nullopt;
    }

    foxglove::schemas::ArrowPrimitive arrow;
    arrow.pose = make_pose(position);

    const Eigen::Vector3d vel_dir = velocity.normalized();
    const Eigen::Vector3d default_dir(1, 0, 0);
    Eigen::Quaterniond q;

    if ((vel_dir + default_dir).norm() < PARALLEL_THRESHOLD) {
        q = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
    } else {
        q = Eigen::Quaterniond::FromTwoVectors(default_dir, vel_dir);
    }

    arrow.pose->orientation = make_quaternion(q);
    arrow.shaft_length      = length * Velocity::ARROW_SHAFT_RATIO;
    arrow.shaft_diameter    = Velocity::ARROW_SHAFT_DIAMETER;
    arrow.head_length       = length * Velocity::ARROW_HEAD_RATIO;
    arrow.head_diameter     = Velocity::ARROW_HEAD_DIAMETER;
    arrow.color             = Velocity::LINEAR;

    return arrow;
}

/// @brief Create angular velocity arrow (along Z axis)
inline std::optional<foxglove::schemas::ArrowPrimitive>
    angular_arrow(const foxglove::schemas::Vector3& position, double v_yaw) noexcept {
    using namespace tactical;

    const double length = std::abs(v_yaw) / M_PI;
    if (length <= Velocity::ARROW_MIN_LENGTH) {
        return std::nullopt;
    }

    foxglove::schemas::ArrowPrimitive arrow;
    arrow.pose = make_pose(position);

    const double sign       = v_yaw >= 0 ? -1.0 : 1.0;
    arrow.pose->orientation = make_quaternion(0.0, sign * 0.7071, 0.0, 0.7071);
    arrow.shaft_length      = length * Velocity::ARROW_SHAFT_RATIO;
    arrow.shaft_diameter    = Velocity::ARROW_SHAFT_DIAMETER;
    arrow.head_length       = length * Velocity::ARROW_HEAD_RATIO;
    arrow.head_diameter     = Velocity::ARROW_HEAD_DIAMETER;
    arrow.color             = Velocity::ANGULAR;

    return arrow;
}
} // namespace Vel

/// @brief Create entity with velocity arrows
/// @param pos Origin position for arrows
/// @param linear_vel Linear velocity vector
/// @param angular_vel_yaw Angular velocity (yaw axis)
/// @param timestamp_ns Entity timestamp for TF interpolation
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity velocity_arrows(
    const Eigen::Vector3d& pos, const Eigen::Vector3d& linear_vel, double angular_vel_yaw,
    uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    const auto vec3_pos = make_vector3(pos);

    auto entity = EntityBuilder::create<Frame>("l3", "velocity").timestamp(timestamp_ns).build();

    if (auto linear_arrow = Vel::linear_arrow(vec3_pos, linear_vel)) {
        entity.arrows.push_back(*linear_arrow);
    }

    if (auto angular_arrow = Vel::angular_arrow(vec3_pos, angular_vel_yaw)) {
        entity.arrows.push_back(*angular_arrow);
    }

    return entity;
}

// ============================================================================
// UNCERTAINTY VISUALIZATION
// ============================================================================

/// @brief Create uncertainty sphere for covariance visualization
/// @param pos Center position
/// @param scale Scale factor for sphere size
/// @param timestamp_ns Entity timestamp for TF interpolation
template <fast_tf::frame Frame>
inline foxglove::schemas::SceneEntity
    uncertainty_sphere(const Eigen::Vector3d& pos, double scale, uint64_t timestamp_ns) noexcept {
    using namespace tactical;

    const double scaled_size = L3::TARGET_SIZE * L3::UNCERTAINTY_SCALE * scale;

    return EntityBuilder::create<Frame>("l3", "uncertainty")
        .position(pos)
        .size(scaled_size)
        .color(L3::UNCERTAINTY_ORANGE)
        .sphere()
        .metadata("scale", fmt::format("{:.3f}", scale))
        .timestamp(timestamp_ns)
        .build();
}

} // namespace patterns

// ============================================================================
// PUBLISH HELPERS
// ============================================================================

/// @brief Publish scene entities if non-empty
template <typename SceneMessageType>
inline void publish_scene_if_nonempty(
    const std::shared_ptr<FoxgloveServer>& server,
    std::vector<foxglove::schemas::SceneEntity> entities) noexcept {
    if (!entities.empty()) {
        SceneMessageType msg;
        msg.scene.entities = std::move(entities);
        server->enqueue_message(std::move(msg));
    }
}

/// @brief Publish scene entities to specified channel
inline void publish_scene(
    const std::shared_ptr<FoxgloveServer>& server, std::string_view channel_id,
    std::vector<foxglove::schemas::SceneEntity> entities, uint64_t timestamp_ns) noexcept {
    if (!entities.empty()) {
        SceneMessage msg;
        msg.scene.entities = std::move(entities);
        server->enqueue_message(std::move(msg));
    }
}

} // namespace fcs::visualization::builder

// ============================================================================
// CONVENIENCE USING DECLARATIONS
// ============================================================================

namespace fcs::visualization {

/// @brief Shorthand for commonly used builder components
namespace viz = builder;
namespace tac = tactical;

} // namespace fcs::visualization
