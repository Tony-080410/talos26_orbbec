#pragma once

#include "channel_registry.hpp"
#include "foxglove_server.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "system_helpers.hpp"
#include "tactical_palette.hpp"

#include "core/channel_topics.hpp"
#include "frame.hpp"
#include "scheduler/scheduler.hpp"

#include <memory>

namespace fcs::foxglove::systems::detail {

// ============================================================================
// Base Registration Patterns - 基础系统注册模式
// ============================================================================
//
// 这个文件提供了所有分层系统注册共享的基础设施和模式。
// 包括：
// - 命名空间别名 (viz, tac)
// - 通用系统注册宏
// - 标准系统 lambda 模式
// ============================================================================

// Pull in fcs::detail helpers for use in lambda functions
using ::fcs::detail::foxglove_ready;
using ::fcs::detail::publish_json_message;
using ::fcs::detail::spherical_to_cartesian;

} // namespace fcs::foxglove::systems::detail

// Shorthand for commonly used namespaces (at file scope for use in other files)
namespace viz = ::fcs::visualization::builder;
namespace tac = ::fcs::visualization::tactical;

namespace fcs::foxglove::systems::detail {

// ---------------------------------------------------------------------------
// System Registration Helper Macros
// ---------------------------------------------------------------------------

/// @brief 注册一个固定频率的图像发布系统
#define REGISTER_FIXED_RATE_IMAGE_SYSTEM(Name, Freq, ...) \
    app.template add_system<talos::fixed_rate<Freq>>(Name, __VA_ARGS__)

/// @brief 注册一个固定频率的 JSON 发布系统
#define REGISTER_FIXED_RATE_JSON_SYSTEM(Name, Freq, ...) \
    app.template add_system<talos::fixed_rate<Freq>>(Name, __VA_ARGS__)

/// @brief 注册一个固定频率的场景发布系统
#define REGISTER_FIXED_RATE_SCENE_SYSTEM(Name, Freq, ...) \
    app.template add_system<talos::fixed_rate<Freq>>(Name, __VA_ARGS__)

/// @brief 注册一个线程池计算系统
#define REGISTER_POOL_COMPUTE_SYSTEM(Name, ...) \
    app.template add_system<talos::pool_compute>(Name, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Standard System Lambda Signature Patterns
// ---------------------------------------------------------------------------

// Image publisher with camera calibration and TF
// [](spmc<AtLegacyImageFrame, AtLegacyImageChannelTopic> img_in,
//    res<std::shared_ptr<FoxgloveServer>> server,
//    res<CameraConfig> cam,
//    spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTopic> det_in,
//    res<fast_tf::CoordinateSystem> tf_buffer) { ... }

// Scene publisher from any tracked channel
// [](spmc<ChannelType, ChannelTopic> input,
//    res<std::shared_ptr<FoxgloveServer>> server) { ... }

// JSON publisher from any tracked channel
// [](spmc<ChannelType, ChannelTopic> input,
//    res<std::shared_ptr<FoxgloveServer>> server) { ... }

} // namespace fcs::foxglove::systems::detail

// ---------------------------------------------------------------------------
// Layered Registration Functions - 分层注册函数声明
// ---------------------------------------------------------------------------

namespace fcs::foxglove::systems {

// Pull in commonly used helpers to this namespace
using ::fcs::detail::foxglove_ready;
using ::fcs::detail::publish_json_message;

/// @brief Register L1 sensor layer systems (image publishing)
template <typename App>
void register_l1_sensor_systems(App& app);

/// @brief Register L2 perception layer systems (detection, measurement)
template <typename App>
void register_l2_perception_systems(App& app);

/// @brief Register L3 estimation layer systems (tracking, association)
template <typename App>
void register_l3_estimation_systems(App& app);

/// @brief Register L4 planning layer systems (gimbal, MPC)
template <typename App>
void register_l4_planning_systems(App& app);

/// @brief Register rune-specific visualization systems
template <typename App>
void register_rune_systems(App& app);

/// @brief Register debug visualization systems
template <typename App>
void register_debug_systems(App& app);

} // namespace fcs::foxglove::systems
