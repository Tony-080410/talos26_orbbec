#pragma once

#include "foxglove_server.hpp"
#include "foxglove_types.hpp"

#include <cmath>

namespace fcs::detail {

// ============================================================================
// System Helper Functions - 共享的系统辅助函数
// ============================================================================
//
// 这些函数在多个 foxglove 发布系统中共享使用。
// 放置在独立头文件中，避免重复定义和编译依赖。
// ============================================================================

/// @brief Check if Foxglove server is ready and channel has new data
[[nodiscard]] inline bool
    foxglove_ready(const std::shared_ptr<FoxgloveServer>& server, const auto& channel) noexcept {
    return server && server->is_initialized() && channel.has_new();
}

/// @brief Publish JSON message to Foxglove server
template <typename MessageType>
inline void publish_json_message(
    const std::shared_ptr<FoxgloveServer>& server, const nlohmann::json& json_obj) noexcept {
    MessageType msg;
    msg.payload = json_to_bytes(json_obj.dump());
    server->enqueue_message(std::move(msg));
}

/// @brief Convert spherical coordinates to Cartesian Vector3
[[nodiscard]] inline foxglove::schemas::Vector3
    spherical_to_cartesian(double distance, double yaw, double pitch) noexcept {
    foxglove::schemas::Vector3 v;
    v.x = distance * ::cos(pitch) * ::cos(yaw);
    v.y = distance * ::cos(pitch) * ::sin(yaw);
    v.z = distance * ::sin(pitch);
    return v;
}

} // namespace fcs::detail
