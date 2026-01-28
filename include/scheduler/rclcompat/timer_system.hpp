#pragma once

#include "ecs/execution_policy.hpp"
#include "rclcompat/system.hpp"
#include "scheduler/rclcompat/system_builder.hpp"
#include <spdlog/spdlog.h>

namespace talos::rclcompat {

/**
 * @brief Fixed-frequency timer system
 *
 * Runs on a dedicated fixed_rate thread at a fixed frequency.
 * Inherits from RclSubSystemBase to enable Publisher auto-binding mechanism.
 *
 * ## Template parameters
 *
 * - `FrequencyHz`: Execution frequency in Hz
 * - `Affinity`: CPU affinity (-1 = no binding)
 * - `Priority`: Thread priority (0 = default)
 */
template <std::uint32_t FrequencyHz, std::int32_t Affinity = -1, std::int32_t Priority = 0>
class RclTimerSystem : public RclSubSystemBase {
public:
    using Callback = std::function<void()>;
    using Policy   = talos::fixed_rate<FrequencyHz, Affinity, Priority>;

    explicit RclTimerSystem(std::string name, Callback callback) noexcept
        : callback_(std::move(callback)) {
        meta_ = SystemMetaBuilder(std::move(name)).policy(make_policy_info<Policy>()).build();
    }

    /**
     * @brief Track a publisher that the callback may write to
     *
     * Used for wake chain computation.
     */
    void track_publisher(
        std::atomic<bool>* write_flag, std::type_index type, std::type_index topic) noexcept {
        tracked_write_flags_.push_back(write_flag);
        meta_.spmc_channels.push_back(
            ChannelMeta{.type = type, .topic = topic, .kind = channel_kind::spmc_writer});
    }

    void bind([[maybe_unused]] World& world) noexcept override {
        // Timer typically doesn't need to bind channels
    }

    bool run([[maybe_unused]] World& world) noexcept override {
        // Set thread context to enable Publisher auto-binding
        CallbackContextScope scope(this);
        callback_();

        // Check if any publisher wrote data
        bool any_written = false;
        for (auto* flag : tracked_write_flags_) {
            if (flag->exchange(false, std::memory_order_acq_rel)) {
                any_written = true;
            }
        }
        return any_written;
    }

    const SystemMeta& meta() const noexcept override { return meta_; }

private:
    SystemMeta meta_;
    Callback callback_;
    std::vector<std::atomic<bool>*> tracked_write_flags_;
};

} // namespace talos::rclcompat
