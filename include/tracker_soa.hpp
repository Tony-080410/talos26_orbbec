#pragma once

#include "fast_tf/frame.hpp"

#include <array>
#include <string>
#include <string_view>

namespace talos {
/**
 * @brief Tracker state machine enumerator
 *
 * State transitions:
 * IDLE --[detection]--> DETECTING --[confirmed]--> TRACKING
 *                                                      |
 *                           <-- [miss] --> TEMP_LOST --+
 *                           [timeout] -> IDLE
 */
enum TrackerState : uint8_t {
    IDLE      = 0, // No target tracked
    DETECTING = 1, // Candidate detected, waiting for confirmation
    TRACKING  = 2, // Actively tracking target
    TEMP_LOST = 3  // Lost tracking, predicting only
};

const std::array<std::string, 4> TRACKER_STATE_NAMES = {
    "IDLE", "DETECTING", "TRACKING", "TEMP_LOST"};

constexpr std::string_view tracker_state(const TrackerState c) noexcept {
    return TRACKER_STATE_NAMES[c];
}

/**
 * @brief Tracker state - cpu cache friendly
 */
struct alignas(64) TrackerSoA {
    std::atomic<TrackerState> state{IDLE};
    uint8_t detecting_count{0};                              // Frames since DETECTING started
    uint8_t lost_count{0};                                   // Frames since tracking lost
    uint8_t generation{0};                                   // Version counter for ABA prevention

    fast_tf::Fuck<fast_tf::world_fuxk_frame, std::array<float, 3>>
        pos{};                                               // [x, y, z] world position
    fast_tf::Fuck<fast_tf::world_fuxk_frame, std::array<float, 3>>
        vel{};                                               // [vx, vy, vz] linear velocity

    fast_tf::Fuck<fast_tf::world_fuxk_frame, float> yaw{};   // Rotation angle (radians)
    fast_tf::Fuck<fast_tf::world_fuxk_frame, float> v_yaw{}; // Angular velocity (rad/s)

    // Shape (2D)
    float radius[2]{1.0f, 1.0f};      // [radius_x, radius_y] armor dimensions ratio

    uint32_t max_temp_lost_count{10}; // 动态丢失阈值, 4 bytes (offset 44-48)
    uint64_t last_timestamp_ns{0};    // 上次更新时间戳 (ns), 8 bytes (offset 48-56, 8字节对齐)

    explicit TrackerSoA() noexcept = default;
    // 禁止拷贝
    TrackerSoA(const TrackerSoA&)            = delete;
    TrackerSoA& operator=(const TrackerSoA&) = delete;

    TrackerSoA(TrackerSoA&&)            = default;
    TrackerSoA& operator=(TrackerSoA&&) = default;

    auto detect() noexcept {
        if (state != IDLE) {
            return false;
        }
        state = DETECTING;
        return true;
    }

    auto track() noexcept {
        if (state == DETECTING || state == TEMP_LOST) {
            return false;
        }
        state = TRACKING;
        return true;
    }

    auto temporary_lost() noexcept {
        if (state != TRACKING) {
            return false;
        }
        state = TEMP_LOST;
        return true;
    }

    auto idle() noexcept {
        if (state == IDLE) {
            return false;
        }
        state = IDLE;
        return true;
    }

    /**
     * @brief Increment generation counter for ABA prevention
     */
    void next_generation() {
        generation++; // Wrap-around is intentional and safe
    }
};
static_assert(sizeof(TrackerSoA) <= 64, "ArmorSlots exceeds 64B budget");

} // namespace talos
