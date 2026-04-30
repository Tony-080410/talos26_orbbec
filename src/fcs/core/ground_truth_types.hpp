#pragma once

#include <cstdint>

namespace fcs::core {

/// Ground truth target (robot or outpost)
struct GroundTruthTarget {
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    uint8_t team{0};            // 0=Red, 1=Blue
    uint8_t armor_label{0};     // ArmorLabel discriminant
    uint8_t is_outpost{0};
    uint8_t _pad1{0};
    float position[3]{0, 0, 0}; // xyz in ROS odom frame
    float vyaw{0};              // angular velocity (rad/s)
    float yaw{0};               // current yaw angle (rad)
};

/// Ground truth rune internal state
struct GroundTruthRune {
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};
    uint8_t team{0};
    uint8_t rune_mode{0};       // 0=Small, 1=Large
    uint8_t mechanism_state{0}; // 0=Inactive, 1=Activating, 2=Activated, 3=Failed
    uint8_t _pad1{0};
    float r_center_odom[3]{0, 0, 0};
    float radius{0};
    float current_angle{0};
    float v_roll{0};
    int32_t direction{0};
    float sin_amplitude{0};
    float sin_omega{0};
    float sin_phase{0};
    float sin_offset{0};
    float relative_time{0};
    int32_t blade_id{-1};
    uint8_t target_activations[5]{0, 0, 0, 0, 0};
};

} // namespace fcs::core
