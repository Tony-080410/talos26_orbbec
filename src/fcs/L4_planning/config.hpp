#pragma once

#include "L4_planning/planning/fire_decision.hpp"
#include "core/trajectory/config.hpp"

namespace fcs::L4 {

/// Aimer configuration (shared by all target types)
struct AimerConfig {
    /// System delay compensation (seconds)
    /// Includes: camera exposure, image processing, communication, gimbal response
    double delay{0.01};

    /// Phase-FSM thresholds (rad/s), aligned with prior implementations.
    double single_whole_up{2.5};
    double single_whole_down{1.0};
    double whole_pair_up{7.5};
    double whole_pair_down{6.5};
    double pair_center_up{16.5};
    double pair_center_down{15.0};

    /// Consecutive frames required to transfer between aim phases.
    int transfer_threshold{50};

    /// Front-window limit for single-armor locking (degrees).
    double front_window_deg{90.0};

    /// Robot whole-car directional armor windows (degrees), aligned with WUST semantics.
    double coming_angle_deg{60.0};
    double leaving_angle_deg{20.0};

    /// Phase-aware fire gate shaping, aligned with WUST semantics.
    double yaw_limit_deg{40.0};
    double min_enable_yaw_deg{0.25};
};

/// L4 planning configuration (Aimer + plan adapter architecture)
struct L4Config {
    // Fire decision configuration shared by L4 aiming and L5 fire gating.
    FireDecisionConfig fire_decision{};

    // Shared aimer configuration (used for Robot/Outpost/Rune)
    AimerConfig aimer{};
};

} // namespace fcs::L4
