#pragma once

#include "L4_planning/planning/fire_decision.hpp"
#include "core/armor_types.hpp"
#include "core/trajectory/config.hpp"

namespace fcs::L4 {

struct TargetSelectionWeights {
    double image_center{5.0};
    double track_state{3.0};
    double tof{1.5};
    double distance{1.0};
    double gimbal_effort{1.0};
    double armor_name{0.5};
};

struct TargetSelectionRefs {
    double image_center_ref_px{300.0};
    double tof_ref_s{0.35};
    double distance_ref_m{10.0};
    double yaw_effort_ref_deg{20.0};
    double pitch_effort_ref_deg{10.0};
};

struct TargetSelectionArmorNameScore {
    double sentry{0.5};
    double one{0.5};
    double two{0.5};
    double three{0.5};
    double four{0.5};
    double five{0.5};
    double outpost{0.5};
    double base{0.5};
    double base_large{0.5};
    double invalid{0.0};

    [[nodiscard]] double score(ArmorName name) const noexcept {
        switch (name) {
        case ArmorName::Sentry: return sentry;
        case ArmorName::One: return one;
        case ArmorName::Two: return two;
        case ArmorName::Three: return three;
        case ArmorName::Four: return four;
        case ArmorName::Five: return five;
        case ArmorName::Outpost: return outpost;
        case ArmorName::Base: return base;
        case ArmorName::BaseLarge: return base_large;
        case ArmorName::Invalid: return invalid;
        }
        return invalid;
    }
};

struct TargetSelectionConfig {
    double temp_lost_state_score{0.05};
    double optical_stale_timeout_s{0.25};
    double switch_margin{0.08};
    TargetSelectionWeights weights{};
    TargetSelectionRefs refs{};
    TargetSelectionArmorNameScore armor_name_score{};
};

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
    int transfer_threshold_single_whole{50};

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

    /// Weighted multi-target selection configuration.
    TargetSelectionConfig target_selection{};
};

/// L4 planning configuration (Aimer + plan adapter architecture)
struct L4Config {
    // Fire decision configuration shared by L4 aiming and L5 fire gating.
    FireDecisionConfig fire_decision{};

    // Shared aimer configuration (used for Robot/Outpost/Rune)
    AimerConfig aimer{};
};

} // namespace fcs::L4
