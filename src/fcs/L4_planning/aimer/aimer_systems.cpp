#include "L4_planning/aimer/aimer_systems.hpp"
#include "L4_planning/aimer/aimer.hpp"

#include "L2_perception/armor/at_legacy_readback_roi.hpp"
#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/aimer/types.hpp"
#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "L4_planning/selected_target_snapshot.hpp"
#include "L4_planning/target_selection_trace.hpp"
#include "L4_planning/trajectory_builder.hpp"
#include "camera_config.hpp"
#include "core/channel_topics.hpp"
#include "core/target_key.hpp"
#include "core/time.hpp"
#include "core/trajectory/resource.hpp"
#include "euler.hpp"
#include "scheduler/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <expected>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <vector>

namespace fcs::L4 {
static constexpr double kMaxVisualizationDistance = 15.0;

namespace {

struct TargetPhaseState {
    ArmorAimPhase phase{ArmorAimPhase::SingleArmor};
    int overflow_count{0};
    std::optional<int> jumped_selected_armor_id{};
};

struct TargetSelectionScores {
    double total{0.0};
    double image_center{0.0};
    double track_state{0.0};
    double tof{0.0};
    double distance{0.0};
    double gimbal_effort{0.0};
    double armor_name{0.0};
};

struct CandidateEvaluation {
    core::TargetKey key{};
    L3::TrackerOutput tracker{};
    TargetPrediction prediction{};
    TargetSelectionScores scores{};
    TargetSelectionTraceEntry trace_entry{};
};

struct GimbalEffortMetrics {
    double yaw_delta_rad{std::numeric_limits<double>::infinity()};
    double pitch_delta_rad{std::numeric_limits<double>::infinity()};
    double score{0.0};
};

struct ReadbackRoiTimestamps {
    uint64_t projection_timestamp_ns{0};
    uint64_t freshness_timestamp_ns{0};
};

void update_phase_fsm(
    const AimerConfig& cfg, double abs_v_yaw, bool target_jumped, ArmorAimPhase& phase,
    int& overflow_count) noexcept {
    if (!target_jumped) {
        phase          = ArmorAimPhase::SingleArmor;
        overflow_count = 0;
        return;
    }

    switch (phase) {
    case ArmorAimPhase::SingleArmor:
        overflow_count = (abs_v_yaw > cfg.single_whole_armor_high_v_yaw()) ? overflow_count + 1 : 0;
        if (overflow_count > cfg.single_to_whole_armor_frames) {
            phase          = ArmorAimPhase::WholeCarArmor;
            overflow_count = 0;
        }
        break;

    case ArmorAimPhase::WholeCarArmor:
        if (abs_v_yaw > cfg.whole_armor_center_high_v_yaw()) {
            ++overflow_count;
        } else if (abs_v_yaw < cfg.single_whole_armor_low_v_yaw()) {
            --overflow_count;
        } else {
            overflow_count = 0;
        }
        if (std::abs(overflow_count) > cfg.phase_transition_frames) {
            phase =
                (overflow_count > 0) ? ArmorAimPhase::WholeCarCenter : ArmorAimPhase::SingleArmor;
            overflow_count = 0;
        }
        break;

    case ArmorAimPhase::WholeCarCenter:
        overflow_count = (abs_v_yaw < cfg.whole_armor_center_low_v_yaw()) ? overflow_count + 1 : 0;
        if (overflow_count > cfg.phase_transition_frames) {
            phase          = ArmorAimPhase::WholeCarArmor;
            overflow_count = 0;
        }
        break;
    }
}

[[nodiscard]] double clamp01(double value) noexcept { return std::clamp(value, 0.0, 1.0); }

[[nodiscard]] double elapsed_seconds(uint64_t start_ns, uint64_t end_ns) noexcept {
    if (end_ns <= start_ns) {
        return 0.0;
    }
    return static_cast<double>(end_ns - start_ns) * 1e-9;
}

[[nodiscard]] double
    optical_age_seconds(const L3::TrackerOutput& tracker, uint64_t current_ns) noexcept {
    if (tracker.last_observation_timestamp_ns == 0) {
        return 0.0;
    }
    return elapsed_seconds(tracker.last_observation_timestamp_ns, current_ns);
}

[[nodiscard]] core::TargetKey make_target_key(const L3::TrackerOutput& tracker) noexcept {
    return core::TargetKey{tracker.target_name, tracker.target_color};
}

[[nodiscard]] bool tracker_supports_readback_roi(const L3::TrackerOutput& tracker) noexcept {
    return tracker.is_tracking() && tracker.target_name != ArmorName::Invalid
        && (tracker.is_robot() || tracker.is_outpost());
}

[[nodiscard]] std::optional<L3::TrackerOutput> pick_readback_tracker_fallback(
    const std::optional<L3::TrackerOutputs>& trackers,
    const std::optional<core::TargetKey>& preferred_key) noexcept {
    if (!trackers) {
        return std::nullopt;
    }

    if (preferred_key.has_value()) {
        for (const auto& tracker : *trackers) {
            if (tracker_supports_readback_roi(tracker)
                && make_target_key(tracker) == *preferred_key) {
                return tracker;
            }
        }
    }

    const L3::TrackerOutput* best = nullptr;
    for (const auto& tracker : *trackers) {
        if (!tracker_supports_readback_roi(tracker)) {
            continue;
        }

        if (best == nullptr) {
            best = &tracker;
            continue;
        }

        const bool tracker_has_finite = std::isfinite(tracker.last_image_center_distance_px);
        const bool best_has_finite    = std::isfinite(best->last_image_center_distance_px);
        if (tracker_has_finite != best_has_finite) {
            if (tracker_has_finite) {
                best = &tracker;
            }
            continue;
        }

        if (tracker_has_finite
            && tracker.last_image_center_distance_px < best->last_image_center_distance_px) {
            best = &tracker;
            continue;
        }

        if (tracker.last_observation_timestamp_ns > best->last_observation_timestamp_ns) {
            best = &tracker;
        }
    }

    if (best == nullptr) {
        return std::nullopt;
    }
    return *best;
}

[[nodiscard]] ReadbackRoiTimestamps
    resolve_readback_roi_timestamps(const SelectedTargetSnapshot& snapshot) noexcept {
    const uint64_t projection_timestamp_ns =
        snapshot.tracker.timestamp_ns != 0 ? snapshot.tracker.timestamp_ns : snapshot.timestamp_ns;
    const uint64_t freshness_timestamp_ns = snapshot.tracker.last_observation_timestamp_ns != 0
                                              ? snapshot.tracker.last_observation_timestamp_ns
                                              : projection_timestamp_ns;
    return {
        .projection_timestamp_ns = projection_timestamp_ns,
        .freshness_timestamp_ns  = freshness_timestamp_ns,
    };
}

[[nodiscard]] std::optional<Eigen::Vector3d>
    tracker_center_world_position(const L3::TrackerOutput& tracker) noexcept {
    if (const auto* robot_state = tracker.robot_state()) {
        return robot_state->position;
    }
    if (const auto* outpost_state = tracker.outpost_state()) {
        return Eigen::Vector3d{
            outpost_state->position.x(),
            outpost_state->position.y(),
            (outpost_state->z[0] + outpost_state->z[1] + outpost_state->z[2]) / 3.0,
        };
    }
    return std::nullopt;
}

[[nodiscard]] bool contains_target_key(
    const std::vector<core::TargetKey>& keys, const core::TargetKey& key) noexcept {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

[[nodiscard]] double target_abs_v_yaw(const L3::TrackerOutput& tracker) noexcept {
    if (const auto* robot_state = tracker.robot_state()) {
        return std::abs(robot_state->v_yaw);
    }
    if (const auto* outpost_state = tracker.outpost_state()) {
        return std::abs(outpost_state->v_yaw);
    }
    return 0.0;
}

[[nodiscard]] std::expected<TargetPrediction, std::string> aim_tracker_target(
    const Aimer& aimer, const L3::TrackerOutput& tracker, const ArmorAimContext& context,
    const Aimer::GimbalTransform& gimbal, const Aimer::MuzzleTransform& muzzle, uint64_t current_ns,
    double bullet_speed, const core::trajectory::solver::TrajectorySolver& solver) noexcept {
    if (const auto* robot_state = tracker.robot_state()) {
        return aimer.aim(
            *robot_state, context, gimbal, muzzle, tracker.timestamp_ns, current_ns, 0.0,
            bullet_speed, solver);
    }
    if (const auto* outpost_state = tracker.outpost_state()) {
        return aimer.aim(
            *outpost_state, context, gimbal, muzzle, tracker.timestamp_ns, current_ns, 0.0,
            bullet_speed, solver);
    }
    return std::unexpected("tracker output does not contain a supported target state");
}

[[nodiscard]] double score_image_center(
    const TargetSelectionConfig& cfg, const L3::TrackerOutput& tracker,
    uint64_t current_ns) noexcept {
    if (!std::isfinite(tracker.last_image_center_distance_px)
        || tracker.last_observation_timestamp_ns == 0 || cfg.refs.image_center_ref_px <= 1e-6) {
        return 0.0;
    }

    double score = 1.0 - tracker.last_image_center_distance_px / cfg.refs.image_center_ref_px;
    score        = clamp01(score);

    if (tracker.status == L3::TrackerStatus::TempLost) {
        if (cfg.optical_stale_timeout_s <= 1e-6) {
            return 0.0;
        }
        const double age = elapsed_seconds(tracker.last_observation_timestamp_ns, current_ns);
        score *= clamp01(1.0 - age / cfg.optical_stale_timeout_s);
    }

    return score;
}

[[nodiscard]] double
    score_track_state(const TargetSelectionConfig& cfg, const L3::TrackerOutput& tracker) noexcept {
    return tracker.status == L3::TrackerStatus::Tracking ? 1.0 : cfg.temp_lost_state_score;
}

[[nodiscard]] double
    score_tof(const TargetSelectionConfig& cfg, const TargetPrediction& prediction) noexcept {
    if (!std::isfinite(prediction.flying_time) || cfg.refs.tof_ref_s <= 1e-6) {
        return 0.0;
    }
    return clamp01(1.0 - prediction.flying_time / cfg.refs.tof_ref_s);
}

[[nodiscard]] double
    score_distance(const TargetSelectionConfig& cfg, const TargetPrediction& prediction) noexcept {
    if (!std::isfinite(prediction.distance) || cfg.refs.distance_ref_m <= 1e-6) {
        return 0.0;
    }
    return clamp01(1.0 - prediction.distance / cfg.refs.distance_ref_m);
}

[[nodiscard]] GimbalEffortMetrics compute_gimbal_effort_metrics(
    const TargetSelectionConfig& cfg, double current_yaw, double current_pitch,
    const TargetPrediction& prediction) noexcept {
    GimbalEffortMetrics metrics;
    const double yaw_ref   = cfg.refs.yaw_effort_ref_deg * std::numbers::pi / 180.0;
    const double pitch_ref = cfg.refs.pitch_effort_ref_deg * std::numbers::pi / 180.0;
    if (yaw_ref <= 1e-6 || pitch_ref <= 1e-6) {
        return metrics;
    }

    metrics.yaw_delta_rad =
        std::abs(std::remainder(prediction.aim_yaw - current_yaw, 2.0 * std::numbers::pi));
    const double command_pitch_in_tf = -prediction.aim_pitch;
    metrics.pitch_delta_rad          = std::abs(command_pitch_in_tf - current_pitch);
    const double effort =
        0.5 * (metrics.yaw_delta_rad / yaw_ref + metrics.pitch_delta_rad / pitch_ref);
    metrics.score = clamp01(1.0 - effort);
    return metrics;
}

[[nodiscard]] TargetSelectionScores score_candidate(
    const TargetSelectionConfig& cfg, const L3::TrackerOutput& tracker,
    const TargetPrediction& prediction, double current_yaw, double current_pitch,
    uint64_t current_ns) noexcept {
    TargetSelectionScores scores;
    scores.image_center = score_image_center(cfg, tracker, current_ns);
    scores.track_state  = score_track_state(cfg, tracker);
    scores.tof          = score_tof(cfg, prediction);
    scores.distance     = score_distance(cfg, prediction);
    scores.gimbal_effort =
        compute_gimbal_effort_metrics(cfg, current_yaw, current_pitch, prediction).score;
    scores.armor_name = clamp01(cfg.armor_name_score.score(tracker.target_name));

    const double weighted_sum =
        cfg.weights.image_center * scores.image_center
        + cfg.weights.track_state * scores.track_state + cfg.weights.tof * scores.tof
        + cfg.weights.distance * scores.distance + cfg.weights.gimbal_effort * scores.gimbal_effort
        + cfg.weights.armor_name * scores.armor_name;
    const double total_weight = cfg.weights.image_center + cfg.weights.track_state + cfg.weights.tof
                              + cfg.weights.distance + cfg.weights.gimbal_effort
                              + cfg.weights.armor_name;

    scores.total = total_weight > 1e-6 ? weighted_sum / total_weight : 0.0;
    return scores;
}

[[nodiscard]] TargetSelectionTraceEntry build_scored_trace_entry(
    const TargetSelectionConfig& cfg, const L3::TrackerOutput& tracker,
    const TargetPrediction& prediction, const TargetSelectionScores& scores, double current_yaw,
    double current_pitch, uint64_t current_ns, bool was_previously_selected) noexcept {
    const auto effort_metrics =
        compute_gimbal_effort_metrics(cfg, current_yaw, current_pitch, prediction);

    TargetSelectionTraceEntry entry;
    entry.target_name              = tracker.target_name;
    entry.target_color             = tracker.target_color;
    entry.track_status             = tracker.status;
    entry.aim_valid                = true;
    entry.was_previously_selected  = was_previously_selected;
    entry.target_center            = tracker_center_world_position(tracker);
    entry.image_center_distance_px = tracker.last_image_center_distance_px;
    entry.optical_age_s            = optical_age_seconds(tracker, current_ns);
    entry.tof_s                    = prediction.flying_time;
    entry.distance_m               = prediction.distance;
    entry.yaw_effort_deg           = effort_metrics.yaw_delta_rad * 180.0 / std::numbers::pi;
    entry.pitch_effort_deg         = effort_metrics.pitch_delta_rad * 180.0 / std::numbers::pi;

    entry.image_center_score  = scores.image_center;
    entry.track_state_score   = scores.track_state;
    entry.tof_score           = scores.tof;
    entry.distance_score      = scores.distance;
    entry.gimbal_effort_score = scores.gimbal_effort;
    entry.armor_name_score    = scores.armor_name;

    entry.image_center_weighted  = cfg.weights.image_center * scores.image_center;
    entry.track_state_weighted   = cfg.weights.track_state * scores.track_state;
    entry.tof_weighted           = cfg.weights.tof * scores.tof;
    entry.distance_weighted      = cfg.weights.distance * scores.distance;
    entry.gimbal_effort_weighted = cfg.weights.gimbal_effort * scores.gimbal_effort;
    entry.armor_name_weighted    = cfg.weights.armor_name * scores.armor_name;
    entry.weighted_sum = entry.image_center_weighted + entry.track_state_weighted
                       + entry.tof_weighted + entry.distance_weighted + entry.gimbal_effort_weighted
                       + entry.armor_name_weighted;
    entry.total_weight = cfg.weights.image_center + cfg.weights.track_state + cfg.weights.tof
                       + cfg.weights.distance + cfg.weights.gimbal_effort + cfg.weights.armor_name;
    entry.total_score  = scores.total;
    return entry;
}

[[nodiscard]] TargetSelectionTraceEntry build_rejected_trace_entry(
    const L3::TrackerOutput& tracker, uint64_t current_ns, std::string reason,
    bool was_previously_selected) {
    TargetSelectionTraceEntry entry;
    entry.target_name              = tracker.target_name;
    entry.target_color             = tracker.target_color;
    entry.track_status             = tracker.status;
    entry.aim_valid                = false;
    entry.was_previously_selected  = was_previously_selected;
    entry.aim_error                = std::move(reason);
    entry.target_center            = tracker_center_world_position(tracker);
    entry.image_center_distance_px = tracker.last_image_center_distance_px;
    entry.optical_age_s            = optical_age_seconds(tracker, current_ns);
    return entry;
}

} // namespace

void register_aimer_systems(talos::Scheduler& scheduler, L4Config&& config) {
    auto& world = scheduler.world();
    if (!world.has_resource<L2::AtLegacyReadbackRoiConfig>()) {
        world.insert_resource(L2::AtLegacyReadbackRoiConfig{});
    }
    if (!world.has_resource<L2::AtLegacyReadbackRoiCache>()) {
        world.insert_resource(L2::AtLegacyReadbackRoiCache{});
    }
    if (!world.has_resource<CameraConfig>()) {
        world.insert_resource(CameraConfig{});
    }

    world.insert_resource(Aimer(config.aimer));
    world.insert_resource(std::move(config));

    scheduler.add_system<talos::fixed_rate<250>>(
        "l4_aimer",
        [phase_states =
             std::unordered_map<core::TargetKey, TargetPhaseState, core::TargetKeyHash>{},
         selected_target_key = std::optional<core::TargetKey>{}](
            talos::res<L4Config> cfg, talos::res<Aimer> aimer,
            core::trajectory::trajectory_solver solver,
            core::trajectory::bullet_speed bullet_speed_,
            talos::spmc<L3::TrackerOutputs, TrackerOutputChannelTopic> tracker_in,
            talos::spmc<energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic> rune_in,
            talos::res<fast_tf::CoordinateSystem> tf_buffer,
            talos::spmc_mut<SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic>
                selected_target_out,
            talos::spmc_mut<TargetSelectionTrace, TargetSelectionTraceChannelTopic>
                target_selection_out,
            talos::spmc_mut<GimbalPlan, GimbalPlanChannelTopic> aimer_out) mutable {
            std::expected<TargetPrediction, std::string> aim_result = std::unexpected("none");

            const uint64_t current_ns = fcs::clock::now_ns();

            auto transforms = lookup_gimbal_muzzle_transforms(*tf_buffer, current_ns);
            if (!transforms) [[unlikely]] {
                SPDLOG_WARN("Transform lookup failed: {}", transforms.error());
                return;
            }
            const auto& [gimbal, muzzle, timestamp_ns] = *transforms;

            const auto& solver_ref           = **solver;
            const double bullet_speed        = bullet_speed_->bullet_speed;
            const auto current_gimbal        = gimbal.euler_rot();
            const double cur_pitch           = current_gimbal.pitch;
            const double cur_yaw             = current_gimbal.yaw;
            const auto previous_selected_key = selected_target_key;

            // Try Robot/Outpost from tracker (multi-target)
            const auto tracker_vec = tracker_in.read_current();
            const auto fallback_readback_tracker =
                pick_readback_tracker_fallback(tracker_vec, previous_selected_key);
            std::vector<core::TargetKey> active_target_keys;
            std::vector<CandidateEvaluation> candidates;
            std::vector<TargetSelectionTraceEntry> rejected_candidates;
            active_target_keys.reserve(tracker_vec ? tracker_vec->size() : 0U);
            candidates.reserve(tracker_vec ? tracker_vec->size() : 0U);
            rejected_candidates.reserve(tracker_vec ? tracker_vec->size() : 0U);

            if (tracker_vec) {
                for (const auto& tracker_output : *tracker_vec) {
                    if (!tracker_output.is_tracking()) {
                        continue;
                    }

                    const auto key = make_target_key(tracker_output);
                    active_target_keys.push_back(key);
                    const bool was_previously_selected =
                        previous_selected_key.has_value() && key == *previous_selected_key;

                    auto& phase_state = phase_states[key];
                    update_phase_fsm(
                        cfg->aimer, target_abs_v_yaw(tracker_output), tracker_output.target_jumped,
                        phase_state.phase, phase_state.overflow_count);

                    ArmorAimContext aim_context;
                    aim_context.target_jumped      = tracker_output.target_jumped;
                    aim_context.phase              = phase_state.phase;
                    aim_context.preferred_armor_id = tracker_output.last_armor_id;
                    if (tracker_output.target_jumped
                        && phase_state.jumped_selected_armor_id.has_value()) {
                        aim_context.preferred_armor_id = phase_state.jumped_selected_armor_id;
                    }

                    auto prediction = aim_tracker_target(
                        *aimer, tracker_output, aim_context, gimbal, muzzle, current_ns,
                        bullet_speed, solver_ref);
                    if (!prediction) {
                        rejected_candidates.push_back(build_rejected_trace_entry(
                            tracker_output, current_ns, prediction.error(),
                            was_previously_selected));
                        continue;
                    }

                    CandidateEvaluation candidate;
                    candidate.key        = key;
                    candidate.tracker    = tracker_output;
                    candidate.prediction = *prediction;
                    candidate.scores     = score_candidate(
                        cfg->aimer.target_selection, tracker_output, *prediction, cur_yaw,
                        cur_pitch, current_ns);
                    candidate.trace_entry = build_scored_trace_entry(
                        cfg->aimer.target_selection, tracker_output, *prediction, candidate.scores,
                        cur_yaw, cur_pitch, current_ns, was_previously_selected);
                    candidates.push_back(std::move(candidate));
                }
            }

            if (active_target_keys.empty()) {
                phase_states.clear();
                selected_target_key.reset();
            } else {
                for (auto it = phase_states.begin(); it != phase_states.end();) {
                    if (!contains_target_key(active_target_keys, it->first)) {
                        it = phase_states.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            std::optional<CandidateEvaluation> selected_candidate;
            TargetSelectionTrace target_selection_trace;
            target_selection_trace.timestamp_ns  = current_ns;
            target_selection_trace.switch_margin = cfg->aimer.target_selection.switch_margin;
            if (previous_selected_key.has_value()) {
                target_selection_trace.had_previous_target   = true;
                target_selection_trace.previous_target_name  = previous_selected_key->name;
                target_selection_trace.previous_target_color = previous_selected_key->color;
            }

            if (!candidates.empty()) {
                std::ranges::sort(
                    candidates, [](const CandidateEvaluation& lhs, const CandidateEvaluation& rhs) {
                        return lhs.scores.total > rhs.scores.total;
                    });

                const size_t best_index = 0;
                std::optional<size_t> current_index;
                if (selected_target_key.has_value()) {
                    for (size_t i = 0; i < candidates.size(); ++i) {
                        if (candidates[i].key == *selected_target_key) {
                            current_index = i;
                            break;
                        }
                    }
                }

                size_t selected_index    = best_index;
                bool kept_current_target = false;
                if (current_index.has_value() && *current_index != best_index
                    && candidates[best_index].scores.total
                           <= candidates[*current_index].scores.total
                                  + cfg->aimer.target_selection.switch_margin) {
                    selected_index      = *current_index;
                    kept_current_target = true;
                }

                std::optional<size_t> runner_up_index;
                if (kept_current_target) {
                    runner_up_index = best_index;
                } else {
                    for (size_t i = 0; i < candidates.size(); ++i) {
                        if (i != selected_index) {
                            runner_up_index = i;
                            break;
                        }
                    }
                }

                selected_candidate = candidates[selected_index];
                aim_result         = selected_candidate->prediction;

                target_selection_trace.kept_current_target = kept_current_target;

                target_selection_trace.candidates.reserve(
                    candidates.size() + rejected_candidates.size());
                for (size_t i = 0; i < candidates.size(); ++i) {
                    auto entry      = candidates[i].trace_entry;
                    entry.rank      = static_cast<int>(i + 1);
                    entry.selected  = i == selected_index;
                    entry.runner_up = runner_up_index.has_value() && i == *runner_up_index;
                    target_selection_trace.candidates.push_back(std::move(entry));
                }
            } else if (!fallback_readback_tracker.has_value()) {
                selected_target_key.reset();
            }

            for (auto& entry : rejected_candidates) {
                entry.rank = static_cast<int>(target_selection_trace.candidates.size() + 1);
                target_selection_trace.candidates.push_back(std::move(entry));
            }

            // Try Rune (fallback)
            if (!aim_result) {
                const auto rune_state = rune_in.read_current();
                if (rune_state && rune_state->tracking_valid) {
                    aim_result = aimer->aim(
                        *rune_state, gimbal, muzzle, current_ns, bullet_speed, solver_ref);
                }
            }

            if (aim_result) {
                aim_result->aimer_fire_advice = is_on_target_phase_aware(
                    cfg->fire_decision, aim_result->aim_phase, aim_result->diff_yaw,
                    cfg->aimer.yaw_limit_deg, cfg->aimer.min_enable_yaw_deg, cur_yaw, cur_pitch,
                    aim_result->aim_yaw, -aim_result->aim_pitch, aim_result->distance);

                if (selected_candidate) {
                    auto& phase_state              = phase_states[selected_candidate->key];
                    const bool tracker_jump_active = selected_candidate->tracker.is_tracking()
                                                  && selected_candidate->tracker.target_jumped;
                    if (aim_result->aim_phase == ArmorAimPhase::SingleArmor
                        && !tracker_jump_active) {
                        phase_state.jumped_selected_armor_id.reset();
                    } else {
                        phase_state.jumped_selected_armor_id = aim_result->selected_armor_id;
                    }
                    selected_target_key = selected_candidate->key;
                } else {
                    selected_target_key.reset();
                }

                SelectedTargetSnapshot selected_target;
                selected_target.timestamp_ns = aim_result->timestamp_ns;
                selected_target.source       = (aim_result->target_type == AimerTargetType::Rune)
                                                 ? GimbalPlanSource::Rune
                                                 : GimbalPlanSource::Armor;
                selected_target.distance     = aim_result->distance;
                selected_target.predicted_future_ns     = aim_result->predicted_future_ns;
                selected_target.aim_phase               = aim_result->aim_phase;
                selected_target.selected_armor_id       = aim_result->selected_armor_id;
                selected_target.rough_selected_armor_id = aim_result->rough_selected_armor_id;
                selected_target.aimer_fire_advice       = aim_result->aimer_fire_advice;
                selected_target.fire_advice             = aim_result->aimer_fire_advice;
                if (selected_candidate) {
                    selected_target.tracker = selected_candidate->tracker;
                }

                GimbalPlan plan;
                plan.timestamp_ns            = aim_result->timestamp_ns;
                plan.yaw                     = aim_result->aim_yaw;
                plan.pitch                   = aim_result->aim_pitch;
                plan.diff_yaw                = aim_result->diff_yaw;
                plan.aim_phase               = aim_result->aim_phase;
                plan.selected_armor_id       = aim_result->selected_armor_id;
                plan.rough_selected_armor_id = aim_result->rough_selected_armor_id;
                plan.aim_yaw                 = aim_result->aim_yaw;
                plan.aim_pitch               = aim_result->aim_pitch;
                plan.distance                = aim_result->distance;
                plan.aimer_fire_advice       = aim_result->aimer_fire_advice;
                plan.fire_advice             = aim_result->aimer_fire_advice;
                plan.predicted_future        = aim_result->predicted_future_ns;
                plan.predicted_position      = aim_result->predicted_position;
                plan.source                  = (aim_result->target_type == AimerTargetType::Rune)
                                                 ? GimbalPlanSource::Rune
                                                 : GimbalPlanSource::Armor;
                if (selected_candidate && selected_candidate->tracker.is_tracking()) {
                    PlannerSeed seed;
                    seed.state_timestamp_ns    = selected_candidate->tracker.timestamp_ns;
                    seed.target_jumped         = selected_candidate->tracker.target_jumped;
                    seed.tracker_last_armor_id = selected_candidate->tracker.last_armor_id;
                    seed.aim_phase             = aim_result->aim_phase;
                    seed.selected_armor_id     = aim_result->selected_armor_id;
                    if (const auto* robot_state = selected_candidate->tracker.robot_state()) {
                        seed.state = *robot_state;
                    } else if (
                        const auto* outpost_state = selected_candidate->tracker.outpost_state()) {
                        seed.state = *outpost_state;
                    }
                    if (!std::holds_alternative<std::monostate>(seed.state)) {
                        plan.planner_seed = std::move(seed);

                        // Build reference trajectory for L5 consumption
                        const TrajectoryBuildContext build_ctx{
                            .current_ns        = current_ns,
                            .gimbal            = gimbal,
                            .muzzle            = muzzle,
                            .trajectory_solver = &solver_ref,
                            .bullet_speed      = bullet_speed,
                        };
                        auto ref_traj = build_reference_trajectory(
                            *plan.planner_seed, *aimer, cfg->reference_trajectory, build_ctx);
                        if (ref_traj) {
                            plan.reference_trajectory = std::move(*ref_traj);
                        } else {
                            SPDLOG_WARN(
                                "Failed to build reference trajectory: {}", ref_traj.error());
                        }
                    }
                }
                VisualizationDebugData viz_debug;
                viz_debug.trajectory = solver_ref.generate_trajectory(
                    aim_result->aim_pitch, bullet_speed, kMaxVisualizationDistance);
                for (auto& p : viz_debug.trajectory) {
                    const double x = p.first;
                    const double z = p.second;
                    p.first        = x * std::cos(plan.pitch) + z * std::sin(plan.pitch);
                    p.second       = -x * std::sin(plan.pitch) + z * std::cos(plan.pitch);
                }
                plan.viz_debug = viz_debug;
                selected_target_out.write(std::move(selected_target));
                target_selection_out.write(std::move(target_selection_trace));
                aimer_out.write(std::move(plan));
            } else {
                const uint64_t output_timestamp_ns = fcs::clock::now_ns();
                SelectedTargetSnapshot selected_target;
                selected_target.timestamp_ns = output_timestamp_ns;
                if (fallback_readback_tracker.has_value()) {
                    selected_target.source  = GimbalPlanSource::Armor;
                    selected_target.tracker = *fallback_readback_tracker;
                    selected_target_key     = make_target_key(*fallback_readback_tracker);
                } else {
                    selected_target_key.reset();
                }
                GimbalPlan plan;
                plan.timestamp_ns = output_timestamp_ns;
                plan.distance     = -1;
                selected_target_out.write(std::move(selected_target));
                target_selection_out.write(std::move(target_selection_trace));
                aimer_out.write(std::move(plan));
            }
        });

    scheduler.add_system<talos::pool_compute>(
        "l4_optimal_target_readback_roi",
        [](talos::spmc<SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic>
               selected_target_in,
           talos::res<fast_tf::CoordinateSystem> tf_buffer, talos::res<CameraConfig> camera_config,
           talos::res<L2::AtLegacyReadbackRoiConfig> readback_roi_config,
           talos::res_mut<L2::AtLegacyReadbackRoiCache> readback_roi_cache) {
            auto has_readback_tracker = [](const SelectedTargetSnapshot& snapshot) noexcept {
                return snapshot.source == GimbalPlanSource::Armor
                    && tracker_supports_readback_roi(snapshot.tracker);
            };
            if (!selected_target_in.has_new()) {
                return;
            }

            const auto selected_target = selected_target_in.read();
            if (!selected_target) {
                return;
            }

            if (!readback_roi_config->enabled || !has_readback_tracker(*selected_target)) {
                readback_roi_cache->invalidate(selected_target->timestamp_ns);
                return;
            }

            const auto roi_timestamps = resolve_readback_roi_timestamps(*selected_target);
            const auto roi            = L2::project_tracker_box_to_image(
                selected_target->tracker, roi_timestamps.projection_timestamp_ns, *tf_buffer,
                *camera_config, readback_roi_config->box_size_m);
            if (!roi) {
                readback_roi_cache->invalidate(roi_timestamps.freshness_timestamp_ns);
                return;
            }

            readback_roi_cache->store(
                L2::AtLegacyReadbackRoiSnapshot{
                    .valid        = true,
                    .timestamp_ns = roi_timestamps.freshness_timestamp_ns,
                    .raw_roi      = *roi,
                });
        });
}

} // namespace fcs::L4
