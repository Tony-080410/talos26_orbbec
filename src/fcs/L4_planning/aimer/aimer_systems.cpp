#include "L4_planning/aimer/aimer_systems.hpp"
#include "L3_estimation/ldm_naive/types.hpp"
#include "L4_planning/aimer/aimer.hpp"
#include "L4_planning/aimer/fsm.hpp"

#include "L2_perception/armor/readback_roi.hpp"
#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/aimer/types.hpp"
#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/config.hpp"
#include "L4_planning/control_intent.hpp"
#include "L4_planning/plan_source.hpp"
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
#include <frame.hpp>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <vector>

namespace fcs::L4 {

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
    entry.weighted_sum           = entry.image_center_weighted + entry.track_state_weighted
                       + entry.tof_weighted + entry.distance_weighted + entry.gimbal_effort_weighted
                       + entry.armor_name_weighted;
    entry.total_weight = cfg.weights.image_center + cfg.weights.track_state + cfg.weights.tof
                       + cfg.weights.distance + cfg.weights.gimbal_effort + cfg.weights.armor_name;
    entry.total_score = scores.total;
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
    if (!world.has_resource<L2::ArmorReadbackRoiConfig>()) {
        world.insert_resource(L2::ArmorReadbackRoiConfig{});
    }
    if (!world.has_resource<L2::TrackerReadbackCache>()) {
        world.insert_resource(L2::TrackerReadbackCache{});
    }

    world.insert_resource(Aimer(config.aimer));
    world.insert_resource(std::move(config));

    scheduler.add_system<talos::fixed_rate<250>>(
        "l4_aimer",
        [phase_states =
             std::unordered_map<core::TargetKey, TargetPhaseState, core::TargetKeyHash>{},
         selected_target_key  = std::optional<core::TargetKey>{},
         ldm_state_cache      = std::optional<L3::ldm::LdmState>{},
         rune_state_cache     = std::optional<energy_meter::EnergyMeterState>{},
         rune_last_aim_angles = std::optional<std::pair<double, double>>{}](
            talos::res<L4Config> cfg, talos::res<Aimer> aimer,
            core::trajectory::trajectory_solver solver,
            core::trajectory::bullet_speed bullet_speed_,
            talos::spmc<L3::TrackerOutputs, TrackerOutputChannelTopic> tracker_in,
            talos::spmc<energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic> rune_in,
            talos::spmc<L3::ldm::LdmState> ldm_in, talos::res<fast_tf::CoordinateSystem> tf_buffer,
            talos::spmc_mut<SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic>
                selected_target_out,
            talos::spmc_mut<TargetSelectionTrace, TargetSelectionTraceChannelTopic>
                target_selection_out,
            talos::spmc_mut<ControlIntent, ControlIntentChannelTopic> control_out) mutable {
            std::expected<TargetPrediction, std::string> aim_result = std::unexpected("none");

            const uint64_t current_ns = fcs::clock::now_ns();

            auto transforms = lookup_gimbal_muzzle_transforms(*tf_buffer, current_ns);
            if (!transforms) [[unlikely]] {
                SPDLOG_WARN("tf lookup failed: {}", transforms.error());
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
                    advance_armor_aim_phase(
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

            // Try Rune (cached state persists through brief tracking dropouts)
            if (!aim_result) {
                const auto rune_state = rune_in.read_current();
                if (rune_state && rune_state->tracking_valid) {
                    rune_state_cache = *rune_state;
                }

                if (rune_state_cache) {
                    const double staleness =
                        elapsed_seconds(rune_state_cache->timestamp_ns, current_ns);
                    if (staleness < 0.5) {
                        aim_result = aimer->aim(
                            *rune_state_cache, gimbal, muzzle, current_ns, bullet_speed,
                            solver_ref);
                    } else {
                        rune_state_cache.reset();
                    }
                }

                // LDM (overrides rune if available)
                const auto ldm_state = ldm_in.read();
                if (ldm_state && ldm_state->is_tracking()) {
                    ldm_state_cache = *ldm_state;
                    aim_result      = aimer->aim(
                        *ldm_state, gimbal, muzzle, current_ns, bullet_speed, solver_ref);
                } else if (!aim_result) {
                    ldm_state_cache.reset();
                }
            }

            if (aim_result) {
                // Cache rune aim angles for the no-target fallback path
                if (aim_result->target_type == AimerTargetType::Rune
                    || aim_result->target_type == AimerTargetType::Ldm) {
                    rune_last_aim_angles = {aim_result->aim_yaw, aim_result->aim_pitch};
                }
                if (selected_candidate) {
                    aim_result->armor_type =
                        cls_to_armor_type(selected_candidate->tracker.target_name);
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
                if (selected_candidate) {
                    selected_target.tracker = selected_candidate->tracker;
                }

                const bool is_ldm_target = aim_result->target_type == AimerTargetType::Ldm
                                        && ldm_state_cache.has_value()
                                        && ldm_state_cache->is_tracking();

                // Build ControlIntent: Track if trajectory built, Shot otherwise.
                ControlIntent intent = [&]() -> ControlIntent {
                    std::optional<std::string> degradation;
                    if (selected_candidate && selected_candidate->tracker.is_tracking()) {
                        PlannerSeed seed;
                        seed.state_timestamp_ns    = selected_candidate->tracker.timestamp_ns;
                        seed.target_jumped         = selected_candidate->tracker.target_jumped;
                        seed.tracker_last_armor_id = selected_candidate->tracker.last_armor_id;
                        seed.aim_phase             = aim_result->aim_phase;
                        seed.selected_armor_id     = aim_result->selected_armor_id;
                        seed.armor_type            = aim_result->armor_type;
                        if (const auto* rs = selected_candidate->tracker.robot_state()) {
                            seed.state = *rs;
                        } else if (const auto* os = selected_candidate->tracker.outpost_state()) {
                            seed.state = *os;
                        }
                        if (!std::holds_alternative<std::monostate>(seed.state)) {
                            const TrajectoryBuildContext build_ctx{
                                .current_ns        = current_ns,
                                .gimbal            = gimbal,
                                .muzzle            = muzzle,
                                .trajectory_solver = &solver_ref,
                                .bullet_speed      = bullet_speed,
                            };
                            auto ref_traj = build_reference_trajectory(
                                seed, *aimer, cfg->reference_trajectory, build_ctx);
                            if (ref_traj) {
                                return TrackCommand{
                                    .timestamp_ns = aim_result->timestamp_ns,
                                    .trajectory   = std::move(*ref_traj),
                                };
                            }
                            degradation = ref_traj.error();
                            SPDLOG_WARN(
                                "Failed to build reference trajectory: {}", ref_traj.error());
                        }
                    } else if (is_ldm_target) {
                        const TrajectoryBuildContext build_ctx{
                            .current_ns        = current_ns,
                            .gimbal            = gimbal,
                            .muzzle            = muzzle,
                            .trajectory_solver = &solver_ref,
                            .bullet_speed      = bullet_speed,
                        };
                        auto ref_traj = build_ldm_reference_trajectory(
                            *ldm_state_cache, *aimer, cfg->reference_trajectory, build_ctx);
                        if (ref_traj) {
                            return TrackCommand{
                                .timestamp_ns = aim_result->timestamp_ns,
                                .trajectory   = std::move(*ref_traj),
                            };
                        }
                        degradation = ref_traj.error();
                        SPDLOG_WARN(
                            "Failed to build LDM reference trajectory: {}", ref_traj.error());
                    }
                    // Have aim angles but no trajectory: direct shot.
                    // degradation carries the trajectory-build error when this is a
                    // fallback; nullopt means this is an intentional direct-shot.
                    return ShotCommand{
                        .timestamp_ns       = aim_result->timestamp_ns,
                        .yaw                = aim_result->aim_yaw,
                        .pitch              = aim_result->aim_pitch,
                        .distance           = aim_result->distance,
                        .degradation_reason = std::move(degradation),
                    };
                }();

                selected_target_out.write(std::move(selected_target));
                target_selection_out.write(std::move(target_selection_trace));
                control_out.write(std::move(intent));
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
                selected_target_out.write(std::move(selected_target));
                target_selection_out.write(std::move(target_selection_trace));
                control_out.write(HoldCommand{.timestamp_ns = output_timestamp_ns});
            }
        });

    scheduler.add_system<talos::pool_compute>(
        "l4_optimal_target_readback_roi",
        [](talos::spmc<SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic>
               selected_target_in,
           talos::res<L2::ArmorReadbackRoiConfig> readback_roi_config,
           talos::res_mut<L2::TrackerReadbackCache> readback_cache) {
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
                readback_cache->invalidate(selected_target->timestamp_ns);
                return;
            }

            const auto roi_timestamps = resolve_readback_roi_timestamps(*selected_target);
            readback_cache->store(
                L2::TrackerReadbackSnapshot{
                    .valid                   = true,
                    .timestamp_ns            = roi_timestamps.freshness_timestamp_ns,
                    .projection_timestamp_ns = roi_timestamps.projection_timestamp_ns,
                    .selected_armor_id       = selected_target->selected_armor_id,
                    .rough_selected_armor_id = selected_target->rough_selected_armor_id,
                    .tracker                 = selected_target->tracker,
                });
        });
}

} // namespace fcs::L4
