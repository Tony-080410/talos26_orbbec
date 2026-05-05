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

// Decision — 三种目标源输出的统一类型
struct AimDecision {
    ControlIntent intent;
    SelectedTargetSnapshot snapshot;
    TargetSelectionTrace trace;
};

// AimContext — 三个决策函数共享的只读上下文
struct AimContext {
    const Aimer::GimbalTransform& gimbal;
    const Aimer::MuzzleTransform& muzzle;
    uint64_t current_ns;
    double bullet_speed;
    const core::trajectory::solver::TrajectorySolver& solver;
    const Aimer& aimer;
};

// 从 TargetPrediction 构建 ControlIntent
[[nodiscard]] ControlIntent build_control_intent(
    const TargetPrediction& pred, const AimContext& ctx,
    const ReferenceTrajectoryConfig* ref_traj_cfg, const std::optional<PlannerSeed>& seed,
    const std::optional<L3::ldm::LdmState>& ldm_for_track) noexcept {
    const TrajectoryBuildContext build_ctx{
        .current_ns        = ctx.current_ns,
        .gimbal            = ctx.gimbal,
        .muzzle            = ctx.muzzle,
        .trajectory_solver = &ctx.solver,
        .bullet_speed      = ctx.bullet_speed,
    };

    std::optional<std::string> degradation;

    if (seed.has_value() && ref_traj_cfg) {
        auto traj = build_reference_trajectory(*seed, ctx.aimer, *ref_traj_cfg, build_ctx);
        if (traj) {
            return TrackCommand{
                .timestamp_ns = pred.timestamp_ns,
                .trajectory   = std::move(*traj),
            };
        }
        degradation = traj.error();
        SPDLOG_WARN("Failed to build reference trajectory: {}", *degradation);
    } else if (ldm_for_track.has_value() && ldm_for_track->is_tracking() && ref_traj_cfg) {
        auto traj =
            build_ldm_reference_trajectory(*ldm_for_track, ctx.aimer, *ref_traj_cfg, build_ctx);
        if (traj) {
            return TrackCommand{
                .timestamp_ns = pred.timestamp_ns,
                .trajectory   = std::move(*traj),
            };
        }
        degradation = traj.error();
        SPDLOG_WARN("Failed to build LDM reference trajectory: {}", *degradation);
    }

    return ShotCommand{
        .timestamp_ns       = pred.timestamp_ns,
        .yaw                = pred.aim_yaw,
        .pitch              = pred.aim_pitch,
        .distance           = pred.distance,
        .degradation_reason = std::move(degradation),
    };
}

// try_armor — 多目标 tracker → 评分 → 选优 → 弹道
[[nodiscard]] std::optional<AimDecision> try_armor(
    const std::optional<L3::TrackerOutputs>& trackers, const AimContext& ctx,
    const AimerConfig& aimer_cfg, const TargetSelectionConfig& sel_cfg,
    const ReferenceTrajectoryConfig& ref_traj_cfg,
    std::unordered_map<core::TargetKey, TargetPhaseState, core::TargetKeyHash>& phase_states,
    std::optional<core::TargetKey>& selected_key) {
    const auto prev_key    = selected_key;
    const auto cur_euler   = ctx.gimbal.euler_rot();
    const double cur_yaw   = cur_euler.yaw;
    const double cur_pitch = cur_euler.pitch;

    std::vector<core::TargetKey> active_keys;
    std::vector<CandidateEvaluation> candidates;
    std::vector<TargetSelectionTraceEntry> rejected;

    const size_t reserve_n = trackers ? trackers->size() : 0U;
    active_keys.reserve(reserve_n);
    candidates.reserve(reserve_n);
    rejected.reserve(reserve_n);

    if (trackers) {
        for (const auto& t : *trackers) {
            if (!t.is_tracking())
                continue;

            const auto key = make_target_key(t);
            active_keys.push_back(key);
            const bool was_prev = prev_key.has_value() && key == *prev_key;

            auto& ps = phase_states[key];
            advance_armor_aim_phase(
                aimer_cfg, target_abs_v_yaw(t), t.target_jumped, ps.phase, ps.overflow_count);

            ArmorAimContext aim_ctx;
            aim_ctx.target_jumped      = t.target_jumped;
            aim_ctx.phase              = ps.phase;
            aim_ctx.preferred_armor_id = t.last_armor_id;
            if (t.target_jumped && ps.jumped_selected_armor_id.has_value()) {
                aim_ctx.preferred_armor_id = ps.jumped_selected_armor_id;
            }

            auto pred = aim_tracker_target(
                ctx.aimer, t, aim_ctx, ctx.gimbal, ctx.muzzle, ctx.current_ns, ctx.bullet_speed,
                ctx.solver);
            if (!pred) {
                rejected.push_back(
                    build_rejected_trace_entry(t, ctx.current_ns, pred.error(), was_prev));
                continue;
            }

            CandidateEvaluation c;
            c.key         = key;
            c.tracker     = t;
            c.prediction  = *pred;
            c.scores      = score_candidate(sel_cfg, t, *pred, cur_yaw, cur_pitch, ctx.current_ns);
            c.trace_entry = build_scored_trace_entry(
                sel_cfg, t, *pred, c.scores, cur_yaw, cur_pitch, ctx.current_ns, was_prev);
            candidates.push_back(std::move(c));
        }
    }

    // Cleanup stale phase states
    if (active_keys.empty()) {
        phase_states.clear();
        selected_key.reset();
    } else {
        std::erase_if(phase_states, [&](const auto& kv) {
            return !contains_target_key(active_keys, kv.first);
        });
    }

    if (candidates.empty())
        return std::nullopt;

    // Sort by score descending
    std::ranges::sort(candidates, std::greater<>{}, [](const CandidateEvaluation& c) noexcept {
        return c.scores.total;
    });

    // Switch-margin selection
    size_t selected_idx = 0;
    bool kept           = false;
    std::optional<size_t> current_idx;
    if (selected_key.has_value()) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].key == *selected_key) {
                current_idx = i;
                break;
            }
        }
    }
    if (current_idx.has_value() && *current_idx != 0
        && candidates[0].scores.total
               <= candidates[*current_idx].scores.total + sel_cfg.switch_margin) {
        selected_idx = *current_idx;
        kept         = true;
    }

    std::optional<size_t> runner_up;
    if (kept) {
        runner_up = 0;
    } else {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i != selected_idx) {
                runner_up = i;
                break;
            }
        }
    }

    auto& sel = candidates[selected_idx];

    // Build trace
    TargetSelectionTrace trace{
        .timestamp_ns        = ctx.current_ns,
        .switch_margin       = sel_cfg.switch_margin,
        .kept_current_target = kept,
    };
    if (prev_key.has_value()) {
        trace.had_previous_target   = true;
        trace.previous_target_name  = prev_key->name;
        trace.previous_target_color = prev_key->color;
    }
    trace.candidates.reserve(candidates.size() + rejected.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto e      = candidates[i].trace_entry;
        e.rank      = static_cast<int>(i + 1);
        e.selected  = i == selected_idx;
        e.runner_up = runner_up.has_value() && i == *runner_up;
        trace.candidates.push_back(std::move(e));
    }
    for (auto& e : rejected) {
        e.rank = static_cast<int>(trace.candidates.size() + 1);
        trace.candidates.push_back(std::move(e));
    }

    // Update persistent state
    sel.prediction.armor_type = cls_to_armor_type(sel.tracker.target_name);
    auto& ps                  = phase_states[sel.key];
    const bool jump_active    = sel.tracker.is_tracking() && sel.tracker.target_jumped;
    if (sel.prediction.aim_phase == ArmorAimPhase::SingleArmor && !jump_active) {
        ps.jumped_selected_armor_id.reset();
    } else {
        ps.jumped_selected_armor_id = sel.prediction.selected_armor_id;
    }
    selected_key = sel.key;

    // Build PlannerSeed for trajectory
    std::optional<PlannerSeed> seed;
    if (sel.tracker.is_tracking()) {
        PlannerSeed s;
        s.state_timestamp_ns    = sel.tracker.timestamp_ns;
        s.target_jumped         = sel.tracker.target_jumped;
        s.tracker_last_armor_id = sel.tracker.last_armor_id;
        s.aim_phase             = sel.prediction.aim_phase;
        s.selected_armor_id     = sel.prediction.selected_armor_id;
        s.armor_type            = sel.prediction.armor_type;
        if (const auto* rs = sel.tracker.robot_state()) {
            s.state = *rs;
        } else if (const auto* os = sel.tracker.outpost_state()) {
            s.state = *os;
        } else {
            s.state = std::monostate{};
        }
        if (!std::holds_alternative<std::monostate>(s.state)) {
            seed = std::move(s);
        }
    }

    SelectedTargetSnapshot snap;
    snap.timestamp_ns            = sel.prediction.timestamp_ns;
    snap.source                  = GimbalPlanSource::Armor;
    snap.distance                = sel.prediction.distance;
    snap.predicted_future_ns     = sel.prediction.predicted_future_ns;
    snap.aim_phase               = sel.prediction.aim_phase;
    snap.selected_armor_id       = sel.prediction.selected_armor_id;
    snap.rough_selected_armor_id = sel.prediction.rough_selected_armor_id;
    snap.tracker                 = sel.tracker;

    return AimDecision{
        .intent   = build_control_intent(sel.prediction, ctx, &ref_traj_cfg, seed, std::nullopt),
        .snapshot = std::move(snap),
        .trace    = std::move(trace),
    };
}

// try_rune — EnergyMeterState → 直接射击
[[nodiscard]] std::optional<AimDecision> try_rune(
    const std::optional<energy_meter::EnergyMeterState>& rune_in, const AimContext& ctx,
    std::optional<energy_meter::EnergyMeterState>& state_cache,
    std::optional<std::pair<double, double>>& last_aim_angles) {
    if (rune_in && rune_in->tracking_valid) {
        state_cache = *rune_in;
    }
    if (!state_cache)
        return std::nullopt;

    if (elapsed_seconds(state_cache->timestamp_ns, ctx.current_ns) >= 0.5) {
        state_cache.reset();
        return std::nullopt;
    }

    auto pred = ctx.aimer.aim(
        *state_cache, ctx.gimbal, ctx.muzzle, ctx.current_ns, ctx.bullet_speed, ctx.solver);
    if (!pred)
        return std::nullopt;

    last_aim_angles = {pred->aim_yaw, pred->aim_pitch};

    SelectedTargetSnapshot snap;
    snap.timestamp_ns            = pred->timestamp_ns;
    snap.source                  = GimbalPlanSource::Rune;
    snap.distance                = pred->distance;
    snap.predicted_future_ns     = pred->predicted_future_ns;
    snap.aim_phase               = pred->aim_phase;
    snap.selected_armor_id       = pred->selected_armor_id;
    snap.rough_selected_armor_id = pred->rough_selected_armor_id;

    return AimDecision{
        .intent =
            build_control_intent(*pred, ctx, /*ref_traj_cfg=*/nullptr, std::nullopt, std::nullopt),
        .snapshot = std::move(snap),
        .trace    = TargetSelectionTrace{.timestamp_ns = ctx.current_ns},
    };
}

// try_ldm — LdmState → 弹道或直接射击
[[nodiscard]] std::optional<AimDecision> try_ldm(
    const std::optional<L3::ldm::LdmState>& ldm_in, const AimContext& ctx,
    const ReferenceTrajectoryConfig& ref_traj_cfg, std::optional<L3::ldm::LdmState>& state_cache) {
    if (ldm_in && ldm_in->is_tracking()) {
        state_cache = *ldm_in;
    }
    if (!state_cache || !state_cache->is_tracking()) {
        // Router handles cache lifecycle — don't clear here
        return std::nullopt;
    }

    auto pred = ctx.aimer.aim(
        *state_cache, ctx.gimbal, ctx.muzzle, ctx.current_ns, ctx.bullet_speed, ctx.solver);
    if (!pred)
        return std::nullopt;

    SelectedTargetSnapshot snap;
    snap.timestamp_ns            = pred->timestamp_ns;
    snap.source                  = GimbalPlanSource::Armor;
    snap.distance                = pred->distance;
    snap.predicted_future_ns     = pred->predicted_future_ns;
    snap.aim_phase               = pred->aim_phase;
    snap.selected_armor_id       = pred->selected_armor_id;
    snap.rough_selected_armor_id = pred->rough_selected_armor_id;

    return AimDecision{
        .intent   = build_control_intent(*pred, ctx, &ref_traj_cfg, std::nullopt, *state_cache),
        .snapshot = std::move(snap),
        .trace    = TargetSelectionTrace{.timestamp_ns = ctx.current_ns},
    };
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
         selected_key = std::optional<core::TargetKey>{},
         rune_cache   = std::optional<energy_meter::EnergyMeterState>{},
         rune_angles  = std::optional<std::pair<double, double>>{},
         ldm_cache    = std::optional<L3::ldm::LdmState>{}](
            talos::res<L4Config> cfg, talos::res<Aimer> aimer,
            core::trajectory::trajectory_solver solver,
            core::trajectory::bullet_speed bullet_speed_,
            talos::spmc<L3::TrackerOutputs, TrackerOutputChannelTopic> tracker_in,
            talos::spmc<energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic> rune_in,
            talos::spmc<L3::ldm::LdmState> ldm_in, talos::res<fast_tf::CoordinateSystem> tf_buffer,
            talos::spmc_mut<SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic> out_snap,
            talos::spmc_mut<TargetSelectionTrace, TargetSelectionTraceChannelTopic> out_trace,
            talos::spmc_mut<ControlIntent, ControlIntentChannelTopic> out_ctrl) mutable {
            const uint64_t now_ns = fcs::clock::now_ns();
            const auto xf         = lookup_gimbal_muzzle_transforms(*tf_buffer, now_ns);
            if (!xf) [[unlikely]] {
                SPDLOG_WARN("tf lookup failed: {}", xf.error());
                return;
            }
            const auto& [gimbal, muzzle, ts_ns] = *xf;

            const AimContext ctx{
                .gimbal       = gimbal,
                .muzzle       = muzzle,
                .current_ns   = now_ns,
                .bullet_speed = bullet_speed_->bullet_speed,
                .solver       = **solver,
                .aimer        = *aimer,
            };

            const auto tracker_vec = tracker_in.read_current();
            const auto fallback    = pick_readback_tracker_fallback(tracker_vec, selected_key);

            auto decision = try_armor(
                tracker_vec, ctx, cfg->aimer, cfg->aimer.target_selection,
                cfg->reference_trajectory, phase_states, selected_key);

            if (!decision) {
                decision = try_rune(rune_in.read_current(), ctx, rune_cache, rune_angles);
                if (auto ldm_d =
                        try_ldm(ldm_in.read(), ctx, cfg->reference_trajectory, ldm_cache)) {
                    decision = std::move(ldm_d);
                } else if (!decision) {
                    // Neither Rune nor LDM succeeded: clear LDM cache
                    ldm_cache.reset();
                }
                if (decision) {
                    // Non-armor target won: clear armor selection state
                    selected_key.reset();
                }
            }

            if (decision) {
                out_snap.write(std::move(decision->snapshot));
                out_trace.write(std::move(decision->trace));
                out_ctrl.write(std::move(decision->intent));
            } else {
                const uint64_t out_ns = fcs::clock::now_ns();
                SelectedTargetSnapshot snap{.timestamp_ns = out_ns};
                if (fallback.has_value()) {
                    snap.source  = GimbalPlanSource::Armor;
                    snap.tracker = *fallback;
                    selected_key = make_target_key(*fallback);
                } else {
                    selected_key.reset();
                }
                out_snap.write(std::move(snap));
                out_trace.write(TargetSelectionTrace{.timestamp_ns = out_ns});
                out_ctrl.write(HoldCommand{.timestamp_ns = out_ns});
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
