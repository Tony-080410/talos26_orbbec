#include "L4_planning/aimer/aimer_systems.hpp"
#include "L4_planning/aimer/aimer.hpp"

#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/aimer/types.hpp"
#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "core/trajectory/resource.hpp"
#include "euler.hpp"
#include "scheduler/scheduler.hpp"

#include <cmath>
#include <optional>
#include <spdlog/spdlog.h>

namespace fcs::L4 {
static constexpr double kMaxVisualizationDistance = 15.0;

namespace {

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
        overflow_count = (abs_v_yaw > cfg.single_whole_up) ? overflow_count + 1 : 0;
        if (overflow_count > cfg.transfer_threshold_single_whole) {
            phase          = ArmorAimPhase::WholeCarArmor;
            overflow_count = 0;
        }
        break;

    case ArmorAimPhase::WholeCarArmor:
        if (abs_v_yaw > cfg.whole_pair_up) {
            ++overflow_count;
        } else if (abs_v_yaw < cfg.single_whole_down) {
            --overflow_count;
        } else {
            overflow_count = 0;
        }
        if (std::abs(overflow_count) > cfg.transfer_threshold) {
            phase = (overflow_count > 0) ? ArmorAimPhase::WholeCarPair : ArmorAimPhase::SingleArmor;
            overflow_count = 0;
        }
        break;

    case ArmorAimPhase::WholeCarPair:
        if (abs_v_yaw > cfg.pair_center_up) {
            ++overflow_count;
        } else if (abs_v_yaw < cfg.whole_pair_down) {
            --overflow_count;
        } else {
            overflow_count = 0;
        }
        if (std::abs(overflow_count) > cfg.transfer_threshold) {
            phase =
                (overflow_count > 0) ? ArmorAimPhase::WholeCarCenter : ArmorAimPhase::WholeCarArmor;
            overflow_count = 0;
        }
        break;

    case ArmorAimPhase::WholeCarCenter:
        overflow_count = (abs_v_yaw < cfg.pair_center_down) ? overflow_count + 1 : 0;
        if (overflow_count > cfg.transfer_threshold) {
            phase          = ArmorAimPhase::WholeCarPair;
            overflow_count = 0;
        }
        break;
    }
}

} // namespace

void register_aimer_systems(talos::Scheduler& scheduler, L4Config&& config) {
    scheduler.world().insert_resource(Aimer(config.aimer));
    scheduler.world().insert_resource(std::move(config));

    scheduler.add_system<talos::pool_compute>(
        "l4_aimer",
        [phase_state = ArmorAimPhase::SingleArmor, phase_overflow_count = 0,
         jumped_selected_armor_id = std::optional<int>{}](
            talos::res<L4Config> cfg, talos::res<Aimer> aimer,
            core::trajectory::trajectory_solver solver,
            core::trajectory::bullet_speed bullet_speed_,
            talos::spmc<L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
            talos::spmc<energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic> rune_in,
            talos::res<fast_tf::CoordinateSystem> tf_buffer,
            talos::spmc_mut<GimbalPlan, GimbalPlanChannelTopic> aimer_out) mutable {
            std::expected<TargetPrediction, std::string> aim_result = std::unexpected("none");

            const uint64_t current_ns = fcs::clock::now_ns();

            auto transforms = lookup_gimbal_muzzle_transforms(*tf_buffer, current_ns);
            if (!transforms) {
                SPDLOG_WARN("Transform lookup failed: {}", transforms.error());
                return;
            }
            const auto& [gimbal, muzzle, timestamp_ns] = *transforms;
            (void)timestamp_ns;

            if (!solver->get()) {
                SPDLOG_ERROR("Aimer: trajectory solver not available");
                return;
            }
            const auto& solver_ref    = **solver;
            const double bullet_speed = bullet_speed_->bullet_speed;

            // Try Robot/Outpost from tracker
            const auto tracker_output = tracker_in.read();
            if (tracker_output && tracker_output->is_tracking()) {
                update_phase_fsm(
                    cfg->aimer,
                    std::abs(
                        tracker_output->is_robot() ? tracker_output->robot_state()->v_yaw
                                                   : tracker_output->outpost_state()->v_yaw),
                    tracker_output->target_jumped, phase_state, phase_overflow_count);

                ArmorAimContext aim_context;
                aim_context.target_jumped      = tracker_output->target_jumped;
                aim_context.phase              = phase_state;
                aim_context.preferred_armor_id = tracker_output->last_armor_id;
                if (tracker_output->target_jumped && jumped_selected_armor_id.has_value()) {
                    aim_context.preferred_armor_id = jumped_selected_armor_id;
                }

                if (tracker_output->is_robot()) {
                    if (const auto robot_state = tracker_output->robot_state()) {
                        aim_result = aimer->aim(
                            *robot_state, aim_context, gimbal, muzzle, tracker_output->timestamp_ns,
                            current_ns, 0.0, bullet_speed, solver_ref);
                    }
                } else if (tracker_output->is_outpost()) {
                    if (const auto outpost_state = tracker_output->outpost_state()) {
                        aim_result = aimer->aim(
                            *outpost_state, aim_context, gimbal, muzzle,
                            tracker_output->timestamp_ns, current_ns, 0.0, bullet_speed,
                            solver_ref);
                    }
                }
            } else {
                phase_state          = ArmorAimPhase::SingleArmor;
                phase_overflow_count = 0;
                jumped_selected_armor_id.reset();
            }

            // Try Rune (fallback)
            if (!aim_result) {
                const auto rune_state = rune_in.read();
                if (rune_state && rune_state->tracking_valid) {
                    aim_result = aimer->aim(
                        *rune_state, gimbal, muzzle, current_ns, bullet_speed, solver_ref);
                }
            }

            if (aim_result) {
                const auto euler     = gimbal.euler_rot();
                const auto cur_pitch = euler.pitch;
                const auto cur_yaw   = euler.yaw;

                aim_result->aimer_fire_advice = is_on_target_phase_aware(
                    cfg->fire_decision, aim_result->aim_phase, aim_result->diff_yaw,
                    cfg->aimer.yaw_limit_deg, cfg->aimer.min_enable_yaw_deg, cur_yaw, cur_pitch,
                    aim_result->aim_yaw, -aim_result->aim_pitch, aim_result->distance);

                const bool tracker_jump_active = tracker_output && tracker_output->is_tracking()
                                              && tracker_output->target_jumped;
                if (aim_result->aim_phase == ArmorAimPhase::SingleArmor && !tracker_jump_active) {
                    jumped_selected_armor_id.reset();
                } else {
                    jumped_selected_armor_id = aim_result->selected_armor_id;
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
                if (tracker_output && tracker_output->is_tracking()) {
                    PlannerSeed seed;
                    seed.state_timestamp_ns    = tracker_output->timestamp_ns;
                    seed.target_jumped         = tracker_output->target_jumped;
                    seed.tracker_last_armor_id = tracker_output->last_armor_id;
                    seed.aim_phase             = aim_result->aim_phase;
                    seed.selected_armor_id     = aim_result->selected_armor_id;
                    if (const auto robot_state = tracker_output->robot_state()) {
                        seed.state = *robot_state;
                    } else if (const auto outpost_state = tracker_output->outpost_state()) {
                        seed.state = *outpost_state;
                    }
                    if (!std::holds_alternative<std::monostate>(seed.state)) {
                        plan.planner_seed = std::move(seed);
                    }
                }
                VisualizationDebugData viz_debug;
                viz_debug.trajectory = solver->get()->generate_trajectory(
                    aim_result->aim_pitch, bullet_speed, kMaxVisualizationDistance);
                for (auto& p : viz_debug.trajectory) {
                    const double x = p.first;
                    const double z = p.second;
                    p.first        = x * std::cos(plan.pitch) + z * std::sin(plan.pitch);
                    p.second       = -x * std::sin(plan.pitch) + z * std::cos(plan.pitch);
                }
                plan.viz_debug = viz_debug;
                aimer_out.write(std::move(plan));
            } else {
                GimbalPlan plan;
                plan.distance = -1;
                aimer_out.write(std::move(plan));
            }
        });
}

} // namespace fcs::L4
