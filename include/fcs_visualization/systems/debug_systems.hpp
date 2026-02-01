#pragma once

#include "base.hpp"
#include "fcs/L3_estimation/tracker/vis_helpers.hpp"
#include "fcs_visualization/foxglove_types.hpp"
#include "fcs_visualization/scene_builder.hpp"
#include "fcs_visualization/system_helpers.hpp"

#include "fcs/L2_perception/rune/rune_config.hpp"
#include "fcs/L2_perception/rune/types.hpp"
#include "fcs/L3_estimation/energy_meter/types.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/types.hpp"
#include "fcs/L5_weapon/fire_control.hpp"
#include "fcs/core/channel_topics.hpp"
#include "fcs/core/types.hpp"
#include "math/fast_tf/frame.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler.hpp"

#include <magic_enum.hpp>
#include <nlohmann/json.hpp>

namespace fcs::foxglove::systems {

using namespace ::fcs::L3::vis;

/// @brief Register debug visualization systems
///
/// This includes all debug publishing systems for L2, L3, and L4 layers:
/// - L2: PnP solver metrics, NN confidence scores
/// - L3: Association phase, permutation, filter innovation/covariance, Viterbi, state machine
/// - L4: MPC horizon, cost, constraints
/// - Misc: Energy meter state, performance statistics
template <typename App>
void register_debug_systems(App& app, talos::Scheduler* scheduler_ptr = nullptr) {

    // =========================================================================
    // L2 DEBUG: PnP Solver Metrics
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l2_pnp_pub",
        [](talos::spmc<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic> meas_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, meas_in)) {
                return;
            }

            auto batch = meas_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json pnp_json;
            pnp_json["timestamp_ns"] = batch->timestamp_ns;
            pnp_json["measurements"] = nlohmann::json::array();

            for (size_t i = 0; i < batch->measurements.size(); ++i) {
                const auto& m = batch->measurements[i];
                nlohmann::json obj;
                obj["index"]      = i;
                obj["name"]       = magic_enum::enum_name(m.name);
                const auto t      = m.transform->translation();
                obj["position"]   = {t.x(), t.y(), t.z()};
                const auto rpy    = m.transform->euler_rot().rpy();
                obj["rpy"]        = {std::get<0>(rpy), std::get<1>(rpy), std::get<2>(rpy)};
                obj["confidence"] = m.confidence;
                pnp_json["measurements"].push_back(obj);
            }

            detail::publish_json_message<PnPSolverMessage>(*server, pnp_json);
        });

    // =========================================================================
    // L2 DEBUG: NN Confidence Scores
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l2_nn_confidence_pub",
        [](talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTopic> det_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, det_in)) {
                return;
            }

            auto batch = det_in.read();
            if (!batch) {
                return;
            }

            nlohmann::json nn_json;
            nn_json["timestamp_ns"] = batch->timestamp_ns;
            nn_json["detections"]   = nlohmann::json::array();

            for (size_t i = 0; i < batch->detections.size(); ++i) {
                const auto& det = batch->detections[i];
                nlohmann::json obj;
                obj["index"]      = i;
                obj["confidence"] = det.confidence;
                obj["color"]      = magic_enum::enum_name(det.color);
                obj["type"]       = magic_enum::enum_name(det.type);
                obj["name"]       = magic_enum::enum_name(det.name);
                nn_json["detections"].push_back(obj);
            }

            detail::publish_json_message<NNConfidenceMessage>(*server, nn_json);
        });

    // =========================================================================
    // L3 DEBUG: Association Phase Probabilities
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l3_association_phase_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->association_debug) {
                return;
            }

            const auto& debug = *output->association_debug;
            nlohmann::json phase_json;
            phase_json["timestamp_ns"] = output->timestamp_ns;
            phase_json["target_name"]  = magic_enum::enum_name(output->target_name);
            phase_json["phase_probs"]  = debug.phase_probs;
            phase_json["map_phase"]    = debug.map_phase;
            phase_json["is_ambiguous"] = debug.is_ambiguous;
            phase_json["n_phases"]     = debug.phase_probs.size();

            detail::publish_json_message<AssociationPhaseMessage>(*server, phase_json);
        });

    // =========================================================================
    // L3 DEBUG: Association Permutation Rankings
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l3_association_permutation_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->association_debug) {
                return;
            }

            const auto& debug = *output->association_debug;
            nlohmann::json perm_json;
            perm_json["timestamp_ns"] = output->timestamp_ns;
            perm_json["permutations"] = nlohmann::json::array();

            for (const auto& perm_info : debug.top_permutations) {
                nlohmann::json p;
                p["perm"]           = perm_info.perm;
                p["log_prob"]       = perm_info.log_prob;
                p["gate_passed"]    = perm_info.gate_passed;
                p["mahalanobis_sq"] = perm_info.mahalanobis_sq;
                perm_json["permutations"].push_back(p);
            }

            detail::publish_json_message<AssociationPermutationMessage>(*server, perm_json);
        });

    // =========================================================================
    // L3 DEBUG: Filter Innovation Vectors
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l3_filter_innovation_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->filter_debug) {
                return;
            }

            const auto& debug = *output->filter_debug;
            nlohmann::json innov_json;
            innov_json["timestamp_ns"] = output->timestamp_ns;

            if (output->is_robot()) {
                innov_json["state_type"] = "robot";
                const auto& state        = *output->robot_state();
                innov_json["position"]   = {
                    state.position.x(), state.position.y(), state.position.z()};
                innov_json["velocity"] = {
                    state.velocity.x(), state.velocity.y(), state.velocity.z()};
                innov_json["yaw"] = state.yaw;
            } else if (output->is_outpost()) {
                innov_json["state_type"] = "outpost";
                const auto& state        = *output->outpost_state();
                innov_json["position"]   = {state.position.x(), state.position.y()};
                innov_json["z"]          = {state.z[0], state.z[1], state.z[2]};
                innov_json["yaw"]        = state.yaw;
            }

            if (debug.innovation.size() > 0) {
                std::vector<double> innov_vec(
                    debug.innovation.data(), debug.innovation.data() + debug.innovation.size());
                innov_json["innovation"] = innov_vec;
            }
            innov_json["covariance_trace"] = debug.covariance_trace;

            detail::publish_json_message<FilterInnovationMessage>(*server, innov_json);
        });

    // =========================================================================
    // L3 DEBUG: Filter Covariance Trace
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l3_filter_covariance_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->filter_debug) {
                return;
            }

            const auto& debug = *output->filter_debug;
            nlohmann::json cov_json;
            cov_json["timestamp_ns"]    = output->timestamp_ns;
            cov_json["cov_trace"]       = debug.covariance_trace;
            cov_json["yaw_uncertainty"] = debug.yaw_uncertainty;

            detail::publish_json_message<FilterCovarianceMessage>(*server, cov_json);
        });

    // =========================================================================
    // L3 DEBUG: Filter Covariance 3D Scene
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l3_filter_covariance_scene_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in))
                return;

            auto output = tracker_in.read();
            if (!output || !output->is_tracking())
                return;

            std::vector<::foxglove::schemas::SceneEntity> entities;

            const auto pos_vec = L3::vis::get_tracker_position(*output);
            const Eigen::Vector3d pos(pos_vec.x, pos_vec.y, pos_vec.z);

            entities.push_back(
                viz::patterns::uncertainty_sphere<fast_tf::odom>(pos, 1.0, output->timestamp_ns));

            viz::publish_scene_if_nonempty<FilterCovarianceSceneMessage>(
                *server, std::move(entities));
        });

    // =========================================================================
    // L3 DEBUG: Viterbi Smaller Trellis
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l3_viterbi_trellis_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->association_debug) {
                return;
            }

            const auto& debug = *output->association_debug;
            nlohmann::json viterbi_json;
            viterbi_json["timestamp_ns"] = output->timestamp_ns;
            viterbi_json["n_phases"]     = debug.phase_probs.size();
            viterbi_json["phase_probs"]  = debug.phase_probs;
            viterbi_json["map_phase"]    = debug.map_phase;
            viterbi_json["is_ambiguous"] = debug.is_ambiguous;

            detail::publish_json_message<ViterbiTrellisMessage>(*server, viterbi_json);
        });

    // =========================================================================
    // L3 DEBUG: State Machine Status
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l3_state_machine_pub",
        [](talos::spmc<::fcs::L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, tracker_in)) {
                return;
            }

            auto output = tracker_in.read();
            if (!output || !output->state_machine_debug) {
                return;
            }

            const auto& debug = *output->state_machine_debug;
            nlohmann::json sm_json;
            sm_json["timestamp_ns"]        = output->timestamp_ns;
            sm_json["status"]              = magic_enum::enum_name(output->status);
            sm_json["is_tracking"]         = output->is_tracking();
            sm_json["target_name"]         = magic_enum::enum_name(output->target_name);
            sm_json["time_in_state"]       = debug.time_in_state;
            sm_json["detecting_count"]     = debug.detecting_count;
            sm_json["lost_time"]           = debug.lost_time;
            sm_json["consecutive_updates"] = debug.consecutive_updates;

            detail::publish_json_message<StateMachineMessage>(*server, sm_json);
        });

    // =========================================================================
    // L4 DEBUG: Full MPC Prediction Horizon
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l4_mpc_horizon_pub",
        [](talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return;
            }

            nlohmann::json horizon_json;
            horizon_json["timestamp_ns"] = plan->timestamp_ns;
            horizon_json["mpc_dt"]       = plan->mpc_dt;
            horizon_json["current_idx"]  = plan->mpc_current_idx;
            horizon_json["horizon"]      = plan->ref_yaw.size();
            horizon_json["time"]         = nlohmann::json::array();

            for (size_t k = 0; k < plan->ref_yaw.size(); ++k) {
                const double t = (static_cast<int>(k) - plan->mpc_current_idx) * plan->mpc_dt;
                horizon_json["time"].push_back(t);
            }
            horizon_json["ref_yaw"]       = plan->ref_yaw;
            horizon_json["ref_pitch"]     = plan->ref_pitch;
            horizon_json["opt_yaw"]       = plan->opt_yaw;
            horizon_json["opt_pitch"]     = plan->opt_pitch;
            horizon_json["ref_yaw_vel"]   = plan->ref_yaw_vel;
            horizon_json["ref_pitch_vel"] = plan->ref_pitch_vel;
            horizon_json["opt_yaw_vel"]   = plan->opt_yaw_vel;
            horizon_json["opt_pitch_vel"] = plan->opt_pitch_vel;

            detail::publish_json_message<MpcHorizonMessage>(*server, horizon_json);
        });

    // =========================================================================
    // L4 DEBUG: MPC Cost Evolution
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l4_mpc_cost_pub",
        [](talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return;
            }

            nlohmann::json cost_json;
            cost_json["timestamp_ns"]         = plan->timestamp_ns;
            cost_json["total_tracking_error"] = plan->tracking_error;
            cost_json["total_cost"]           = plan->total_cost;
            cost_json["mpc_iterations"]       = plan->mpc_iterations;
            cost_json["mpc_solve_time_us"]    = plan->mpc_solve_time_us;
            cost_json["mpc_converged"]        = plan->mpc_converged;
            cost_json["cost_yaw_pos"]         = plan->cost_yaw_pos;
            cost_json["cost_yaw_vel"]         = plan->cost_yaw_vel;
            cost_json["cost_pitch_pos"]       = plan->cost_pitch_pos;
            cost_json["cost_pitch_vel"]       = plan->cost_pitch_vel;

            detail::publish_json_message<MpcCostMessage>(*server, cost_json);
        });

    // =========================================================================
    // L4 DEBUG: MPC Constraint Margins
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_l4_mpc_constraint_pub",
        [](talos::spmc<::fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, plan_in)) {
                return;
            }

            auto plan = plan_in.read();
            if (!plan || plan->ref_yaw.empty()) {
                return;
            }

            nlohmann::json constraint_json;
            constraint_json["timestamp_ns"]     = plan->timestamp_ns;
            constraint_json["yaw_acc_margin"]   = plan->constraint_yaw_acc_margin;
            constraint_json["pitch_acc_margin"] = plan->constraint_pitch_acc_margin;

            detail::publish_json_message<MpcConstraintMessage>(*server, constraint_json);
        });

    // =========================================================================
    // ENERGY METER STATE PUBLISHER (L3)
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_debug_energy_meter_pub",
        [](talos::spmc<fcs::energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic> state_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server) {
            if (!detail::foxglove_ready(*server, state_in)) {
                return;
            }

            auto state = state_in.read();
            if (!state) {
                return;
            }

            nlohmann::json em_json;
            em_json["timestamp_ns"]   = state->timestamp_ns;
            em_json["tracking_valid"] = state->tracking_valid;
            em_json["r_center_odom"]  = {
                state->r_center_odom.x(), state->r_center_odom.y(), state->r_center_odom.z()};
            em_json["radius"]                  = state->radius;
            em_json["roll"]                    = state->roll;
            em_json["v_roll"]                  = state->v_roll;
            em_json["blade_id"]                = state->blade_id;
            em_json["predicted_position_odom"] = {
                state->predicted_position_odom.x(), state->predicted_position_odom.y(),
                state->predicted_position_odom.z()};
            em_json["predicted_future_ns"] = state->predicted_future_ns;
            em_json["is_big_rune"]         = state->is_big_rune;
            em_json["model_valid"]         = state->model_valid;
            em_json["fit_error"]           = state->fit_error;

            detail::publish_json_message<EnergyMeterMessage>(*server, em_json);
        });

    // =========================================================================
    // PERFORMANCE STATISTICS PUBLISHER (1 Hz, fixed_rate_silent)
    // =========================================================================

    if (scheduler_ptr) {
        app.template add_system<talos::fixed_rate_silent<100>>(
            "foxglove_debug_perf_stats_pub",
            [scheduler_ptr](talos::res<std::shared_ptr<FoxgloveServer>> server) {
                auto& server_ptr = *server; // 引用绑定
                if (!server_ptr->is_initialized()) {
                    return;
                }

                // Get scheduler performance statistics JSON
                std::string json_str = scheduler_ptr->get_stats_json();

                // Send to foxglove
                PerfStatsMessage msg;
                msg.data = json_to_bytes(json_str);
                server_ptr->enqueue_message(std::move(msg));
            });
    }
}

} // namespace fcs::foxglove::systems
