#include "L5_weapon/enhanced/trajectory_optimizer.hpp"

#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "core/math/normalize.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <utility>

// =============================================================================
// Optimized MPC trajectory optimizer for ARM Cortex-A55.
//
// Uses SmallMpcSolver (float, fixed-size Eigen) instead of vendored TinyMPC
// (double, dynamic Eigen). This eliminates:
//   1. Software-emulated double-precision FP (~20x overhead on A55)
//   2. Dynamic matrix allocation overhead per ADMM iteration
//   3. The 5-allocation RAII wrapper needed to paper over TinyMPC's memory leaks
//
// The ADMM algorithm is identical — same Riccati backward/forward pass,
// same slack/dual update, same termination check. Just without the overhead.
// =============================================================================

namespace fcs::L5 {
namespace {

using fcs::core::math::normalize_angle;

using ReferenceTrajectory = core::trajectory::ReferenceTrajectory;

struct FireGateStatus {
    int delay_steps{0};
    double delay_gate_threshold{0.0};
    double delay_gate_forward_error{0.0};
    double delay_gate_backward_error{0.0};
    bool delay_gate_forward_ok{false};
    bool delay_gate_backward_ok{false};
    bool phase_gate_ok{false};
    bool final_fire_gate_ok{false};
};

[[nodiscard]] Eigen::Vector3d
    spherical_to_cartesian(double distance, double yaw, double pitch) noexcept {
    return Eigen::Vector3d{
        distance * std::cos(pitch) * std::cos(yaw),
        distance * std::cos(pitch) * std::sin(yaw),
        distance * std::sin(pitch),
    };
}

[[nodiscard]] FireGateStatus compute_fire_gate(
    const L4::FireDecisionConfig& fire_decision, const L4::AimerConfig& aimer_config,
    const L4::GimbalPlan& plan, double current_yaw, double current_pitch, double mpc_dt,
    int center_index, const ReferenceTrajectory& trajectory, const SmallMpcSolver& yaw_solver,
    const SmallMpcSolver& pitch_solver) noexcept {
    FireGateStatus status;
    status.delay_gate_threshold = fire_decision.delay_gate_error_rad;

    const int horizon = static_cast<int>(trajectory.distances.size());
    if (horizon <= 0) {
        return status;
    }

    const int sample_index      = std::clamp(center_index, 0, horizon - 1);
    const auto reference_angles = [&](int index) noexcept {
        return DelayGateAngles{
            .yaw   = normalize_angle(trajectory.state(0, index) + trajectory.yaw_origin),
            .pitch = trajectory.state(2, index),
        };
    };
    const auto optimized_angles = [&](int index) noexcept {
        return DelayGateAngles{
            .yaw = normalize_angle(
                static_cast<double>(yaw_solver.state(0, index)) + trajectory.yaw_origin),
            .pitch = static_cast<double>(pitch_solver.state(0, index)),
        };
    };

    status.delay_steps =
        std::max(0, static_cast<int>(std::lround(aimer_config.delay / std::max(mpc_dt, 1e-6))));
    const DualDelayGateErrors delay_errors = compute_dual_delay_gate_errors(
        horizon, sample_index, status.delay_steps, reference_angles, optimized_angles);
    status.delay_gate_forward_error  = delay_errors.forward;
    status.delay_gate_backward_error = delay_errors.backward;
    status.delay_gate_forward_ok  = status.delay_gate_forward_error < status.delay_gate_threshold;
    status.delay_gate_backward_ok = status.delay_gate_backward_error < status.delay_gate_threshold;

    const double optimized_yaw = normalize_angle(
        static_cast<double>(yaw_solver.state(0, sample_index)) + trajectory.yaw_origin);
    // is_on_target compares against TF pitch convention; optimizer pitch follows command
    // convention.
    const double optimized_pitch = -static_cast<double>(pitch_solver.state(0, sample_index));
    const double distance        = std::max(trajectory.distances[sample_index], 1e-6);
    const double diff_yaw        = trajectory.diff_yaws[sample_index];

    status.phase_gate_ok = L4::is_on_target_phase_aware(
        fire_decision, plan.aim_phase, diff_yaw, aimer_config.yaw_limit_deg,
        aimer_config.min_enable_yaw_deg, current_yaw, current_pitch, optimized_yaw, optimized_pitch,
        distance);
    status.final_fire_gate_ok =
        status.delay_gate_forward_ok && status.delay_gate_backward_ok && status.phase_gate_ok;
    return status;
}

[[nodiscard]] WeaponVisualizationDebugData build_debug_data(
    int horizon_back, const ReferenceTrajectory& trajectory, const SmallMpcSolver& yaw_solver,
    const SmallMpcSolver& pitch_solver, const FireGateStatus& fire_gate) {
    WeaponVisualizationDebugData debug;
    const int horizon               = static_cast<int>(trajectory.distances.size());
    debug.center_index              = horizon_back;
    debug.lookahead_index           = debug.center_index;
    debug.delay_steps               = fire_gate.delay_steps;
    debug.delay_gate_threshold      = fire_gate.delay_gate_threshold;
    debug.delay_gate_forward_error  = fire_gate.delay_gate_forward_error;
    debug.delay_gate_backward_error = fire_gate.delay_gate_backward_error;
    debug.delay_gate_forward_ok     = fire_gate.delay_gate_forward_ok;
    debug.delay_gate_backward_ok    = fire_gate.delay_gate_backward_ok;
    debug.phase_gate_ok             = fire_gate.phase_gate_ok;
    debug.final_fire_gate_ok        = fire_gate.final_fire_gate_ok;
    debug.reference_plan.reserve(horizon);
    debug.optimized_plan.reserve(horizon);

    for (int k = 0; k < horizon; ++k) {
        debug.reference_plan.push_back(
            TrajectoryPlanSample{
                .yaw      = normalize_angle(trajectory.state(0, k) + trajectory.yaw_origin),
                .pitch    = trajectory.state(2, k),
                .distance = trajectory.distances[k],
                .tof      = trajectory.time_of_flights[k],
                .diff_yaw = trajectory.diff_yaws[k],
                .selected_armor_id       = trajectory.selected_armor_ids[k],
                .rough_selected_armor_id = trajectory.rough_selected_armor_ids[k],
                .world_position          = trajectory.positions[k],
            });

        const Eigen::Vector3d reconstructed_reference = spherical_to_cartesian(
            trajectory.distances[k], debug.reference_plan.back().yaw,
            debug.reference_plan.back().pitch);
        const Eigen::Vector3d world_bias = trajectory.positions[k] - reconstructed_reference;
        const double optimized_yaw =
            normalize_angle(static_cast<double>(yaw_solver.state(0, k)) + trajectory.yaw_origin);
        const double optimized_pitch = static_cast<double>(pitch_solver.state(0, k));
        const Eigen::Vector3d reconstructed_optimized =
            spherical_to_cartesian(trajectory.distances[k], optimized_yaw, optimized_pitch);
        debug.optimized_plan.push_back(
            TrajectoryPlanSample{
                .yaw                     = optimized_yaw,
                .pitch                   = optimized_pitch,
                .distance                = trajectory.distances[k],
                .tof                     = trajectory.time_of_flights[k],
                .diff_yaw                = trajectory.diff_yaws[k],
                .selected_armor_id       = trajectory.selected_armor_ids[k],
                .rough_selected_armor_id = trajectory.rough_selected_armor_ids[k],
                .world_position          = reconstructed_optimized + world_bias,
            });
    }

    return debug;
}

} // namespace

TinyMpcTrajectoryOptimizer::TinyMpcTrajectoryOptimizer(
    const WeaponControllerConfig& config,
    const L4::ReferenceTrajectoryConfig& trajectory_cfg) noexcept
    : config_(config)
    , trajectory_cfg_(trajectory_cfg) {
    if (config_.enabled) {
        initialize_solvers();
    }
}

WeaponCommand TinyMpcTrajectoryOptimizer::passthrough(
    const L4::GimbalPlan& plan, uint64_t command_timestamp_ns) const noexcept {
    WeaponCommand cmd;
    cmd.timestamp_ns      = command_timestamp_ns;
    cmd.plan_timestamp_ns = plan.timestamp_ns;
    cmd.plan_yaw          = plan.yaw;
    cmd.plan_pitch        = plan.pitch;
    cmd.plan_fire         = plan.fire_advice;
    cmd.yaw               = plan.yaw;
    cmd.pitch             = plan.pitch;
    cmd.ref_yaw           = plan.aim_yaw;
    cmd.ref_pitch         = plan.aim_pitch;
    cmd.yaw_vel           = plan.yaw_vel;
    cmd.pitch_vel         = plan.pitch_vel;
    cmd.yaw_accel         = plan.yaw_acc;
    cmd.pitch_accel       = plan.pitch_acc;
    cmd.fire              = plan.fire_advice;
    cmd.distance          = plan.distance;
    return cmd;
}

std::expected<WeaponCommand, std::string> TinyMpcTrajectoryOptimizer::optimize(
    const L4::GimbalPlan& plan, const L4::FireDecisionConfig& fire_decision,
    const L4::AimerConfig& aimer_config, const fast_tf::CoordinateSystem& tf_buffer,
    uint64_t command_timestamp_ns) noexcept {
    if (!config_.enabled) {
        return std::unexpected("mpc_weapon disabled");
    }
    if (!ready_) {
        return std::unexpected("mpc solvers are not ready");
    }
    if (plan.source != L4::GimbalPlanSource::Armor) {
        return std::unexpected("only armor planner seeds are supported");
    }
    if (!plan.reference_trajectory) {
        return std::unexpected("reference trajectory is missing");
    }

    // Reject stale reference trajectories to avoid solving on outdated data.
    const double reference_age_s =
        static_cast<double>(
            static_cast<int64_t>(command_timestamp_ns) - static_cast<int64_t>(plan.timestamp_ns))
        * 1e-9;
    if (reference_age_s > config_.reference_age_threshold_s) {
        return std::unexpected(
            "reference trajectory is stale (" + std::to_string(reference_age_s) + "s > "
            + std::to_string(config_.reference_age_threshold_s) + "s)");
    }

    const ReferenceTrajectory& reference = *plan.reference_trajectory;

    auto transforms = L4::lookup_gimbal_muzzle_transforms(tf_buffer, command_timestamp_ns);
    if (!transforms) {
        return std::unexpected(transforms.error());
    }

    const auto current_gimbal  = transforms->gimbal.euler_rot();
    const double current_yaw   = normalize_angle(current_gimbal.yaw);
    const double current_pitch = current_gimbal.pitch;

    // --- seed yaw solver ---
    yaw_solver_.set_x0(reference.state(0, 0), reference.state(1, 0));
    const int N = reference.state.cols();
    for (int i = 0; i < N; ++i) {
        yaw_solver_.set_ref_col(
            i, static_cast<float>(reference.state(0, i)),
            static_cast<float>(reference.state(1, i)));
    }
    const bool yaw_ok = yaw_solver_.solve();
    if (!yaw_ok) {
        return std::unexpected("yaw MPC did not converge");
    }

    // --- seed pitch solver ---
    pitch_solver_.set_x0(reference.state(2, 0), reference.state(3, 0));
    for (int i = 0; i < N; ++i) {
        pitch_solver_.set_ref_col(
            i, static_cast<float>(reference.state(2, i)),
            static_cast<float>(reference.state(3, i)));
    }
    const bool pitch_ok = pitch_solver_.solve();
    if (!pitch_ok) {
        return std::unexpected("pitch MPC did not converge");
    }

    const int center_index  = trajectory_cfg_.horizon_back;
    const int control_index = std::min(center_index, reference.horizon() - 2);

    WeaponCommand cmd;
    cmd.timestamp_ns      = command_timestamp_ns;
    cmd.plan_timestamp_ns = plan.timestamp_ns;
    cmd.plan_yaw          = plan.yaw;
    cmd.plan_pitch        = plan.pitch;
    cmd.plan_fire         = plan.fire_advice;
    cmd.ref_yaw   = normalize_angle(reference.state(0, center_index) + reference.yaw_origin);
    cmd.ref_pitch = reference.state(2, center_index);
    cmd.yaw       = normalize_angle(
        static_cast<double>(yaw_solver_.state(0, center_index)) + reference.yaw_origin);
    cmd.pitch = std::clamp(
        static_cast<double>(pitch_solver_.state(0, center_index)), config_.pitch_min,
        config_.pitch_max);
    cmd.yaw_vel                    = static_cast<double>(yaw_solver_.state(1, center_index));
    cmd.pitch_vel                  = static_cast<double>(pitch_solver_.state(1, center_index));
    cmd.yaw_accel                  = static_cast<double>(yaw_solver_.input(control_index));
    cmd.pitch_accel                = static_cast<double>(pitch_solver_.input(control_index));
    cmd.distance                   = reference.distances[center_index];
    cmd.tof                        = reference.time_of_flights[center_index];
    const FireGateStatus fire_gate = compute_fire_gate(
        fire_decision, aimer_config, plan, current_yaw, current_pitch, trajectory_cfg_.dt,
        center_index, reference, yaw_solver_, pitch_solver_);
    cmd.fire = plan.fire_advice && fire_gate.final_fire_gate_ok;
    if (config_.enable_debug) {
        cmd.viz_debug = build_debug_data(
            trajectory_cfg_.horizon_back, reference, yaw_solver_, pitch_solver_, fire_gate);
    }

    const auto finite = [](double value) noexcept { return std::isfinite(value); };
    if (!finite(cmd.yaw) || !finite(cmd.pitch) || !finite(cmd.yaw_vel) || !finite(cmd.pitch_vel)
        || !finite(cmd.yaw_accel) || !finite(cmd.pitch_accel) || !finite(cmd.distance)
        || !finite(cmd.tof)) {
        return std::unexpected("MPC output contains non-finite values");
    }

    return cmd;
}

void TinyMpcTrajectoryOptimizer::initialize_solvers() noexcept {
    const int horizon = trajectory_cfg_.horizon_ahead + trajectory_cfg_.horizon_back + 1;
    const float dt    = static_cast<float>(trajectory_cfg_.dt);
    const float rho   = static_cast<float>(config_.mpc.rho);

    auto make_axis = [&](const MpcAxisConfig& axis,
                         bool pitch) -> std::expected<SmallMpcSolver, std::string> {
        SmallMpcSolver::AxisConfig cfg;
        cfg.q_pos              = static_cast<float>(axis.q_pos);
        cfg.q_vel              = static_cast<float>(axis.q_vel);
        cfg.r                  = static_cast<float>(axis.r);
        cfg.max_acc            = static_cast<float>(axis.max_acc);
        cfg.enable_state_bound = pitch;

        if (pitch) {
            cfg.state_min = static_cast<float>(config_.pitch_min);
            cfg.state_max = static_cast<float>(config_.pitch_max);
        }

        auto solver = SmallMpcSolver::create(dt, horizon, rho, cfg);
        if (!solver) {
            return std::unexpected("Failed to create solver: " + solver.error());
        }

        solver->set_settings(static_cast<float>(config_.mpc.abs_tol), config_.mpc.max_iterations);
        return solver;
    };

    auto yaw_result = make_axis(config_.mpc.yaw, false);
    if (!yaw_result) {
        ready_ = false;
        return;
    }
    yaw_solver_ = std::move(*yaw_result);

    auto pitch_result = make_axis(config_.mpc.pitch, true);
    if (!pitch_result) {
        yaw_solver_ = SmallMpcSolver{};
        ready_      = false;
        return;
    }
    pitch_solver_ = std::move(*pitch_result);

    ready_ = true;
}

} // namespace fcs::L5
