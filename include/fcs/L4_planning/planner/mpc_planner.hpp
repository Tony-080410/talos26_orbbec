#pragma once

#include <algorithm>
#include <expected>
#include <memory>
#include <utility>

#include <Eigen/Core>

#include "math/fast_tf/frame.hpp"
#include "tiny_api.hpp"

#include "fcs/L4_planning/aimer/types.hpp"
#include "fcs/L4_planning/angle_utils.hpp"
#include "fcs/L4_planning/common/conversion.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L4_planning/mpc_planner/solver_handle.hpp"
#include "fcs/L4_planning/planning/fire_decision.hpp"
#include <spdlog/spdlog.h>

// Forward declaration to avoid including toml++ in config headers.
namespace toml::inline v3 {
class table;
}

namespace fcs::L4 {

/// MPC-based gimbal planner (refactored)
///
/// This planner consumes AimerOutput (pure prediction) and performs trajectory
/// optimization using TinyMPC. The gimbal is modeled as a double integrator.
///
/// Key features:
/// - Separate yaw and pitch MPC solvers
/// - Reference trajectory generated from AimerOutput
/// - Fire decision based on trajectory tracking error at lookahead step
/// - Feedforward velocity and acceleration terms for improved tracking
class MpcPlanner {
public:
    explicit MpcPlanner(const MpcPlannerConfig& config);
    ~MpcPlanner() = default;

    // Non-copyable, movable (RAII handles manage resources automatically)
    MpcPlanner(const MpcPlanner&)                = delete;
    MpcPlanner& operator=(const MpcPlanner&)     = delete;
    MpcPlanner(MpcPlanner&&) noexcept            = default;
    MpcPlanner& operator=(MpcPlanner&&) noexcept = default;

    /// Set configuration (lightweight: only stores config, does not rebuild resources)
    void set_config(const MpcPlannerConfig& config) noexcept { config_ = config; }

    /// Sync configuration (heavyweight: applies config and rebuilds resources if needed)
    void sync_config() noexcept;

    /// Plan gimbal trajectory from target prediction
    ///
    /// @param prediction Target prediction from Aimer
    /// @param current_time_ns Current time (ns)
    /// @param bullet_speed Bullet velocity (m/s)
    /// @param gimbal_tf Current gimbal transform
    /// @return Optimized gimbal plan
    [[nodiscard]] std::expected<GimbalPlan, std::string> plan(
        const TargetPrediction& prediction, uint64_t current_time_ns, double bullet_speed,
        const fast_tf::FuckOdom& gimbal_tf) noexcept;

private:
    using TrajectoryMat       = Eigen::Matrix<double, 4, Eigen::Dynamic>;
    using ReferenceTrajectory = std::pair<std::pair<double, Eigen::Vector3d>, TrajectoryMat>;

    // MPC constants (double integrator: [angle, velocity] state, [acceleration] control)
    static constexpr int kStateDim = 2; ///< [angle, angular_velocity]
    static constexpr int kCtrlDim  = 1; ///< [angular_acceleration]

    /// MPC axis configuration (cost weights and constraints)
    struct AxisConfig {
        double q_pos;   ///< Position tracking weight
        double q_vel;   ///< Velocity tracking weight
        double r;       ///< Control effort weight
        double max_acc; ///< Maximum acceleration constraint
    };

    /// Update compensator parameters from config
    void update_compensator_params() noexcept {
        compensator_->velocity        = config_.trajectory.bullet_speed;
        compensator_->gravity         = config_.trajectory.gravity;
        compensator_->iteration_times = config_.trajectory.iteration_times;
        compensator_->cd              = config_.trajectory.cd;
        compensator_->rho             = config_.trajectory.rho;
        compensator_->diameter        = config_.trajectory.diameter;
        compensator_->mass            = config_.trajectory.mass;
        compensator_->rk4_dt          = config_.trajectory.rk4_dt;
        compensator_->rk4_min_steps   = config_.trajectory.rk4_min_steps;
        compensator_->rk4_max_steps   = config_.trajectory.rk4_max_steps;
        compensator_->on_params_updated();
    }

    /// Initialize TinyMPC solvers
    void setup_solvers() noexcept;

    /// Solve both yaw and pitch MPC problems from a reference trajectory.
    void solve_from_reference(const ReferenceTrajectory& traj) noexcept;

    /// Compute cost breakdown (position/velocity errors per horizon step)
    void compute_cost_breakdown(
        const ReferenceTrajectory& traj, double yaw0, std::vector<double>& cost_yaw_pos,
        std::vector<double>& cost_yaw_vel, std::vector<double>& cost_pitch_pos,
        std::vector<double>& cost_pitch_vel, double& total_cost) const noexcept;

    /// Compute constraint margins (distance to acceleration limits)
    void compute_constraint_margins(
        std::vector<double>& yaw_acc_margin, std::vector<double>& pitch_acc_margin) const noexcept;

    /// Fill MPC output fields common to all planning
    void fill_common_output(
        const fast_tf::FuckOdom& gimbal_tf, GimbalPlan& output, const ReferenceTrajectory& traj,
        double yaw0, uint64_t current_time_ns) noexcept;

    /// Copy MPC debug arrays into output for visualization.
    void fill_debug_output(
        GimbalPlan& output, const ReferenceTrajectory& traj, double yaw0) const noexcept;

    /// Generate and rotate ballistic trajectory points for visualization.
    void fill_visualization_trajectory(GimbalPlan& output) const;

    /// Generate reference trajectory from target prediction
    [[nodiscard]] ReferenceTrajectory generate_reference_trajectory(
        const TargetPrediction& prediction, double bullet_speed) const noexcept;

    /// Infinity bound for MPC constraints (effectively no constraint)
    static constexpr double kInfinityBound = 1e17;

    /// Maximum distance for trajectory visualization (meters)
    static constexpr double kMaxVisualizationDistance = 15.0;

    /// Nanoseconds per second conversion factor
    static constexpr double kNanosecondsPerSecond = 1e9;

    MpcPlannerConfig config_;

    // TinyMPC solvers (RAII-managed, yaw and pitch are independent)
    SolverHandle yaw_solver_;
    SolverHandle pitch_solver_;

    // Ballistic compensator
    std::unique_ptr<TrajectoryCompensator> compensator_;

    // Track current solver configuration for change detection
    int current_horizon_{0};
    double current_dt_{0.0};
};

// ============================================================================
// Implementation (header-only for consistency with legacy code)
// ============================================================================

inline MpcPlanner::MpcPlanner(const MpcPlannerConfig& config)
    : config_(config) {
    compensator_ = create_compensator(config.compensator_type);
    update_compensator_params();
    setup_solvers();
}

inline void MpcPlanner::sync_config() noexcept {
    update_compensator_params();

    // Check if solver needs to be recreated (horizon or dt changed)
    const int new_horizon = config_.ponder_ahead + config_.ponder_back;
    const double new_dt   = config_.ponder_dt;

    const bool need_recreate = (new_horizon != current_horizon_) || (new_dt != current_dt_);

    if (need_recreate) {
        SPDLOG_INFO(
            "MpcPlanner: rebuilding solvers (horizon: {}->{}, dt: {}->{})", current_horizon_,
            new_horizon, current_dt_, new_dt);
        yaw_solver_.reset();
        pitch_solver_.reset();
        setup_solvers();
    }
}

inline void MpcPlanner::setup_solvers() noexcept {
    const auto N    = config_.ponder_ahead + config_.ponder_back;
    const double dt = config_.ponder_dt;

    // Double integrator dynamics: x_{k+1} = A*x_k + B*u_k
    Eigen::Matrix2d A;
    A << 1, dt, 0, 1;

    Eigen::Vector2d B;
    B << 0, dt;

    Eigen::Vector2d f = Eigen::Vector2d::Zero();

    auto setup_axis = [&](SolverHandle& handle, const AxisConfig& cfg) noexcept {
        Eigen::Vector2d Q_diag{cfg.q_pos, cfg.q_vel};
        Eigen::Matrix<double, 1, 1> R_diag{cfg.r};

        TinySolver* solver = nullptr;
        tiny_setup(
            &solver, A, B, f, Q_diag.asDiagonal(), R_diag.asDiagonal(), config_.rho, kStateDim,
            kCtrlDim, N, 0);

        Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(kStateDim, N, -kInfinityBound);
        Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(kStateDim, N, kInfinityBound);
        Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(kCtrlDim, N - 1, -cfg.max_acc);
        Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(kCtrlDim, N - 1, cfg.max_acc);
        tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

        solver->settings->max_iter    = config_.max_iterations;
        solver->settings->abs_pri_tol = config_.abs_tol;
        solver->settings->abs_dua_tol = config_.abs_tol;

        // Transfer ownership to RAII handle
        handle.reset(solver);
    };

    setup_axis(
        yaw_solver_,
        {config_.q_yaw_position, config_.q_yaw_velocity, config_.r_yaw, config_.max_yaw_acc});
    setup_axis(
        pitch_solver_, {config_.q_pitch_position, config_.q_pitch_velocity, config_.r_pitch,
                        config_.max_pitch_acc});

    // Update tracked configuration
    current_horizon_ = config_.ponder_ahead + config_.ponder_back;
    current_dt_      = config_.ponder_dt;
}

inline MpcPlanner::ReferenceTrajectory MpcPlanner::generate_reference_trajectory(
    const TargetPrediction& prediction, double bullet_speed) const noexcept {

    const auto N = config_.ponder_ahead + config_.ponder_back;
    TrajectoryMat traj(4, N);

    // Start from the aim point (center of horizon)
    const double aim_yaw   = prediction.aim_yaw;
    const double aim_pitch = prediction.aim_pitch;

    // Generate trajectory by extrapolating from aim point
    // For simplicity, use a stationary reference at the aim point
    for (int i = 0; i < N; ++i) {
        traj(0, i) = aim_yaw;   // yaw
        traj(1, i) = 0.0;       // yaw_vel
        traj(2, i) = aim_pitch; // pitch
        traj(3, i) = 0.0;       // pitch_vel
    }

    // Return with predicted position and total prediction time
    const double total_time =
        prediction.flying_time
        + static_cast<double>(prediction.predicted_future_ns - prediction.timestamp_ns) * 1e-9;
    return std::make_pair(std::make_pair(total_time, prediction.predicted_position), traj);
}

inline void MpcPlanner::solve_from_reference(const ReferenceTrajectory& traj) noexcept {
    const auto N = config_.ponder_ahead + config_.ponder_back;

    auto solve_axis = [&](const SolverHandle& handle, int row_offset) noexcept {
        TinySolver* solver = handle.get();
        Eigen::Vector2d x0 = traj.second.block(row_offset, 0, 2, 1);
        tiny_set_x0(solver, x0);
        solver->work->x    = traj.second.block(row_offset, 0, 2, N);
        solver->work->u    = Eigen::MatrixXd::Zero(kCtrlDim, N - 1);
        solver->work->Xref = traj.second.block(row_offset, 0, 2, N);
        tiny_solve(solver);
    };

    solve_axis(yaw_solver_, 0);   // Yaw: rows 0-1
    solve_axis(pitch_solver_, 2); // Pitch: rows 2-3
}

inline void MpcPlanner::fill_debug_output(
    GimbalPlan& output, const ReferenceTrajectory& traj, double yaw0) const noexcept {
    const auto N  = config_.ponder_ahead + config_.ponder_back;
    const int idx = config_.ponder_back;

    // Create MpcDebugData
    MpcDebugData debug;
    debug.mpc_dt          = config_.ponder_dt;
    debug.mpc_current_idx = idx;

    // Resize all debug arrays
    debug.ref_yaw.resize(N);
    debug.ref_yaw_vel.resize(N);
    debug.ref_pitch.resize(N);
    debug.ref_pitch_vel.resize(N);
    debug.opt_yaw.resize(N);
    debug.opt_yaw_vel.resize(N);
    debug.opt_pitch.resize(N);
    debug.opt_pitch_vel.resize(N);
    debug.ctrl_yaw_acc.resize(N - 1);
    debug.ctrl_pitch_acc.resize(N - 1);

    for (int k = 0; k < N; ++k) {
        debug.ref_yaw[k]       = normalize_rad(traj.second(0, k) + yaw0);
        debug.ref_yaw_vel[k]   = traj.second(1, k);
        debug.ref_pitch[k]     = traj.second(2, k);
        debug.ref_pitch_vel[k] = traj.second(3, k);

        debug.opt_yaw[k]       = normalize_rad(yaw_solver_.get()->work->x(0, k) + yaw0);
        debug.opt_yaw_vel[k]   = yaw_solver_.get()->work->x(1, k);
        debug.opt_pitch[k]     = pitch_solver_.get()->work->x(0, k);
        debug.opt_pitch_vel[k] = pitch_solver_.get()->work->x(1, k);
    }

    for (int k = 0; k < N - 1; ++k) {
        debug.ctrl_yaw_acc[k]   = yaw_solver_.get()->work->u(0, k);
        debug.ctrl_pitch_acc[k] = pitch_solver_.get()->work->u(0, k);
    }

    output.mpc_debug = debug;
}

inline void MpcPlanner::fill_visualization_trajectory(GimbalPlan& output) const {
    VisualizationDebugData viz_debug;
    // Note: Don't set viz_debug.predicted_position because:
    // 1. It has timestamp = predicted_future_ns (future time when bullet hits target)
    // 2. TF interpolation uses current timestamp_ns for visualization
    // 3. For moving targets, time mismatch causes visual misalignment
    // 4. The trajectory itself is sufficient for visualization
    // viz_debug.predicted_position left as std::nullopt (not set)
    viz_debug.predicted_future = output.predicted_future;
    viz_debug.trajectory = compensator_->get_trajectory(kMaxVisualizationDistance, output.pitch);
    for (auto& p : viz_debug.trajectory) {
        const double x = p.first;
        const double z = p.second;
        p.first        = x * std::cos(output.pitch) + z * std::sin(output.pitch);
        p.second       = -x * std::sin(output.pitch) + z * std::cos(output.pitch);
    }
    output.viz_debug = viz_debug;
}

inline void MpcPlanner::fill_common_output(
    const fast_tf::FuckOdom& gimbal_tf, GimbalPlan& output, const ReferenceTrajectory& traj,
    double yaw0, uint64_t current_time_ns) noexcept {
    const int idx       = config_.ponder_back;
    output.timestamp_ns = current_time_ns;

    output.yaw     = normalize_rad(yaw_solver_.get()->work->x(0, idx) + yaw0);
    output.yaw_vel = yaw_solver_.get()->work->x(1, idx);
    output.yaw_acc = yaw_solver_.get()->work->u(0, idx);

    output.pitch =
        std::clamp(pitch_solver_.get()->work->x(0, idx), config_.pitch_min, config_.pitch_max);
    output.pitch_vel = pitch_solver_.get()->work->x(1, idx);
    output.pitch_acc = pitch_solver_.get()->work->u(0, idx);

    output.predicted_position = traj.first.second;
    output.distance           = output.predicted_position->norm();
    output.predicted_future =
        static_cast<uint64_t>(traj.first.first * kNanosecondsPerSecond) + current_time_ns;

    // Tracking error at lookahead step
    const int fire_idx = idx + config_.fire_lookahead;
    if (fire_idx < (config_.ponder_ahead + config_.ponder_back)) {
        const double yaw_error =
            normalize_rad(traj.second(0, fire_idx) - yaw_solver_.get()->work->x(0, fire_idx));
        const double pitch_error =
            normalize_rad(traj.second(2, fire_idx) - pitch_solver_.get()->work->x(0, fire_idx));
        output.tracking_error = std::hypot(yaw_error, pitch_error);
    }

    // Fire decision using FireDecision
    auto [roll, pitch, yaw] = gimbal_tf->euler_rot().rpy();
    (void)roll;
    output.fire_advice = is_on_target(
        {config_.fire_thresh, config_.shooting_range_h, config_.shooting_range_w}, yaw, pitch,
        output.yaw, -output.pitch, output.distance);

    fill_visualization_trajectory(output);
    fill_debug_output(output, traj, yaw0);

    // Populate extended MPC debug fields
    if (output.mpc_debug) {
        output.mpc_debug->mpc_iterations    = yaw_solver_.get()->solution->iter;
        output.mpc_debug->mpc_converged     = yaw_solver_.get()->solution->solved != 0;
        output.mpc_debug->mpc_solve_time_us = 0.0; // TODO: add timing

        compute_cost_breakdown(
            traj, yaw0, output.mpc_debug->cost_yaw_pos, output.mpc_debug->cost_yaw_vel,
            output.mpc_debug->cost_pitch_pos, output.mpc_debug->cost_pitch_vel,
            output.mpc_debug->total_cost);

        compute_constraint_margins(
            output.mpc_debug->constraint_yaw_acc_margin,
            output.mpc_debug->constraint_pitch_acc_margin);
    }
}

inline void MpcPlanner::compute_cost_breakdown(
    const ReferenceTrajectory& traj, [[maybe_unused]] double yaw0,
    std::vector<double>& cost_yaw_pos, std::vector<double>& cost_yaw_vel,
    std::vector<double>& cost_pitch_pos, std::vector<double>& cost_pitch_vel,
    double& total_cost) const noexcept {
    const auto N = config_.ponder_ahead + config_.ponder_back;

    cost_yaw_pos.resize(N);
    cost_yaw_vel.resize(N);
    cost_pitch_pos.resize(N);
    cost_pitch_vel.resize(N);

    total_cost = 0.0;

    for (int k = 0; k < N; ++k) {
        const double yaw_pos_err =
            normalize_rad(traj.second(0, k) - yaw_solver_.get()->work->x(0, k));
        const double yaw_vel_err = traj.second(1, k) - yaw_solver_.get()->work->x(1, k);

        const double pitch_pos_err =
            normalize_rad(traj.second(2, k) - pitch_solver_.get()->work->x(0, k));
        const double pitch_vel_err = traj.second(3, k) - pitch_solver_.get()->work->x(1, k);

        cost_yaw_pos[k]   = config_.q_yaw_position * yaw_pos_err * yaw_pos_err;
        cost_yaw_vel[k]   = config_.q_yaw_velocity * yaw_vel_err * yaw_vel_err;
        cost_pitch_pos[k] = config_.q_pitch_position * pitch_pos_err * pitch_pos_err;
        cost_pitch_vel[k] = config_.q_pitch_velocity * pitch_vel_err * pitch_vel_err;

        total_cost += cost_yaw_pos[k] + cost_yaw_vel[k] + cost_pitch_pos[k] + cost_pitch_vel[k];
    }
}

inline void MpcPlanner::compute_constraint_margins(
    std::vector<double>& yaw_acc_margin, std::vector<double>& pitch_acc_margin) const noexcept {
    const auto N     = config_.ponder_ahead + config_.ponder_back;
    const int n_ctrl = N - 1;

    yaw_acc_margin.resize(n_ctrl);
    pitch_acc_margin.resize(n_ctrl);

    for (int k = 0; k < n_ctrl; ++k) {
        const double yaw_acc   = yaw_solver_.get()->work->u(0, k);
        const double pitch_acc = pitch_solver_.get()->work->u(0, k);

        yaw_acc_margin[k] = std::min(config_.max_yaw_acc - yaw_acc, yaw_acc + config_.max_yaw_acc);
        pitch_acc_margin[k] =
            std::min(config_.max_pitch_acc - pitch_acc, pitch_acc + config_.max_pitch_acc);
    }
}

inline std::expected<GimbalPlan, std::string> MpcPlanner::plan(
    const TargetPrediction& prediction, uint64_t current_time_ns, double bullet_speed,
    const fast_tf::FuckOdom& gimbal_tf) noexcept {

    // Generate reference trajectory from prediction
    auto traj         = generate_reference_trajectory(prediction, bullet_speed);
    const double yaw0 = prediction.aim_yaw; // Use aim yaw as reference offset

    solve_from_reference(traj);

    GimbalPlan output;

    // Copy common fields from prediction to output
    copy_prediction_to_gimbal_plan(prediction, output);

    fill_common_output(gimbal_tf, output, traj, yaw0, current_time_ns);
    return output;
}

} // namespace fcs::L4
