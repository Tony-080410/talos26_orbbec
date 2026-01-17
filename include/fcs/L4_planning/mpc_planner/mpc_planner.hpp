#pragma once

#include <algorithm>
#include <expected>
#include <memory>
#include <numbers>
#include <utility>

#include <Eigen/Core>

#include "tiny_api.hpp"

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/angle_utils.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L4_planning/mpc_planner/solver_handle.hpp"
#include "fcs/L4_planning/mpc_planner/trajectory_generator.hpp"
#include "fcs/L4_planning/mpc_planner/types.hpp"

namespace fcs::L4 {

/// MPC-based gimbal planner
///
/// Uses TinyMPC (ADMM-based QP solver) to compute optimal gimbal trajectories.
/// The gimbal is modeled as a double integrator, and the MPC solves for the
/// optimal acceleration sequence that tracks a reference trajectory generated
/// from target predictions.
///
/// Key features:
/// - Separate yaw and pitch MPC solvers
/// - Reference trajectory spans [-T/2, +T/2] relative to current time
/// - Fire decision based on trajectory tracking error at lookahead step
/// - Feedforward velocity and acceleration terms for improved tracking
class MpcGimbalPlanner {
public:
    explicit MpcGimbalPlanner(const MpcPlannerConfig& config);
    ~MpcGimbalPlanner() = default;

    // Non-copyable, movable (RAII handles manage resources automatically)
    MpcGimbalPlanner(const MpcGimbalPlanner&)                = delete;
    MpcGimbalPlanner& operator=(const MpcGimbalPlanner&)     = delete;
    MpcGimbalPlanner(MpcGimbalPlanner&&) noexcept            = default;
    MpcGimbalPlanner& operator=(MpcGimbalPlanner&&) noexcept = default;

    /// Update configuration
    void set_config(const MpcPlannerConfig& config) noexcept;

    /// Plan gimbal trajectory for robot target
    ///
    /// @param target Tracker output with robot state
    /// @param current_time_ns Current time in nanoseconds
    /// @param bullet_speed Bullet velocity in m/s
    /// @return MPC planner output or error string
    [[nodiscard]] std::expected<MpcPlannerOutput, std::string> plan(
        const L3::TrackerOutput& target, uint64_t current_time_ns, double bullet_speed) noexcept;

    /// Plan gimbal trajectory using a custom muzzle origin (odom frame).
    ///
    /// Use this when tracker positions are in odom/world coordinates but ballistic compensation
    /// expects target position relative to the muzzle origin.
    [[nodiscard]] std::expected<MpcPlannerOutput, std::string> plan(
        const L3::TrackerOutput& target, uint64_t current_time_ns, double bullet_speed,
        const Eigen::Vector3d& muzzle_pos_odom) noexcept;

private:
    using Trajectory = TrajectoryGenerator::Trajectory;

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

    /// Cleanup TinyMPC solvers
    void cleanup_solvers() noexcept;

    /// Warm start: shift previous solution forward by one step
    void warm_start_shift() noexcept;

    /// Solve both yaw and pitch MPC problems from a reference trajectory.
    void solve_from_reference(const Trajectory& traj) noexcept;

    /// Compute cost breakdown (position/velocity errors per horizon step)
    void compute_cost_breakdown(
        const Trajectory& traj, double yaw0, std::vector<double>& cost_yaw_pos,
        std::vector<double>& cost_yaw_vel, std::vector<double>& cost_pitch_pos,
        std::vector<double>& cost_pitch_vel, double& total_cost) const noexcept;

    /// Compute constraint margins (distance to acceleration limits)
    void compute_constraint_margins(
        std::vector<double>& yaw_acc_margin, std::vector<double>& pitch_acc_margin) const noexcept;

    /// Fill MPC output fields common to both robot/outpost planning.
    void fill_common_output(
        MpcPlannerOutput& output, const Trajectory& traj, double yaw0, uint64_t current_time_ns,
        bool clamp_pitch) noexcept;

    /// Copy MPC debug arrays into output for visualization.
    void fill_debug_output(
        MpcPlannerOutput& output, const Trajectory& traj, double yaw0) const noexcept;

    /// Generate and rotate ballistic trajectory points for visualization.
    void fill_visualization_trajectory(MpcPlannerOutput& output) const;

    /// Plan for robot target
    [[nodiscard]] std::expected<MpcPlannerOutput, std::string> plan_robot(
        const L3::RobotTargetState& target, uint64_t timestamp_ns, uint64_t current_time_ns,
        double bullet_speed, const Eigen::Vector3d& muzzle_pos_odom) noexcept;

    /// Plan for outpost target
    [[nodiscard]] std::expected<MpcPlannerOutput, std::string> plan_outpost(
        const L3::OutpostTargetState& target, uint64_t timestamp_ns, uint64_t current_time_ns,
        double bullet_speed, const Eigen::Vector3d& muzzle_pos_odom) noexcept;

    /// Select adaptive delay based on target angular velocity
    [[nodiscard]] inline double select_adaptive_delay(double v_yaw) const noexcept {
        return std::abs(v_yaw) > config_.decision_speed ? config_.high_speed_delay
                                                        : config_.low_speed_delay;
    }

    /// Calculate flying time and prediction time delta
    [[nodiscard]] inline std::pair<double, double> calculate_prediction_times(
        const Eigen::Vector3d& target_rel_pos, uint64_t timestamp_ns, uint64_t current_time_ns,
        double delay) const noexcept {
        const double flying_time = compensator_->get_flying_time(target_rel_pos);
        const double predict_dt =
            static_cast<double>(current_time_ns - timestamp_ns) * kSecondsPerNanosecond + delay
            + flying_time;
        return {flying_time, predict_dt};
    }

    /// Large initial distance for closest armor selection
    static constexpr double kArmorSelectionMinDist = 1e10;

    /// Infinity bound for MPC constraints (effectively no constraint)
    static constexpr double kInfinityBound = 1e17;

    /// Maximum distance for trajectory visualization (meters)
    static constexpr double kMaxVisualizationDistance = 15.0;

    /// Nanoseconds per second conversion factor
    static constexpr double kNanosecondsPerSecond = 1e9;

    /// Seconds per nanosecond conversion factor
    static constexpr double kSecondsPerNanosecond = 1e-9;

    MpcPlannerConfig config_;

    // TinyMPC solvers (RAII-managed, yaw and pitch are independent)
    SolverHandle yaw_solver_;
    SolverHandle pitch_solver_;

    // Trajectory generator
    TrajectoryGenerator trajectory_gen_;

    // Ballistic compensator
    std::unique_ptr<TrajectoryCompensator> compensator_;
};

// ============================================================================
// Implementation
// ============================================================================

inline MpcGimbalPlanner::MpcGimbalPlanner(const MpcPlannerConfig& config)
    : config_(config)
    , trajectory_gen_(config) {
    compensator_ = create_compensator(config.compensator_type);
    update_compensator_params();
    setup_solvers();
}

inline void MpcGimbalPlanner::set_config(const MpcPlannerConfig& config) noexcept {
    // Check if solver needs to be recreated (horizon changed)
    bool need_recreate = (config.horizon != config_.horizon || config.dt != config_.dt);

    config_ = config;
    trajectory_gen_.set_config(config);
    update_compensator_params();

    if (need_recreate) {
        // SolverHandle automatically cleans up when reset
        yaw_solver_.reset();
        pitch_solver_.reset();
        setup_solvers();
    }
}

inline void MpcGimbalPlanner::setup_solvers() noexcept {
    const int N     = config_.horizon;
    const double dt = config_.dt;

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
}

inline void MpcGimbalPlanner::warm_start_shift() noexcept {
    // Shift trajectory and control sequence forward by one step for warm start
    // This provides a better initial guess for the solver
    const int N = config_.horizon;

    auto shift_solver = [N](const SolverHandle& handle) noexcept {
        TinySolver* solver = handle.get();
        if (!solver || !solver->work)
            return;
        // Shift state trajectory: x[:, 0:N-1] = x[:, 1:N]
        solver->work->x.leftCols(N - 1) = solver->work->x.rightCols(N - 1).eval();
        // Shift control sequence: u[:, 0:N-2] = u[:, 1:N-1]
        if (N > 2) {
            solver->work->u.leftCols(N - 2) = solver->work->u.rightCols(N - 2).eval();
        }
        // Reset dual variables for better convergence with new reference
        solver->work->g.setZero();
        solver->work->y.setZero();
    };

    shift_solver(yaw_solver_);
    shift_solver(pitch_solver_);
}

inline void MpcGimbalPlanner::solve_from_reference(const Trajectory& traj) noexcept {
    const int N = config_.horizon;

    auto solve_axis = [&](const SolverHandle& handle, int row_offset) noexcept {
        TinySolver* solver = handle.get();
        Eigen::Vector2d x0 = traj.block(row_offset, 0, 2, 1);
        tiny_set_x0(solver, x0);
        solver->work->x    = traj.block(row_offset, 0, 2, N);
        solver->work->u    = Eigen::MatrixXd::Zero(kCtrlDim, N - 1);
        solver->work->Xref = traj.block(row_offset, 0, 2, N);
        tiny_solve(solver);
    };

    solve_axis(yaw_solver_, 0);   // Yaw: rows 0-1
    solve_axis(pitch_solver_, 2); // Pitch: rows 2-3
}

inline void MpcGimbalPlanner::fill_debug_output(
    MpcPlannerOutput& output, const Trajectory& traj, double yaw0) const noexcept {
    const int N            = config_.horizon;
    const int idx          = config_.half_horizon;
    output.mpc_dt          = config_.dt;
    output.mpc_current_idx = idx;

    // Resize all debug arrays
    output.ref_yaw.resize(N);
    output.ref_yaw_vel.resize(N);
    output.ref_pitch.resize(N);
    output.ref_pitch_vel.resize(N);
    output.opt_yaw.resize(N);
    output.opt_yaw_vel.resize(N);
    output.opt_pitch.resize(N);
    output.opt_pitch_vel.resize(N);
    output.ctrl_yaw_acc.resize(N - 1);
    output.ctrl_pitch_acc.resize(N - 1);

    for (int k = 0; k < N; ++k) {
        output.ref_yaw[k]       = normalize_rad(traj(0, k) + yaw0);
        output.ref_yaw_vel[k]   = traj(1, k);
        output.ref_pitch[k]     = traj(2, k);
        output.ref_pitch_vel[k] = traj(3, k);

        output.opt_yaw[k]       = normalize_rad(yaw_solver_.get()->work->x(0, k) + yaw0);
        output.opt_yaw_vel[k]   = yaw_solver_.get()->work->x(1, k);
        output.opt_pitch[k]     = pitch_solver_.get()->work->x(0, k);
        output.opt_pitch_vel[k] = pitch_solver_.get()->work->x(1, k);
    }

    for (int k = 0; k < N - 1; ++k) {
        output.ctrl_yaw_acc[k]   = yaw_solver_.get()->work->u(0, k);
        output.ctrl_pitch_acc[k] = pitch_solver_.get()->work->u(0, k);
    }
}

inline void MpcGimbalPlanner::fill_visualization_trajectory(MpcPlannerOutput& output) const {
    output.planned_trajectory =
        compensator_->get_trajectory(kMaxVisualizationDistance, output.pitch);
    for (auto& p : output.planned_trajectory) {
        const double x = p.first;
        const double z = p.second;
        p.first        = x * std::cos(output.pitch) + z * std::sin(output.pitch);
        p.second       = -x * std::sin(output.pitch) + z * std::cos(output.pitch);
    }
}

inline void MpcGimbalPlanner::fill_common_output(
    MpcPlannerOutput& output, const Trajectory& traj, double yaw0, uint64_t current_time_ns,
    bool clamp_pitch) noexcept {
    const int idx       = config_.half_horizon;
    output.timestamp_ns = current_time_ns;

    output.yaw     = normalize_rad(yaw_solver_.get()->work->x(0, idx) + yaw0);
    output.yaw_vel = yaw_solver_.get()->work->x(1, idx);
    output.yaw_acc = yaw_solver_.get()->work->u(0, idx);

    if (clamp_pitch) {
        output.pitch =
            std::clamp(pitch_solver_.get()->work->x(0, idx), config_.pitch_min, config_.pitch_max);
    } else {
        output.pitch = pitch_solver_.get()->work->x(0, idx);
    }
    output.pitch_vel = pitch_solver_.get()->work->x(1, idx);
    output.pitch_acc = pitch_solver_.get()->work->u(0, idx);

    const int fire_idx = idx + config_.fire_lookahead;
    if (fire_idx < config_.horizon) {
        const double yaw_error   = traj(0, fire_idx) - yaw_solver_.get()->work->x(0, fire_idx);
        const double pitch_error = traj(2, fire_idx) - pitch_solver_.get()->work->x(0, fire_idx);
        output.tracking_error    = std::hypot(yaw_error, pitch_error);
        output.fire_advice       = output.tracking_error < config_.fire_thresh;
    }

    fill_visualization_trajectory(output);
    fill_debug_output(output, traj, yaw0);

    // Populate extended MPC debug fields
    output.mpc_iterations = yaw_solver_.get()->solution->iter;
    output.mpc_converged  = yaw_solver_.get()->solution->solved != 0;

    compute_cost_breakdown(
        traj, yaw0, output.cost_yaw_pos, output.cost_yaw_vel, output.cost_pitch_pos,
        output.cost_pitch_vel, output.total_cost);

    compute_constraint_margins(
        output.constraint_yaw_acc_margin, output.constraint_pitch_acc_margin);
}

inline void MpcGimbalPlanner::compute_cost_breakdown(
    const Trajectory& traj, double yaw0, std::vector<double>& cost_yaw_pos,
    std::vector<double>& cost_yaw_vel, std::vector<double>& cost_pitch_pos,
    std::vector<double>& cost_pitch_vel, double& total_cost) const noexcept {
    const int N = config_.horizon;

    cost_yaw_pos.resize(N);
    cost_yaw_vel.resize(N);
    cost_pitch_pos.resize(N);
    cost_pitch_vel.resize(N);

    total_cost = 0.0;

    for (int k = 0; k < N; ++k) {
        // Yaw errors
        const double yaw_pos_err = traj(0, k) - yaw_solver_.get()->work->x(0, k);
        const double yaw_vel_err = traj(1, k) - yaw_solver_.get()->work->x(1, k);

        // Pitch errors
        const double pitch_pos_err = traj(2, k) - pitch_solver_.get()->work->x(0, k);
        const double pitch_vel_err = traj(3, k) - pitch_solver_.get()->work->x(1, k);

        // Cost at this step (quadratic)
        cost_yaw_pos[k]   = config_.q_yaw_position * yaw_pos_err * yaw_pos_err;
        cost_yaw_vel[k]   = config_.q_yaw_velocity * yaw_vel_err * yaw_vel_err;
        cost_pitch_pos[k] = config_.q_pitch_position * pitch_pos_err * pitch_pos_err;
        cost_pitch_vel[k] = config_.q_pitch_velocity * pitch_vel_err * pitch_vel_err;

        total_cost += cost_yaw_pos[k] + cost_yaw_vel[k] + cost_pitch_pos[k] + cost_pitch_vel[k];
    }
}

inline void MpcGimbalPlanner::compute_constraint_margins(
    std::vector<double>& yaw_acc_margin, std::vector<double>& pitch_acc_margin) const noexcept {
    const int N      = config_.horizon;
    const int n_ctrl = N - 1;

    yaw_acc_margin.resize(n_ctrl);
    pitch_acc_margin.resize(n_ctrl);

    for (int k = 0; k < n_ctrl; ++k) {
        // Distance from control to acceleration limit (positive = within bounds, negative =
        // violated)
        const double yaw_acc   = yaw_solver_.get()->work->u(0, k);
        const double pitch_acc = pitch_solver_.get()->work->u(0, k);

        yaw_acc_margin[k] = std::min(config_.max_yaw_acc - yaw_acc, yaw_acc + config_.max_yaw_acc);
        pitch_acc_margin[k] =
            std::min(config_.max_pitch_acc - pitch_acc, pitch_acc + config_.max_pitch_acc);
    }
}

inline std::expected<MpcPlannerOutput, std::string> MpcGimbalPlanner::plan(
    const L3::TrackerOutput& target, uint64_t current_time_ns, double bullet_speed) noexcept {
    return plan(target, current_time_ns, bullet_speed, Eigen::Vector3d::Zero());
}

inline std::expected<MpcPlannerOutput, std::string> MpcGimbalPlanner::plan(
    const L3::TrackerOutput& target, uint64_t current_time_ns, double bullet_speed,
    const Eigen::Vector3d& muzzle_pos_odom) noexcept {
    if (!target.is_tracking()) {
        return std::unexpected("Target not tracking");
    }

    // NOTE: warm_start disabled - causes tracking issues when target suddenly turns
    // Each frame now solves independently from reference trajectory start point,
    // matching sp_vision_25 behavior for better stability
    // warm_start_shift();

    if (target.is_robot()) {
        return plan_robot(
            *target.robot_state(), target.timestamp_ns, current_time_ns, bullet_speed,
            muzzle_pos_odom);
    } else if (target.is_outpost()) {
        return plan_outpost(
            *target.outpost_state(), target.timestamp_ns, current_time_ns, bullet_speed,
            muzzle_pos_odom);
    }

    return std::unexpected("Unknown target type");
}

inline std::expected<MpcPlannerOutput, std::string> MpcGimbalPlanner::plan_robot(
    const L3::RobotTargetState& target, uint64_t timestamp_ns, uint64_t current_time_ns,
    double bullet_speed, const Eigen::Vector3d& muzzle_pos_odom) noexcept {

    const double delay = select_adaptive_delay(target.v_yaw);

    // Generate reference trajectory
    double yaw0 = 0.0;
    auto traj   = trajectory_gen_.generate_robot(
        target, timestamp_ns, current_time_ns, delay, *compensator_, bullet_speed, muzzle_pos_odom,
        yaw0);

    solve_from_reference(traj);

    // Extract results at half_horizon (current time)
    const int idx = config_.half_horizon;
    MpcPlannerOutput output;
    fill_common_output(output, traj, yaw0, current_time_ns, true);

    // Distance calculation and predicted position
    // Use horizontal distance (x, y) for selecting closest armor, matching sp_vision_25
    const auto armor_poses = target.armor_poses();
    double min_dist        = kArmorSelectionMinDist;
    int best_idx           = 0;
    for (size_t i = 0; i < armor_poses.size(); ++i) {
        const Eigen::Vector3d rel = armor_poses[i].head<3>() - muzzle_pos_odom;
        const double dist         = std::hypot(rel.x(), rel.y()); // horizontal distance
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = static_cast<int>(i);
        }
    }
    const Eigen::Vector3d rel_now = armor_poses[best_idx].head<3>() - muzzle_pos_odom;
    output.distance               = rel_now.norm();               // 3D distance for output

    // Predicted position (in odom frame) - use the flying time to predict
    const auto [flying_time, predict_dt] =
        calculate_prediction_times(rel_now, timestamp_ns, current_time_ns, delay);

    // Predict target center position and the selected armor position
    Eigen::Vector3d pred_center = target.position + target.velocity * predict_dt;
    double pred_yaw             = target.yaw + target.v_yaw * predict_dt;

    // Use correct radius and height for selected armor (0,2 use radius0/z0; 1,3 use radius1/z1)
    const double r         = (best_idx % 2 == 0) ? target.radius0 : target.radius1;
    const double h         = target.z1 - target.position.z();
    const double armor_z   = pred_center.z() + ((best_idx % 2 == 0) ? 0.0 : h);
    const double armor_yaw = pred_yaw + best_idx * (2.0 * std::numbers::pi / target.armors_num);

    Eigen::Vector3d pred_armor_pos{
        pred_center.x() - r * std::cos(armor_yaw), pred_center.y() - r * std::sin(armor_yaw),
        armor_z};
    output.predicted_position = pred_armor_pos;
    output.predicted_future =
        current_time_ns + static_cast<uint64_t>(flying_time * kNanosecondsPerSecond);

    return output;
}

inline std::expected<MpcPlannerOutput, std::string> MpcGimbalPlanner::plan_outpost(
    const L3::OutpostTargetState& target, uint64_t timestamp_ns, uint64_t current_time_ns,
    double bullet_speed, const Eigen::Vector3d& muzzle_pos_odom) noexcept {

    const double delay = select_adaptive_delay(target.v_yaw);

    double yaw0 = 0.0;
    auto traj   = trajectory_gen_.generate_outpost(
        target, timestamp_ns, current_time_ns, delay, *compensator_, bullet_speed, muzzle_pos_odom,
        yaw0);

    solve_from_reference(traj);

    const int idx = config_.half_horizon;
    MpcPlannerOutput output;
    fill_common_output(output, traj, yaw0, current_time_ns, false);

    // Distance and predicted position for outpost
    Eigen::Vector3d armor_pos_3d{target.position.x(), target.position.y(), target.z[0]};
    const Eigen::Vector3d rel_now = armor_pos_3d - muzzle_pos_odom;
    output.distance               = rel_now.norm();

    const auto [flying_time, predict_dt] =
        calculate_prediction_times(rel_now, timestamp_ns, current_time_ns, delay);

    double pred_yaw    = target.yaw + target.v_yaw * predict_dt;
    constexpr double r = L3::OutpostTargetState::radius;
    Eigen::Vector3d pred_armor_pos{
        target.position.x() + target.velocity.x() * predict_dt - r * std::cos(pred_yaw),
        target.position.y() + target.velocity.y() * predict_dt - r * std::sin(pred_yaw),
        target.z[0]};
    output.predicted_position = pred_armor_pos;
    output.predicted_future =
        current_time_ns + static_cast<uint64_t>(flying_time * kNanosecondsPerSecond);

    return output;
}

} // namespace fcs::L4
