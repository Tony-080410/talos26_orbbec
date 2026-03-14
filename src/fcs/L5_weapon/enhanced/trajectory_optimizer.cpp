#include "L5_weapon/enhanced/trajectory_optimizer.hpp"

#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "tiny_api.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace fcs::L5 {
namespace {

using GimbalTransform = L4::Aimer::GimbalTransform;
using MuzzleTransform = L4::Aimer::MuzzleTransform;
using ReferenceMatrix = Eigen::Matrix<double, 4, Eigen::Dynamic>;

constexpr double kUnlimitedBound = 1e17;

struct ReferenceTrajectory {
    ReferenceMatrix state;
    std::vector<double> distances;
    std::vector<double> time_of_flights;
    std::vector<Eigen::Vector3d> positions;
    double yaw_origin{0.0};
};

struct ReferenceSample {
    double yaw{0.0};
    double pitch{0.0};
    double distance{0.0};
    double time_of_flight{0.0};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
};

void destroy_tiny_solver(TinySolver* solver) noexcept {
    if (solver == nullptr) {
        return;
    }

    delete solver->solution;
    delete solver->settings;
    delete solver->cache;
    delete solver->work;
    delete solver;
}

[[nodiscard]] double normalize_angle(double angle) noexcept {
    return std::remainder(angle, 2.0 * std::numbers::pi);
}

[[nodiscard]] Eigen::Vector3d
    spherical_to_cartesian(double distance, double yaw, double pitch) noexcept {
    return Eigen::Vector3d{
        distance * std::cos(pitch) * std::cos(yaw),
        distance * std::cos(pitch) * std::sin(yaw),
        distance * std::sin(pitch),
    };
}

[[nodiscard]] double
    finite_difference(const std::vector<double>& samples, int index, double dt) noexcept {
    const int size = static_cast<int>(samples.size());
    if (size <= 1) {
        return 0.0;
    }
    if (index <= 0) {
        return (samples[1] - samples[0]) / dt;
    }
    if (index >= size - 1) {
        return (samples[size - 1] - samples[size - 2]) / dt;
    }
    return (samples[index + 1] - samples[index - 1]) / (2.0 * dt);
}

[[nodiscard]] double
    finite_difference_angle(const std::vector<double>& samples, int index, double dt) noexcept {
    const int size = static_cast<int>(samples.size());
    if (size <= 1) {
        return 0.0;
    }
    if (index <= 0) {
        return normalize_angle(samples[1] - samples[0]) / dt;
    }
    if (index >= size - 1) {
        return normalize_angle(samples[size - 1] - samples[size - 2]) / dt;
    }
    return normalize_angle(samples[index + 1] - samples[index - 1]) / (2.0 * dt);
}

template <typename TargetState>
[[nodiscard]] std::expected<ReferenceSample, std::string> build_reference_sample(
    const L4::Aimer& aimer, const TargetState& state, uint64_t state_timestamp_ns,
    uint64_t current_ns, double sample_offset_s, const GimbalTransform& gimbal,
    const MuzzleTransform& muzzle, const core::trajectory::solver::TrajectorySolver& solver,
    double bullet_speed) noexcept {
    const auto prediction = aimer.aim(
        state, gimbal, muzzle, state_timestamp_ns, current_ns, sample_offset_s, bullet_speed,
        solver);
    if (!prediction) {
        return std::unexpected(prediction.error());
    }

    return ReferenceSample{
        .yaw            = prediction->aim_yaw,
        .pitch          = prediction->aim_pitch,
        .distance       = prediction->distance,
        .time_of_flight = prediction->flying_time,
        .position       = prediction->predicted_position,
    };
}

[[nodiscard]] std::expected<ReferenceTrajectory, std::string> build_reference_trajectory(
    const WeaponControllerConfig& config, const L4::Aimer& aimer, const L4::PlannerSeed& seed,
    uint64_t current_ns, const GimbalTransform& gimbal, const MuzzleTransform& muzzle,
    const core::trajectory::solver::TrajectorySolver& solver, double bullet_speed) noexcept {
    const int horizon = config.mpc.horizon_ahead + config.mpc.horizon_back;

    ReferenceTrajectory trajectory{
        .state           = ReferenceMatrix(4, horizon),
        .distances       = std::vector<double>(horizon, 0.0),
        .time_of_flights = std::vector<double>(horizon, 0.0),
        .positions       = std::vector<Eigen::Vector3d>(horizon, Eigen::Vector3d::Zero()),
    };

    std::vector<double> yaw_samples(horizon, 0.0);
    std::vector<double> pitch_samples(horizon, 0.0);

    for (int k = 0; k < horizon; ++k) {
        const double sample_offset =
            (static_cast<double>(k) - config.mpc.horizon_back) * config.mpc.dt;

        std::expected<ReferenceSample, std::string> sample = std::unexpected("empty planner seed");
        if (const auto* robot = seed.robot_state()) {
            sample = build_reference_sample(
                aimer, *robot, seed.state_timestamp_ns, current_ns, sample_offset, gimbal, muzzle,
                solver, bullet_speed);
        } else if (const auto* outpost = seed.outpost_state()) {
            sample = build_reference_sample(
                aimer, *outpost, seed.state_timestamp_ns, current_ns, sample_offset, gimbal, muzzle,
                solver, bullet_speed);
        }

        if (!sample) {
            return std::unexpected(sample.error());
        }

        yaw_samples[k]          = sample->yaw;
        pitch_samples[k]        = std::clamp(sample->pitch, config.pitch_min, config.pitch_max);
        trajectory.distances[k] = sample->distance;
        trajectory.time_of_flights[k] = sample->time_of_flight;
        trajectory.positions[k]       = sample->position;
    }

    trajectory.yaw_origin = yaw_samples[config.mpc.horizon_back];

    for (int k = 0; k < horizon; ++k) {
        trajectory.state(0, k) = normalize_angle(yaw_samples[k] - trajectory.yaw_origin);
        trajectory.state(1, k) = finite_difference_angle(yaw_samples, k, config.mpc.dt);
        trajectory.state(2, k) = pitch_samples[k];
        trajectory.state(3, k) = finite_difference(pitch_samples, k, config.mpc.dt);
    }

    return trajectory;
}

[[nodiscard]] bool solved_ok(const TinySolver* solver, int status) noexcept {
    return solver && status == 0 && solver->solution && solver->solution->solved == 1;
}

[[nodiscard]] bool compute_fire_gate(
    const L4::FireDecisionConfig& fire_decision, int center_index,
    const ReferenceTrajectory& trajectory, const TinySolver* yaw_solver,
    const TinySolver* pitch_solver) noexcept {
    const int sample_index =
        std::min(center_index, static_cast<int>(trajectory.distances.size()) - 1);
    const double reference_yaw =
        normalize_angle(trajectory.state(0, sample_index) + trajectory.yaw_origin);
    const double optimized_yaw =
        normalize_angle(yaw_solver->work->x(0, sample_index) + trajectory.yaw_origin);
    const double reference_pitch = trajectory.state(2, sample_index);
    const double optimized_pitch = pitch_solver->work->x(0, sample_index);
    const double distance        = std::max(trajectory.distances[sample_index], 1e-6);

    return L4::is_on_target(
        fire_decision, optimized_yaw, optimized_pitch, reference_yaw, reference_pitch, distance);
}

[[nodiscard]] WeaponVisualizationDebugData build_debug_data(
    const WeaponControllerConfig& config, const ReferenceTrajectory& trajectory,
    const TinySolver* yaw_solver, const TinySolver* pitch_solver) {
    WeaponVisualizationDebugData debug;
    const int horizon     = static_cast<int>(trajectory.distances.size());
    debug.center_index    = config.mpc.horizon_back;
    debug.lookahead_index = debug.center_index;
    debug.reference_plan.reserve(horizon);
    debug.optimized_plan.reserve(horizon);

    for (int k = 0; k < horizon; ++k) {
        debug.reference_plan.push_back(
            TrajectoryPlanSample{
                .yaw            = normalize_angle(trajectory.state(0, k) + trajectory.yaw_origin),
                .pitch          = trajectory.state(2, k),
                .distance       = trajectory.distances[k],
                .tof            = trajectory.time_of_flights[k],
                .world_position = trajectory.positions[k],
            });

        const Eigen::Vector3d reconstructed_reference = spherical_to_cartesian(
            trajectory.distances[k], debug.reference_plan.back().yaw,
            debug.reference_plan.back().pitch);
        const Eigen::Vector3d world_bias = trajectory.positions[k] - reconstructed_reference;
        const double optimized_yaw =
            normalize_angle(yaw_solver->work->x(0, k) + trajectory.yaw_origin);
        const double optimized_pitch = pitch_solver->work->x(0, k);
        const Eigen::Vector3d reconstructed_optimized =
            spherical_to_cartesian(trajectory.distances[k], optimized_yaw, optimized_pitch);
        debug.optimized_plan.push_back(
            TrajectoryPlanSample{
                .yaw            = optimized_yaw,
                .pitch          = optimized_pitch,
                .distance       = trajectory.distances[k],
                .tof            = trajectory.time_of_flights[k],
                .world_position = reconstructed_optimized + world_bias,
            });
    }

    return debug;
}

} // namespace

TinySolverHandle::~TinySolverHandle() noexcept { reset(); }

TinySolverHandle::TinySolverHandle(TinySolverHandle&& other) noexcept
    : solver_(std::exchange(other.solver_, nullptr)) {}

TinySolverHandle& TinySolverHandle::operator=(TinySolverHandle&& other) noexcept {
    if (this != &other) {
        reset(std::exchange(other.solver_, nullptr));
    }
    return *this;
}

TinySolver** TinySolverHandle::put() noexcept {
    reset();
    return &solver_;
}

void TinySolverHandle::reset(TinySolver* solver) noexcept {
    destroy_tiny_solver(solver_);
    solver_ = solver;
}

TinyMpcTrajectoryOptimizer::TinyMpcTrajectoryOptimizer(
    const WeaponControllerConfig& config) noexcept
    : config_(config) {
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
    const L4::GimbalPlan& plan, const L4::Aimer& aimer, const L4::FireDecisionConfig& fire_decision,
    const fast_tf::CoordinateSystem& tf_buffer,
    const core::trajectory::solver::TrajectorySolver& solver, double bullet_speed,
    uint64_t command_timestamp_ns) noexcept {
    if (!config_.enabled) {
        return std::unexpected("mpc_weapon disabled");
    }
    if (!ready_) {
        return std::unexpected("tinympc solvers are not ready");
    }
    if (plan.source != L4::GimbalPlanSource::Armor) {
        return std::unexpected("only armor planner seeds are supported");
    }
    if (!plan.planner_seed) {
        return std::unexpected("planner seed is missing");
    }
    if (bullet_speed <= 1e-6) {
        return std::unexpected("invalid bullet speed");
    }

    const uint64_t current_ns = command_timestamp_ns;

    auto transforms = L4::lookup_gimbal_muzzle_transforms(tf_buffer, current_ns);
    if (!transforms) {
        return std::unexpected(transforms.error());
    }

    auto reference = build_reference_trajectory(
        config_, aimer, *plan.planner_seed, current_ns, transforms->gimbal, transforms->muzzle,
        solver, bullet_speed);
    if (!reference) {
        return std::unexpected(reference.error());
    }

    TinySolver* const yaw_solver   = yaw_solver_.get();
    TinySolver* const pitch_solver = pitch_solver_.get();

    Eigen::Vector2d x0;
    x0 << reference->state(0, 0), reference->state(1, 0);
    if (tiny_set_x0(yaw_solver, x0) != 0) {
        return std::unexpected("failed to set yaw MPC initial condition");
    }
    yaw_solver->work->Xref = reference->state.block(0, 0, 2, reference->state.cols());
    const int yaw_status   = tiny_solve(yaw_solver);
    if (!solved_ok(yaw_solver, yaw_status)) {
        return std::unexpected("yaw MPC did not converge");
    }

    x0 << reference->state(2, 0), reference->state(3, 0);
    if (tiny_set_x0(pitch_solver, x0) != 0) {
        return std::unexpected("failed to set pitch MPC initial condition");
    }
    pitch_solver->work->Xref = reference->state.block(2, 0, 2, reference->state.cols());
    const int pitch_status   = tiny_solve(pitch_solver);
    if (!solved_ok(pitch_solver, pitch_status)) {
        return std::unexpected("pitch MPC did not converge");
    }

    const int center_index  = config_.mpc.horizon_back;
    const int control_index = std::min(center_index, static_cast<int>(reference->state.cols()) - 2);

    WeaponCommand cmd = passthrough(plan, current_ns);
    cmd.ref_yaw       = normalize_angle(reference->state(0, center_index) + reference->yaw_origin);
    cmd.ref_pitch     = reference->state(2, center_index);
    cmd.yaw = normalize_angle(yaw_solver->work->x(0, center_index) + reference->yaw_origin);
    cmd.pitch =
        std::clamp(pitch_solver->work->x(0, center_index), config_.pitch_min, config_.pitch_max);
    cmd.yaw_vel     = yaw_solver->work->x(1, center_index);
    cmd.pitch_vel   = pitch_solver->work->x(1, center_index);
    cmd.yaw_accel   = yaw_solver->work->u(0, control_index);
    cmd.pitch_accel = pitch_solver->work->u(0, control_index);
    cmd.distance    = reference->distances[center_index];
    cmd.tof         = reference->time_of_flights[center_index];
    cmd.fire        = plan.fire_advice
                   && compute_fire_gate(fire_decision, center_index, *reference, yaw_solver, pitch_solver);
    if (config_.enable_debug) {
        cmd.viz_debug = build_debug_data(config_, *reference, yaw_solver, pitch_solver);
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
    yaw_solver_.reset();
    pitch_solver_.reset();

    const int horizon = config_.mpc.horizon_ahead + config_.mpc.horizon_back;
    const double dt   = config_.mpc.dt;

    const Eigen::Matrix2d A             = (Eigen::Matrix2d{} << 1.0, dt, 0.0, 1.0).finished();
    const Eigen::Matrix<double, 2, 1> B = (Eigen::Matrix<double, 2, 1>{} << 0.0, dt).finished();
    const Eigen::Vector2d f             = Eigen::Vector2d::Zero();

    auto setup_axis = [&](TinySolverHandle& solver_handle, const MpcAxisConfig& axis,
                          bool pitch_axis) {
        TinySolverHandle solver;
        const Eigen::Matrix2d Q             = Eigen::Vector2d{axis.q_pos, axis.q_vel}.asDiagonal();
        const Eigen::Matrix<double, 1, 1> R = Eigen::Matrix<double, 1, 1>::Constant(axis.r);

        if (tiny_setup(solver.put(), A, B, f, Q, R, config_.mpc.rho, 2, 1, horizon, 0) != 0) {
            return false;
        }

        Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, horizon, -kUnlimitedBound);
        Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, horizon, kUnlimitedBound);
        if (pitch_axis) {
            x_min.row(0).setConstant(config_.pitch_min);
            x_max.row(0).setConstant(config_.pitch_max);
        }

        const Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, horizon - 1, -axis.max_acc);
        const Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, horizon - 1, axis.max_acc);

        if (tiny_set_bound_constraints(solver.get(), x_min, x_max, u_min, u_max) != 0) {
            return false;
        }

        solver.get()->settings->max_iter          = config_.mpc.max_iterations;
        solver.get()->settings->abs_pri_tol       = config_.mpc.abs_tol;
        solver.get()->settings->abs_dua_tol       = config_.mpc.abs_tol;
        solver.get()->settings->check_termination = 1;
        solver_handle                             = std::move(solver);
        return true;
    };

    ready_ = setup_axis(yaw_solver_, config_.mpc.yaw, false)
          && setup_axis(pitch_solver_, config_.mpc.pitch, true);
    if (!ready_) {
        yaw_solver_.reset();
        pitch_solver_.reset();
    }
}

} // namespace fcs::L5
