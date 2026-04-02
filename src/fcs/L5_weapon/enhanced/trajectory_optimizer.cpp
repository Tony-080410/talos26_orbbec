#include "L5_weapon/enhanced/trajectory_optimizer.hpp"

#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "core/math/normalize.hpp"
#include "tiny_api.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <utility>

// =============================================================================
// TinyMPC: Vendored C library (3dparty/tinympc) with broken memory management.
// See destroy_tiny_solver() and create_tiny_solver() for full invariant documentation.
// =============================================================================
namespace fcs::L5 {
namespace {
// ==============================================================================
// TinyMPC invariant provenance (vendored library - do not modify)
// ==============================================================================
//
// ALLOCATION INVARIANTS (tiny_api.cpp:25-29):
//   - Uses 5 separate `new` calls WITHOUT exception handling
//   - Allocation order: solution → cache → settings → work → solver
//   - If ANY allocation throws std::bad_alloc, PREVIOUS allocations LEAK
//   - NO exception safety guarantees
//
// ERROR PATH INVARIANTS (tiny_api.cpp:64-66, 137-139):
//   - Dimension check failure: Returns non-zero status WITHOUT freeing allocations
//   - Cache init failure: Returns non-zero status WITHOUT freeing allocations
//   - ALL error paths leak all 5 allocated objects
//
// SUCCESS PATH INVARIANTS (ONLY if no exceptions thrown):
//   - solver != nullptr
//   - solver->solution != nullptr (TinySolution with Eigen matrices)
//   - solver->cache != nullptr (TinyCache with Eigen matrices)
//   - solver->settings != nullptr (TinySettings struct)
//   - solver->work != nullptr (TinyWorkspace with many Eigen matrices)
//
// EXCEPTION SAFETY: NONE
//   - No try-catch blocks anywhere in tinympc
//   - Eigen matrix destruction can throw std::bad_alloc
//   - Basic exception guarantee violated
//
// THREAD SAFETY: UNDEFINED
//   - No synchronization primitives
//   - No documentation about concurrent access
//   - ASSUME NOT THREAD-SAFE
//
// WRAPPER CONTRACT:
//   1. Must check for nullptr on ALL 5 internal pointers before deletion
//   2. Must wrap delete in try-catch to preserve noexcept (Eigen can throw)
//   3. Must clean up allocations if tiny_setup() returns error status
//   4. Must verify all 5 internal pointers are non-null on success
//   5. Must document as "not thread-safe"
using fcs::core::math::normalize_angle;

using ReferenceTrajectory = core::trajectory::ReferenceTrajectory;

constexpr double kUnlimitedBound = 1e17;

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

// ==============================================================================
// RAII CLEANUP FUNCTION: Handles all tinympc invariant violations
// ==============================================================================
void destroy_tiny_solver(TinySolver* solver) noexcept {
    if (solver == nullptr) {
        return;
    }

    // tinympc invariant analysis requires individual nullptr checks:
    // 1. Partial allocation: new may have thrown partway through
    // 2. Error path leak: tiny_setup() returned error but allocated all 5
    // 3. Eigen destruction safety: std::bad_alloc can be thrown during ~Matrix()
    //
    // Strategy: Check each pointer, wrap delete in try-catch for noexcept safety
    auto safe_delete = [](auto* ptr) noexcept {
        if (ptr != nullptr) {
            try {
                delete ptr;
            } catch (...) {
                // Eigen threw during matrix destruction
                // In production: log this failure with spdlog
                // We swallow to preserve noexcept as required by C++ Core Guidelines
            }
        }
    };

    // Delete in REVERSE construction order (work → settings → cache → solution → solver)
    // This follows RAII best practices even though tinympc doesn't~
    safe_delete(solver->work);     // Allocated 4th, contains many Eigen matrices
    safe_delete(solver->settings); // Allocated 3rd, plain struct
    safe_delete(solver->cache);    // Allocated 2nd, contains Eigen matrices
    safe_delete(solver->solution); // Allocated 1st, contains Eigen matrices
    safe_delete(solver);           // Allocated 5th, plain struct
}

[[nodiscard]] Eigen::Vector3d
    spherical_to_cartesian(double distance, double yaw, double pitch) noexcept {
    return Eigen::Vector3d{
        distance * std::cos(pitch) * std::cos(yaw),
        distance * std::cos(pitch) * std::sin(yaw),
        distance * std::sin(pitch),
    };
}

[[nodiscard]] bool solved_ok(const TinySolver* solver, int status) noexcept {
    return solver && status == 0 && solver->solution && solver->solution->solved == 1;
}

[[nodiscard]] FireGateStatus compute_fire_gate(
    const L4::FireDecisionConfig& fire_decision, const L4::AimerConfig& aimer_config,
    const L4::GimbalPlan& plan, double current_yaw, double current_pitch, double mpc_dt,
    int center_index, const ReferenceTrajectory& trajectory, const TinySolver* yaw_solver,
    const TinySolver* pitch_solver) noexcept {
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
            .yaw   = normalize_angle(yaw_solver->solution->x(0, index) + trajectory.yaw_origin),
            .pitch = pitch_solver->solution->x(0, index),
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

    const double optimized_yaw =
        normalize_angle(yaw_solver->solution->x(0, sample_index) + trajectory.yaw_origin);
    // is_on_target compares against TF pitch convention; optimizer pitch follows command
    // convention.
    const double optimized_pitch = -pitch_solver->solution->x(0, sample_index);
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
    int horizon_back, const ReferenceTrajectory& trajectory, const TinySolver* yaw_solver,
    const TinySolver* pitch_solver, const FireGateStatus& fire_gate) {
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
            normalize_angle(yaw_solver->solution->x(0, k) + trajectory.yaw_origin);
        const double optimized_pitch = pitch_solver->solution->x(0, k);
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

TinySolverHandle::~TinySolverHandle() noexcept { reset(); }

TinySolverHandle::TinySolverHandle(TinySolverHandle&& other) noexcept
    : solver_(std::exchange(other.solver_, nullptr)) {}

TinySolverHandle& TinySolverHandle::operator=(TinySolverHandle&& other) noexcept {
    if (this != &other) {
        reset(std::exchange(other.solver_, nullptr));
    }
    return *this;
}

void TinySolverHandle::reset(TinySolver* solver) noexcept {
    destroy_tiny_solver(solver_);
    solver_ = solver;
}

// ==============================================================================
// FACTORY FUNCTION: Exception-safe creation with invariant verification
// ==============================================================================
[[nodiscard]] std::expected<TinySolverHandle, std::string> create_tiny_solver(
    const Eigen::Matrix2d& A, const Eigen::Matrix<double, 2, 1>& B, const Eigen::Vector2d& f,
    const Eigen::Matrix2d& Q, const Eigen::Matrix<double, 1, 1>& R, double rho, int nx, int nu,
    int horizon) noexcept {

    TinySolver* raw_solver = nullptr;

    // ============================================================
    // TINYMPC BEHAVIOR ANALYSIS:
    // ============================================================
    //
    // tiny_setup() allocation sequence (tiny_api.cpp:25-29):
    //   1. new TinySolution()
    //   2. new TinyCache()         ← Can throw std::bad_alloc
    //   3. new TinySettings()      ← Can throw std::bad_alloc
    //   4. new TinyWorkspace()     ← Can throw std::bad_alloc
    //   5. new TinySolver()        ← Can throw std::bad_alloc
    //
    // Error paths (tiny_api.cpp:64-66, 137-139):
    //   - Dimension mismatch → returns non-zero, LEAKS all 5 objects
    //   - Cache init failure → returns non-zero, LEAKS all 5 objects
    //
    // Contract:
    //   - Return value 0 = success, all 5 pointers valid
    //   - Return value non-zero = failure, but raw_solver might STILL be non-null (leaked!)
    //
    // ============================================================

    const int status = tiny_setup(&raw_solver, A, B, f, Q, R, rho, nx, nu, horizon, 0);

    // ============================================================
    // ERROR PATH HANDLING: Clean up tinympc's leaks
    // ============================================================
    if (status != 0) {
        // tiny_setup failed, but it may have allocated the 5 objects anyway
        // tinympc BUG: Returns error without freeing allocations
        // destroy_tiny_solver handles nullptr checks + safe deletion
        destroy_tiny_solver(raw_solver);

        return std::unexpected("tiny_setup failed with status: " + std::to_string(status));
    }

    // ============================================================
    // SUCCESS PATH VERIFICATION: Defensive invariant checking
    // ============================================================
    if (raw_solver == nullptr) {
        // tinympc returned status=0 but didn't set the output parameter
        // This is a tinympc bug - it should NEVER happen
        return std::unexpected("tiny_setup returned success but solver is null (tinympc bug)");
    }

    // tinympc invariant: On success, ALL internal pointers MUST be non-null
    // But we're defensive because tinympc has proven untrustworthy~
    const bool fully_initialized = (raw_solver->solution != nullptr)
                                && (raw_solver->settings != nullptr)
                                && (raw_solver->cache != nullptr) && (raw_solver->work != nullptr);

    if (!fully_initialized) {
        // Partial initialization detected - tinympc is even more broken than we thought
        destroy_tiny_solver(raw_solver);

        return std::unexpected(
            "tiny_setup returned partially initialized solver (tinympc invariant violation)");
    }

    // ============================================================
    // CONSTRUCTION SUCCESS: Transfer ownership to RAII handle
    // ============================================================
    TinySolverHandle handle;
    handle.reset(raw_solver); // Transfer ownership, handle will cleanup on destruction
    return handle;
}

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
        return std::unexpected("tinympc solvers are not ready");
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

    TinySolver* const yaw_solver   = yaw_solver_.get();
    TinySolver* const pitch_solver = pitch_solver_.get();

    Eigen::Vector2d x0;
    x0 << reference.state(0, 0), reference.state(1, 0);
    if (tiny_set_x0(yaw_solver, x0) != 0) {
        return std::unexpected("failed to set yaw MPC initial condition");
    }
    yaw_solver->work->Xref = reference.state.block(0, 0, 2, reference.state.cols());
    const int yaw_status   = tiny_solve(yaw_solver);
    if (!solved_ok(yaw_solver, yaw_status)) {
        return std::unexpected("yaw MPC did not converge");
    }

    x0 << reference.state(2, 0), reference.state(3, 0);
    if (tiny_set_x0(pitch_solver, x0) != 0) {
        return std::unexpected("failed to set pitch MPC initial condition");
    }
    pitch_solver->work->Xref = reference.state.block(2, 0, 2, reference.state.cols());
    const int pitch_status   = tiny_solve(pitch_solver);
    if (!solved_ok(pitch_solver, pitch_status)) {
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
    cmd.yaw   = normalize_angle(yaw_solver->solution->x(0, center_index) + reference.yaw_origin);
    cmd.pitch = std::clamp(
        pitch_solver->solution->x(0, center_index), config_.pitch_min, config_.pitch_max);
    cmd.yaw_vel                    = yaw_solver->solution->x(1, center_index);
    cmd.pitch_vel                  = pitch_solver->solution->x(1, center_index);
    cmd.yaw_accel                  = yaw_solver->solution->u(0, control_index);
    cmd.pitch_accel                = pitch_solver->solution->u(0, control_index);
    cmd.distance                   = reference.distances[center_index];
    cmd.tof                        = reference.time_of_flights[center_index];
    const FireGateStatus fire_gate = compute_fire_gate(
        fire_decision, aimer_config, plan, current_yaw, current_pitch, trajectory_cfg_.dt,
        center_index, reference, yaw_solver, pitch_solver);
    cmd.fire = plan.fire_advice && fire_gate.final_fire_gate_ok;
    if (config_.enable_debug) {
        cmd.viz_debug = build_debug_data(
            trajectory_cfg_.horizon_back, reference, yaw_solver, pitch_solver, fire_gate);
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

    const int horizon = trajectory_cfg_.horizon_ahead + trajectory_cfg_.horizon_back + 1;
    const double dt   = trajectory_cfg_.dt;

    const Eigen::Matrix2d A             = (Eigen::Matrix2d{} << 1.0, dt, 0.0, 1.0).finished();
    const Eigen::Matrix<double, 2, 1> B = (Eigen::Matrix<double, 2, 1>{} << 0.0, dt).finished();
    const Eigen::Vector2d f             = Eigen::Vector2d::Zero();

    auto setup_axis = [&](const MpcAxisConfig& axis,
                          bool pitch_axis) -> std::expected<TinySolverHandle, std::string> {
        const Eigen::Matrix2d Q             = Eigen::Vector2d{axis.q_pos, axis.q_vel}.asDiagonal();
        const Eigen::Matrix<double, 1, 1> R = Eigen::Matrix<double, 1, 1>::Constant(axis.r);

        // Use factory function for exception-safe creation
        auto solver_result = create_tiny_solver(A, B, f, Q, R, config_.mpc.rho, 2, 1, horizon);
        if (!solver_result) {
            return std::unexpected("Failed to create solver: " + solver_result.error());
        }
        auto solver = std::move(*solver_result);

        Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, horizon, -kUnlimitedBound);
        Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, horizon, kUnlimitedBound);
        if (pitch_axis) {
            x_min.row(0).setConstant(config_.pitch_min);
            x_max.row(0).setConstant(config_.pitch_max);
        }

        const Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, horizon - 1, -axis.max_acc);
        const Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, horizon - 1, axis.max_acc);

        if (tiny_set_bound_constraints(solver.get(), x_min, x_max, u_min, u_max) != 0) {
            return std::unexpected("Failed to set bound constraints");
        }

        solver.get()->settings->max_iter          = config_.mpc.max_iterations;
        solver.get()->settings->abs_pri_tol       = config_.mpc.abs_tol;
        solver.get()->settings->abs_dua_tol       = config_.mpc.abs_tol;
        solver.get()->settings->check_termination = 1;

        return solver;
    };

    auto yaw_result = setup_axis(config_.mpc.yaw, false);
    if (!yaw_result) {
        ready_ = false;
        return;
    }
    yaw_solver_ = std::move(*yaw_result);

    auto pitch_result = setup_axis(config_.mpc.pitch, true);
    if (!pitch_result) {
        yaw_solver_.reset();
        ready_ = false;
        return;
    }
    pitch_solver_ = std::move(*pitch_result);

    ready_ = true;
}

} // namespace fcs::L5
