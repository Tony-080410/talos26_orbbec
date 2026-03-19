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

// =============================================================================
// TINYMPC INVARIANT PROVENANCE (VENDORED LIBRARY - DO NOT MODIFY)
// =============================================================================
//
// VENDORED LIBRARY: 3dparty/tinympc (C API with broken memory management)
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
//
// =============================================================================
namespace fcs::L5 {
namespace {
// ==============================================================================
// TINYMPC INVARIANT PROVENANCE
// ==============================================================================
//
// VENDORED LIBRARY: 3dparty/tinympc (DO NOT MODIFY)
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
// ==============================================================================

using GimbalTransform = L4::Aimer::GimbalTransform;
using MuzzleTransform = L4::Aimer::MuzzleTransform;
using ReferenceMatrix = Eigen::Matrix<double, 4, Eigen::Dynamic>;

constexpr double kUnlimitedBound = 1e17;

struct ReferenceTrajectory {
    ReferenceMatrix state;
    std::vector<double> distances;
    std::vector<double> diff_yaws;
    std::vector<double> time_of_flights;
    std::vector<Eigen::Vector3d> positions;
    std::vector<int> selected_armor_ids;
    std::vector<int> rough_selected_armor_ids;
    double yaw_origin{0.0};
};

struct ReferenceSample {
    double yaw{0.0};
    double pitch{0.0};
    double distance{0.0};
    double diff_yaw{0.0};
    double time_of_flight{0.0};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    int selected_armor_id{0};
    int rough_selected_armor_id{0};
};

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
    const L4::Aimer& aimer, const TargetState& state, L4::ArmorAimContext context,
    uint64_t state_timestamp_ns, uint64_t current_ns, double sample_offset_s,
    const GimbalTransform& gimbal, const MuzzleTransform& muzzle,
    const core::trajectory::solver::TrajectorySolver& solver, double bullet_speed) noexcept {
    const auto prediction = aimer.aim(
        state, context, gimbal, muzzle, state_timestamp_ns, current_ns, sample_offset_s,
        bullet_speed, solver);
    if (!prediction) {
        return std::unexpected(prediction.error());
    }

    return ReferenceSample{
        .yaw                     = prediction->aim_yaw,
        .pitch                   = prediction->aim_pitch,
        .distance                = prediction->distance,
        .diff_yaw                = prediction->diff_yaw,
        .time_of_flight          = prediction->flying_time,
        .position                = prediction->predicted_position,
        .selected_armor_id       = prediction->selected_armor_id,
        .rough_selected_armor_id = prediction->rough_selected_armor_id,
    };
}

[[nodiscard]] std::expected<ReferenceTrajectory, std::string> build_reference_trajectory(
    const WeaponControllerConfig& config, const L4::Aimer& aimer, const L4::PlannerSeed& seed,
    uint64_t current_ns, const GimbalTransform& gimbal, const MuzzleTransform& muzzle,
    const core::trajectory::solver::TrajectorySolver& solver, double bullet_speed) noexcept {
    const int horizon = config.mpc.horizon_ahead + config.mpc.horizon_back + 1;

    ReferenceTrajectory trajectory{
        .state                    = ReferenceMatrix(4, horizon),
        .distances                = std::vector<double>(horizon, 0.0),
        .diff_yaws                = std::vector<double>(horizon, 0.0),
        .time_of_flights          = std::vector<double>(horizon, 0.0),
        .positions                = std::vector<Eigen::Vector3d>(horizon, Eigen::Vector3d::Zero()),
        .selected_armor_ids       = std::vector<int>(horizon, 0),
        .rough_selected_armor_ids = std::vector<int>(horizon, 0),
    };

    std::vector<double> yaw_samples(horizon, 0.0);
    std::vector<double> pitch_samples(horizon, 0.0);
    L4::ArmorAimContext context;
    context.target_jumped      = seed.target_jumped;
    context.phase              = seed.aim_phase;
    context.preferred_armor_id = seed.selected_armor_id;

    for (int k = 0; k < horizon; ++k) {
        const double sample_offset =
            (static_cast<double>(k) - config.mpc.horizon_back) * config.mpc.dt;

        std::expected<ReferenceSample, std::string> sample = std::unexpected("empty planner seed");
        if (const auto* robot = seed.robot_state()) {
            sample = build_reference_sample(
                aimer, *robot, context, seed.state_timestamp_ns, current_ns, sample_offset, gimbal,
                muzzle, solver, bullet_speed);
        } else if (const auto* outpost = seed.outpost_state()) {
            sample = build_reference_sample(
                aimer, *outpost, context, seed.state_timestamp_ns, current_ns, sample_offset,
                gimbal, muzzle, solver, bullet_speed);
        }

        if (!sample) {
            return std::unexpected(sample.error());
        }

        context.preferred_armor_id = sample->selected_armor_id;

        yaw_samples[k]          = sample->yaw;
        pitch_samples[k]        = std::clamp(sample->pitch, config.pitch_min, config.pitch_max);
        trajectory.distances[k] = sample->distance;
        trajectory.diff_yaws[k] = sample->diff_yaw;
        trajectory.time_of_flights[k]          = sample->time_of_flight;
        trajectory.positions[k]                = sample->position;
        trajectory.selected_armor_ids[k]       = sample->selected_armor_id;
        trajectory.rough_selected_armor_ids[k] = sample->rough_selected_armor_id;
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

    const int sample_index = std::clamp(center_index, 0, horizon - 1);
    const auto clamp_index = [&](int index) noexcept { return std::clamp(index, 0, horizon - 1); };
    const auto reference_angles = [&](int index) noexcept {
        return std::pair{
            normalize_angle(trajectory.state(0, index) + trajectory.yaw_origin),
            trajectory.state(2, index),
        };
    };
    const auto optimized_angles = [&](int index) noexcept {
        return std::pair{
            normalize_angle(yaw_solver->solution->x(0, index) + trajectory.yaw_origin),
            pitch_solver->solution->x(0, index),
        };
    };
    const auto delay_error = [&](int index) noexcept {
        const auto [reference_yaw, reference_pitch] = reference_angles(index);
        const auto [optimized_yaw, optimized_pitch] = optimized_angles(index);
        return std::hypot(
            normalize_angle(reference_yaw - optimized_yaw), reference_pitch - optimized_pitch);
    };

    status.delay_steps =
        std::max(0, static_cast<int>(std::lround(aimer_config.delay / std::max(mpc_dt, 1e-6))));
    const int forward_index          = clamp_index(sample_index + status.delay_steps);
    const int backward_index         = clamp_index(sample_index - status.delay_steps);
    status.delay_gate_forward_error  = delay_error(forward_index);
    status.delay_gate_backward_error = delay_error(backward_index);
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
    const WeaponControllerConfig& config, const ReferenceTrajectory& trajectory,
    const TinySolver* yaw_solver, const TinySolver* pitch_solver, const FireGateStatus& fire_gate) {
    WeaponVisualizationDebugData debug;
    const int horizon               = static_cast<int>(trajectory.distances.size());
    debug.center_index              = config.mpc.horizon_back;
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
        // We MUST check and clean up if raw_solver is non-null

        if (raw_solver != nullptr) {
            // tinympc leaked all 5 internal pointers - clean them up
            auto safe_delete = [](auto* ptr) noexcept {
                if (ptr != nullptr) {
                    try {
                        delete ptr;
                    } catch (...) { /* swallow to preserve noexcept */
                    }
                }
            };

            safe_delete(raw_solver->work);
            safe_delete(raw_solver->settings);
            safe_delete(raw_solver->cache);
            safe_delete(raw_solver->solution);
            safe_delete(raw_solver);
        }

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
        // Clean up the partial mess and report error
        auto safe_delete = [](auto* ptr) noexcept {
            if (ptr != nullptr) {
                try {
                    delete ptr;
                } catch (...) { /* swallow */
                }
            }
        };

        safe_delete(raw_solver->work);
        safe_delete(raw_solver->settings);
        safe_delete(raw_solver->cache);
        safe_delete(raw_solver->solution);
        safe_delete(raw_solver);

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

    const auto current_gimbal  = transforms->gimbal.euler_rot();
    const double current_yaw   = normalize_angle(current_gimbal.yaw);
    const double current_pitch = current_gimbal.pitch;
    const int center_index     = config_.mpc.horizon_back;
    const int control_index = std::min(center_index, static_cast<int>(reference->state.cols()) - 2);

    WeaponCommand cmd = passthrough(plan, current_ns);
    cmd.ref_yaw       = normalize_angle(reference->state(0, center_index) + reference->yaw_origin);
    cmd.ref_pitch     = reference->state(2, center_index);
    cmd.yaw   = normalize_angle(yaw_solver->solution->x(0, center_index) + reference->yaw_origin);
    cmd.pitch = std::clamp(
        pitch_solver->solution->x(0, center_index), config_.pitch_min, config_.pitch_max);
    cmd.yaw_vel                    = yaw_solver->solution->x(1, center_index);
    cmd.pitch_vel                  = pitch_solver->solution->x(1, center_index);
    cmd.yaw_accel                  = yaw_solver->solution->u(0, control_index);
    cmd.pitch_accel                = pitch_solver->solution->u(0, control_index);
    cmd.distance                   = reference->distances[center_index];
    cmd.tof                        = reference->time_of_flights[center_index];
    const FireGateStatus fire_gate = compute_fire_gate(
        fire_decision, aimer.config(), plan, current_yaw, current_pitch, config_.mpc.dt,
        center_index, *reference, yaw_solver, pitch_solver);
    cmd.fire = plan.fire_advice && fire_gate.final_fire_gate_ok;
    if (config_.enable_debug) {
        cmd.viz_debug = build_debug_data(config_, *reference, yaw_solver, pitch_solver, fire_gate);
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

    const int horizon = config_.mpc.horizon_ahead + config_.mpc.horizon_back + 1;
    const double dt   = config_.mpc.dt;

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
