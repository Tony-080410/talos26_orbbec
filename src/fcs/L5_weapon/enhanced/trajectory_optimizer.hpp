#pragma once

#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/fire_control.hpp"
#include "core/trajectory/reference_trajectory.hpp"
#include "frame.hpp"
#include "tiny_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <numbers>
#include <string>

namespace fcs::L5 {

struct DelayGateAngles {
    double yaw{0.0};
    double pitch{0.0};
};

struct DualDelayGateErrors {
    double forward{0.0};
    double backward{0.0};
};

[[nodiscard]] inline double delay_gate_tracking_error(
    const DelayGateAngles& reference, const DelayGateAngles& optimized) noexcept {
    const double yaw_error = std::remainder(reference.yaw - optimized.yaw, 2.0 * std::numbers::pi);
    return std::hypot(yaw_error, reference.pitch - optimized.pitch);
}

template <typename ReferenceFn, typename OptimizedFn>
[[nodiscard]] inline DualDelayGateErrors compute_dual_delay_gate_errors(
    int horizon, int sample_index, int delay_steps, ReferenceFn&& reference_angles,
    OptimizedFn&& optimized_angles) noexcept {
    if (horizon <= 0) {
        return {};
    }

    const auto clamp_index = [horizon](int index) noexcept {
        return std::clamp(index, 0, horizon - 1);
    };
    const int center_index   = clamp_index(sample_index);
    const int forward_index  = clamp_index(center_index + delay_steps);
    const int backward_index = clamp_index(center_index - delay_steps);

    return {
        .forward = delay_gate_tracking_error(
            reference_angles(forward_index), optimized_angles(forward_index)),
        .backward = delay_gate_tracking_error(
            reference_angles(backward_index), optimized_angles(backward_index)),
    };
}

/// RAII wrapper for TinySolver from tinympc C API
///
/// This class owns a TinySolver* pointer and ensures all its internal resources
/// (solution, settings, cache, work) are properly destroyed when the handle goes out of scope.
///
/// ## TINYMPC INVARIANT PROVENANCE
///
/// The vendored tinympc library (3dparty/tinympc) has SEVERE memory safety issues:
///
/// **Allocation Strategy (tiny_api.cpp:25-29):**
///   - Uses 5 separate `new` calls WITHOUT exception handling
///   - Allocation order: solution → cache → settings → work → solver
///   - If ANY `new` throws std::bad_alloc, PREVIOUS allocations LEAK
///
/// **Error Path Behavior (tiny_api.cpp:64-66, 137-139):**
///   - Dimension check failure: Returns non-zero WITHOUT freeing allocations
///   - Cache init failure: Returns non-zero WITHOUT freeing allocations
///   - ALL error paths leak all 5 allocated objects
///
/// **Success Path Invariants (ONLY if no exceptions):**
///   - solver != nullptr
///   - solver->solution != nullptr
///   - solver->cache != nullptr
///   - solver->settings != nullptr
///   - solver->work != nullptr
///
/// **Wrapper Safety Guarantees:**
///   - Checks ALL 5 internal pointers for nullptr before deletion
///   - Wraps delete in try-catch to preserve noexcept (Eigen can throw std::bad_alloc)
///   - Cleans up allocations if tiny_setup() returns error status
///   - Verifies all 5 internal pointers are non-null on success
///   - Documented as NOT thread-safe
///
/// ## Usage
///
/// ```cpp
/// // Factory function (recommended)
/// auto solver = create_tiny_solver(A, B, f, Q, R, rho, nx, nu, horizon);
/// if (!solver) {
///     std::cerr << solver.error() << std::endl;
///     return;
/// }
///
/// // Use solver
/// tiny_set_x0(solver->get(), x0);
/// tiny_solve(solver->get());
///
/// // Automatic cleanup when solver goes out of scope
/// ```
class TinySolverHandle {
public:
    TinySolverHandle() = default;
    ~TinySolverHandle() noexcept;

    TinySolverHandle(const TinySolverHandle&)            = delete;
    TinySolverHandle& operator=(const TinySolverHandle&) = delete;

    TinySolverHandle(TinySolverHandle&& other) noexcept;
    TinySolverHandle& operator=(TinySolverHandle&& other) noexcept;

    [[nodiscard]] TinySolver* get() const noexcept { return solver_; }
    [[nodiscard]] explicit operator bool() const noexcept { return solver_ != nullptr; }

    /// Reset the handle, taking ownership of a new raw pointer.
    /// If a solver was previously owned, it will be destroyed properly.
    void reset(TinySolver* solver = nullptr) noexcept;

private:
    TinySolver* solver_{nullptr};

    // Factory function has access to private constructor
    friend std::expected<TinySolverHandle, std::string> create_tiny_solver(
        const Eigen::Matrix2d& A, const Eigen::Matrix<double, 2, 1>& B, const Eigen::Vector2d& f,
        const Eigen::Matrix2d& Q, const Eigen::Matrix<double, 1, 1>& R, double rho, int nx, int nu,
        int horizon) noexcept;
};

/// Factory function to create a TinySolver with RAII ownership.
///
/// Calls tiny_setup() and wraps the result in TinySolverHandle.
/// Handles all tinympc memory leak scenarios:
/// - Cleans up allocations if tiny_setup() returns error status
/// - Verifies all 5 internal pointers are non-null on success
/// - Returns the handle on success, or error message on failure.
///
/// Thread safety: NOT thread-safe. Do not call concurrently on shared data.
[[nodiscard]] std::expected<TinySolverHandle, std::string> create_tiny_solver(
    const Eigen::Matrix2d& A, const Eigen::Matrix<double, 2, 1>& B, const Eigen::Vector2d& f,
    const Eigen::Matrix2d& Q, const Eigen::Matrix<double, 1, 1>& R, double rho, int nx, int nu,
    int horizon) noexcept;

/// TinyMPC-based trajectory optimizer for the L5 weapon layer.
///
/// Consumes a pre-built reference trajectory from L4, solves independent
/// yaw/pitch MPC problems, and emits the center-horizon command with an
/// L5 fire gate.
class TinyMpcTrajectoryOptimizer {
public:
    explicit TinyMpcTrajectoryOptimizer(
        const WeaponControllerConfig& config,
        const L4::ReferenceTrajectoryConfig& trajectory_cfg) noexcept;
    ~TinyMpcTrajectoryOptimizer() noexcept = default;

    TinyMpcTrajectoryOptimizer(const TinyMpcTrajectoryOptimizer&)            = delete;
    TinyMpcTrajectoryOptimizer& operator=(const TinyMpcTrajectoryOptimizer&) = delete;
    TinyMpcTrajectoryOptimizer(TinyMpcTrajectoryOptimizer&&)                 = delete;
    TinyMpcTrajectoryOptimizer& operator=(TinyMpcTrajectoryOptimizer&&)      = delete;

    [[nodiscard]] bool enabled() const noexcept { return config_.enabled; }

    [[nodiscard]] WeaponCommand
        passthrough(const L4::GimbalPlan& plan, uint64_t command_timestamp_ns) const noexcept;

    [[nodiscard]] std::expected<WeaponCommand, std::string> optimize(
        const L4::GimbalPlan& plan, const L4::FireDecisionConfig& fire_decision,
        const L4::AimerConfig& aimer_config, const fast_tf::CoordinateSystem& tf_buffer,
        uint64_t command_timestamp_ns) noexcept;

private:
    WeaponControllerConfig config_;
    L4::ReferenceTrajectoryConfig trajectory_cfg_;
    TinySolverHandle yaw_solver_;
    TinySolverHandle pitch_solver_;
    bool ready_{false};

    void initialize_solvers() noexcept;
};

} // namespace fcs::L5
