#pragma once

#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/fire_control.hpp"
#include "core/trajectory/reference_trajectory.hpp"
#include "frame.hpp"
#include "small_mpc_solver.hpp"

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

/// TinyMPC-based trajectory optimizer for the L5 weapon layer.
///
/// Consumes a pre-built reference trajectory from L4, solves independent
/// yaw/pitch MPC problems via SmallMpcSolver, and emits the center-horizon
/// command with an L5 fire gate.
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
    SmallMpcSolver yaw_solver_;
    SmallMpcSolver pitch_solver_;
    bool ready_{false};

    void initialize_solvers() noexcept;
};

} // namespace fcs::L5
