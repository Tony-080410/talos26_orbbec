#pragma once

#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/fire_control.hpp"
#include "core/trajectory/reference_trajectory.hpp"
#include "dual_small_mpc_solver.hpp"
#include "frame.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <numbers>
#include <string>

namespace fcs::L5 {

/// TinyMPC-based trajectory optimizer for the L5 weapon layer.
///
/// Consumes a pre-built reference trajectory from L4, solves independent
/// yaw/pitch MPC problems via DualSmallMpcSolver, and emits the center-horizon
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
    DualSmallMpcSolver batched_solver_;
    bool ready_{false};

    void initialize_solvers() noexcept;
};

} // namespace fcs::L5
