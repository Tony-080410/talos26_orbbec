#pragma once

#include "L4_planning/aimer/aimer.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/fire_control.hpp"
#include "core/trajectory/solver/solver_interfaces.hpp"
#include "frame.hpp"
#include "tiny_api.hpp"

#include <cstdint>
#include <expected>
#include <string>

namespace fcs::L5 {

class TinySolverHandle {
public:
    TinySolverHandle() = default;
    ~TinySolverHandle() noexcept;

    TinySolverHandle(const TinySolverHandle&)            = delete;
    TinySolverHandle& operator=(const TinySolverHandle&) = delete;

    TinySolverHandle(TinySolverHandle&& other) noexcept;
    TinySolverHandle& operator=(TinySolverHandle&& other) noexcept;

    [[nodiscard]] TinySolver* get() const noexcept { return solver_; }
    [[nodiscard]] TinySolver** put() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return solver_ != nullptr; }

    void reset(TinySolver* solver = nullptr) noexcept;

private:
    TinySolver* solver_{nullptr};
};

/// TinyMPC-based trajectory optimizer for the L5 weapon layer.
///
/// The optimizer rebuilds a future reference horizon from the L4 planner seed,
/// solves independent yaw/pitch MPC problems, and emits the center-horizon
/// command with an L5 fire gate.
class TinyMpcTrajectoryOptimizer {
public:
    explicit TinyMpcTrajectoryOptimizer(const WeaponControllerConfig& config) noexcept;
    ~TinyMpcTrajectoryOptimizer() noexcept = default;

    TinyMpcTrajectoryOptimizer(const TinyMpcTrajectoryOptimizer&)            = delete;
    TinyMpcTrajectoryOptimizer& operator=(const TinyMpcTrajectoryOptimizer&) = delete;
    TinyMpcTrajectoryOptimizer(TinyMpcTrajectoryOptimizer&&)                 = delete;
    TinyMpcTrajectoryOptimizer& operator=(TinyMpcTrajectoryOptimizer&&)      = delete;

    [[nodiscard]] bool enabled() const noexcept { return config_.enabled; }

    [[nodiscard]] WeaponCommand
        passthrough(const L4::GimbalPlan& plan, uint64_t command_timestamp_ns) const noexcept;

    [[nodiscard]] std::expected<WeaponCommand, std::string> optimize(
        const L4::GimbalPlan& plan, const L4::Aimer& aimer,
        const L4::FireDecisionConfig& fire_decision, const fast_tf::CoordinateSystem& tf_buffer,
        const core::trajectory::solver::TrajectorySolver& solver, double bullet_speed,
        uint64_t command_timestamp_ns) noexcept;

private:
    WeaponControllerConfig config_;
    TinySolverHandle yaw_solver_;
    TinySolverHandle pitch_solver_;
    bool ready_{false};

    void initialize_solvers() noexcept;
};

} // namespace fcs::L5
