#include "fcs/L4_planning/rune_planner/rune_planner_systems.hpp"

#include "fcs/L3_estimation/energy_meter/types.hpp"
#include "fcs/L4_planning/angle_utils.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L4_planning/trajectory/solver_interfaces.hpp"
#include "fcs/core/channel_topics.hpp"
#include "fcs/core/time.hpp"
#include "math/fast_tf/frame.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"
#include "toml_helper.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>

namespace fcs::L4 {
namespace {

constexpr double kMaxVisualizationDistance = 15.0;

[[nodiscard]] size_t solver_config_hash(const MpcPlannerConfig& cfg) noexcept {
    size_t hash             = 0;
    const auto hash_combine = [&hash](auto value) {
        hash ^= std::hash<decltype(value)>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    };

    hash_combine(static_cast<int>(cfg.compensator_type));
    const auto& traj = cfg.trajectory;
    hash_combine(traj.iteration_times);
    hash_combine(traj.bullet_speed);
    hash_combine(traj.gravity);
    hash_combine(traj.resistance);
    hash_combine(traj.cd);
    hash_combine(traj.rho);
    hash_combine(traj.diameter);
    hash_combine(traj.mass);
    hash_combine(traj.rk4_dt);
    hash_combine(traj.rk4_min_steps);
    hash_combine(traj.rk4_max_steps);
    hash_combine(traj.enable_lookup_table);
    hash_combine(traj.table_range_max);
    hash_combine(traj.table_range_step);
    hash_combine(traj.table_pitch_min);
    hash_combine(traj.table_pitch_max);
    hash_combine(traj.table_pitch_step);
    return hash;
}

auto is_on_target(
    const MpcPlannerConfig& cfg, double cur_yaw, double cur_pitch, double target_yaw,
    double target_pitch, double distance) noexcept -> bool {
    const double range_yaw =
        std::max(std::abs(std::atan2(cfg.shooting_range_w / 2.0, distance)), cfg.fire_thresh);
    const double range_pitch =
        std::max(std::abs(std::atan2(cfg.shooting_range_h / 2.0, distance)), cfg.fire_thresh);

    return std::abs(cur_yaw - target_yaw) < range_yaw
        && std::abs(cur_pitch - target_pitch) < range_pitch;
}

} // namespace

std::expected<void, std::string> RunePlannerConfig::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, enabled);
    return {};
}

void register_rune_planning_systems(talos::Scheduler& scheduler, RunePlannerConfig&& config) {
    scheduler.world().insert_resource(config);

    scheduler.add_system<talos::pool_compute>(
        "l4_rune_planner",
        [solver = std::shared_ptr<trajectory::IBallisticSolver>{}, last_config_hash = size_t{0},
         last_timestamp_ns = uint64_t{0}, last_yaw = 0.0, last_pitch = 0.0](
            talos::res<RunePlannerConfig> cfg, talos::res<MpcPlannerConfig> mpc_cfg,
            talos::subscribe<fcs::energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic>
                state_in,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::publish<GimbalPlan, RuneGimbalPlanChannelTopic> plan_out) mutable {
            if (!cfg->enabled) {
                return;
            }

            const size_t current_config_hash = solver_config_hash(*mpc_cfg);

            // Recreate solver when config changes
            if (!solver || last_config_hash != current_config_hash) {
                solver = std::shared_ptr<trajectory::IBallisticSolver>(
                    trajectory::create_solver(mpc_cfg->compensator_type, mpc_cfg->trajectory));
                last_config_hash = current_config_hash;
            }

            const auto state = state_in.read();
            if (!state || !state->tracking_valid) {
                return;
            }

            const uint64_t current_time_ns = fcs::clock::now_ns();

            auto muzzle_tf =
                fast_tf::lookup_clamped<fast_tf::odom_fuxk_frame, fast_tf::muzzle_link_fuxk_frame>(
                    **tf_buffer, fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>{},
                    current_time_ns);
            auto gimbal_tf =
                fast_tf::lookup_clamped<fast_tf::odom_fuxk_frame, fast_tf::gimbal_link_fuxk_frame>(
                    **tf_buffer, fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>{},
                    current_time_ns);

            if (!muzzle_tf) {
                SPDLOG_DEBUG("rune planner: muzzle_tf lookup failed: {}", muzzle_tf.error());
                return;
            }
            if (!gimbal_tf) {
                SPDLOG_DEBUG("rune planner: gimbal_tf lookup failed: {}", gimbal_tf.error());
                return;
            }

            const Eigen::Vector3d origin_odom = (*muzzle_tf)->translation();
            const Eigen::Vector3d target_odom = state->predicted_position_odom;
            const Eigen::Vector3d rel         = target_odom - origin_odom;

            const double distance = rel.norm();
            if (!std::isfinite(distance) || distance <= 0.0) {
                return;
            }

            // Use new solver API - solve returns AimSolution value object
            const auto solution = solver->solve(rel, mpc_cfg->trajectory.bullet_speed);
            if (!solution.is_valid()) {
                return;
            }
            double pitch = std::clamp(solution.pitch, mpc_cfg->pitch_min, mpc_cfg->pitch_max);

            GimbalPlan plan;
            plan.timestamp_ns       = current_time_ns;
            plan.yaw                = normalize_rad(solution.yaw);
            plan.pitch              = pitch;
            plan.distance           = distance;
            plan.predicted_future   = state->predicted_future_ns;
            plan.predicted_position = target_odom;

            // Initialize visualization debug data
            // Note: Don't set viz_debug.predicted_position because:
            // 1. It has timestamp = predicted_future_ns (future time)
            // 2. TF interpolation uses current timestamp_ns
            // 3. For moving targets, time mismatch causes visual misalignment
            // 4. The trajectory itself is sufficient for visualization
            plan.viz_debug = VisualizationDebugData{};
            // viz_debug->predicted_position left as std::nullopt (not set)
            plan.viz_debug->predicted_future = state->predicted_future_ns;
            plan.viz_debug->trajectory       = solver->generate_trajectory(
                plan.pitch, mpc_cfg->trajectory.bullet_speed, kMaxVisualizationDistance);
            for (auto& p : plan.viz_debug->trajectory) {
                const double x = p.first;
                const double z = p.second;
                p.first        = x * std::cos(plan.pitch) + z * std::sin(plan.pitch);
                p.second       = -x * std::sin(plan.pitch) + z * std::cos(plan.pitch);
            }

            if (last_timestamp_ns > 0 && current_time_ns > last_timestamp_ns) {
                const double dt = static_cast<double>(current_time_ns - last_timestamp_ns) * 1e-9;
                if (dt > 0.0) {
                    plan.yaw_vel   = normalize_rad(plan.yaw - last_yaw) / dt;
                    plan.pitch_vel = (plan.pitch - last_pitch) / dt;
                }
            }

            last_timestamp_ns = current_time_ns;
            last_yaw          = plan.yaw;
            last_pitch        = plan.pitch;

            const auto [roll, pitch_now, yaw_now] = (*gimbal_tf)->euler_rot().rpy();
            (void)roll;
            plan.fire_advice =
                is_on_target(*mpc_cfg, yaw_now, pitch_now, plan.yaw, -plan.pitch, distance);

            plan_out.write(plan);
        });
}

} // namespace fcs::L4
