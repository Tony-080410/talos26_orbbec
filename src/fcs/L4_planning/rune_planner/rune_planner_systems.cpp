#include "fcs/L4_planning/rune_planner/rune_planner_systems.hpp"

#include "fcs/L3_estimation/energy_meter/types.hpp"
#include "fcs/L4_planning/angle_utils.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/time.hpp"
#include "math/fast_tf/frame.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"
#include "toml_helper.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace fcs::L4 {
namespace {

constexpr double kMaxVisualizationDistance = 15.0;

auto sync_compensator(TrajectoryCompensator& compensator, const TrajectoryConfig& traj_cfg) noexcept
    -> void {
    compensator.velocity        = traj_cfg.bullet_speed;
    compensator.gravity         = traj_cfg.gravity;
    compensator.iteration_times = traj_cfg.iteration_times;
    compensator.cd              = traj_cfg.cd;
    compensator.rho             = traj_cfg.rho;
    compensator.diameter        = traj_cfg.diameter;
    compensator.mass            = traj_cfg.mass;
    compensator.rk4_dt          = traj_cfg.rk4_dt;
    compensator.rk4_min_steps   = traj_cfg.rk4_min_steps;
    compensator.rk4_max_steps   = traj_cfg.rk4_max_steps;
    compensator.on_params_updated();
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
        [compensator = std::shared_ptr<TrajectoryCompensator>{}, last_timestamp_ns = uint64_t{0},
         last_yaw = 0.0, last_pitch = 0.0](
            talos::res<RunePlannerConfig> cfg, talos::res<MpcPlannerConfig> mpc_cfg,
            talos::spmc<fcs::energy_meter::EnergyMeterState, EnergyMeterStateChannelTag> state_in,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::spmc_mut<GimbalPlan, RuneGimbalPlanChannelTag> plan_out) mutable {
            if (!cfg->enabled) {
                return;
            }

            if (!compensator || mpc_cfg.is_changed()) {
                compensator = std::shared_ptr<TrajectoryCompensator>(
                    create_compensator(mpc_cfg->compensator_type));
                sync_compensator(*compensator, mpc_cfg->trajectory);
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
                SPDLOG_DEBUG(
                    "rune planner: muzzle_tf lookup failed: {}",
                    fast_tf::to_string(muzzle_tf.error()));
                return;
            }
            if (!gimbal_tf) {
                SPDLOG_DEBUG(
                    "rune planner: gimbal_tf lookup failed: {}",
                    fast_tf::to_string(gimbal_tf.error()));
                return;
            }

            const Eigen::Vector3d origin_odom = (*muzzle_tf)->translation();
            const Eigen::Vector3d target_odom = state->predicted_position_odom;
            const Eigen::Vector3d rel         = target_odom - origin_odom;

            const double distance = rel.norm();
            if (!std::isfinite(distance) || distance <= 0.0) {
                return;
            }

            double yaw   = 0.0;
            double pitch = 0.0;
            if (!compensator->compensate(rel, yaw, pitch, mpc_cfg->trajectory.bullet_speed)) {
                return;
            }
            pitch = std::clamp(pitch, mpc_cfg->pitch_min, mpc_cfg->pitch_max);

            GimbalPlan plan;
            plan.timestamp_ns       = current_time_ns;
            plan.yaw                = normalize_rad(yaw);
            plan.pitch              = pitch;
            plan.distance           = distance;
            plan.predicted_future   = state->predicted_future_ns;
            plan.predicted_position = target_odom;

            plan.trajectory = compensator->get_trajectory(kMaxVisualizationDistance, plan.pitch);
            for (auto& p : plan.trajectory) {
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
