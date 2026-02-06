#include "fcs/L5_weapon/enhanced/weapon_systems.hpp"

#include "L4_planning/config.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L4_planning/trajectory/solver_interfaces.hpp"
#include "fcs/L5_weapon/config.hpp"
#include "fcs/L5_weapon/enhanced/weapon_controller.hpp"
#include "fcs/core/channel_topics.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

#include <functional>
#include <memory>

namespace fcs::L5 {

namespace {

[[nodiscard]] size_t solver_config_hash(const fcs::L4::MpcPlannerConfig& cfg) noexcept {
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

} // namespace

void register_enhanced_weapon_system(talos::Scheduler& scheduler, WeaponControllerConfig&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::fixed_rate<250>>(
        "enhanced_weapon_control",
        [controller = std::shared_ptr<EnhancedWeaponController>{}, last_solver_hash = size_t{0}](
            talos::res<L4::L4Config> l4_config, talos::res<WeaponControllerConfig> config,
            talos::spmc<fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTopic> weapon_out) mutable {
            const size_t current_solver_hash = solver_config_hash(l4_config->mpc_planner);

            if (!controller) {
                auto comp_type = l4_config->mpc_planner.compensator_type;
                auto solver    = fcs::L4::trajectory::create_solver(
                    comp_type, l4_config->mpc_planner.trajectory);
                controller = std::make_shared<EnhancedWeaponController>(*config, std::move(solver));
                last_solver_hash = current_solver_hash;
            }

            if (config.is_changed()) {
                controller->set_config(*config);
            }

            if (l4_config.is_changed() && current_solver_hash != last_solver_hash) {
                controller->sync_config(
                    fcs::L4::trajectory::create_solver(
                        l4_config->mpc_planner.compensator_type,
                        l4_config->mpc_planner.trajectory));
                last_solver_hash = current_solver_hash;
            }

            auto plan = plan_in.read();
            if (!plan) {
                return;
            }

            double target_distance   = plan->distance;
            WeaponCommand weapon_cmd = controller->generate_command(
                *plan, target_distance, l4_config->mpc_planner.trajectory.bullet_speed);
            weapon_out.write(weapon_cmd);
        });
}

} // namespace fcs::L5
