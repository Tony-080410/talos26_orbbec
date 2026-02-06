#include "fcs/L4_planning/plan_mux/plan_mux_systems.hpp"

#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/core/channel_topics.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"
#include "toml_helper.hpp"

namespace fcs::L4 {

std::expected<void, std::string> PlanMuxConfig::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, mode);
    return {};
}

void register_plan_mux_systems(::talos::Scheduler& scheduler, PlanMuxConfig&& config) {
    scheduler.world().insert_resource(std::move(config));

    scheduler.add_system<talos::pool_compute>(
        "l4_plan_mux", [](talos::res<PlanMuxConfig> config,
                          talos::spmc<GimbalPlan, ArmorGimbalPlanChannelTopic> armor_plan_in,
                          talos::spmc<GimbalPlan, RuneGimbalPlanChannelTopic> rune_plan_in,
                          talos::spmc_mut<GimbalPlan, GimbalPlanChannelTopic> final_plan_out) {
            const auto armor_plan = armor_plan_in.read();
            const auto rune_plan  = rune_plan_in.read();

            const GimbalPlan* selected = nullptr;

            switch (config->mode) {
            case TargetMode::Armor: selected = armor_plan ? &*armor_plan : nullptr; break;
            case TargetMode::Rune: selected = rune_plan ? &*rune_plan : nullptr; break;
            case TargetMode::Auto:
                // Select plan with latest timestamp, preferring rune if timestamps are equal
                if (armor_plan && rune_plan) {
                    selected = (rune_plan->timestamp_ns >= armor_plan->timestamp_ns) ? &*rune_plan
                                                                                     : &*armor_plan;
                } else if (rune_plan) {
                    selected = &*rune_plan;
                } else if (armor_plan) {
                    selected = &*armor_plan;
                }
                break;
            default: break; // Unknown mode, no plan selected
            }

            if (selected != nullptr) {
                final_plan_out.write(*selected);
            }
        });
}

} // namespace fcs::L4
