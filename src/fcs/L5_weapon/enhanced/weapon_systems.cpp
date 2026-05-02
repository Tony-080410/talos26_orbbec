#include "L5_weapon/enhanced/weapon_systems.hpp"

#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/control_intent.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/enhanced/trajectory_optimizer.hpp"
#include "L5_weapon/fire_control.hpp"
#include "L5_weapon/fire_decision.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L5 {

namespace {
/// std::visit helper for variant exhaustiveness.
template <typename... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

/// Apply fire gate: compute is_on_target and stamp it into the command.
[[nodiscard]] WeaponCommand fire_gate(
    WeaponCommand cmd, const FireDecisionConfig& fire_cfg, double cur_yaw,
    double cur_pitch) noexcept {
    auto result = is_on_target(
        fire_cfg, cur_yaw, -cur_pitch, cmd.plan_yaw, cmd.plan_pitch, cmd.plan_distance);
    cmd.fire                 = result.fire;
    cmd.yaw_error            = result.yaw_error;
    cmd.pitch_error          = result.pitch_error;
    cmd.shooting_range_yaw   = result.shooting_range_yaw;
    cmd.shooting_range_pitch = result.shooting_range_pitch;
    cmd.ref_yaw              = cmd.plan_yaw;
    cmd.ref_pitch            = cmd.plan_pitch;
    return cmd;
}

/// Build a fallback passthrough command from a TrackCommand's trajectory center.
[[nodiscard]] WeaponCommand
    track_fallback(const L4::TrackCommand& track, uint64_t command_timestamp_ns) noexcept {
    const auto aim = track.trajectory.center_aim_point();
    WeaponCommand cmd;
    cmd.timestamp_ns      = command_timestamp_ns;
    cmd.plan_timestamp_ns = track.timestamp_ns;
    cmd.plan_yaw          = aim.yaw;
    cmd.plan_pitch        = aim.pitch;
    cmd.plan_distance     = aim.distance;
    cmd.yaw               = aim.yaw;
    cmd.pitch             = aim.pitch;
    cmd.distance          = aim.distance;
    return cmd;
}
} // namespace

void register_enhanced_weapon_system(talos::Scheduler& scheduler, L5Config&& config) {
    const auto& trajectory_cfg = scheduler.world().get_res<L4::L4Config>()->reference_trajectory;
    scheduler.world().insert_resource(config);
    auto optimizer =
        std::make_shared<TinyMpcTrajectoryOptimizer>(config.mpc_weapon, trajectory_cfg);

    scheduler.add_system<talos::fixed_rate<250>>(
        "enhanced_weapon_control",
        [optimizer, fire_cfg = config.fire_decision](
            talos::spmc<fcs::L4::ControlIntent, ControlIntentChannelTopic> intent_in,
            talos::res<fast_tf::CoordinateSystem> tf_buffer,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTopic> weapon_out) mutable {
            auto intent = intent_in.read();
            if (!intent) {
                return;
            }

            const uint64_t now = fcs::clock::now_ns();

            std::visit(
                overloaded{
                    [&](const L4::TrackCommand& cmd) {
                        auto opt         = optimizer->optimize(cmd, now);
                        WeaponCommand wc = opt ? *opt : track_fallback(cmd, now);
                        auto tf          = L4::lookup_gimbal_transform(*tf_buffer, now);
                        if (tf) {
                            auto [r, p, y] = tf->euler_rot().rpy();
                            wc             = fire_gate(wc, fire_cfg, y, p);
                        }
                        weapon_out.write(wc);
                        if (!opt) {
                            SPDLOG_WARN("MPC optimization failed: {}", opt.error());
                        }
                    },
                    [&](const L4::ShotCommand& cmd) {
                        auto wc = optimizer->passthrough(cmd, now);
                        auto tf = L4::lookup_gimbal_transform(*tf_buffer, now);
                        if (tf) {
                            auto [r, p, y] = tf->euler_rot().rpy();
                            wc             = fire_gate(wc, fire_cfg, y, p);
                        } else {
                            SPDLOG_ERROR("Gimbal transform lookup failed: {}", tf.error());
                        }
                        weapon_out.write(wc);
                    },
                    [&](const L4::HoldCommand&) {
                        // No target — don't emit a weapon command.
                        // Gimbal stays at current position.
                        auto wc         = WeaponCommand{};
                        wc.timestamp_ns = now;
                        wc.distance     = -1.0;
                        weapon_out.write(wc);
                    },
                },
                *intent);
        });
}

} // namespace fcs::L5
