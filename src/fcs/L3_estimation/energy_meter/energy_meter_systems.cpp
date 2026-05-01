#include "L3_estimation/energy_meter/energy_meter_systems.hpp"

#include "L2_perception/rune/types.hpp"
#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "L3_estimation/energy_meter_solver/types.hpp"
#include "L3_estimation/energy_meter_solver/voter.hpp"
#include "core/channel_topics.hpp"
#include "scheduler/scheduler.hpp"

#include <cmath>
#include <cstdint>
#include <memory>

#include <spdlog/spdlog.h>

namespace fcs::energy_meter {

void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config) {
    scheduler.world().insert_resource(std::move(config));

    scheduler.world().insert_resource(std::make_shared<::energy_meter::Tracker>());

    scheduler.add_system<talos::pool_compute>(
        "l3_energy_meter",
        [last_timestamp_ns = uint64_t{0}, tracking_frames = int{0},
         voter = ::energy_meter::Voter{}](
            talos::res<EnergyMeterL3Config> cfg,
            talos::res_mut<std::shared_ptr<::energy_meter::Tracker>> tracker,
            talos::subscribe<fcs::rune::RuneObservation, RuneObservationChannelTopic> obs_in,
            talos::publish<EnergyMeterState, EnergyMeterStateChannelTopic> state_out) mutable {
            auto obs = obs_in.read();
            if (!obs || !obs->valid || obs->target_positions_odom.empty()
                || obs->target_quats_odom.empty()) {
                return;
            }

            (*tracker)->set_params(cfg->tracker);

            double dt = 0.0;
            if (last_timestamp_ns > 0 && obs->timestamp_ns > last_timestamp_ns) {
                dt = static_cast<double>(obs->timestamp_ns - last_timestamp_ns) / 1e9;
            }
            last_timestamp_ns = obs->timestamp_ns;

            // Reset on large gap
            if (dt > 2.0) {
                (*tracker)->state = ::energy_meter::Tracker::IDLE;
                (*tracker)->reset_ekf();
                last_timestamp_ns = 0;
                tracking_frames   = 0;
                voter.reset();
                SPDLOG_WARN("energy_meter: large gap {:.1f}s, resetting", dt);
                return;
            }

            // ── EKF tracker step ──

            if (!(*tracker)->has_ekf()) {
                (*tracker)->first_meet_u(
                    obs->r_center_odom.translation(), obs->target_positions_odom[0],
                    obs->target_quats_odom[0]);
            }

            const double observed_radius =
                (obs->target_positions_odom[0] - obs->r_center_odom.translation()).norm();
            (*tracker)->setRadius(observed_radius);

            if (dt > 0.0) {
                (*tracker)->step(
                    dt, obs->r_center_odom.translation(), obs->target_positions_odom,
                    obs->target_quats_odom);
            } else {
                (*tracker)->update(
                    obs->r_center_odom.translation(), obs->target_positions_odom,
                    obs->target_quats_odom);
            }

            // ── Big/small rune detection ──

            if (!voter.determined()) {
                if (const auto decision = voter.update(obs->target_positions_odom.size())) {
                    SPDLOG_INFO(
                        "靶数投票决策: {}, big={} small={}, frames={}",
                        decision->is_big ? "大符" : "小符", decision->big_votes,
                        decision->small_votes, decision->frames);
                }

                return;
            }

            // ── Output ──

            const auto& x       = (*tracker)->get_state();
            const double roll   = x(::energy_meter::THETA);
            const double yaw    = x(::energy_meter::YAW);
            const double pitch  = x(::energy_meter::PITCH);
            const double t_val  = x(::energy_meter::TAU);
            const double a_val  = (*tracker)->get_a();
            const double w_val  = (*tracker)->get_w();
            const double b_val  = (*tracker)->get_b();
            const int dir       = (*tracker)->get_dir();
            const double radius = (*tracker)->getRadius();

            const bool is_tracking =
                ((*tracker)->state == ::energy_meter::Tracker::TRACKING
                 || (*tracker)->state == ::energy_meter::Tracker::TEMP_LOST);

            if (is_tracking) {
                tracking_frames++;
            } else {
                tracking_frames = 0;
            }

            // Compute blade position from state (confirmed blade, not blade 0)
            const double blade_roll =
                roll + static_cast<double>((*tracker)->getCurrentBladeId()) * 2.0 * M_PI / 5.0;
            const double sr = std::sin(blade_roll);
            const double cr = std::cos(blade_roll);
            Eigen::Vector3d blade_local(0.0, -radius * sr, radius * cr);

            const double sp = std::sin(pitch);
            const double cp = std::cos(pitch);
            blade_local     = Eigen::Vector3d(
                blade_local.x() * cp + blade_local.z() * sp, blade_local.y(),
                -blade_local.x() * sp + blade_local.z() * cp);

            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const Eigen::Vector3d blade_offset(
                blade_local.x() * cy - blade_local.y() * sy,
                blade_local.x() * sy + blade_local.y() * cy, blade_local.z());

            const Eigen::Vector3d r_center(
                x(::energy_meter::XC), x(::energy_meter::YC), x(::energy_meter::ZC));

            const Eigen::Quaterniond yaw_quat(std::cos(yaw / 2), 0.0, 0.0, std::sin(yaw / 2));
            const Eigen::Quaterniond pitch_quat(std::cos(pitch / 2), 0.0, std::sin(pitch / 2), 0.0);
            const Eigen::Quaterniond roll_quat(
                std::cos(blade_roll / 2), std::sin(blade_roll / 2), 0.0, 0.0);

            EnergyMeterState out;
            out.timestamp_ns   = obs->timestamp_ns;
            out.tracking_valid = is_tracking;
            out.r_center_odom  = r_center;
            out.radius         = radius;
            out.yaw            = yaw;
            out.pitch          = pitch;
            out.roll           = roll;
            out.t              = t_val;
            out.a              = a_val;
            out.omega          = w_val;
            out.b              = b_val;
            out.blade_id       = (*tracker)->getCurrentBladeId();
            out.direction      = dir;
            out.model_valid    = is_tracking && tracking_frames > 5;
            out.is_big_rune    = voter.is_big();
            out.position       = r_center + blade_offset;
            out.quat           = yaw_quat * pitch_quat * roll_quat;

            state_out.write(out);
        });
}

} // namespace fcs::energy_meter
