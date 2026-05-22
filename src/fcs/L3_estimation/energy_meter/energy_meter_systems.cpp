#include "L3_estimation/energy_meter/energy_meter_systems.hpp"

#include "L2_perception/rune/types.hpp"
#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "L3_estimation/energy_meter_solver/types.hpp"
#include "L3_estimation/energy_meter_solver/voter.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "scheduler/scheduler.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <variant>

#include <spdlog/spdlog.h>

namespace fcs::energy_meter {

namespace {

template <typename TrackerT>
EnergyMeterState build_state_from_tracker(
    const TrackerT& tracker, uint64_t timestamp_ns, bool is_big, bool obs_valid) {
    const double roll       = tracker.current_roll();
    const double yaw        = tracker.current_yaw();
    const double pitch      = tracker.current_pitch();
    const double radius     = tracker.getRadius();
    const int blade_id      = tracker.getCurrentBladeId();
    const double blade_roll = roll + static_cast<double>(blade_id) * 2.0 * M_PI / 5.0;

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
        blade_local.x() * cy - blade_local.y() * sy, blade_local.x() * sy + blade_local.y() * cy,
        blade_local.z());

    EnergyMeterState out;
    out.timestamp_ns  = timestamp_ns;
    out.tracker_state = static_cast<TrackerState>(tracker.state);
    out.tracking_valid =
        tracker.state == TrackerT::TRACKING || tracker.state == TrackerT::TEMP_LOST;
    out.r_center_odom = tracker.current_center();
    out.radius        = radius;
    out.yaw           = yaw;
    out.pitch         = pitch;
    out.roll          = roll;
    out.t             = tracker.current_time();
    out.blade_id      = blade_id;
    out.direction     = tracker.getDirection();
    out.is_big_rune   = is_big;
    out.obs_valid     = obs_valid;
    out.position      = out.r_center_odom + blade_offset;
    out.quat          = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
             * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(blade_roll, Eigen::Vector3d::UnitX());

    if constexpr (std::is_same_v<TrackerT, ::energy_meter::BigTracker>) {
        const auto& fit = tracker.big_fit();
        out.a           = fit.a;
        out.omega       = fit.omega;
        out.b           = fit.b;
        if (fit.valid && std::abs(fit.omega) > 1e-9) {
            out.t = tracker.current_time() + fit.phase / fit.omega;
        }
        out.model_valid = tracker.model_valid();
    } else {
        out.a           = 0.0;
        out.omega       = ::energy_meter::FIXED_RUNE_SPEED;
        out.b           = ::energy_meter::FIXED_RUNE_SPEED;
        out.model_valid = tracker.model_valid();
    }

    return out;
}

EnergyMeterState build_state(
    const ::energy_meter::AnyTracker& tracker_var, uint64_t timestamp_ns, bool is_big,
    bool obs_valid) {
    return std::visit(
        [&](const auto& tk) -> EnergyMeterState {
            using T = std::decay_t<decltype(tk)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return EnergyMeterState{};
            } else {
                return build_state_from_tracker(tk, timestamp_ns, is_big, obs_valid);
            }
        },
        tracker_var);
}

} // namespace

void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.world().insert_resource(std::make_shared<::energy_meter::AnyTracker>());

    scheduler.add_system<talos::pool_compute>(
        "l3_energy_meter",
        [last_timestamp_ns = uint64_t{0}, tracking_frames = int{0},
         voter = ::energy_meter::Voter{}](
            talos::res<EnergyMeterL3Config> cfg,
            talos::res_mut<std::shared_ptr<::energy_meter::AnyTracker>> tracker_ptr,
            talos::subscribe<fcs::rune::RuneObservation, RuneObservationChannelTopic> obs_in,
            talos::publish<EnergyMeterState, EnergyMeterStateChannelTopic> state_out) mutable {
            auto obs = obs_in.read();

            const bool obs_valid = obs && obs->valid && !obs->target_positions_odom.empty()
                                && !obs->target_quats_odom.empty();

            const uint64_t frame_ns = obs ? obs->timestamp_ns : fcs::clock::now_ns();
            double dt               = 0.0;
            if (last_timestamp_ns > 0 && frame_ns > last_timestamp_ns) {
                dt = static_cast<double>(frame_ns - last_timestamp_ns) * 1e-9;
            }
            auto& tracker_var = **tracker_ptr;

            // IDLE 状态且无有效观测，直接返回
            bool is_idle = false;
            std::visit(
                [&](const auto& tk) {
                    using T = std::decay_t<decltype(tk)>;
                    if constexpr (std::is_same_v<T, std::monostate>) {
                        is_idle = true;
                    } else {
                        is_idle = (tk.state == T::IDLE);
                    }
                },
                tracker_var);

            if (is_idle && !obs_valid) {
                return;
            }

            last_timestamp_ns = frame_ns;

            // Reset on large gap
            if (dt > cfg->reset_vote_time) {
                **tracker_ptr     = ::energy_meter::AnyTracker{};
                last_timestamp_ns = 0;
                tracking_frames   = 0;
                voter.reset();
                SPDLOG_WARN("energy_meter: large gap {:.1f}s, resetting", dt);
                return;
            }

            // ── No valid observation: advance state machine, publish lost state ──
            if (!obs_valid) {
                std::visit(
                    [&](auto& tk) {
                        using T = std::decay_t<decltype(tk)>;
                        if constexpr (!std::is_same_v<T, std::monostate>) {
                            if (dt > 0.0) {
                                tk.predict(dt);
                                tk.update(Eigen::Vector3d::Zero(), {}, {});
                            }
                        }
                    },
                    tracker_var);

                if (!std::holds_alternative<std::monostate>(tracker_var) && voter.determined()) {
                    state_out.write(build_state(tracker_var, frame_ns, voter.is_big(), false));
                }
                return;
            }

            if (!voter.determined()) {
                if (const auto decision = voter.update(obs->target_positions_odom.size())) {
                    SPDLOG_INFO(
                        "靶数投票决策: {}, big={} small={}, frames={}",
                        decision->is_big ? "大符" : "小符", decision->big_votes,
                        decision->small_votes, decision->frames);

                    if (decision->is_big) {
                        tracker_var = ::energy_meter::BigTracker{};
                    } else {
                        tracker_var = ::energy_meter::SmallTracker{};
                    }
                }
            }

            if (std::holds_alternative<std::monostate>(tracker_var)) {
                return;
            }

            auto advance_tracker = [&](auto& tk, const auto& model_params) {
                using T = std::decay_t<decltype(tk)>;
                typename T::Params p;
                p.lost_thres               = cfg->lost_thres;
                p.tracking_thres           = cfg->tracking_thres;
                p.matcher_gate             = cfg->matcher_gate;
                p.angle_match_gate_rad     = cfg->angle_match_gate_rad;
                p.blade_unlock_frames      = cfg->blade_unlock_frames;
                p.direction_lock_frames    = cfg->direction_lock_frames;
                p.min_fit_samples          = cfg->min_fit_samples;
                p.max_fit_samples          = cfg->max_fit_samples;
                p.fit_iterations           = cfg->fit_iterations;
                p.fit_window_sec           = cfg->fit_window_sec;
                p.fit_inlier_threshold_rad = cfg->fit_inlier_threshold_rad;
                p.fit_penalty_a            = cfg->fit_penalty_a;
                p.fit_penalty_w            = cfg->fit_penalty_w;
                p.fit_temporal_weight      = cfg->fit_temporal_weight;
                p.fit_filter_alpha         = cfg->fit_filter_alpha;
                p.model_params             = model_params;
                tk.set_params(p);

                if (!tk.has_filter()) {
                    tk.first_meet_u(
                        obs->r_center_odom.translation(), obs->target_positions_odom.front(),
                        obs->target_quats_odom.front());
                }

                if (dt > 0.0) {
                    tk.step(
                        dt, obs->r_center_odom.translation(), obs->target_positions_odom,
                        obs->target_quats_odom);
                } else {
                    tk.update(
                        obs->r_center_odom.translation(), obs->target_positions_odom,
                        obs->target_quats_odom);
                }
            };

            if (auto* tk = std::get_if<::energy_meter::BigTracker>(&tracker_var)) {
                advance_tracker(*tk, cfg->big_model);
            } else if (auto* tk = std::get_if<::energy_meter::SmallTracker>(&tracker_var)) {
                advance_tracker(*tk, cfg->small_model);
            }

            state_out.write(build_state(tracker_var, frame_ns, voter.is_big(), true));
        });
}

} // namespace fcs::energy_meter
