#include "L3_estimation/energy_meter/energy_meter_systems.hpp"

#include "L2_perception/rune/types.hpp"
#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "L3_estimation/energy_meter_solver/types.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "scheduler/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include <spdlog/spdlog.h>

namespace fcs::energy_meter {

namespace {

constexpr auto ns_to_sec(uint64_t ns) -> double { return static_cast<double>(ns) * 1e-9; }

} // namespace

void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config) {
    scheduler.world().insert_resource(std::move(config));

    scheduler.world().insert_resource(std::make_shared<::energy_meter::Tracker>());
    scheduler.world().insert_resource(std::make_shared<::energy_meter::AngleFitter>(config.fitter));

    scheduler.add_system<talos::pool_compute>(
        "l3_energy_meter",
        [last_timestamp_ns = uint64_t{0}, start_time_s = double{0.0}, has_start_time = false,
         fit_counter = int{0}](
            talos::res<EnergyMeterL3Config> cfg,
            talos::res_mut<std::shared_ptr<::energy_meter::Tracker>> tracker,
            talos::res_mut<std::shared_ptr<::energy_meter::AngleFitter>> fitter,
            talos::subscribe<fcs::rune::RuneObservation, RuneObservationChannelTopic> obs_in,
            talos::publish<EnergyMeterState, EnergyMeterStateChannelTopic> state_out) mutable {
            auto obs = obs_in.read();
            if (!obs || !obs->valid || obs->target_positions_odom.empty()
                || obs->target_quats_odom.empty()) {
                return;
            }

            (*tracker)->set_params(cfg->tracker);

            const double timestamp_s = ns_to_sec(obs->timestamp_ns);
            double dt                = 0.0;
            if (last_timestamp_ns > 0 && obs->timestamp_ns > last_timestamp_ns) {
                dt = static_cast<double>(obs->timestamp_ns - last_timestamp_ns) / 1e9;
            }
            last_timestamp_ns = obs->timestamp_ns;

            // Reset on large gap
            if (dt > 2.0) {
                (*tracker)->state = ::energy_meter::Tracker::IDLE;
                (*tracker)->energy_ukf.reset();
                last_timestamp_ns = 0;
                (*fitter)->reset();
                has_start_time = false;
                fit_counter    = 0;
                SPDLOG_WARN("energy_meter: large gap {:.1f}s, resetting", dt);
                return;
            }

            // ── ISRCKF tracker step ──

            if (!(*tracker)->energy_ukf.has_value()) {
                (*tracker)->first_meet_u(
                    obs->r_center_odom, obs->target_positions_odom[0], obs->target_quats_odom[0]);
            }

            if (dt > 0.0) {
                (*tracker)->step(dt, obs->target_positions_odom, obs->target_quats_odom);
            } else {
                (*tracker)->update(obs->target_positions_odom, obs->target_quats_odom);
            }

            const auto& x       = (*tracker)->get_state();
            const double roll   = x(::energy_meter::ROLL);
            const double v_roll = x(::energy_meter::V_ROLL);

            // ── Timing ──

            if (!has_start_time) {
                has_start_time = true;
                start_time_s   = timestamp_s;
            }
            const double relative_time = timestamp_s - start_time_s;

            // ── Direction voting (rm_power_rune semantics) ──

            (*fitter)->updateDirection(roll);

            // ── Fitting data ──

            (*fitter)->addObservation(relative_time, std::abs(roll));

            // ── Fitting ──

            const auto direction = (*fitter)->getDirection();

            // Only fit when direction is resolved (mirrors rm_power_rune: skip predict if
            // UNKNOWN)
            if (direction != ::energy_meter::Direction::UNKNOWN) {
                // Throttle fitting: every ~10 frames (~333ms at 30fps, matching rm_power_rune's
                // background thread interval)
                fit_counter++;
                if (fit_counter >= 10
                    && (*fitter)->getDataPointCount() >= cfg->fitter.min_fit_data_size) {
                    fit_counter = 0;
                    if (cfg->is_big_rune) {
                        (*fitter)->fitBigRune();
                    } else {
                        (*fitter)->fitSmallRune();
                    }
                }
            }

            // ── Prediction ──

            const uint64_t now_ns           = fcs::clock::now_ns();
            const double image_delay_s      = ns_to_sec(now_ns - obs->timestamp_ns);
            const double prediction_delay_s = image_delay_s + cfg->control_delay_s;
            const double predict_dt_s       = std::max(0.0, prediction_delay_s);

            // Compute current angle from observation (same as before)
            const Eigen::Quaterniond q = obs->target_quats_odom[0].normalized();
            const Eigen::Matrix3d rot  = q.toRotationMatrix();
            const Eigen::Vector3d u    = rot.col(2);
            const Eigen::Vector3d v    = rot.col(1);

            const Eigen::Vector3d rel  = obs->target_positions_odom[0] - obs->r_center_odom;
            const double radius        = rel.norm();
            const double current_angle = std::atan2(rel.dot(v), rel.dot(u));

            // Compute predicted roll using rm_power_rune model semantics
            double predicted_roll = current_angle;

            if (direction == ::energy_meter::Direction::STABLE) {
                // No rotation
            } else if (cfg->is_big_rune && (*fitter)->isModelValid()) {
                // Big rune: model-based prediction
                double delta_angle =
                    (*fitter)->predictDeltaAngle(relative_time, relative_time + predict_dt_s);

                if (direction == ::energy_meter::Direction::ANTI_CLOCKWISE) {
                    delta_angle = -delta_angle;
                }
                predicted_roll = current_angle + delta_angle;
            } else if (!cfg->is_big_rune) {
                // Small rune: constant speed rotation
                double rotation_speed = cfg->fitter.small_rune_rotation_speed;
                double delta_angle    = rotation_speed * predict_dt_s;
                if (direction == ::energy_meter::Direction::ANTI_CLOCKWISE) {
                    delta_angle = -delta_angle;
                }
                predicted_roll = current_angle + delta_angle;
            } else {
                // Big rune but model not yet valid: fall back to linear prediction
                predicted_roll = current_angle + v_roll * predict_dt_s;
            }

            const Eigen::Vector3d predicted_pos = obs->r_center_odom
                                                + radius * std::cos(predicted_roll) * u
                                                - radius * std::sin(predicted_roll) * v;

            // ── Output ──

            const auto model = (*fitter)->getModel();

            EnergyMeterState out;
            out.timestamp_ns = obs->timestamp_ns;
            out.tracking_valid =
                ((*tracker)->state == ::energy_meter::Tracker::TRACKING
                 || (*tracker)->state == ::energy_meter::Tracker::TEMP_LOST);
            out.r_center_odom           = obs->r_center_odom.eval();
            out.radius                  = radius;
            out.roll                    = roll;
            out.v_roll                  = v_roll;
            out.blade_id                = (*tracker)->getCurrentBladeId();
            out.predicted_position_odom = predicted_pos;
            out.predicted_future_ns = obs->timestamp_ns + static_cast<uint64_t>(predict_dt_s * 1e9);
            out.is_big_rune         = cfg->is_big_rune;
            out.model_valid         = model.is_valid;
            out.fit_error           = model.fit_error;

            state_out.write(out);
        });
}

} // namespace fcs::energy_meter
