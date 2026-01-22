#include "fcs/L3_estimation/energy_meter/energy_meter_systems.hpp"

#include "fcs/L2_perception/rune/types.hpp"
#include "fcs/L3_estimation/energy_meter_solver/angle_fitter.hpp"
#include "fcs/L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "fcs/L3_estimation/energy_meter_solver/types.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/time.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"
#include "toml_helper.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace fcs::energy_meter {

std::expected<void, std::string> EnergyMeterL3Config::read_from(const toml::table& tbl) noexcept {
    // TODO: use READ macro for a robust parsing
    if (const toml::node* node = tbl.get("tracker")) {
        const auto tracker_tbl = node->as_table();
        if (!tracker_tbl) {
            return std::unexpected("tracker: expected table");
        }

        if (auto v = toml_helper::take_optional<int>(*tracker_tbl, "lost_thres"); !v) {
            return std::unexpected(v.error());
        } else if (*v) {
            tracker.lost_thres = **v;
        }

        if (auto v = toml_helper::take_optional<int>(*tracker_tbl, "tracking_thres"); !v) {
            return std::unexpected(v.error());
        } else if (*v) {
            tracker.tracking_thres = **v;
        }

        if (auto v = toml_helper::take_optional<double>(*tracker_tbl, "matcher_gate"); !v) {
            return std::unexpected(v.error());
        } else if (*v) {
            tracker.matcher_gate = **v;
        }

        if (const toml::node* energy_node = tracker_tbl->get("energy_params")) {
            const auto energy_tbl = energy_node->as_table();
            if (!energy_tbl) {
                return std::unexpected("energy_params: expected table");
            }

            if (auto v = toml_helper::take_optional<double>(*energy_tbl, "sigma_xy"); !v) {
                return std::unexpected(v.error());
            } else if (*v) {
                tracker.energy_params.sigma_xy = **v;
            }

            if (auto v = toml_helper::take_optional<double>(*energy_tbl, "sigma_z"); !v) {
                return std::unexpected(v.error());
            } else if (*v) {
                tracker.energy_params.sigma_z = **v;
            }

            if (auto v = toml_helper::take_optional<double>(*energy_tbl, "sigma_roll"); !v) {
                return std::unexpected(v.error());
            } else if (*v) {
                tracker.energy_params.sigma_roll = **v;
            }

            if (auto v = toml_helper::take_optional<double>(*energy_tbl, "radius"); !v) {
                return std::unexpected(v.error());
            } else if (*v) {
                tracker.energy_params.radius = **v;
            }

            if (auto v = toml_helper::take_optional<double>(*energy_tbl, "meas_dist_k"); !v) {
                return std::unexpected(v.error());
            } else if (*v) {
                tracker.energy_params.meas_dist_k = **v;
            }

            if (auto unread = toml_helper::ensure_all_read(*energy_tbl, "energy_params"); !unread) {
                return std::unexpected(unread.error());
            }

            (void)const_cast<toml::table&>(*tracker_tbl).erase("energy_params");
        }

        if (auto unread = toml_helper::ensure_all_read(*tracker_tbl, "tracker"); !unread) {
            return std::unexpected(unread.error());
        }

        (void)const_cast<toml::table&>(tbl).erase("tracker");
    }

    READ_OPT(tbl, smooth_alpha);
    READ_OPT(tbl, min_data_points);
    READ_OPT(tbl, control_delay_s);
    READ_OPT(tbl, is_big_rune);
    return {};
}

namespace {

constexpr auto ns_to_sec(uint64_t ns) -> double { return static_cast<double>(ns) * 1e-9; }

} // namespace

void register_energy_meter_systems(::talos::Scheduler& scheduler, EnergyMeterL3Config&& config) {
    scheduler.world().insert_resource(std::move(config));

    scheduler.world().insert_resource(std::make_shared<::energy_meter::Tracker>());
    scheduler.world().insert_resource(
        std::make_shared<::energy_meter::AngleFitter>(::energy_meter::EnergyMeterConfig{
            .rune_type       = config.is_big_rune ? "big" : "small",
            .predict_time    = 0.0,
            .gimbal_frame    = "",
            .world_frame     = "",
            .smooth_alpha    = config.smooth_alpha,
            .min_data_points = config.min_data_points,
            .control_delay   = config.control_delay_s,
        }));

    scheduler.add_system<talos::pool_compute>(
        "l3_energy_meter",
        [last_timestamp_ns = uint64_t{0}](
            talos::res<EnergyMeterL3Config> cfg,
            talos::res_mut<std::shared_ptr<::energy_meter::Tracker>> tracker,
            talos::res_mut<std::shared_ptr<::energy_meter::AngleFitter>> fitter,
            talos::spmc<fcs::rune::RuneObservation, RuneObservationChannelTag> obs_in,
            talos::spmc_mut<EnergyMeterState, EnergyMeterStateChannelTag> state_out) mutable {
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

            (*fitter)->setFilteredVelocity(v_roll);
            ::energy_meter::AngleObservation ao;
            ao.timestamp        = timestamp_s;
            ao.absolute_angle   = roll;
            ao.continuous_angle = roll;
            ao.velocity         = v_roll;
            ao.blade_offset     = (*tracker)->getCurrentBladeId();
            (void)(*fitter)->addObservation(ao);

            if (cfg->is_big_rune) {
                (void)(*fitter)->fitAngleModelLSQ();
                (*fitter)->applyPosteriorConstraints();
            } else {
                (void)(*fitter)->fitLinearModelPublic();
            }

            const auto model = (*fitter)->getModel();

            const uint64_t now_ns           = fcs::clock::now_ns();
            const double image_delay_s      = ns_to_sec(now_ns - obs->timestamp_ns);
            const double prediction_delay_s = image_delay_s + cfg->control_delay_s;
            const double predict_dt_s       = std::max(0.0, prediction_delay_s);

            const Eigen::Quaterniond q = obs->target_quats_odom[0].normalized();
            const Eigen::Matrix3d rot  = q.toRotationMatrix();
            const Eigen::Vector3d u    = rot.col(2);
            const Eigen::Vector3d v    = rot.col(1);

            const Eigen::Vector3d rel = obs->target_positions_odom[0] - obs->r_center_odom;
            const double radius       = rel.norm();

            const double current_angle  = std::atan2(rel.dot(v), rel.dot(u));
            const double predicted_roll = current_angle - v_roll * predict_dt_s;

            const Eigen::Vector3d predicted_pos = obs->r_center_odom
                                                + radius * std::cos(predicted_roll) * u
                                                + radius * std::sin(predicted_roll) * v;

            EnergyMeterState out;
            out.timestamp_ns = obs->timestamp_ns;
            out.tracking_valid =
                ((*tracker)->state == ::energy_meter::Tracker::TRACKING
                 || (*tracker)->state == ::energy_meter::Tracker::TEMP_LOST);
            out.r_center_odom           = obs->r_center_odom;
            out.radius                  = radius;
            out.roll                    = roll;
            out.v_roll                  = v_roll;
            out.blade_id                = (*tracker)->getCurrentBladeId();
            out.predicted_position_odom = predicted_pos;
            out.predicted_future_ns = obs->timestamp_ns + static_cast<uint64_t>(predict_dt_s * 1e9);
            out.is_big_rune         = model.is_big_rune;
            out.model_valid         = model.is_valid;
            out.fit_error           = model.fit_error;

            state_out.write(out);
        });
}

} // namespace fcs::energy_meter
