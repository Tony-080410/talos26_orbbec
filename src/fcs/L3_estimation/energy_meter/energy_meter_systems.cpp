#include "L3_estimation/energy_meter/energy_meter_systems.hpp"

#include "L2_perception/rune/types.hpp"
#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "L3_estimation/energy_meter_solver/types.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "scheduler/scheduler.hpp"

#include <cmath>
#include <memory>

#include <spdlog/spdlog.h>

namespace fcs::energy_meter {

namespace {

constexpr auto ns_to_sec(uint64_t ns) -> double { return static_cast<double>(ns) * 1e-9; }

} // namespace

static constexpr int VOTE_MIN_VALID  = 8;  // 最少有效投票数
static constexpr int VOTE_MAX_FRAMES = 20; // 最多等待帧数
void register_energy_meter_systems(talos::Scheduler& scheduler, EnergyMeterL3Config&& config) {
    scheduler.world().insert_resource(std::move(config));

    scheduler.world().insert_resource(std::make_shared<::energy_meter::Tracker>());
    scheduler.world().insert_resource(std::make_shared<::energy_meter::AngleFitter>(config.fitter));

    scheduler.add_system<talos::pool_compute>(
        "l3_energy_meter",
        [last_timestamp_ns = uint64_t{0}, start_time_s = double{0.0}, has_start_time = false,
         fit_counter = int{0}, model_type_determined = false, vote_big = 0, vote_small = 0,
         detection_frame_count = 0, is_big = true](
            talos::res<EnergyMeterL3Config> cfg,
            talos::res_mut<std::shared_ptr<::energy_meter::Tracker>> tracker,
            talos::res_mut<std::shared_ptr<::energy_meter::AngleFitter>> fitter,
            talos::subscribe<fcs::rune::RuneObservation, RuneObservationChannelTopic> obs_in,
            talos::publish<EnergyMeterState, EnergyMeterStateChannelTopic> state_out) mutable {
            auto obs = obs_in.read();
            if (!obs || !obs->valid || obs->target_positions_odom.empty()
                || obs->target_quats_odom.empty()) {
                EnergyMeterState out;
                out.timestamp_ns       = obs->timestamp_ns;
                out.tracking_valid     = false;
                out.model_valid        = false;
                out.blade_id_confirmed = false;

                state_out.write(out);
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
                has_start_time        = false;
                fit_counter           = 0;
                vote_big              = 0;
                vote_small            = 0;
                model_type_determined = false;
                detection_frame_count = 0;
                is_big                = true;
                (*tracker)->reset_blade_debounce();
                SPDLOG_WARN("energy_meter: large gap {:.1f}s, resetting", dt);
                return;
            }

            // ── ISRCKF tracker step ──

            if (!(*tracker)->energy_ukf.has_value()) {
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

            const auto& x       = (*tracker)->get_state();
            const double roll   = x(::energy_meter::ROLL);
            const double v_roll = x(::energy_meter::V_ROLL);
            const double yaw    = x(::energy_meter::YAW);
            const double pitch  = x(::energy_meter::PITCH);

            // ── Timing ──

            if (!has_start_time) {
                has_start_time = true;
                start_time_s   = timestamp_s;
            }
            const double relative_time = timestamp_s - start_time_s;

            // ── Fitting data ──

            (*fitter)->addObservation(relative_time, std::abs(roll));

            if (!model_type_determined) {
                detection_frame_count++;
                const int num_targets = static_cast<int>(obs->target_positions_odom.size());

                // ---- 靶数投票 ----
                if (num_targets == 2) {
                    vote_big++;
                } else if (num_targets == 1) {
                    vote_small++;
                }
                // num_targets == 0: 漏检，不投票

                const int total_votes = vote_big + vote_small;
                const bool vote_ready =
                    (total_votes >= VOTE_MIN_VALID) || (detection_frame_count >= VOTE_MAX_FRAMES);

                // ---- 投票决策：启动 UKF ----
                if (vote_ready && !model_type_determined) {
                    model_type_determined = true;

                    is_big = (vote_big > vote_small);

                    SPDLOG_INFO(
                        "靶数投票决策: {}, big={} small={}, frames={}", is_big ? "大符" : "小符",
                        vote_big, vote_small, detection_frame_count);
                }

                // ── Direction voting (rm_power_rune semantics) ──
                (*fitter)->updateDirection(roll);

                return;
            }

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
                    if (is_big) {
                        (*fitter)->fitBigRune();
                    } else {
                        (*fitter)->fitSmallRune();
                    }
                }
            }

            int dir;
            if (direction == ::energy_meter::Direction::CLOCKWISE) {
                dir = 1;
            } else if (direction == ::energy_meter::Direction::ANTI_CLOCKWISE) {
                dir = -1;
            } else {
                dir = 0;
            }

            // ── Output ──

            const auto model = (*fitter)->getModel();

            EnergyMeterState out;
            out.timestamp_ns = obs->timestamp_ns;
            out.tracking_valid =
                ((*tracker)->state == ::energy_meter::Tracker::TRACKING
                 || (*tracker)->state == ::energy_meter::Tracker::TEMP_LOST);
            out.r_center_odom = Eigen::Vector3d(
                x(::energy_meter::XC), x(::energy_meter::YC),
                x(::energy_meter::ZC)); // obs->r_center_odom.translation();

            // const Eigen::Vector3d ukf_center(
            //     x(::energy_meter::XC), x(::energy_meter::YC), x(::energy_meter::ZC));
            // const Eigen::Vector3d diff = ukf_center - obs->r_center_odom.translation();
            // const double diff_norm     = diff.norm();

            // SPDLOG_INFO(
            //     "UKF: ({}, {}, {}) OBS: ({}, {}, {})", ukf_center.x(), ukf_center.y(),
            //     ukf_center.z(), obs->r_center_odom.translation().x(),
            //     obs->r_center_odom.translation().y(), obs->r_center_odom.translation().z());

            // SPDLOG_INFO("DIFF: {}m", diff_norm);

            //  SPDLOG_INFO("yaw: {}m", x(::energy_meter::YAW));

            // 使用tracker动态计算的radius（基于观测更新），而非硬编码0.7
            // 这对上下位置（roll≈±90°）的精度尤其重要，因为Z坐标=r_center_z±radius
            const double radius    = (*tracker)->getRadius();
            out.radius             = radius;
            out.yaw                = yaw;
            out.pitch              = pitch;
            out.roll               = roll;
            out.v_roll             = v_roll;
            out.blade_id           = (*tracker)->getCurrentBladeId();
            out.blade_id_confirmed = (*tracker)->isBladeIdConfirmed();

            // 基于 UKF 的 roll、pitch、yaw 计算预测位置
            // 叶片在竖直平面 (Y-Z) 内绕水平轴 (X) 旋转 roll 角度
            const double sr = std::sin(roll);
            const double cr = std::cos(roll);
            Eigen::Vector3d blade_local(0.0, -radius * sr, radius * cr);

            // 应用 pitch 旋转 (绕 Y 轴)

            const double sp = std::sin(pitch);
            const double cp = std::cos(pitch);
            blade_local     = Eigen::Vector3d(
                blade_local.x() * cp + blade_local.z() * sp, blade_local.y(),
                -blade_local.x() * sp + blade_local.z() * cp);

            // 应用 yaw 旋转 (绕 Z 轴)，再平移到圆心
            const double cy = std::cos(yaw);
            const double sy = std::sin(yaw);
            const Eigen::Vector3d blade_offset(
                blade_local.x() * cy - blade_local.y() * sy,
                blade_local.x() * sy + blade_local.y() * cy, blade_local.z());
            out.position = out.r_center_odom + blade_offset;

            // 预测姿态: yaw(绕Z) * pitch(绕Y) * roll(绕X)
            const Eigen::Quaterniond yaw_quat(std::cos(yaw / 2), 0.0, 0.0, std::sin(yaw / 2));
            const Eigen::Quaterniond pitch_quat(std::cos(pitch / 2), 0.0, std::sin(pitch / 2), 0.0);
            const Eigen::Quaterniond roll_quat(std::cos(roll / 2), std::sin(roll / 2), 0.0, 0.0);
            out.quat = yaw_quat * pitch_quat * roll_quat;

            out.relative_time = relative_time;
            out.is_big_rune   = is_big;
            if (relative_time < cfg->min_fit_time) {
                out.model_valid = false;
            } else {
                out.model_valid = model.is_valid;
            }

            out.fit_error = model.fit_error;

            out.direction = dir;

            out.sin_amplitude  = model.a;
            out.sin_omega      = model.omega;
            out.sin_offset     = model.b;
            out.sin_phase      = model.t0;
            out.sin_const_term = model.c;

            state_out.write(out);
        });
}

} // namespace fcs::energy_meter
