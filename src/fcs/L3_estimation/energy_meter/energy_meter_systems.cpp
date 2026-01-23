#include "fcs/L3_estimation/energy_meter/energy_meter_systems.hpp"

#include "fcs/L2_perception/rune/types.hpp"
#include "fcs/L3_estimation/energy_meter_solver/angle_fitter.hpp"
#include "fcs/L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"
#include "fcs/L3_estimation/energy_meter_solver/types.hpp"
#include "fcs/core/channel_topics.hpp"
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
    READ_OPT(tbl, angle_fitter);
    READ_OPT(tbl, tracker);
    READ_OPT(tbl, control_delay_s);
    READ_OPT(tbl, is_big_rune);
    return {};
}

std::expected<void, std::string> AngleFitterConfig::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, smooth_alpha);
    READ_OPT(tbl, min_data_points);
    return {};
}

} // namespace fcs::energy_meter

// Define Tracker::Params::read_from in the correct namespace
namespace energy_meter {

std::expected<void, std::string> Tracker::Params::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, lost_thres);
    READ_OPT(tbl, tracking_thres);
    READ_OPT(tbl, matcher_gate);
    READ_OPT(tbl, energy_params);
    return {};
}

std::expected<void, std::string> EnergyMeter::Params::read_from(const toml::table& tbl) noexcept {
    READ_OPT(tbl, sigma_xy);
    READ_OPT(tbl, sigma_z);
    READ_OPT(tbl, sigma_roll);
    READ_OPT(tbl, radius);
    READ_OPT(tbl, meas_dist_k);
    return {};
}

} // namespace energy_meter

namespace fcs::energy_meter {

namespace {

constexpr auto ns_to_sec(uint64_t ns) -> double { return static_cast<double>(ns) * 1e-9; }

} // namespace

// 模型类型判断状态
bool model_type_determined_           = false; // 是否已确定模型类型
int detection_frame_count_            = 0;     // 检测帧计数
static constexpr int DETECTION_FRAMES = 20;    // 前20帧用于判断类型

// 防误判机制
static constexpr double SWITCH_THRESHOLD    = 1.5; // v
int switch_cooldown_count_                  = 0;   // 切换后的冷却帧计数
static constexpr int SWITCH_COOLDOWN_FRAMES = 30;  // 切换后等待30帧再检测
int frame_fps_                              = 0;
static constexpr int FPS_THRESHOLD          = 100;

// 速度统计（用于判断大小能量机关）
int velocity_frame_count_                  = 0;
double velocity_max_                       = -std::numeric_limits<double>::max();
double velocity_min_                       = std::numeric_limits<double>::max();
static constexpr int VELOCITY_SKIP_FRAMES_ = 5;

void resetVelocityStats() {
    velocity_frame_count_ = 0;
    velocity_max_         = -std::numeric_limits<double>::max();
    velocity_min_         = std::numeric_limits<double>::max();
}

void register_energy_meter_systems(::talos::Scheduler& scheduler, EnergyMeterL3Config&& config) {
    scheduler.world().insert_resource(std::move(config));

    scheduler.world().insert_resource(std::make_shared<::energy_meter::Tracker>());
    scheduler.world().insert_resource(
        std::make_shared<::energy_meter::AngleFitter>(
            config.angle_fitter.smooth_alpha, config.angle_fitter.min_data_points));

    scheduler.add_system<talos::pool_compute>(
        "l3_energy_meter",
        [last_timestamp_ns = uint64_t{0}](
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

            // 如果间隔超过 2s，重置所有状态
            if (dt > 2) {
                (*tracker)->state = ::energy_meter::Tracker::IDLE;
                (*tracker)->energy_ukf.reset();
                last_timestamp_ns = 0;
                // 重置拟合器和类型判断状态
                (*fitter)->reset();
                model_type_determined_ = false;
                detection_frame_count_ = 0;
                switch_cooldown_count_ = 0;
                resetVelocityStats();
                frame_fps_ = 0;
                std::cerr << "时间间隔过大, " << dt << "重置类型判断" << std::endl;
                return;
            }

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

            // 速度统计：从第5帧开始记录最大值和最小值
            velocity_frame_count_++;
            if (velocity_frame_count_ > VELOCITY_SKIP_FRAMES_) {
                double vel = v_roll;

                if (vel > velocity_max_)
                    velocity_max_ = vel;
                if (vel < velocity_min_)
                    velocity_min_ = vel;
            }
            // 拟合逻辑
            if ((*fitter)->getDataPointCount() >= 15) {

                if (!model_type_determined_) {
                    // 前20帧：判断模型类型
                    detection_frame_count_++;

                    (*fitter)->fitAngleModelLSQ();
                    (*fitter)->applyPosteriorConstraints();

                    // 通过速度变化范围判断大小符
                    // 小符: 速度恒定，max-min 很小
                    // 大符: 速度变化大，max-min 较大
                    double velocity_range                     = velocity_max_ - velocity_min_;
                    constexpr double VELOCITY_RANGE_THRESHOLD = 0.5; // 速度变化阈值
                    bool velocity_suggests_big_rune =
                        (std::abs(velocity_range) > VELOCITY_RANGE_THRESHOLD);

                    if (!velocity_suggests_big_rune) {
                        // 速度变化小，判定为小符
                        (*fitter)->fitLinearModelPublic();
                    }

                    // 20帧后锁定类型
                    if (detection_frame_count_ >= DETECTION_FRAMES) {
                        model_type_determined_ = true;
                        auto final_model       = (*fitter)->getModel();
                        std::cerr << "模型类型已确定: " << final_model.is_big_rune
                                  << ", 速度范围: " << velocity_range << "max= " << velocity_max_
                                  << "min= " << velocity_min_ << std::endl;
                        resetVelocityStats();
                        frame_fps_ = 0;
                    }

                } else {
                    // 类型已确定，只更新参数不切换类型
                    auto model = (*fitter)->getModel();
                    if (model.is_big_rune) {
                        (*fitter)->fitAngleModelLSQ();
                        (*fitter)->applyPosteriorConstraints();
                    } else {
                        (*fitter)->fitLinearModelPublic();
                    }

                    // 防误判机制：冷却期结束后，误差超过10 rad时切换类型
                    if (switch_cooldown_count_ > 0) {
                        switch_cooldown_count_--;
                    } else {
                        frame_fps_++;
                        if (frame_fps_ > FPS_THRESHOLD) {
                            double velocity_range = velocity_max_ - velocity_min_;
                            auto updated_model    = (*fitter)->getModel();
                            if (velocity_range > SWITCH_THRESHOLD && !updated_model.is_big_rune) {
                                std::cerr << "速度差值过大 " << velocity_range
                                          << " ，切换类型: 小符 -> 大符" << std::endl;

                                (*fitter)->fitAngleModelLSQ();
                                (*fitter)->applyPosteriorConstraints();

                                // 启动冷却
                                switch_cooldown_count_ = SWITCH_COOLDOWN_FRAMES;

                            } else if (
                                velocity_range < SWITCH_THRESHOLD && updated_model.is_big_rune) {
                                std::cerr << "速度差值过小 " << velocity_range
                                          << " ，切换类型: 大符 ->小符" << std::endl;

                                (*fitter)->fitLinearModelPublic();

                                // 启动冷却
                                switch_cooldown_count_ = SWITCH_COOLDOWN_FRAMES;
                            }
                            frame_fps_ = 0;
                            resetVelocityStats();
                        }
                    }
                }
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
