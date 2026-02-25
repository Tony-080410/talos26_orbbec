#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "L3_estimation/tracker/motion_model.hpp"
#include "L3_estimation/tracker/tracker.hpp"
#include "L3_estimation/tracker/util.hpp"
#include "core/types.hpp"
#include "frame.hpp"
#include "matrix.hpp"

namespace {

constexpr double kManeuverTargetXYMeters        = 0.05;
constexpr double kTightTrackXYMeters            = 0.02;
constexpr double kSharpTurnGraceSec             = 1.0;
constexpr double kSharpTurnHeadingRateRadPerSec = 0.40;

using fcs::L3::RoboCKF;
using fcs::L3::RobotModel;
using fcs::L3::Tracker;
using fcs::L3::TrackerConfig;
using namespace fcs::L3;

struct NoiseProfile {
    double yaw_std{0.0025};
    double pitch_std{0.0020};
    double dist_std{0.0080};
    double armor_yaw_std{0.0030};
};

enum class RobotScenarioKind : uint8_t {
    EasyTranslation,
    SmallGyro,
    TranslatingSmallGyro,
    AggressiveManeuver,
};

struct ScenarioSpec {
    std::string name;
    RobotScenarioKind kind{RobotScenarioKind::EasyTranslation};
    int total_steps{0};
    int warmup_steps{0};
    int startup_steps{0};
    double dt{0.01};
    int seeds{1};

    double settle_xy{0.12};
    double settle_yaw{0.30};
    double settle_vyaw{1.20};
    int settle_streak{50};
};

struct StepErrors {
    double xy{0.0};
    double z{0.0};
    double pos{0.0};
    double yaw{0.0};
    double vyaw{0.0};
    double r0{0.0};
    double r1{0.0};
    double h{0.0};
};

struct SeedRunMetrics {
    bool finite_state{true};
    bool settled{false};
    int total_steps{0};
    int update_ok_count{0};
    int robot_output_count{0};
    int settled_step{-1};

    double startup_peak_xy{0.0};
    double steady_peak_xy{0.0};
    double max_xy{0.0};
    double max_pos{0.0};
    double max_yaw{0.0};
    double max_vyaw{0.0};

    double xy_rmse{0.0};
    double pos_rmse{0.0};
    double z_rmse{0.0};
    double yaw_rmse{0.0};
    double vyaw_rmse{0.0};
    double r0_rmse{0.0};
    double r1_rmse{0.0};
    double h_rmse{0.0};

    double steady_xy_rmse{0.0};
    double steady_yaw_rmse{0.0};
    double steady_vyaw_rmse{0.0};

    double p50_xy{0.0};
    double p95_xy{0.0};
    double p99_xy{0.0};
    double p95_yaw{0.0};
    double p95_vyaw{0.0};

    int maneuver_samples{0};
    double maneuver_xy_rmse{0.0};
    double maneuver_yaw_rmse{0.0};
    double maneuver_vyaw_rmse{0.0};
    double maneuver_p95_xy{0.0};
    double maneuver_p95_vyaw{0.0};
    double maneuver_peak_xy{0.0};
    int accuracy_samples{0};
    double accuracy_p95_xy{0.0};
    double accuracy_peak_xy{0.0};

    std::array<int, 4> armor_seen_count{0, 0, 0, 0};
    std::array<double, 4> armor_xy_mean{0.0, 0.0, 0.0, 0.0};

    std::string summary_line(const ScenarioSpec& spec, int seed_id) const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        oss << "[TrackerKFAccuracy][" << spec.name << "][seed=" << seed_id << "] "
            << "ok_rate="
            << (total_steps ? static_cast<double>(update_ok_count) / total_steps : 0.0)
            << " robot_rate="
            << (total_steps ? static_cast<double>(robot_output_count) / total_steps : 0.0)
            << " xy_rmse=" << xy_rmse << " steady_xy_rmse=" << steady_xy_rmse
            << " p95_xy=" << p95_xy << " p99_xy=" << p99_xy << " yaw_rmse=" << yaw_rmse
            << " vyaw_rmse=" << vyaw_rmse << " maneuver_xy_rmse=" << maneuver_xy_rmse
            << " maneuver_p95_xy=" << maneuver_p95_xy << " maneuver_peak_xy=" << maneuver_peak_xy
            << " acc_p95_xy=" << accuracy_p95_xy << " max_xy=" << max_xy
            << " settle_s=" << (settled ? (spec.dt * settled_step) : -1.0);
        return oss.str();
    }
};

struct AggregateMetrics {
    bool finite_state{true};
    bool all_settled{true};
    int seeds{0};
    int total_steps{0};
    int update_ok_count{0};
    int robot_output_count{0};
    int settled_count{0};
    int min_accuracy_samples{std::numeric_limits<int>::max()};

    double mean_xy_rmse{0.0};
    double mean_pos_rmse{0.0};
    double mean_yaw_rmse{0.0};
    double mean_vyaw_rmse{0.0};
    double mean_steady_xy_rmse{0.0};
    double mean_steady_yaw_rmse{0.0};
    double mean_steady_vyaw_rmse{0.0};
    double mean_maneuver_xy_rmse{0.0};
    double mean_maneuver_yaw_rmse{0.0};
    double mean_maneuver_vyaw_rmse{0.0};

    double worst_p95_xy{0.0};
    double worst_p99_xy{0.0};
    double worst_max_xy{0.0};
    double worst_max_pos{0.0};
    double worst_p95_yaw{0.0};
    double worst_p95_vyaw{0.0};
    double worst_maneuver_p95_xy{0.0};
    double worst_maneuver_p95_vyaw{0.0};
    double worst_maneuver_peak_xy{0.0};
    double worst_accuracy_p95_xy{0.0};
    double worst_accuracy_peak_xy{0.0};
    double worst_startup_peak_xy{0.0};
    double worst_steady_peak_xy{0.0};
    double worst_settle_time_s{0.0};

    double update_success_rate() const {
        return total_steps > 0
                 ? static_cast<double>(update_ok_count) / static_cast<double>(total_steps)
                 : 0.0;
    }
    double robot_output_rate() const {
        return total_steps > 0
                 ? static_cast<double>(robot_output_count) / static_cast<double>(total_steps)
                 : 0.0;
    }
};

struct MetricsAccumulator {
    explicit MetricsAccumulator(const ScenarioSpec& spec)
        : spec_(spec) {}

    void note_update_result(bool ok) {
        ++total_steps;
        if (ok) {
            ++update_ok_count;
        }
    }

    void observe(
        int step, int armor_id, const RobotTargetState& st, const RoboCKF::VecX& x_true,
        bool is_maneuver, bool count_for_accuracy_metric) noexcept {
        ++robot_output_count;

        const StepErrors e = compute_errors(st, x_true);
        finite_state &= st.position.allFinite() && st.velocity.allFinite() && std::isfinite(st.yaw)
                     && std::isfinite(st.v_yaw) && std::isfinite(st.radius0)
                     && std::isfinite(st.radius1) && std::isfinite(st.z1);

        add_error(xy_errors_, e.xy);
        add_error(pos_errors_, e.pos);
        add_error(yaw_errors_, e.yaw);
        add_error(vyaw_errors_, e.vyaw);

        sum_sq_xy_ += e.xy * e.xy;
        sum_sq_pos_ += e.pos * e.pos;
        sum_sq_z_ += e.z * e.z;
        sum_sq_yaw_ += e.yaw * e.yaw;
        sum_sq_vyaw_ += e.vyaw * e.vyaw;
        sum_sq_r0_ += e.r0 * e.r0;
        sum_sq_r1_ += e.r1 * e.r1;
        sum_sq_h_ += e.h * e.h;
        ++count_;

        max_xy_   = std::max(max_xy_, e.xy);
        max_pos_  = std::max(max_pos_, e.pos);
        max_yaw_  = std::max(max_yaw_, e.yaw);
        max_vyaw_ = std::max(max_vyaw_, e.vyaw);

        if (step < spec_.startup_steps) {
            startup_peak_xy_ = std::max(startup_peak_xy_, e.xy);
        }
        if (step >= spec_.warmup_steps) {
            steady_xy_errors_.push_back(e.xy);
            steady_yaw_errors_.push_back(e.yaw);
            steady_vyaw_errors_.push_back(e.vyaw);
            sum_sq_xy_steady_ += e.xy * e.xy;
            sum_sq_yaw_steady_ += e.yaw * e.yaw;
            sum_sq_vyaw_steady_ += e.vyaw * e.vyaw;
            ++steady_count_;
            steady_peak_xy_ = std::max(steady_peak_xy_, e.xy);
        }

        if (is_maneuver) {
            maneuver_xy_errors_.push_back(e.xy);
            maneuver_vyaw_errors_.push_back(e.vyaw);
            sum_sq_xy_maneuver_ += e.xy * e.xy;
            sum_sq_yaw_maneuver_ += e.yaw * e.yaw;
            sum_sq_vyaw_maneuver_ += e.vyaw * e.vyaw;
            ++maneuver_count_;
            maneuver_peak_xy_ = std::max(maneuver_peak_xy_, e.xy);
        }

        if (count_for_accuracy_metric) {
            accuracy_xy_errors_.push_back(e.xy);
            ++accuracy_count_;
            accuracy_peak_xy_ = std::max(accuracy_peak_xy_, e.xy);
        }

        if (armor_id >= 0 && armor_id < 4) {
            armor_seen_count_[static_cast<size_t>(armor_id)] += 1;
            armor_xy_sum_[static_cast<size_t>(armor_id)] += e.xy;
        }

        update_settle(step, e);
    }

    SeedRunMetrics finalize() const {
        SeedRunMetrics out;
        out.finite_state       = finite_state;
        out.total_steps        = total_steps;
        out.update_ok_count    = update_ok_count;
        out.robot_output_count = robot_output_count;
        out.startup_peak_xy    = startup_peak_xy_;
        out.steady_peak_xy     = steady_peak_xy_;
        out.max_xy             = max_xy_;
        out.max_pos            = max_pos_;
        out.max_yaw            = max_yaw_;
        out.max_vyaw           = max_vyaw_;
        out.settled            = settled_step_.has_value();
        out.settled_step       = settled_step_.value_or(-1);

        out.xy_rmse   = rmse(sum_sq_xy_, count_);
        out.pos_rmse  = rmse(sum_sq_pos_, count_);
        out.z_rmse    = rmse(sum_sq_z_, count_);
        out.yaw_rmse  = rmse(sum_sq_yaw_, count_);
        out.vyaw_rmse = rmse(sum_sq_vyaw_, count_);
        out.r0_rmse   = rmse(sum_sq_r0_, count_);
        out.r1_rmse   = rmse(sum_sq_r1_, count_);
        out.h_rmse    = rmse(sum_sq_h_, count_);

        out.steady_xy_rmse   = rmse(sum_sq_xy_steady_, steady_count_);
        out.steady_yaw_rmse  = rmse(sum_sq_yaw_steady_, steady_count_);
        out.steady_vyaw_rmse = rmse(sum_sq_vyaw_steady_, steady_count_);

        out.p50_xy             = quantile(xy_errors_, 0.50);
        out.p95_xy             = quantile(xy_errors_, 0.95);
        out.p99_xy             = quantile(xy_errors_, 0.99);
        out.p95_yaw            = quantile(yaw_errors_, 0.95);
        out.p95_vyaw           = quantile(vyaw_errors_, 0.95);
        out.maneuver_samples   = maneuver_count_;
        out.maneuver_xy_rmse   = rmse(sum_sq_xy_maneuver_, maneuver_count_);
        out.maneuver_yaw_rmse  = rmse(sum_sq_yaw_maneuver_, maneuver_count_);
        out.maneuver_vyaw_rmse = rmse(sum_sq_vyaw_maneuver_, maneuver_count_);
        out.maneuver_p95_xy    = quantile(maneuver_xy_errors_, 0.95);
        out.maneuver_p95_vyaw  = quantile(maneuver_vyaw_errors_, 0.95);
        out.maneuver_peak_xy   = maneuver_peak_xy_;
        out.accuracy_samples   = accuracy_count_;
        if (accuracy_count_ > 0) {
            out.accuracy_p95_xy  = quantile(accuracy_xy_errors_, 0.95);
            out.accuracy_peak_xy = accuracy_peak_xy_;
        } else {
            out.accuracy_p95_xy  = std::numeric_limits<double>::infinity();
            out.accuracy_peak_xy = std::numeric_limits<double>::infinity();
        }

        out.armor_seen_count = armor_seen_count_;
        for (size_t i = 0; i < out.armor_xy_mean.size(); ++i) {
            out.armor_xy_mean[i] =
                out.armor_seen_count[i] > 0
                    ? armor_xy_sum_[i] / static_cast<double>(out.armor_seen_count[i])
                    : 0.0;
        }
        return out;
    }

private:
    static StepErrors compute_errors(const RobotTargetState& st, const RoboCKF::VecX& x_true) {
        StepErrors e;
        const double ex = st.position.x() - x_true[XC];
        const double ey = st.position.y() - x_true[YC];
        const double ez = st.position.z() - x_true[Z0];
        e.xy            = std::hypot(ex, ey);
        e.z             = std::abs(ez);
        e.pos           = std::sqrt(ex * ex + ey * ey + ez * ez);
        e.yaw           = std::abs(shortest_rad(st.yaw, x_true[YAW]));
        e.vyaw          = std::abs(st.v_yaw - x_true[V_YAW]);
        e.r0            = std::abs(st.radius0 - x_true[R_0]);
        e.r1            = std::abs(st.radius1 - x_true[R_1]);
        e.h             = std::abs((st.z1 - st.position.z()) - x_true[H]);
        return e;
    }

    static void add_error(std::vector<double>& dst, double v) { dst.push_back(v); }

    static double rmse(double sum_sq, int n) { return n > 0 ? std::sqrt(sum_sq / n) : 0.0; }

    static double quantile(const std::vector<double>& values, double q) {
        if (values.empty()) {
            return 0.0;
        }
        std::vector<double> tmp = values;
        q                       = std::clamp(q, 0.0, 1.0);
        const size_t idx =
            static_cast<size_t>(std::llround(q * static_cast<double>(tmp.size() - 1)));
        std::nth_element(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(idx), tmp.end());
        return tmp[idx];
    }

    void update_settle(int step, const StepErrors& e) noexcept {
        const bool good =
            e.xy <= spec_.settle_xy && e.yaw <= spec_.settle_yaw && e.vyaw <= spec_.settle_vyaw;
        if (good) {
            ++settle_streak_;
            if (!settled_step_.has_value() && settle_streak_ >= spec_.settle_streak) {
                settled_step_ = step - spec_.settle_streak + 1;
            }
        } else {
            settle_streak_ = 0;
        }
    }

    const ScenarioSpec& spec_;

    bool finite_state{true};
    int total_steps{0};
    int update_ok_count{0};
    int robot_output_count{0};

    int count_{0};
    int steady_count_{0};
    int maneuver_count_{0};
    int accuracy_count_{0};
    int settle_streak_{0};
    std::optional<int> settled_step_;

    double sum_sq_xy_{0.0};
    double sum_sq_pos_{0.0};
    double sum_sq_z_{0.0};
    double sum_sq_yaw_{0.0};
    double sum_sq_vyaw_{0.0};
    double sum_sq_r0_{0.0};
    double sum_sq_r1_{0.0};
    double sum_sq_h_{0.0};

    double sum_sq_xy_steady_{0.0};
    double sum_sq_yaw_steady_{0.0};
    double sum_sq_vyaw_steady_{0.0};
    double sum_sq_xy_maneuver_{0.0};
    double sum_sq_yaw_maneuver_{0.0};
    double sum_sq_vyaw_maneuver_{0.0};

    double startup_peak_xy_{0.0};
    double steady_peak_xy_{0.0};
    double maneuver_peak_xy_{0.0};
    double accuracy_peak_xy_{0.0};
    double max_xy_{0.0};
    double max_pos_{0.0};
    double max_yaw_{0.0};
    double max_vyaw_{0.0};

    std::vector<double> xy_errors_;
    std::vector<double> pos_errors_;
    std::vector<double> yaw_errors_;
    std::vector<double> vyaw_errors_;
    std::vector<double> steady_xy_errors_;
    std::vector<double> steady_yaw_errors_;
    std::vector<double> steady_vyaw_errors_;
    std::vector<double> maneuver_xy_errors_;
    std::vector<double> maneuver_vyaw_errors_;
    std::vector<double> accuracy_xy_errors_;

    std::array<int, 4> armor_seen_count_{0, 0, 0, 0};
    std::array<double, 4> armor_xy_sum_{0.0, 0.0, 0.0, 0.0};
};

struct TruthDynamics {
    double ax{0.0};
    double ay{0.0};
    double az{0.0};
    double yaw_rate{0.0};
    double yaw_acc{0.0};
    double xy_heading_rate{0.0};
    bool is_maneuver{false};
    bool is_sharp_turn_xy{false};
};

struct NoiseGenerator {
    explicit NoiseGenerator(uint32_t seed)
        : rng(seed) {}

    double gaussian(double sigma) { return sigma * normal(rng); }

    std::mt19937 rng;
    std::normal_distribution<double> normal{0.0, 1.0};
};

auto make_tracker_config_for_accuracy() -> TrackerConfig {
    TrackerConfig cfg{};
    cfg.robot.lost_threshold     = 1.0;
    cfg.robot.tracking_threshold = 1;
    cfg.robot.matcher_gate       = 120.0;
    cfg.robot.model.sigma_a_xy   = 2.0;
    cfg.robot.model.sigma_a_z    = 2.0;
    cfg.robot.model.sigma_a_yaw  = 6.0;
    cfg.robot.model.sigma_r0     = 0.0;
    cfg.robot.model.sigma_r1     = 0.0;
    cfg.robot.model.sigma_h      = 0.0;
    // Match synthetic benchmark noise scale to evaluate model/tracker quality instead of legacy
    // robust defaults tuned for real detector outliers.
    cfg.robot.model.meas_yaw_var_floor       = 1.5e-4;
    cfg.robot.model.meas_pitch_var_floor     = 1.5e-4;
    cfg.robot.model.meas_dist_var_floor      = 2.0e-3;
    cfg.robot.model.meas_dist_delta_angle_k  = 4.0e-3;
    cfg.robot.model.meas_armor_yaw_var_floor = 6.0e-4;
    cfg.robot.model.meas_armor_yaw_range_k   = 1.0e-3;
    // Center-space virtual measurement update uses an explicitly separated noise model.
    cfg.robot.model.center_meas_yaw_var_floor       = cfg.robot.model.meas_yaw_var_floor;
    cfg.robot.model.center_meas_pitch_var_floor     = cfg.robot.model.meas_pitch_var_floor;
    cfg.robot.model.center_meas_dist_var_floor      = 4.0e-4;
    cfg.robot.model.center_meas_armor_yaw_var_floor = cfg.robot.model.meas_armor_yaw_var_floor;
    cfg.robot.model.center_meas_armor_yaw_range_k   = cfg.robot.model.meas_armor_yaw_range_k;
    cfg.robot.model.center_meas_cov_inflation       = 1.2;
    cfg.outpost.lost_threshold                      = 2.0;
    cfg.outpost.tracking_threshold                  = 1;
    cfg.outpost.matcher_gate                        = 4.0;
    // Provide geometry prior to the tracker initialization path (now used by ISRCKF init too).
    cfg.robot_inekf.radius0 = 0.23;
    cfg.robot_inekf.radius1 = 0.245;
    cfg.robot_inekf.height  = 0.06;
    return cfg;
}

void init_truth_state(const ScenarioSpec& spec, RoboCKF::VecX& x_true) {
    x_true.setZero();
    x_true[XC]    = 1.8;
    x_true[YC]    = 0.6;
    x_true[Z0]    = 0.26;
    x_true[YAW]   = 0.3;
    x_true[R_0]   = 0.23;
    x_true[R_1]   = 0.245;
    x_true[H]     = 0.06;
    x_true[VZ]    = 0.0;
    x_true[V_YAW] = 0.0;

    switch (spec.kind) {
    case RobotScenarioKind::EasyTranslation:
        x_true[VX]    = 0.45;
        x_true[VY]    = -0.20;
        x_true[VZ]    = 0.01;
        x_true[YAW]   = 0.2;
        x_true[V_YAW] = 0.85; // moderate rotation, not "small gyro"
        break;
    case RobotScenarioKind::SmallGyro:
        x_true[VX]    = 0.0;
        x_true[VY]    = 0.0;
        x_true[VZ]    = 0.0;
        x_true[YAW]   = 0.0;
        x_true[V_YAW] = 6.28;
        break;
    case RobotScenarioKind::TranslatingSmallGyro:
        x_true[VX]    = 0.35;
        x_true[VY]    = 0.18;
        x_true[VZ]    = 0.01;
        x_true[YAW]   = -0.4;
        x_true[V_YAW] = 6.28;
        break;
    case RobotScenarioKind::AggressiveManeuver:
        x_true[VX]    = 0.0;
        x_true[VY]    = 0.0;
        x_true[VZ]    = 0.0;
        x_true[YAW]   = -0.2;
        x_true[V_YAW] = 0.0;
        break;
    }
}

TruthDynamics
    advance_truth_state(const ScenarioSpec& spec, RoboCKF::VecX& x_true, int step) noexcept {
    const double t = spec.dt * static_cast<double>(step);
    TruthDynamics dyn;
    dyn.yaw_rate = x_true[V_YAW];

    switch (spec.kind) {
    case RobotScenarioKind::EasyTranslation:
        dyn.ax       = 0.12 * std::sin(0.45 * t);
        dyn.ay       = 0.10 * std::cos(0.37 * t);
        dyn.az       = 0.02 * std::sin(0.18 * t);
        dyn.yaw_rate = 0.85 + 0.15 * std::sin(0.25 * t);
        dyn.yaw_acc  = 0.15 * 0.25 * std::cos(0.25 * t);
        break;
    case RobotScenarioKind::SmallGyro:
        dyn.ax       = 0.0;
        dyn.ay       = 0.0;
        dyn.az       = 0.0;
        dyn.yaw_rate = 6.28 + 0.30 * std::sin(0.22 * t);
        dyn.yaw_acc  = 0.30 * 0.22 * std::cos(0.22 * t);
        x_true[VX]   = 0.0;
        x_true[VY]   = 0.0;
        x_true[VZ]   = 0.0;
        break;
    case RobotScenarioKind::TranslatingSmallGyro:
        dyn.ax       = 0.20 * std::sin(0.55 * t) + 0.06 * std::cos(1.05 * t);
        dyn.ay       = -0.16 * std::cos(0.42 * t) + 0.05 * std::sin(0.83 * t);
        dyn.az       = 0.03 * std::sin(0.21 * t);
        dyn.yaw_rate = 6.28 + 0.35 * std::sin(0.24 * t);
        dyn.yaw_acc  = 0.35 * 0.24 * std::cos(0.24 * t);
        break;
    case RobotScenarioKind::AggressiveManeuver: {
        // 8-second cycle with acceleration reversals and spin direction changes.
        const double phase = std::fmod(t, 8.0);
        if (phase < 1.5) {
            dyn.ax       = 1.20;
            dyn.ay       = 0.40;
            dyn.az       = 0.03 * std::sin(1.8 * t);
            dyn.yaw_rate = 1.0 + 4.5 * phase;
            dyn.yaw_acc  = 4.5;
        } else if (phase < 3.0) {
            dyn.ax       = -1.50;
            dyn.ay       = 0.90;
            dyn.az       = 0.02 * std::cos(1.5 * t);
            dyn.yaw_rate = 7.75 - 2.2 * (phase - 1.5);
            dyn.yaw_acc  = -2.2;
        } else if (phase < 4.5) {
            dyn.ax       = -1.10;
            dyn.ay       = -1.30;
            dyn.az       = -0.03 * std::sin(1.3 * t);
            dyn.yaw_rate = 4.45 - 8.0 * (phase - 3.0);
            dyn.yaw_acc  = -8.0;
        } else if (phase < 6.0) {
            dyn.ax       = 1.60;
            dyn.ay       = -0.70;
            dyn.az       = 0.02 * std::sin(0.8 * t);
            dyn.yaw_rate = -7.55 + 6.5 * (phase - 4.5);
            dyn.yaw_acc  = 6.5;
        } else {
            dyn.ax       = 0.70 * std::sin(2.4 * t) + 0.35 * std::cos(1.1 * t);
            dyn.ay       = -0.60 * std::cos(2.0 * t) + 0.30 * std::sin(1.7 * t);
            dyn.az       = 0.03 * std::sin(0.9 * t);
            dyn.yaw_rate = 2.2 + 3.8 * std::sin(1.6 * t);
            dyn.yaw_acc  = 3.8 * 1.6 * std::cos(1.6 * t);
        }
        break;
    }
    }

    if (spec.kind != RobotScenarioKind::SmallGyro) {
        x_true[VX] += dyn.ax * spec.dt;
        x_true[VY] += dyn.ay * spec.dt;
        x_true[VZ] += dyn.az * spec.dt;
        x_true[VX] = std::clamp(x_true[VX], -2.6, 2.6);
        x_true[VY] = std::clamp(x_true[VY], -2.6, 2.6);
        x_true[VZ] = std::clamp(x_true[VZ], -0.4, 0.4);
    }

    x_true[XC] += x_true[VX] * spec.dt;
    x_true[YC] += x_true[VY] * spec.dt;
    x_true[Z0] += x_true[VZ] * spec.dt;
    x_true[V_YAW] = dyn.yaw_rate;
    x_true[YAW] += x_true[V_YAW] * spec.dt;

    {
        const double vx       = x_true[VX];
        const double vy       = x_true[VY];
        const double v2       = vx * vx + vy * vy;
        const double speed_xy = std::sqrt(v2);
        dyn.xy_heading_rate   = (v2 > 1e-9) ? std::abs(vx * dyn.ay - vy * dyn.ax) / v2 : 0.0;
        dyn.is_sharp_turn_xy =
            (speed_xy > 0.30) && (dyn.xy_heading_rate > kSharpTurnHeadingRateRadPerSec);
    }

    dyn.is_maneuver = (std::hypot(dyn.ax, dyn.ay) > 0.35) || (std::abs(dyn.yaw_acc) > 0.70)
                   || (std::abs(dyn.yaw_rate) > 5.5);
    return dyn;
}

int select_visible_armor_id(const ScenarioSpec& spec, const RoboCKF::VecX& x_true) noexcept {
    constexpr double kAngleStep = 2.0 * std::numbers::pi / 4.0;
    switch (spec.kind) {
    case RobotScenarioKind::EasyTranslation: {
        // Moderate rotation: still track the "front-most" armor to exercise phase switching.
        const int visible_id = static_cast<int>(std::llround(x_true[YAW] / kAngleStep)) % 4;
        return (visible_id + 4) % 4;
    }
    case RobotScenarioKind::SmallGyro:
    case RobotScenarioKind::TranslatingSmallGyro: {
        const int visible_id = static_cast<int>(std::llround(x_true[YAW] / kAngleStep)) % 4;
        return (visible_id + 4) % 4;
    }
    case RobotScenarioKind::AggressiveManeuver: {
        const int visible_id = static_cast<int>(std::llround(x_true[YAW] / kAngleStep)) % 4;
        return (visible_id + 4) % 4;
    }
    }
    return 0;
}

auto make_robot_batch_from_truth(
    const RobotModel& model, const RoboCKF::VecX& x_true, int armor_id, int step,
    NoiseGenerator& noise, const NoiseProfile& profile) -> fcs::AtLegacyArmorMeasurementBatch {
    RoboCKF::VecZ z = model.h(x_true, armor_id);

    const double t = static_cast<double>(step);
    z[0] += 0.004 * std::sin(0.11 * t) + noise.gaussian(profile.yaw_std);
    z[1] += 0.003 * std::cos(0.07 * t) + noise.gaussian(profile.pitch_std);
    z[2] += 0.010 * std::sin(0.05 * t) + noise.gaussian(profile.dist_std);
    z[3] += 0.004 * std::cos(0.13 * t) + noise.gaussian(profile.armor_yaw_std);

    const Eigen::Vector3d xyz = ypd2xyz({z[0], z[1], std::max(0.05, z[2])});

    fcs::AtLegacyArmorMeasurement meas;
    meas.name                     = fcs::ArmorName::Two;
    meas.color                    = fcs::ArmorColor::Red;
    meas.type                     = fcs::ArmorType::Small;
    meas.distance_to_image_center = 0.0f;
    meas.transform                = fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>{
        fast_tf::TransformMatrix4d::from_rpy(0.0, 0.0, z[3], xyz.x(), xyz.y(), xyz.z())};

    fcs::AtLegacyArmorMeasurementBatch batch;
    batch.measurements.push_back(meas);
    return batch;
}

auto run_one_seed(const ScenarioSpec& spec, int seed_id) -> SeedRunMetrics {
    const TrackerConfig cfg = make_tracker_config_for_accuracy();
    Tracker tracker(cfg);

    RobotModel truth_model;
    RoboCKF::VecX x_true;
    init_truth_state(spec, x_true);

    NoiseGenerator noise(static_cast<uint32_t>(0xC0FFEE + 97 * seed_id));
    const NoiseProfile profile{};

    const int first_armor_id = select_visible_armor_id(spec, x_true);
    auto batch0 =
        make_robot_batch_from_truth(truth_model, x_true, first_armor_id, 0, noise, profile);
    [[maybe_unused]] const auto first = tracker.first_meet(batch0);

    MetricsAccumulator acc(spec);
    double sharp_turn_grace_remaining_s = 0.0;
    for (int step = 1; step < spec.total_steps; ++step) {
        const auto dyn     = advance_truth_state(spec, x_true, step);
        const int armor_id = select_visible_armor_id(spec, x_true);

        if (dyn.is_sharp_turn_xy) {
            sharp_turn_grace_remaining_s = kSharpTurnGraceSec;
        } else if (sharp_turn_grace_remaining_s > 0.0) {
            sharp_turn_grace_remaining_s = std::max(0.0, sharp_turn_grace_remaining_s - spec.dt);
        }
        const bool count_for_accuracy_metric = (sharp_turn_grace_remaining_s <= 0.0);

        tracker.predict(spec.dt);
        auto batch =
            make_robot_batch_from_truth(truth_model, x_true, armor_id, step, noise, profile);
        const bool ok = tracker.update(batch);
        acc.note_update_result(ok);

        const auto out = tracker.get_output();
        if (!out.is_robot()) {
            continue;
        }
        const auto* st = out.robot_state();
        if (st == nullptr) {
            continue;
        }
        acc.observe(step, armor_id, *st, x_true, dyn.is_maneuver, count_for_accuracy_metric);
    }

    return acc.finalize();
}

auto aggregate_metrics(const ScenarioSpec& spec, const std::vector<SeedRunMetrics>& runs)
    -> AggregateMetrics {
    AggregateMetrics agg;
    agg.seeds = static_cast<int>(runs.size());
    if (runs.empty()) {
        agg.min_accuracy_samples = 0;
        return agg;
    }

    for (const auto& r : runs) {
        agg.finite_state &= r.finite_state;
        agg.all_settled &= r.settled;
        agg.total_steps += r.total_steps;
        agg.update_ok_count += r.update_ok_count;
        agg.robot_output_count += r.robot_output_count;
        agg.settled_count += (r.settled ? 1 : 0);
        agg.min_accuracy_samples = std::min(agg.min_accuracy_samples, r.accuracy_samples);

        agg.mean_xy_rmse += r.xy_rmse;
        agg.mean_pos_rmse += r.pos_rmse;
        agg.mean_yaw_rmse += r.yaw_rmse;
        agg.mean_vyaw_rmse += r.vyaw_rmse;
        agg.mean_steady_xy_rmse += r.steady_xy_rmse;
        agg.mean_steady_yaw_rmse += r.steady_yaw_rmse;
        agg.mean_steady_vyaw_rmse += r.steady_vyaw_rmse;
        agg.mean_maneuver_xy_rmse += r.maneuver_xy_rmse;
        agg.mean_maneuver_yaw_rmse += r.maneuver_yaw_rmse;
        agg.mean_maneuver_vyaw_rmse += r.maneuver_vyaw_rmse;

        agg.worst_p95_xy            = std::max(agg.worst_p95_xy, r.p95_xy);
        agg.worst_p99_xy            = std::max(agg.worst_p99_xy, r.p99_xy);
        agg.worst_max_xy            = std::max(agg.worst_max_xy, r.max_xy);
        agg.worst_max_pos           = std::max(agg.worst_max_pos, r.max_pos);
        agg.worst_p95_yaw           = std::max(agg.worst_p95_yaw, r.p95_yaw);
        agg.worst_p95_vyaw          = std::max(agg.worst_p95_vyaw, r.p95_vyaw);
        agg.worst_maneuver_p95_xy   = std::max(agg.worst_maneuver_p95_xy, r.maneuver_p95_xy);
        agg.worst_maneuver_p95_vyaw = std::max(agg.worst_maneuver_p95_vyaw, r.maneuver_p95_vyaw);
        agg.worst_maneuver_peak_xy  = std::max(agg.worst_maneuver_peak_xy, r.maneuver_peak_xy);
        agg.worst_accuracy_p95_xy   = std::max(agg.worst_accuracy_p95_xy, r.accuracy_p95_xy);
        agg.worst_accuracy_peak_xy  = std::max(agg.worst_accuracy_peak_xy, r.accuracy_peak_xy);
        agg.worst_startup_peak_xy   = std::max(agg.worst_startup_peak_xy, r.startup_peak_xy);
        agg.worst_steady_peak_xy    = std::max(agg.worst_steady_peak_xy, r.steady_peak_xy);
        if (r.settled) {
            agg.worst_settle_time_s = std::max(agg.worst_settle_time_s, spec.dt * r.settled_step);
        }
    }

    const double n = static_cast<double>(runs.size());
    agg.mean_xy_rmse /= n;
    agg.mean_pos_rmse /= n;
    agg.mean_yaw_rmse /= n;
    agg.mean_vyaw_rmse /= n;
    agg.mean_steady_xy_rmse /= n;
    agg.mean_steady_yaw_rmse /= n;
    agg.mean_steady_vyaw_rmse /= n;
    agg.mean_maneuver_xy_rmse /= n;
    agg.mean_maneuver_yaw_rmse /= n;
    agg.mean_maneuver_vyaw_rmse /= n;
    return agg;
}

void print_report(
    const ScenarioSpec& spec, const std::vector<SeedRunMetrics>& runs,
    const AggregateMetrics& agg) {
    for (size_t i = 0; i < runs.size(); ++i) {
        std::cout << runs[i].summary_line(spec, static_cast<int>(i)) << '\n';
    }

    std::cout << std::fixed << std::setprecision(4) << "[TrackerKFAccuracy][" << spec.name
              << "][aggregate] "
              << "seeds=" << agg.seeds << " ok_rate=" << agg.update_success_rate()
              << " robot_rate=" << agg.robot_output_rate() << " mean_xy_rmse=" << agg.mean_xy_rmse
              << " mean_steady_xy_rmse=" << agg.mean_steady_xy_rmse
              << " mean_maneuver_xy_rmse=" << agg.mean_maneuver_xy_rmse
              << " mean_yaw_rmse=" << agg.mean_yaw_rmse << " mean_vyaw_rmse=" << agg.mean_vyaw_rmse
              << " worst_p95_xy=" << agg.worst_p95_xy
              << " worst_maneuver_p95_xy=" << agg.worst_maneuver_p95_xy
              << " worst_maneuver_p95_vyaw=" << agg.worst_maneuver_p95_vyaw
              << " worst_p99_xy=" << agg.worst_p99_xy
              << " worst_acc_p95_xy=" << agg.worst_accuracy_p95_xy
              << " worst_max_xy=" << agg.worst_max_xy
              << " worst_maneuver_peak_xy=" << agg.worst_maneuver_peak_xy
              << " worst_acc_peak_xy=" << agg.worst_accuracy_peak_xy
              << " worst_startup_peak_xy=" << agg.worst_startup_peak_xy
              << " worst_steady_peak_xy=" << agg.worst_steady_peak_xy
              << " settled=" << agg.settled_count << "/" << agg.seeds
              << " min_acc_samples=" << agg.min_accuracy_samples
              << " worst_settle_s=" << agg.worst_settle_time_s << '\n';
}

auto run_scenario_suite(const ScenarioSpec& spec) -> AggregateMetrics {
    std::vector<SeedRunMetrics> runs;
    runs.reserve(static_cast<size_t>(spec.seeds));
    for (int seed = 0; seed < spec.seeds; ++seed) {
        runs.push_back(run_one_seed(spec, seed));
    }
    const auto agg = aggregate_metrics(spec, runs);
    print_report(spec, runs, agg);
    return agg;
}

} // namespace

TEST(TrackerKFAccuracy, RobotTargetEasyTranslation) {
    const ScenarioSpec spec{
        .name          = "robot_easy_translation",
        .kind          = RobotScenarioKind::EasyTranslation,
        .total_steps   = 2600,
        .warmup_steps  = 600,
        .startup_steps = 250,
        .dt            = 0.01,
        .seeds         = 3,
        .settle_xy     = 0.15,
        .settle_yaw    = 0.50,
        .settle_vyaw   = 1.20,
        .settle_streak = 60,
    };
    const auto agg = run_scenario_suite(spec);

    EXPECT_TRUE(agg.finite_state);
    EXPECT_GT(agg.update_success_rate(), 0.995);
    EXPECT_GT(agg.robot_output_rate(), 0.995);
    EXPECT_GE(agg.settled_count, 2);
    EXPECT_LT(agg.mean_steady_xy_rmse, 0.10);
    EXPECT_LT(agg.worst_p95_xy, 0.22);
    EXPECT_LT(agg.worst_p99_xy, 0.28);
    EXPECT_LT(agg.worst_startup_peak_xy, 0.55);
}

TEST(TrackerKFAccuracy, RobotTargetSmallGyro) {
    const ScenarioSpec spec{
        .name          = "robot_small_gyro",
        .kind          = RobotScenarioKind::SmallGyro,
        .total_steps   = 4500,
        .warmup_steps  = 900,
        .startup_steps = 350,
        .dt            = 0.01,
        .seeds         = 6,
        .settle_xy     = 0.20,
        .settle_yaw    = 0.70,
        .settle_vyaw   = 1.50,
        .settle_streak = 80,
    };
    const auto agg = run_scenario_suite(spec);

    EXPECT_TRUE(agg.finite_state);
    EXPECT_GT(agg.update_success_rate(), 0.995);
    EXPECT_GT(agg.robot_output_rate(), 0.995);
    EXPECT_GE(agg.settled_count, 5);
    EXPECT_GT(agg.min_accuracy_samples, 0);
    EXPECT_LT(agg.mean_maneuver_xy_rmse, 0.12);
    EXPECT_LT(agg.worst_accuracy_p95_xy, kManeuverTargetXYMeters);
    EXPECT_LT(agg.worst_accuracy_peak_xy, kManeuverTargetXYMeters);
    EXPECT_LT(agg.worst_maneuver_p95_vyaw, 1.40);
    EXPECT_LT(agg.worst_startup_peak_xy, 0.65);
}

TEST(TrackerKFAccuracy, RobotTargetTranslatingSmallGyro) {
    const ScenarioSpec spec{
        .name          = "robot_translating_small_gyro",
        .kind          = RobotScenarioKind::TranslatingSmallGyro,
        .total_steps   = 5000,
        .warmup_steps  = 1000,
        .startup_steps = 450,
        .dt            = 0.01,
        .seeds         = 6,
        .settle_xy     = 0.24,
        .settle_yaw    = 0.80,
        .settle_vyaw   = 1.70,
        .settle_streak = 100,
    };
    const auto agg = run_scenario_suite(spec);

    EXPECT_TRUE(agg.finite_state);
    EXPECT_GT(agg.update_success_rate(), 0.992);
    EXPECT_GT(agg.robot_output_rate(), 0.992);
    EXPECT_GE(agg.settled_count, 4);
    EXPECT_GT(agg.min_accuracy_samples, 0);
    EXPECT_LT(agg.mean_maneuver_xy_rmse, 0.22);
    EXPECT_LT(agg.worst_accuracy_p95_xy, kTightTrackXYMeters);
    EXPECT_LT(agg.worst_accuracy_peak_xy, kTightTrackXYMeters);
    EXPECT_LT(agg.worst_maneuver_p95_vyaw, 1.80);
    EXPECT_LT(agg.worst_startup_peak_xy, 0.80);
}

TEST(TrackerKFAccuracy, RobotTargetAggressiveManeuverFocus) {
    const ScenarioSpec spec{
        .name          = "robot_aggressive_maneuver",
        .kind          = RobotScenarioKind::AggressiveManeuver,
        .total_steps   = 5200,
        .warmup_steps  = 900,
        .startup_steps = 500,
        .dt            = 0.01,
        .seeds         = 4,
        .settle_xy     = 0.30,
        .settle_yaw    = 1.00,
        .settle_vyaw   = 2.40,
        .settle_streak = 80,
    };
    const auto agg = run_scenario_suite(spec);

    EXPECT_TRUE(agg.finite_state);
    EXPECT_GT(agg.update_success_rate(), 0.985);
    EXPECT_GT(agg.robot_output_rate(), 0.985);
    EXPECT_GE(agg.settled_count, 3);
    EXPECT_GT(agg.min_accuracy_samples, 0);
    EXPECT_LT(agg.mean_maneuver_xy_rmse, 0.35);
    EXPECT_LT(agg.worst_accuracy_p95_xy, kManeuverTargetXYMeters);
    EXPECT_LT(agg.worst_accuracy_peak_xy, kManeuverTargetXYMeters);
    EXPECT_LT(agg.worst_maneuver_p95_vyaw, 3.20);
    EXPECT_LT(agg.worst_maneuver_peak_xy, 1.10);
    EXPECT_LT(agg.worst_settle_time_s, 1.20);
}
