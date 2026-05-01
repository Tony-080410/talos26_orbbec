#pragma once

#include "L3_estimation/energy_meter_solver/motion_model.hpp"

#include "L3_estimation/tracker/extended_kalman_filter.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <optional>
#include <vector>

namespace energy_meter {

struct Tracker {
    using Model = EnergyMeterModel;
    using JetX  = ceres::Jet<double, Model::NX>;

    using PredictFunc = std::function<void(const JetX*, JetX*)>;
    using MeasureFunc = std::function<void(const JetX*, JetX*)>;
    using EKF         = ExtendedKalmanFilter<Model::NX, Model::NZ, PredictFunc, MeasureFunc>;

    struct Params {
        int lost_thres{10};
        int tracking_thres{3};
        double matcher_gate{25.0};
        Model::Params model_params{};
    } params;

    enum State : uint8_t {
        IDLE,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } state{IDLE};

    Tracker() = default;

    explicit Tracker(const Params& p) { set_params(p); }

    void set_params(const Params& p) {
        params        = p;
        model_.params = params.model_params;
    }

    // ── EKF access ──

    bool has_ekf() const { return ekf_.has_value(); }

    void reset_ekf() { ekf_.reset(); }

    auto get_state() const -> const Model::VecX& {
        static Model::VecX zero = Model::VecX::Zero();
        return ekf_ ? ekf_->X() : zero;
    }

    auto get_a() const -> double {
        const auto& x = get_state();
        return to_a(x(ALPHA));
    }

    auto get_w() const -> double {
        const auto& x = get_state();
        return to_w(x(BETA));
    }

    auto get_b() const -> double { return 2.090 - get_a(); }

    auto get_t() const -> double {
        const auto& x = get_state();
        return x(TAU);
    }

    auto get_dir() const -> int { return dir_; }

    auto getRadius() const -> double { return model_.params.radius; }

    void setRadius(double radius) { model_.params.radius = radius; }

    auto getCurrentBladeId() const -> int { return armor_id_; }

    // ── Lifecycle ──

    bool first_meet_u(
        const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
        const Eigen::Quaterniond& target_quat);

    void predict(double dt);

    bool update(
        const Eigen::Vector3d& rcenter_position,
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

    void step(
        double dt, const Eigen::Vector3d& rcenter_position,
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

    /// Reset model parameters (T→0, re-init ALPHA/BETA) on big-rune activation.
    void reset_on_activate(double observed_roll);

private:
    void state_machine(bool found);
    void init_ekf(const Model::VecX& x0, const Model::MatXX& P0);

    Model model_{};
    std::optional<EKF> ekf_;

    // Direction: +1 CW, -1 ACW. Detected from raw observations, fixed per match.
    int dir_{1};
    double prev_observed_roll_{0.0};
    int dir_votes_pos_{0};
    int dir_votes_neg_{0};
    int dir_detect_count_{0};
    bool dir_locked_{false};
    static constexpr int DIR_DETECT_FRAMES = 20;

    // Blade ID matching
    int detecting_count_{0};
    int lost_count_{0};

    // EKF wrapper state
    double dt_{0.0};
    int armor_id_{0};
};

} // namespace energy_meter
