#pragma once

#include "L3_estimation/energy_meter_solver/isrckf.hpp"
#include "L3_estimation/energy_meter_solver/motion_model.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <expected>

#include <array>
#include <map>
#include <optional>
#include <vector>

namespace energy_meter {

struct Tracker {
    struct Params {
        int lost_thres{10};
        int tracking_thres{3};
        double matcher_gate{25.0};
        EnergyMeter::Params energy_params{};
    } params;

    enum State : uint8_t {
        IDLE,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } state{IDLE};

    Tracker() = default;

    explicit Tracker(const Params& param) { set_params(param); }

    void set_params(const Params& p) {
        params               = p;
        energy_model_.params = params.energy_params;
    }

    using EnergyUKF = srukf::ISRCKF<EnergyMeter, EnergyMeter::NX, EnergyMeter::NZ>;

    std::optional<EnergyUKF> energy_ukf;

    auto get_measurement() const noexcept -> const auto& { return measurement_; }

    auto get_state() const -> const EnergyUKF::VecX& {
        static EnergyUKF::VecX zero = EnergyUKF::VecX::Zero();
        return energy_ukf.has_value() ? energy_ukf->x() : zero;
    }

    auto getCurrentBladeId() const -> int { return current_matched_blade_id_; }

    auto isBladeIdConfirmed() const -> bool {
        return blade_id_confirmed_;  // 只有防抖通过后才为 true
    }

    auto getRadius() const -> double { return energy_model_.params.radius; }

    void setRadius(double radius) { energy_model_.params.radius = radius; }

    bool first_meet_u(
        const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
        const Eigen::Quaterniond& target_quat);

    void step(
        double dts, const Eigen::Vector3d& rcenter_position,
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

    void predict(double dts);

    bool update(
        const Eigen::Vector3d& rcenter_position,
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

    void reset_blade_debounce() {
        confirmed_blade_id_       = -1;
        pending_blade_id_         = -1;
        consecutive_change_count_ = 0;
        blade_id_confirmed_       = false;
    }

private:
    void state_machine(bool found);
    void set_measurement(const EnergyUKF::VecZ& z);

    std::array<double, EnergyMeter::NZ> measurement_{};
    EnergyMeter energy_model_{};

    int detecting_count_{0};
    std::map<int, std::string> voter_{};
    int lost_count_{0};
    int current_matched_blade_id_{0};
    int confirmed_blade_id_{-1};
    int pending_blade_id_{-1};
    int consecutive_change_count_{0};
    bool blade_id_confirmed_{false}; // 只有防抖通过后才为 true
};

} // namespace energy_meter
