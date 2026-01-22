#pragma once

#include "fcs/L3_estimation/energy_meter_solver/isrckf.hpp"
#include "fcs/L3_estimation/energy_meter_solver/motion_model.hpp"

#include <Eigen/Dense>
#include <expected>
#include <string>

#include <array>
#include <map>
#include <optional>
#include <vector>

// Forward declaration to avoid including toml++ in headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace energy_meter {

struct Tracker {
    struct Params {
        int lost_thres{10};
        int tracking_thres{3};
        double matcher_gate{25.0};
        EnergyMeter::Params energy_params{};

        std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
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

    auto getRadius() const -> double { return energy_model_.params.radius; }

    bool first_meet_u(
        const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
        const Eigen::Quaterniond& target_quat);

    void step(
        double dts, const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

    void predict(double dts);

    bool update(
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

private:
    void state_machine(bool found);
    void set_measurement(const EnergyUKF::VecZ& z);

    std::array<double, EnergyMeter::NZ> measurement_{};
    EnergyMeter energy_model_{};

    int detecting_count_{0};
    std::map<int, std::string> voter_{};
    int lost_count_{0};
    int current_matched_blade_id_{0};
};

} // namespace energy_meter
