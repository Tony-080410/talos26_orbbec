#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"

#include "L3_estimation/energy_meter_solver/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace energy_meter {

bool Tracker::update(
    const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {
    measurement_.fill(0.0);

    if (!energy_ukf.has_value() || target_positions.empty()) {
        state_machine(false);
        return false;
    }

    const auto& pos  = target_positions[0];
    const auto& quat = target_quats[0];

    using VecZ = EnergyMeter::VecZ;
    VecZ meas;
    meas[0] = pos.x();
    meas[1] = pos.y();
    meas[2] = pos.z();
    meas[3] = orientation2roll(quat);

    const auto& x_pre        = energy_ukf->x();
    constexpr int armors_num = 5;

    double best_cost = std::numeric_limits<double>::max();
    int best_id      = -1;

    for (int id = 0; id < armors_num; ++id) {
        VecZ z_pred = energy_model_.h(x_pre, id);
        VecZ nu     = meas - z_pred;
        nu[3]       = normalize_rad(nu[3]);

        const auto R = energy_model_.R_sqrt(z_pred);
        const Eigen::Matrix<double, EnergyMeter::NZ, EnergyMeter::NZ> Rinv = R.inverse();
        const double d2 = (nu.transpose() * Rinv * nu)(0, 0);

        if (std::isfinite(d2) && d2 < params.matcher_gate && d2 < best_cost) {
            best_cost = d2;
            best_id   = id;
        }
    }

    if (best_id < 0) {
        state_machine(false);
        return false;
    }

    state_machine(true);
    current_matched_blade_id_ = best_id;

    EnergyUKF::VecZ z;
    z[0] = pos.x();
    z[1] = pos.y();
    z[2] = pos.z();
    z[3] = unwrap_rad(x_pre[ROLL] + best_id * 2 * M_PI / 5, meas[3]);

    energy_model_.armor_id = best_id;
    set_measurement(z);
    energy_ukf->update(z);

    return true;
}

void Tracker::step(
    double dts, const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {
    predict(dts);
    update(target_positions, target_quats);
}

void Tracker::set_measurement(const EnergyUKF::VecZ& z) {
    measurement_[0] = z[0];
    measurement_[1] = z[1];
    measurement_[2] = z[2];
    measurement_[3] = z[3];
}

void Tracker::predict(double dts) {
    if (energy_ukf.has_value()) {
        energy_ukf->predict(dts);
    }
}

bool Tracker::first_meet_u(
    const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
    const Eigen::Quaterniond& target_quat) {
    const double initial_roll = orientation2roll(target_quat);

    const double xc = r_center.x();
    const double yc = r_center.y();
    const double zc = r_center.z();
    const double v0 = 0.0;

    const double observed_radius = (target_pos - r_center).norm();
    if (observed_radius > 0.1) {
        energy_model_.params.radius = observed_radius;
    }

    EnergyMeter::VecX xp0{xc, yc, zc, initial_roll, v0};
    EnergyUKF::MatXX p0 = EnergyUKF::MatXX::Identity();
    energy_ukf.emplace(energy_model_, xp0, p0);

    state_machine(true);
    return true;
}

void Tracker::state_machine(bool found) {
    switch (state) {
    case IDLE:
        if (found) {
            ++detecting_count_;
            if (detecting_count_ > 0) {
                state = DETECTING;
            }
        } else {
            energy_ukf.reset();
            detecting_count_ = 0;
        }
        break;

    case DETECTING:
        if (found) {
            ++detecting_count_;
            if (detecting_count_ > params.tracking_thres) {
                state            = TRACKING;
                detecting_count_ = 0;
            }
        } else {
            detecting_count_ = 0;
            energy_ukf.reset();
            state = IDLE;
        }
        break;

    case TRACKING:
        if (!found) {
            state       = TEMP_LOST;
            lost_count_ = 1;
        }
        break;

    case TEMP_LOST:
        if (found) {
            lost_count_ = 0;
            state       = TRACKING;
        } else {
            ++lost_count_;
            if (lost_count_ > params.lost_thres) {
                lost_count_      = 0;
                detecting_count_ = 0;
                energy_ukf.reset();
                state = IDLE;
            }
        }
        break;
    }
}

} // namespace energy_meter
