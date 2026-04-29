#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"

#include "L3_estimation/energy_meter_solver/types.hpp"
#include "euler.hpp"

#include <cmath>
#include <limits>

namespace energy_meter {

bool Tracker::update(
    const Eigen::Vector3d& rcenter_position, const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {
    measurement_.fill(0.0);

    if (!energy_ukf.has_value() || target_positions.empty()) {
        state_machine(false);
        return false;
    }

    // ── Target selection: pick the target closest to predicted roll ──
    size_t selected_idx = 0;
    if (target_quats.size() >= 2) {
        const auto& x = energy_ukf->x();
        const double predict_roll =
            (confirmed_blade_id_ >= 0)
                ? normalize_rad(
                      x(ROLL) + static_cast<double>(confirmed_blade_id_) * 2.0 / 5.0 * M_PI)
                : x(ROLL); // 初始状态直接用预测 roll 比较

        double best_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < target_quats.size(); ++i) {
            const auto [roll_i, pitch_i, yaw_i] = math_fuxk::rpy(target_quats[i]);
            const double diff                   = std::abs(normalize_rad(predict_roll - roll_i));
            if (diff < best_diff) {
                best_diff    = diff;
                selected_idx = i;
            }
        }
    }

    const auto& pos               = rcenter_position;
    const auto [roll, pitch, yaw] = math_fuxk::rpy(target_quats[selected_idx]);

    using VecZ = EnergyMeter::VecZ;
    VecZ meas;
    meas[0] = pos.x();
    meas[1] = pos.y();
    meas[2] = pos.z();
    meas[3] = yaw;
    meas[4] = pitch;
    meas[5] = roll;

    const auto& x_pre        = energy_ukf->x();
    constexpr int armors_num = 5;

    double best_cost = std::numeric_limits<double>::max();
    int best_id      = -1;

    for (int id = 0; id < armors_num; ++id) {
        VecZ z_pred = energy_model_.h(x_pre, id);
        VecZ nu     = meas - z_pred;
        nu[5]       = normalize_rad(nu[5]);

        // R is diagonal, so Rinv * nu = nu[i] / R[i,i]
        // d2 = nu' * Rinv * nu = sum(nu[i]^2 / R[i,i])
        const auto R    = energy_model_.R_sqrt(z_pred);
        const double d2 = (nu.array().square() / R.diagonal().array()).sum();

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

    // ── Blade ID debounce: confirm only after 2 consecutive frames ──
    // 初始状态（confirmed == -1）立即确认第一个匹配的 blade_id
    if (confirmed_blade_id_ < 0) {
        confirmed_blade_id_ = best_id;
        pending_blade_id_   = best_id;
        // 1 个靶立即确认，2 个靶需要防抖验证
        blade_id_confirmed_       = (target_positions.size() < 2);
        consecutive_change_count_ = blade_id_confirmed_ ? 0 : 1;
    } else if (best_id == confirmed_blade_id_) {
        consecutive_change_count_ = 0;
        blade_id_confirmed_       = true;
    } else if (target_positions.size() == 2) {
        blade_id_confirmed_ = false;
        if (best_id == pending_blade_id_) {
            consecutive_change_count_++;
        } else {
            pending_blade_id_         = best_id;
            consecutive_change_count_ = 1;
        }
        if (consecutive_change_count_ >= 2) {
            confirmed_blade_id_       = pending_blade_id_;
            consecutive_change_count_ = 0;
            blade_id_confirmed_       = true;
        }
    } else {
        // 1 个靶时立即更新
        confirmed_blade_id_ = best_id;
        pending_blade_id_   = best_id;
        blade_id_confirmed_ = true;
    }
    current_matched_blade_id_ = confirmed_blade_id_;

    EnergyUKF::VecZ z;
    z[0] = pos.x();
    z[1] = pos.y();
    z[2] = pos.z();
    z[3] = yaw;
    z[4] = pitch;
    z[5] = unwrap_rad(x_pre[ROLL] + best_id * 2 * M_PI / 5, meas[5]);

    energy_model_.armor_id = best_id;
    set_measurement(z);
    energy_ukf->update(z);

    return true;
}

void Tracker::step(
    double dts, const Eigen::Vector3d& rcenter_position,
    const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {
    predict(dts);
    update(rcenter_position, target_positions, target_quats);
}

void Tracker::set_measurement(const EnergyUKF::VecZ& z) {
    measurement_[0] = z[0];
    measurement_[1] = z[1];
    measurement_[2] = z[2];
    measurement_[3] = z[3];
    measurement_[4] = z[4];
    measurement_[5] = z[5];
}

void Tracker::predict(double dts) {
    if (energy_ukf.has_value()) {
        energy_ukf->predict(dts);
    }
}

bool Tracker::first_meet_u(
    const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
    const Eigen::Quaterniond& target_quat) {
    const auto [roll, pitch, yaw] = math_fuxk::rpy(target_quat);

    const double xc = r_center.x();
    const double yc = r_center.y();
    const double zc = r_center.z();
    const double v0 = 0.0;

    const double observed_radius = (target_pos - r_center).norm();

    energy_model_.params.radius = observed_radius;

    EnergyMeter::VecX xp0{xc, yc, zc, yaw, pitch, roll, v0};
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
