#include "L3_estimation/energy_meter_solver/energy_meter_tracker.hpp"

#include "L3_estimation/energy_meter_solver/types.hpp"
#include "euler.hpp"

#include <cmath>
#include <limits>

namespace energy_meter {

void Tracker::init_ekf(const Model::VecX& x0, const Model::MatXX& P0) {
    auto f = [this](const JetX* x, JetX* xp) {
        Model::template predict_state<JetX>(x, JetX(dt_), dir_, xp);
    };
    auto h = [this](const JetX* x, JetX* z) {
        Model::template measure_state<JetX>(x, armor_id_, z);
    };
    auto q = [this]() -> Model::MatXX { return model_.Q(dt_); };
    auto r = [this](const Model::VecZ& z) -> Model::MatZZ { return model_.R(z); };

    ekf_.emplace(f, h, q, r, P0);
    ekf_->setState(x0);
}

bool Tracker::update(
    const Eigen::Vector3d& rcenter_position, const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {

    if (!ekf_ || target_positions.empty()) {
        state_machine(false);
        return false;
    }

    const auto& pos               = rcenter_position;
    const auto euler              = math_fuxk::rpy(target_quats[0]);
    const auto [roll, pitch, yaw] = euler.rpy();

    using VecZ = Model::VecZ;
    VecZ meas;
    meas[CENTER_X]   = pos.x();
    meas[CENTER_Y]   = pos.y();
    meas[CENTER_Z]   = pos.z();
    meas[RUNE_YAW]   = yaw;
    meas[RUNE_PITCH] = pitch;
    meas[ARMOR_ROLL] = roll;

    const auto& x_pre        = ekf_->X();
    constexpr int armors_num = Model::ARMORS_NUM;

    double best_cost = std::numeric_limits<double>::max();
    int best_id      = -1;

    for (int id = 0; id < armors_num; ++id) {
        VecZ z_pred    = model_.h(x_pre, id);
        VecZ nu        = meas - z_pred;
        nu[ARMOR_ROLL] = normalize_rad(nu[ARMOR_ROLL]);

        const auto R_diag = model_.R_diag(z_pred);
        const double d2   = (nu.array().square() / R_diag.array()).sum();

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

    // Build unwrapped measurement for EKF update
    VecZ z;
    z[CENTER_X]   = pos.x();
    z[CENTER_Y]   = pos.y();
    z[CENTER_Z]   = pos.z();
    z[RUNE_YAW]   = yaw;
    z[RUNE_PITCH] = pitch;
    z[ARMOR_ROLL] = unwrap_rad(x_pre[THETA] + best_id * 2 * M_PI / 5, meas[ARMOR_ROLL]);

    armor_id_ = best_id;
    ekf_->update(z);

    // ── Direction detection from raw observations ──
    // Use the raw observed roll (not EKF state) to avoid feedback bias.
    // The EKF prediction uses dir_ which defaults to +1, contaminating
    // the EKF state's Δθ. Voting over many frames gives robust detection.
    if (!dir_locked_) {
        if (dir_detect_count_ > 0) {
            const double droll         = normalize_rad(meas[ARMOR_ROLL] - prev_observed_roll_);
            constexpr double kMinDelta = 0.005; // ~0.3° minimum to avoid noise
            if (droll > kMinDelta) {
                dir_votes_pos_++;
            } else if (droll < -kMinDelta) {
                dir_votes_neg_++;
            }
        }
        prev_observed_roll_ = meas[ARMOR_ROLL];
        dir_detect_count_++;

        if (dir_detect_count_ >= DIR_DETECT_FRAMES) {
            dir_        = (dir_votes_pos_ >= dir_votes_neg_) ? 1 : -1;
            dir_locked_ = true;
            SPDLOG_INFO(
                "energy_meter: direction locked: dir={}, votes +{}/-{}", dir_, dir_votes_pos_,
                dir_votes_neg_);
        }
    }

    return true;
}

void Tracker::predict(double dt) {
    if (!ekf_) {
        return;
    }
    dt_ = std::max(0.0, dt);
    ekf_->predict();
}

void Tracker::step(
    double dt, const Eigen::Vector3d& rcenter_position,
    const std::vector<Eigen::Vector3d>& target_positions,
    const std::vector<Eigen::Quaterniond>& target_quats) {
    predict(dt);
    update(rcenter_position, target_positions, target_quats);
}

bool Tracker::first_meet_u(
    const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
    const Eigen::Quaterniond& target_quat) {

    const auto euler              = math_fuxk::rpy(target_quat);
    const auto [roll, pitch, yaw] = euler.rpy();

    const double observed_radius = (target_pos - r_center).norm();
    model_.params.radius         = observed_radius;

    // State: [xc, yc, zc, yaw, pitch, θ, τ, α, β]
    // Initialize: θ = observed roll, τ = 0, α = 0 (→ a ≈ 0.9125), β = 0 (→ ω ≈ 1.942)
    Model::VecX x0 = Model::VecX::Zero();
    x0(XC)         = r_center.x();
    x0(YC)         = r_center.y();
    x0(ZC)         = r_center.z();
    x0(YAW)        = yaw;
    x0(PITCH)      = pitch;
    x0(THETA)      = roll;
    x0(TAU)        = 0.0;
    x0(ALPHA)      = 0.0; // sigmoid(0) = 0.5 → a = 0.9125
    x0(BETA)       = 0.0; // sigmoid(0) = 0.5 → ω = 1.942

    Model::MatXX P0 = Model::MatXX::Identity() * 0.1;
    // Large uncertainty on model parameters
    P0(ALPHA, ALPHA) = 2.0;
    P0(BETA, BETA)   = 2.0;
    // Time is known exactly
    P0(TAU, TAU) = 0.0;

    init_ekf(x0, P0);

    // Reset direction detection
    dir_                = 1;
    prev_observed_roll_ = roll;
    dir_detect_count_   = 0;
    dir_votes_pos_      = 0;
    dir_votes_neg_      = 0;
    dir_locked_         = false;

    state_machine(true);
    return true;
}

void Tracker::reset_on_activate(double observed_roll) {
    if (!ekf_) {
        return;
    }

    const auto& x_old = ekf_->X();

    Model::VecX x0 = x_old;
    x0(THETA)      = observed_roll;
    x0(TAU)        = 0.0;
    x0(ALPHA)      = 0.0;
    x0(BETA)       = 0.0;

    Model::MatXX P0 = Model::MatXX::Identity() * 0.1;
    // Keep position/yaw/pitch covariance from previous state
    for (int i = 0; i <= PITCH; ++i) {
        P0(i, i) = ekf_->P()(i, i);
    }
    P0(ALPHA, ALPHA) = 2.0;
    P0(BETA, BETA)   = 2.0;
    P0(TAU, TAU)     = 0.0;

    init_ekf(x0, P0);

    // Reset direction detection
    dir_detect_count_   = 0;
    dir_votes_pos_      = 0;
    dir_votes_neg_      = 0;
    dir_locked_         = false;
    prev_observed_roll_ = observed_roll;
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
            ekf_.reset();
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
            ekf_.reset();
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
                ekf_.reset();
                state = IDLE;
            }
        }
        break;
    }
}

} // namespace energy_meter
