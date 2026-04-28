#pragma once

#include "L3_estimation/tracker/invariant_extended_kalman_filter.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "ldm_kinematic_model.hpp"
#include "types.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <spdlog/spdlog.h>
#include <utility>

namespace fcs::L3::ldm {

struct NaiveLdmConfig {
    uint32_t tracking_threshold{5};
    double lost_threshold{1.0};

    double initial_sigma_rot{0.05};
    double initial_sigma_velocity_body{5.0};
    double initial_sigma_position{0.20};

    LdmKinematic::Params model{};
};

class LdmInEkfTracker {
public:
    using Model           = LdmKinematic;
    using Nominal         = Model::Nominal;
    using CovXi           = Model::CovXi;
    using PoseMeasurement = Model::PoseMeasurement;
    using Params          = Model::Params;

    LdmInEkfTracker() = default;

    void initialize(const Params& params, const Nominal& x0, const CovXi& P0) {
        model_.params = params;

        PredictFunc f = [this](const Nominal& x, double dt) { return model_.f(x, dt); };
        UpdateQFunc q = [this](double dt) { return model_.Q(dt); };

        ekf_ = InEKF(std::move(f), std::move(q), P0);
        ekf_.setState(x0);
        active_ = true;
    }

    void predict(double dt) {
        if (!active_) {
            return;
        }
        ekf_.predict(std::max(0.0, dt));
    }

    void update(const PoseMeasurement& z) {
        if (!active_) {
            return;
        }
        ekf_.update(
            Model::pose_innovation(ekf_.X(), z), Model::pose_update_H(ekf_.X()), model_.R(z));
    }

    [[nodiscard]] bool active() const noexcept { return active_; }

    [[nodiscard]] const Nominal& nominal() const noexcept { return ekf_.X(); }

    [[nodiscard]] const CovXi& P() const noexcept { return ekf_.P(); }

private:
    using PredictFunc = std::function<Nominal(const Nominal&, double)>;
    using UpdateQFunc = std::function<CovXi(double)>;
    using InEKF       = fcs::L3::InvariantExtendedKalmanFilter<Nominal, Model::NZ>;

    bool active_{false};
    Model model_{};
    InEKF ekf_{};
};

static_assert(LdmKinematic::NX == 9);
static_assert(LdmKinematic::NZ == POSE_UPDATE_MAX);

class LdmTracker {
public:
    using Model           = LdmKinematic;
    using Nominal         = Model::Nominal;
    using CovXi           = Model::CovXi;
    using PoseMeasurement = Model::PoseMeasurement;

    LdmTracker() = default;

    explicit LdmTracker(const NaiveLdmConfig& config)
        : config_(config) {}

    void update(uint64_t timestamp_ns, const std::optional<PoseMeasurement>& measurement) {
        if (status_ == TrackerStatus::Idle) {
            if (measurement.has_value()) {
                initialize_(*measurement, timestamp_ns);
                state_machine_(true);
            }
            return;
        }

        predict_to_(timestamp_ns);

        if (measurement.has_value()) {
            target_.update(*measurement);
            last_observation_timestamp_ns_ = timestamp_ns;
            state_machine_(true);
        } else {
            state_machine_(false);
        }
    }

    void reset() noexcept {
        target_                        = LdmInEkfTracker{};
        status_                        = TrackerStatus::Idle;
        detecting_count_               = 0;
        last_timestamp_ns_             = 0;
        last_observation_timestamp_ns_ = 0;
        last_dt_                       = 0.0;
        lost_time_                     = 0.0;
    }

    [[nodiscard]] TrackerStatus status() const noexcept { return status_; }

    [[nodiscard]] bool is_tracking() const noexcept {
        return status_ == TrackerStatus::Tracking || status_ == TrackerStatus::TempLost;
    }

    [[nodiscard]] std::optional<LdmState> get_output() const noexcept {
        if (!target_.active() || status_ == TrackerStatus::Idle) {
            return std::nullopt;
        }

        const auto& X = target_.nominal();

        LdmState output;
        output.timestamp_ns                  = last_timestamp_ns_;
        output.last_observation_timestamp_ns = last_observation_timestamp_ns_;
        output.status                        = status_;
        output.accurate                      = status_ == TrackerStatus::Tracking;
        output.position                      = X.p();
        output.R_world_body                  = X.R();
        output.velocity_body                 = X.v();
        output.velocity_world                = X.R() * X.v();
        return output;
    }

private:
    void initialize_(const PoseMeasurement& measurement, uint64_t timestamp_ns) {
        Nominal::IsometriesType t{};
        t[0] = Eigen::Vector3d::Zero();
        t[1] = measurement.p_world_body;

        const Nominal x0(measurement.R_world_body, t);

        CovXi P0 = CovXi::Zero();
        P0.template block<3, 3>(0, 0).setIdentity();
        P0.template block<3, 3>(0, 0) *= config_.initial_sigma_rot * config_.initial_sigma_rot;
        P0.template block<3, 3>(3, 3).setIdentity();
        P0.template block<3, 3>(3, 3) *=
            config_.initial_sigma_velocity_body * config_.initial_sigma_velocity_body;
        P0.template block<3, 3>(6, 6).setIdentity();
        P0.template block<3, 3>(6, 6) *=
            config_.initial_sigma_position * config_.initial_sigma_position;

        target_.initialize(config_.model, x0, P0);

        last_timestamp_ns_             = timestamp_ns;
        last_observation_timestamp_ns_ = timestamp_ns;
        last_dt_                       = 0.0;
        lost_time_                     = 0.0;
    }

    void predict_to_(uint64_t timestamp_ns) {
        last_dt_ = 0.0;
        if (timestamp_ns > last_timestamp_ns_ && last_timestamp_ns_ != 0) {
            last_dt_ = static_cast<double>(timestamp_ns - last_timestamp_ns_) * 1e-9;
        }

        if (timestamp_ns > last_timestamp_ns_) {
            last_timestamp_ns_ = timestamp_ns;
        }

        target_.predict(last_dt_);
    }

    void state_machine_(bool found) noexcept {
        switch (status_) {
        case TrackerStatus::Idle:
            if (found) {
                ++detecting_count_;
                status_ = TrackerStatus::Detecting;
            } else {
                reset();
            }
            break;

        case TrackerStatus::Detecting:
            if (found) {
                ++detecting_count_;
                if (detecting_count_ > static_cast<int>(config_.tracking_threshold)) {
                    detecting_count_ = 0;
                    status_          = TrackerStatus::Tracking;
                }
            }
            break;

        case TrackerStatus::Tracking:
            if (!found) {
                status_    = TrackerStatus::TempLost;
                lost_time_ = last_dt_;
            }
            break;

        case TrackerStatus::TempLost:
            if (found) {
                lost_time_ = 0.0;
                status_    = TrackerStatus::Tracking;
            } else {
                lost_time_ += last_dt_;
                if (lost_time_ >= config_.lost_threshold) {
                    reset();
                }
            }
            break;
        }
    }

    NaiveLdmConfig config_{};
    TrackerStatus status_{TrackerStatus::Idle};
    LdmInEkfTracker target_{};

    int detecting_count_{0};
    uint64_t last_timestamp_ns_{0};
    uint64_t last_observation_timestamp_ns_{0};
    double last_dt_{0.0};
    double lost_time_{0.0};
};

} // namespace fcs::L3::ldm
