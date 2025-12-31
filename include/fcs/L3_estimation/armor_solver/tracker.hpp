#pragma once

#include "config.hpp"
#include "data_associator.hpp"
#include "motion_model.hpp"
#include "types.hpp"

#include "fcs/core/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace fcs::L3 {

/// Armor tracker using Iterated SR-CKF
/// Supports both 4-armor robots and 3-armor outposts
class Tracker {
public:
    Tracker() noexcept = default;

    explicit Tracker(const TrackerConfig& config) noexcept
        : config_(config) {
        robot_model_.params   = config_.robot_params;
        outpost_model_.params = config_.outpost_params;
    }

    /// Update configuration
    void set_config(const TrackerConfig& config) noexcept {
        config_               = config;
        robot_model_.params   = config_.robot_params;
        outpost_model_.params = config_.outpost_params;
    }

    /// Combined predict + update step
    void step(double dt, const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        predict(dt);
        (void)update(measurements);
    }

    /// Prediction step (time update)
    void predict(double dt) noexcept {
        last_dt_ = std::max(0.0, dt);

        // Check if we've lost the target for too long
        if (status_ == TrackerStatus::TempLost && (lost_time_ + last_dt_) >= config_.lost_thres) {
            reset();
            return;
        }

        if (robot_ukf_.has_value()) {
            robot_ukf_->predict(dt);
            auto& x = *robot_ukf_->x_raw_ptr();
            // Clamp radius and height to prevent divergence
            x[R_0] = std::clamp(x[R_0], 0.17, 0.27);
            x[R_1] = std::clamp(x[R_1], 0.17, 0.27);
            x[H]   = std::clamp(x[H], -0.2, 0.2);
        }

        if (outpost_ukf_.has_value()) {
            outpost_ukf_->predict(dt);
        }
    }

    /// Update step (measurement update)
    /// @return true if matched successfully
    [[nodiscard]] bool update(const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        measurement_.fill(0.0);

        const bool is_outpost = (target_name_ == ArmorName::Outpost);
        const int armors_num  = is_outpost ? 3 : 4;

        if (is_outpost && !outpost_ukf_) {
            return false;
        }
        if (!is_outpost && !robot_ukf_) {
            return false;
        }

        // Use template-based DataAssociator
        typename DataAssociator<RobotModel>::MatchResult match_result;
        std::vector<int> idx;

        if (is_outpost) {
            auto result = outpost_associator_.match(
                measurements, outpost_ukf_->x(), outpost_ukf_->Sx(), outpost_model_, target_name_,
                config_.outpost_matcher_gate, armors_num);

            for (size_t i = 0; i < result.armors.size(); ++i) {
                match_result.armors.push_back(result.armors[i]);
                idx.push_back(result.armor_ids[i]);
            }
        } else {
            auto result = robot_associator_.match(
                measurements, robot_ukf_->x(), robot_ukf_->Sx(), robot_model_, target_name_,
                config_.matcher_gate, armors_num);

            match_result.armors = result.armors;
            idx                 = result.armor_ids;
        }

        if (match_result.armors.empty()) {
            state_machine(false);
            return false;
        }

        state_machine(true);

        // Process matched armors
        for (size_t i = 0; i < match_result.armors.size(); ++i) {
            RoboUKF::VecZ z;
            const auto& meas = match_result.armors[i];

            const auto ypd             = xyz2ypd(meas.transform->translation());
            const double armor_yaw_raw = meas.yaw();

            if (is_outpost) {
                outpost_model_.armor_id = idx[i];
                const auto z_pred       = outpost_model_.h(outpost_ukf_->x(), idx[i]);
                z[0]                    = unwrap_rad(z_pred[0], ypd[0]);
                z[1]                    = unwrap_rad(z_pred[1], ypd[1]);
                z[2]                    = ypd[2];
                z[3]                    = unwrap_rad(z_pred[3], armor_yaw_raw);
                set_measurement(z, false);
                outpost_ukf_->update(z);
            } else {
                robot_model_.armor_id = idx[i];
                const auto z_pred     = robot_model_.h(robot_ukf_->x(), idx[i]);
                z[0]                  = unwrap_rad(z_pred[0], ypd[0]);
                z[1]                  = unwrap_rad(z_pred[1], ypd[1]);
                z[2]                  = ypd[2];
                z[3]                  = unwrap_rad(z_pred[3], armor_yaw_raw);
                bool is_another_pair  = (idx[i] == 1 || idx[i] == 3);
                set_measurement(z, is_another_pair);
                robot_ukf_->update(z);
            }
        }

        return true;
    }

    /// Initialize tracking from first detection
    /// @return Target name if successfully initialized
    [[nodiscard]] std::optional<ArmorName>
        first_meet(const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        if (measurements.empty()) {
            state_machine(false);
            return std::nullopt;
        }

        // Find measurement closest to image center (by distance)
        double min_distance = 1e8;
        int min_idx         = -1;
        for (int i = 0; i < static_cast<int>(measurements.measurements.size()); ++i) {
            const double dist = measurements.measurements[i].distance();
            if (dist < min_distance) {
                min_distance = dist;
                min_idx      = i;
            }
        }

        if (min_idx < 0) {
            state_machine(false);
            return std::nullopt;
        }

        const auto& target = measurements.measurements[min_idx];
        const auto& pos    = target.transform->translation();

        target_name_           = target.name;
        const bool is_outpost  = (target_name_ == ArmorName::Outpost);
        const double armor_yaw = target.yaw();

        // Reset existing filters
        robot_ukf_.reset();
        outpost_ukf_.reset();

        constexpr double v0 = 0.0;

        if (is_outpost) {
            const double x0 = pos.x() + OutpostModel::OUTPOST_RADIUS * std::cos(armor_yaw);
            const double y0 = pos.y() + OutpostModel::OUTPOST_RADIUS * std::sin(armor_yaw);
            const double z0 = pos.z();

            OutpostUKF::VecX xp0{x0, y0, armor_yaw, v0, z0, z0, z0};
            OutpostUKF::MatXX P0 = OutpostUKF::MatXX::Identity();
            outpost_ukf_.emplace(outpost_model_, xp0, P0);
        } else {
            constexpr double kDefaultRadius = 0.23;
            const double x0                 = pos.x() + kDefaultRadius * std::cos(armor_yaw);
            const double y0                 = pos.y() + kDefaultRadius * std::sin(armor_yaw);
            const double z0                 = pos.z();

            RoboUKF::VecX xp0{x0, v0, y0, v0, z0, v0, armor_yaw, v0, kDefaultRadius, kDefaultRadius,
                              0.0};
            RoboUKF::MatXX P0 = RoboUKF::MatXX::Identity();
            robot_ukf_.emplace(robot_model_, xp0, P0);
        }

        state_machine(true);
        return target_name_;
    }

    /// Get current tracker output
    [[nodiscard]] TrackerOutput get_output() const noexcept {
        TrackerOutput output;
        output.status      = status_;
        output.target_name = target_name_;

        if (robot_ukf_) {
            const auto& x = robot_ukf_->x();
            RobotTargetState state;
            state.position   = {x[XC], x[YC], x[Z0]};
            state.velocity   = {x[VX], x[VY], x[VZ]};
            state.yaw        = x[YAW];
            state.v_yaw      = x[V_YAW];
            state.radius0    = x[R_0];
            state.radius1    = x[R_1];
            state.z1         = x[Z0] + x[H];
            state.armors_num = 4;
            output.state     = state;
        } else if (outpost_ukf_) {
            const auto& x = outpost_ukf_->x();
            OutpostTargetState state;
            state.position = {x[O_XC], x[O_YC]};
            state.yaw      = x[O_YAW];
            state.v_yaw    = x[O_VYAW];
            state.z        = {x[O_Z0], x[O_Z1], x[O_Z2]};
            output.state   = state;
        }

        return output;
    }

    /// Reset tracker to IDLE state
    void reset() noexcept {
        robot_ukf_.reset();
        outpost_ukf_.reset();
        target_name_ = ArmorName::Invalid;
        measurement_.fill(0.0);
        detecting_count_ = 0;
        last_dt_         = 0.0;
        lost_time_       = 0.0;
        status_          = TrackerStatus::Idle;
    }

    /// Get current status
    [[nodiscard]] TrackerStatus status() const noexcept { return status_; }

    /// Get target name
    [[nodiscard]] ArmorName target_name() const noexcept { return target_name_; }

    /// Get raw measurement array (for debugging)
    [[nodiscard]] const std::array<double, RobotModel::NZ * 2>& measurement() const noexcept {
        return measurement_;
    }

private:
    TrackerConfig config_;
    TrackerStatus status_{TrackerStatus::Idle};
    ArmorName target_name_{ArmorName::Invalid};

    // Filters
    std::optional<RoboUKF> robot_ukf_;
    std::optional<OutpostUKF> outpost_ukf_;

    // Models
    RobotModel robot_model_;
    OutpostModel outpost_model_;

    // Data associators
    DataAssociator<RobotModel> robot_associator_;
    DataAssociator<OutpostModel> outpost_associator_;

    // State machine variables
    int detecting_count_{0};
    double last_dt_{0.0};
    double lost_time_{0.0};

    // Debug measurement storage
    std::array<double, RobotModel::NZ * 2> measurement_{};

    void state_machine(bool found) noexcept {
        switch (status_) {
        case TrackerStatus::Idle:
            if (found) {
                detecting_count_++;
                if (detecting_count_ > 0) {
                    status_ = TrackerStatus::Detecting;
                }
            } else {
                reset();
            }
            break;

        case TrackerStatus::Detecting:
            if (found) {
                detecting_count_++;
                if (detecting_count_ > config_.tracking_thres) {
                    status_          = TrackerStatus::Tracking;
                    detecting_count_ = 0;
                }
            } else {
                reset();
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
                if (lost_time_ >= config_.lost_thres) {
                    reset();
                }
            }
            break;
        }
    }

    void set_measurement(const RoboUKF::VecZ& z, bool is_another_pair) noexcept {
        if (is_another_pair) {
            measurement_[4] = z[0];
            measurement_[5] = z[1];
            measurement_[6] = z[2];
            measurement_[7] = z[3];
        } else {
            measurement_[0] = z[0];
            measurement_[1] = z[1];
            measurement_[2] = z[2];
            measurement_[3] = z[3];
        }
    }

    [[nodiscard]] static double quaternion_to_yaw(const Eigen::Quaterniond& q) noexcept {
        const double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        const double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }
};

} // namespace fcs::L3
