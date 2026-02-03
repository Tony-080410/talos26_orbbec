#pragma once

#include "config.hpp"
#include "data_associator.hpp"
#include "fcs/core/types.hpp"
#include "inekf.hpp"
#include "inekf_config.hpp"
#include "motion_model.hpp"
#include "spdlog_hook.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <optional>

namespace fcs::L3 {

/// Armor tracker using Iterated SR-CKF
/// Supports both 4-armor robots and 3-armor outposts
class Tracker {
public:
    Tracker() noexcept = default;

    explicit Tracker(const TrackerConfig& config) noexcept
        : config_(config) {
        robot_model_.params   = config_.robot.model;
        outpost_model_.params = config_.outpost.model;
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
        if (status_ == TrackerStatus::TempLost && (lost_time_ + last_dt_) >= lost_threshold) {
            reset();
            return;
        }

        if (robot_ckf_.has_value()) {
            robot_ckf_->predict(dt);
        }

        if (outpost_ckf_.has_value()) {
            outpost_ckf_->predict(dt);
        }

        // InEKF predict (parallel)
        if (robot_inekf_.has_value() && config_.inekf_enabled) {
            robot_inekf_->predict(dt, config_.robot_inekf);
        }
        if (outpost_inekf_.has_value() && config_.inekf_enabled) {
            outpost_inekf_->predict(dt, config_.outpost_inekf);
        }
    }

    /// Update step (measurement update)
    /// @return true if matched successfully
    [[nodiscard]] bool update(const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        measurement_.fill(0.0);

        const bool is_outpost = (target_name_ == ArmorName::Outpost);
        const int armors_num  = is_outpost ? 3 : 4;

        if (is_outpost && !outpost_ckf_) {
            return false;
        }
        if (!is_outpost && !robot_ckf_) {
            return false;
        }

        // Use template-based DataAssociator
        typename DataAssociator<RobotModel>::MatchResult match_result;
        std::vector<int> idx;

        if (is_outpost) {
            auto result = outpost_associator_.match(
                measurements, outpost_ckf_->x(), outpost_ckf_->Sx(), outpost_model_, target_name_,
                matcher_gate, armors_num);

            for (size_t i = 0; i < result.armors.size(); ++i) {
                match_result.armors.push_back(result.armors[i]);
                idx.push_back(result.armor_ids[i]);
            }
        } else {
            auto result = robot_associator_.match(
                measurements, robot_ckf_->x(), robot_ckf_->Sx(), robot_model_, target_name_,
                matcher_gate, armors_num);

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
            RoboCKF::VecZ z;
            const auto& meas = match_result.armors[i];

            const auto ypd             = xyz2ypd(meas.transform->translation());
            const double armor_yaw_raw = meas.yaw();
            const auto& armor_pos_odom = meas.transform->translation();

            if (is_outpost) {
                outpost_model_.armor_id = idx[i];
                const auto z_pred       = outpost_model_.h(outpost_ckf_->x(), idx[i]);
                z[0]                    = unwrap_rad(z_pred[0], ypd[0]);
                z[1]                    = unwrap_rad(z_pred[1], ypd[1]);
                z[2]                    = ypd[2];
                z[3]                    = unwrap_rad(z_pred[3], armor_yaw_raw);
                set_measurement(z, false);
                outpost_ckf_->update(z);

                if (outpost_inekf_.has_value() && config_.inekf_enabled) {
                    const int armor_id = idx[i];
                    constexpr double kAngleStep =
                        2.0 * std::numbers::pi / static_cast<double>(OutpostModel::ARMORS_NUM);
                    const double yaw_offset = static_cast<double>(armor_id) * kAngleStep;

                    if (std::isfinite(armor_yaw_raw)) {
                        const double armor_yaw_pred  = outpost_inekf_->get_yaw() + yaw_offset;
                        const double armor_yaw_meas  = unwrap_rad(armor_yaw_pred, armor_yaw_raw);
                        const double target_yaw_meas = armor_yaw_meas - yaw_offset;
                        outpost_inekf_->update_yaw(target_yaw_meas, config_.outpost_inekf);
                    }

                    outpost_inekf_->update_armor(armor_pos_odom, armor_id, config_.outpost_inekf);
                }
            } else {
                robot_model_.armor_id = idx[i];
                const auto z_pred     = robot_model_.h(robot_ckf_->x(), idx[i]);
                z[0]                  = unwrap_rad(z_pred[0], ypd[0]);
                z[1]                  = unwrap_rad(z_pred[1], ypd[1]);
                z[2]                  = ypd[2];
                z[3]                  = unwrap_rad(z_pred[3], armor_yaw_raw);
                bool is_another_pair  = (idx[i] == 1 || idx[i] == 3);
                set_measurement(z, is_another_pair);
                robot_ckf_->update(z);

                if (robot_inekf_.has_value() && config_.inekf_enabled) {
                    const int armor_id = idx[i];
                    constexpr double kAngleStep =
                        2.0 * std::numbers::pi / static_cast<double>(RobotModel::ARMORS_NUM);
                    const double yaw_offset = static_cast<double>(armor_id) * kAngleStep;

                    // Use direct armor yaw measurement to update target yaw.
                    if (std::isfinite(armor_yaw_raw)) {
                        const double armor_yaw_pred  = robot_inekf_->get_yaw() + yaw_offset;
                        const double armor_yaw_meas  = unwrap_rad(armor_yaw_pred, armor_yaw_raw);
                        const double target_yaw_meas = armor_yaw_meas - yaw_offset;
                        robot_inekf_->update_yaw(target_yaw_meas, config_.robot_inekf);
                    }

                    robot_inekf_->update_armor(armor_pos_odom, armor_id, config_.robot_inekf);
                }
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

        // Find measurement closest to image center (by pixel distance)
        float min_distance = 1e8f;
        int min_idx        = -1;
        for (int i = 0; i < static_cast<int>(measurements.measurements.size()); ++i) {
            const float dist = measurements.measurements[i].distance_to_image_center;
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
        robot_ckf_.reset();
        outpost_ckf_.reset();
        robot_inekf_.reset();
        outpost_inekf_.reset();

        constexpr double v0 = 0.0;

        if (is_outpost) {
            const double x0 = pos.x() + OutpostModel::OUTPOST_RADIUS * std::cos(armor_yaw);
            const double y0 = pos.y() + OutpostModel::OUTPOST_RADIUS * std::sin(armor_yaw);
            const double z0 = pos.z();

            OutpostCKF::VecX xp0{x0, y0, armor_yaw, v0, z0, z0, z0};
            OutpostCKF::MatXX P0 = OutpostCKF::MatXX::Identity();
            outpost_ckf_.emplace(outpost_model_, xp0, P0);
            lost_threshold     = config_.outpost.lost_threshold;
            tracking_threshold = config_.outpost.tracking_threshold;
            matcher_gate       = config_.outpost.matcher_gate;

            // Initialize InEKF (parallel filter)
            if (config_.inekf_enabled) {
                outpost_inekf_.emplace();
                outpost_inekf_->init(Eigen::Vector3d{x0, y0, z0}, armor_yaw, config_.outpost_inekf);
            }
        } else {
            constexpr double kDefaultRadius = 0.23;
            const double x0                 = pos.x() + kDefaultRadius * std::cos(armor_yaw);
            const double y0                 = pos.y() + kDefaultRadius * std::sin(armor_yaw);
            const double z0                 = pos.z();

            RoboCKF::VecX xp0{x0, v0, y0, v0, z0, v0, armor_yaw, v0, kDefaultRadius, kDefaultRadius,
                              0.0};
            RoboCKF::MatXX P0 = RoboCKF::MatXX::Identity();
            robot_ckf_.emplace(robot_model_, xp0, P0);
            lost_threshold     = config_.robot.lost_threshold;
            tracking_threshold = config_.robot.tracking_threshold;
            matcher_gate       = config_.robot.matcher_gate;

            // Initialize InEKF (parallel filter)
            if (config_.inekf_enabled) {
                robot_inekf_.emplace();
                robot_inekf_->init(Eigen::Vector3d{x0, y0, z0}, armor_yaw, config_.robot_inekf);
            }
        }

        state_machine(true);
        return target_name_;
    }

    /// Get current tracker output
    [[nodiscard]] TrackerOutput get_output() const noexcept {
        TrackerOutput output;
        output.status      = status_;
        output.target_name = target_name_;

        if (robot_ckf_) {
            if (robot_inekf_) {
                auto state   = robot_inekf_->get_robot_state();
                output.state = state;
                return output;
            }
            const auto& x = robot_ckf_->x();
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
        } else if (outpost_ckf_) {
            if (outpost_inekf_) {
                auto state   = outpost_inekf_->get_outpost_state();
                output.state = state;
                return output;
            }
            const auto& x = outpost_ckf_->x();
            OutpostTargetState state;
            state.position = {x[O_XC], x[O_YC]};
            state.yaw      = x[O_YAW];
            state.v_yaw    = x[O_VYAW];
            state.z        = {x[O_Z0], x[O_Z1], x[O_Z2]};
            output.state   = state;
        }

        return output;
    }

    /// Get comparison output between InEKF and ISRCKF
    /// Useful for analysis and debugging
    [[nodiscard]] TrackerComparison get_comparison_output() const noexcept {
        TrackerComparison comparison;

        // ISRCKF output
        if (robot_ckf_) {
            const auto& x              = robot_ckf_->x();
            comparison.isrckf.active   = true;
            comparison.isrckf.position = {x[XC], x[YC], x[Z0]};
            comparison.isrckf.velocity = {x[VX], x[VY], x[VZ]};
            comparison.isrckf.yaw      = x[YAW];
            comparison.isrckf.v_yaw    = x[V_YAW];
            // Covariance from robot_ckf_ would need to be added if available
        } else if (outpost_ckf_) {
            const auto& x              = outpost_ckf_->x();
            comparison.isrckf.active   = true;
            comparison.isrckf.position = {x[O_XC], x[O_YC], 0};
            comparison.isrckf.velocity = {0, 0, 0};
            comparison.isrckf.yaw      = x[O_YAW];
            comparison.isrckf.v_yaw    = x[O_VYAW];
        }

        // InEKF output
        if (robot_inekf_.has_value() && robot_inekf_->is_initialized()) {
            comparison.inekf = robot_inekf_->get_filter_output();
        } else if (outpost_inekf_.has_value() && outpost_inekf_->is_initialized()) {
            comparison.inekf = outpost_inekf_->get_filter_output();
        }

        return comparison;
    }

    /// Reset tracker to IDLE state
    void reset() noexcept {
        robot_ckf_.reset();
        outpost_ckf_.reset();
        robot_inekf_.reset();
        outpost_inekf_.reset();
        target_name_ = ArmorName::Invalid;
        measurement_.fill(0.0);
        detecting_count_   = 0;
        last_dt_           = 0.0;
        lost_time_         = 0.0;
        lost_threshold     = 0.0;
        tracking_threshold = 0.0;
        matcher_gate       = 0.0;
        status_            = TrackerStatus::Idle;
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
    std::optional<RoboCKF> robot_ckf_;
    std::optional<OutpostCKF> outpost_ckf_;
    std::optional<InEKF<RobotModel>> robot_inekf_;
    std::optional<InEKF<OutpostModel>> outpost_inekf_;

    // Models
    RobotModel robot_model_;
    OutpostModel outpost_model_;

    /// Timeout before returning to IDLE from TEMP_LOST (seconds), copied from TrackerConfig in
    /// first_meet
    double lost_threshold{0};
    /// Number of consecutive frames required to transition DETECTING -> TRACKING, copied from
    /// TrackerConfig in first_meet
    uint32_t tracking_threshold{0};
    /// Mahalanobis distance gate for target matching, copied from TrackerConfig in first_meet
    double matcher_gate{0};

    /// Data associators
    DataAssociator<RobotModel> robot_associator_;
    DataAssociator<OutpostModel> outpost_associator_;

    /// State machine variables
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
                if (detecting_count_ > tracking_threshold) {
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
                if (lost_time_ >= lost_threshold) {
                    reset();
                }
            }
            break;
        }
    }

    void set_measurement(const RoboCKF::VecZ& z, bool is_another_pair) noexcept {
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
};

} // namespace fcs::L3
