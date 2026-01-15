#pragma once

#include "config.hpp"
#include "data_associator.hpp"
#include "fcs/core/types.hpp"
#include "inekf_config.hpp"
#include "motion_model.hpp"
#include "spdlog_hook.hpp"
#include "types.hpp"

// InEKF tracker using uni_kf framework
#include "inekf_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <optional>

namespace fcs::L3 {

/// Armor tracker = Data association + state machine + filter(s).
///
/// Problem:
///   The detector produces a set of armor measurements every frame. The set can be messy: false
///   positives, missed detections, and multiple plates of the same target. Downstream modules want
///   one continuous target state that can be predicted forward.
///
/// Intuition:
///   - Association decides which detections belong to the track and which `armor_id` they are.
///   - The filter estimates kinematics/geometry and predicts the next frame.
///   - The state machine debounces: confirm before locking, tolerate brief dropouts.
///
/// Analogy:
///   Like following a person in a crowd: first decide who you are following, then keep following
///   through brief occlusions using your motion prediction.
///
/// Precise (implementation facts that define tuning):
///   - Association ALWAYS uses the ISRCKF prediction (x_pred, Sx_pred) via DataAssociator, even if
///     InEKF output is enabled. Therefore `matcher_gate` and `robot/outpost.model.*` remain the
///     primary knobs for match stability.
///   - The ISRCKF measurement is 4D: z = [yaw, pitch, distance, armor_yaw], where
///   yaw/pitch/distance
///     come from `xyz2ypd(armor_position_odom)` and armor_yaw comes from the PnP yaw.
///   - ISRCKF measurement noise R is a heuristic (`RobotModel::R_diag` / `OutpostModel::R_diag`),
///     not a TOML parameter, and it affects both the filter update and association gating.
///   - ISRCKF geometry states are hard-clamped after each predict step:
///       r0,r1 ∈ [0.17,0.27], h ∈ [-0.2,0.2].
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
            auto& x = *robot_ckf_->x_raw_ptr();
            // Clamp radius/height to keep geometry physically plausible and prevent divergence.
            //
            // Problem: wrong associations or spikes can drive geometry to absurd values and then
            //          make association even worse (positive feedback).
            // Intuition: hard bounds keep the filter "sane" even if process noise is large.
            // Analogy: saturating a control signal / enforcing physical joint limits.
            // Precise: these clamps are applied after every ISRCKF predict(), regardless of Q.
            x[R_0] = std::clamp(x[R_0], 0.17, 0.27);
            x[R_1] = std::clamp(x[R_1], 0.17, 0.27);
            x[H]   = std::clamp(x[H], -0.2, 0.2);
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
                outpost_ckf_->update(z);

                if (outpost_inekf_.has_value() && config_.inekf_enabled) {
                    const int armor_id = idx[i];
                    constexpr double kAngleStep =
                        2.0 * std::numbers::pi / static_cast<double>(OutpostModel::ARMORS_NUM);
                    const double yaw_offset = static_cast<double>(armor_id) * kAngleStep;

                    // Use spherical or Cartesian observation mode based on config
                    if (config_.outpost_inekf.use_spherical_observation) {
                        // Spherical mode: update_armor_spherical includes armor_yaw,
                        // so we skip separate update_yaw to avoid double yaw update.
                        const Eigen::Vector3d armor_ypd{z[0], z[1], z[2]};
                        outpost_inekf_->update_armor_spherical(
                            armor_ypd, z[3], armor_id, config_.outpost_inekf);
                    } else {
                        // Cartesian mode: update yaw separately, then position
                        if (std::isfinite(armor_yaw_raw)) {
                            const double armor_yaw_pred = outpost_inekf_->get_yaw() + yaw_offset;
                            const double armor_yaw_meas = unwrap_rad(armor_yaw_pred, armor_yaw_raw);
                            const double target_yaw_meas = armor_yaw_meas - yaw_offset;
                            outpost_inekf_->update_yaw(target_yaw_meas, config_.outpost_inekf);
                        }
                        outpost_inekf_->update_armor(
                            armor_pos_odom, armor_id, config_.outpost_inekf);
                    }
                }
            } else {
                robot_model_.armor_id = idx[i];
                const auto z_pred     = robot_model_.h(robot_ckf_->x(), idx[i]);
                z[0]                  = unwrap_rad(z_pred[0], ypd[0]);
                z[1]                  = unwrap_rad(z_pred[1], ypd[1]);
                z[2]                  = ypd[2];
                z[3]                  = unwrap_rad(z_pred[3], armor_yaw_raw);
                bool is_another_pair  = (idx[i] == 1 || idx[i] == 3);
                robot_ckf_->update(z);

                if (robot_inekf_.has_value() && config_.inekf_enabled) {
                    const int armor_id = idx[i];
                    constexpr double kAngleStep =
                        2.0 * std::numbers::pi / static_cast<double>(RobotModel::ARMORS_NUM);
                    const double yaw_offset = static_cast<double>(armor_id) * kAngleStep;

                    // Use spherical or Cartesian observation mode based on config
                    if (config_.robot_inekf.use_spherical_observation) {
                        // Spherical mode: update_armor_spherical includes armor_yaw,
                        // so we skip separate update_yaw to avoid double yaw update.
                        const Eigen::Vector3d armor_ypd{z[0], z[1], z[2]};
                        robot_inekf_->update_armor_spherical(
                            armor_ypd, z[3], armor_id, config_.robot_inekf);
                    } else {
                        // Cartesian mode: update yaw separately, then position
                        if (std::isfinite(armor_yaw_raw)) {
                            const double armor_yaw_pred = robot_inekf_->get_yaw() + yaw_offset;
                            const double armor_yaw_meas = unwrap_rad(armor_yaw_pred, armor_yaw_raw);
                            const double target_yaw_meas = armor_yaw_meas - yaw_offset;
                            robot_inekf_->update_yaw(target_yaw_meas, config_.robot_inekf);
                        }
                        robot_inekf_->update_armor(armor_pos_odom, armor_id, config_.robot_inekf);
                    }
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

            // Outpost initialization notes.
            //
            // Problem:
            //   OutpostModel includes separate height states (z0/z1/z2), but at the first detection
            //   we do not know their relative offsets.
            // Intuition:
            //   Start with all z_i equal to the observed z0 and let measurement updates separate
            //   them over time.
            // Analogy:
            //   A flat initial guess that the data will sculpt into the correct shape.
            // Precise:
            //   We initialize [z0,z1,z2] = z0.
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
            // Robot initialization notes (easy to misread from TOML).
            //
            // Problem:
            //   Users often tune `tracker.robot_inekf.radius0/radius1/height` expecting the
            //   *tracker* geometry to change. In this implementation those values affect ONLY the
            //   InEKF.
            //
            // Intuition:
            //   The ISRCKF used for association starts with a fixed geometry. If that initial guess
            //   is far from reality, the early match/update can be biased and harder to stabilize.
            //
            // Analogy:
            //   Changing the dashboard does not change the engine: InEKF config does not change
            //   ISRCKF initialization.
            //
            // Precise:
            //   ISRCKF (robot) is initialized with r0=r1=0.23m and h=0.0m via `kDefaultRadius` and
            //   the last state element in xp0. To make ISRCKF geometry configurable, modify these
            //   initial values.
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

    /// Reset tracker to IDLE state
    void reset() noexcept {
        robot_ckf_.reset();
        outpost_ckf_.reset();
        robot_inekf_.reset();
        outpost_inekf_.reset();
        target_name_       = ArmorName::Invalid;
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

private:
    TrackerConfig config_;
    TrackerStatus status_{TrackerStatus::Idle};
    ArmorName target_name_{ArmorName::Invalid};

    // Filters
    std::optional<RoboCKF> robot_ckf_;
    std::optional<OutpostCKF> outpost_ckf_;
    std::optional<uni_kf::TrackerInEKF_Robot> robot_inekf_;
    std::optional<uni_kf::TrackerInEKF_Outpost> outpost_inekf_;

    // Models
    RobotModel robot_model_;
    OutpostModel outpost_model_;

    /// Timeout before returning to IDLE from TEMP_LOST (seconds), copied from TrackerConfig in
    /// first_meet
    double lost_threshold{0};
    /// Number of consecutive "found" frames required to transition DETECTING -> TRACKING (copied
    /// from TrackerConfig in first_meet).
    ///
    /// Problem: debounce false positives before committing to TRACKING.
    /// Intuition: larger → fewer false locks but slower lock; smaller → faster lock but more
    /// flicker. Analogy: button debounce counter. Precise: implemented as `detecting_count_ >
    /// tracking_threshold` (strict '>'), so it requires `tracking_threshold + 1` consecutive hits.
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
};

} // namespace fcs::L3
