#pragma once

#include "config.hpp"
#include "core/types.hpp"
#include "data_associator.hpp"
#include "inekf_config.hpp"
#include "motion_model.hpp"
#include "spdlog_hook.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <optional>
#include <vector>

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
            const double angle_step  = 2.0 * std::numbers::pi / static_cast<double>(armors_num);
            const double phase_shift = outpost_ckf_->x()[O_VYAW] * last_dt_ / angle_step;
            const double phase_std =
                compute_phase_std_(outpost_ckf_->Sx(), O_YAW, O_VYAW, last_dt_, angle_step);
            const auto prior =
                build_mode_prior_(armors_num, outpost_last_armor_id_, phase_shift, phase_std);
            auto result = outpost_associator_.match(
                measurements, outpost_ckf_->x(), outpost_ckf_->Sx(), outpost_model_, target_name_,
                matcher_gate, armors_num, &prior);

            // Pre-allocate for at most armors_num elements
            match_result.armors.reserve(result.armors.size());
            idx.reserve(result.armor_ids.size());
            match_result.costs.reserve(result.costs.size());

            for (size_t i = 0; i < result.armors.size(); ++i) {
                match_result.armors.push_back(result.armors[i]);
                idx.push_back(result.armor_ids[i]);
                match_result.costs.push_back(result.costs[i]);
            }
        } else {
            const double angle_step  = 2.0 * std::numbers::pi / static_cast<double>(armors_num);
            const double phase_shift = robot_ckf_->x()[V_YAW] * last_dt_ / angle_step;
            const double phase_std =
                compute_phase_std_(robot_ckf_->Sx(), YAW, V_YAW, last_dt_, angle_step);
            const auto prior =
                build_mode_prior_(armors_num, robot_last_armor_id_, phase_shift, phase_std);
            auto result = robot_associator_.match(
                measurements, robot_ckf_->x(), robot_ckf_->Sx(), robot_model_, target_name_,
                matcher_gate, armors_num, &prior);

            match_result.armors = result.armors;
            idx                 = result.armor_ids;
            match_result.costs  = result.costs;
        }

        if (match_result.armors.empty()) {
            state_machine(false);
            return false;
        }

        state_machine(true);

        std::optional<size_t> best_match_i;
        if (!idx.empty()) {
            size_t best_i = 0;
            if (match_result.costs.size() == idx.size()) {
                best_i = static_cast<size_t>(std::distance(
                    match_result.costs.begin(),
                    std::min_element(match_result.costs.begin(), match_result.costs.end())));
            }
            best_match_i = best_i;

            if (is_outpost) {
                outpost_last_armor_id_ = idx[best_i];
            } else {
                robot_last_armor_id_ = idx[best_i];
            }
        }

        // Process matched armors
        for (size_t i = 0; i < match_result.armors.size(); ++i) {
            RoboCKF::VecZ z;
            const auto& meas           = match_result.armors[i];
            const auto ypd             = xyz2ypd(meas.transform->translation());
            const double armor_yaw_raw = meas.yaw();

            if (is_outpost) {
                outpost_model_.armor_id = idx[i];
                const auto z_pred       = outpost_model_.h(outpost_ckf_->x(), idx[i]);
                z[0]                    = unwrap_rad(z_pred[0], ypd[0]);
                z[1]                    = unwrap_rad(z_pred[1], ypd[1]);
                z[2]                    = ypd[2];
                z[3]                    = unwrap_rad(z_pred[3], armor_yaw_raw);
                set_measurement(z, false);
                outpost_ckf_->update(z);

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
        robot_last_armor_id_.reset();
        outpost_last_armor_id_.reset();
        constexpr double v0 = 0.0;

        if (is_outpost) {
            const double x0 = pos.x() + OutpostModel::OUTPOST_RADIUS * std::cos(armor_yaw);
            const double y0 = pos.y() + OutpostModel::OUTPOST_RADIUS * std::sin(armor_yaw);
            const double z0 = pos.z();

            OutpostCKF::VecX xp0{x0, y0, armor_yaw, v0, z0, z0, z0};
            OutpostCKF::MatXX P0 = OutpostCKF::MatXX::Identity();
            outpost_ckf_.emplace(outpost_model_, xp0, P0);
            outpost_ckf_->set_max_iters(5);
            outpost_ckf_->set_iter_tol(1e-8);
            // By construction, first_meet initializes state yaw from the observed armor yaw.
            // This establishes a local phase convention where the observed armor is treated as
            // id=0.
            outpost_last_armor_id_ = 0;
            lost_threshold         = config_.outpost.lost_threshold;
            tracking_threshold     = config_.outpost.tracking_threshold;
            matcher_gate           = config_.outpost.matcher_gate;

        } else {
            const double r0_prior = std::clamp(config_.robot_inekf.radius0, 0.10, 0.50);
            const double r1_prior = std::clamp(config_.robot_inekf.radius1, 0.10, 0.50);
            const double h_prior  = std::clamp(config_.robot_inekf.height, -0.20, 0.20);
            // Single-frame first meet cannot disambiguate high/low armor parity reliably.
            // Initialize robot center with a parity-neutral geometric prior instead of assuming
            // the observed armor is id=0 (low armor).
            const double r_init = 0.5 * (r0_prior + r1_prior);
            const double x0     = pos.x() + r_init * std::cos(armor_yaw);
            const double y0     = pos.y() + r_init * std::sin(armor_yaw);
            const double z0     = pos.z() - 0.5 * h_prior;

            RoboCKF::VecX xp0{x0, v0, y0, v0, z0, v0, armor_yaw, v0, r0_prior, r1_prior, h_prior};
            RoboCKF::MatXX P0 = RoboCKF::MatXX::Identity();
            // Faster maneuver convergence: keep geometry tightly constrained, leave kinematics
            // loose.
            P0(VX, VX)       = 9.0;
            P0(VY, VY)       = 9.0;
            P0(VZ, VZ)       = 16.0;
            P0(V_YAW, V_YAW) = 1.0;
            P0(R_0, R_0)     = 1e-2;
            P0(R_1, R_1)     = 1e-2;
            P0(H, H)         = 1e-2;
            robot_ckf_.emplace(robot_model_, xp0, P0);
            robot_ckf_->set_max_iters(5);
            robot_ckf_->set_iter_tol(1e-8);
            // Do not force a first-frame armor id: for alternating high/low armor geometry this
            // can inject a parity error into phase prior and bootstrap state.
            robot_last_armor_id_.reset();
            lost_threshold     = config_.robot.lost_threshold;
            tracking_threshold = config_.robot.tracking_threshold;
            matcher_gate       = config_.robot.matcher_gate;
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

        return comparison;
    }

    /// Reset tracker to IDLE state
    void reset() noexcept {
        robot_ckf_.reset();
        outpost_ckf_.reset();
        robot_last_armor_id_.reset();
        outpost_last_armor_id_.reset();
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
    std::optional<RoboCKF> robot_ckf_;
    std::optional<OutpostCKF> outpost_ckf_; // Models
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
    std::optional<int> robot_last_armor_id_{};
    std::optional<int> outpost_last_armor_id_{};

    template <typename Mat>
    static double diag_var_from_sqrt_cov_(const Mat& Sx, int idx) noexcept {
        if (idx < 0 || idx >= Sx.rows()) {
            return 0.0;
        }
        return Sx.row(idx).squaredNorm();
    }

    template <typename Mat>
    static double compute_phase_std_(
        const Mat& Sx, int yaw_idx, int vyaw_idx, double dt, double angle_step) noexcept {
        (void)yaw_idx;
        if (angle_step <= 0.0) {
            return 1.0;
        }
        const double var_vyaw = diag_var_from_sqrt_cov_(Sx, vyaw_idx);
        // Use inter-frame phase uncertainty (Delta yaw) instead of absolute yaw variance.
        // Absolute yaw carries gauge ambiguity with armor index and would over-weaken mode priors.
        const double phase_var = std::max(0.0, (dt * dt * var_vyaw) / (angle_step * angle_step));
        return std::sqrt(phase_var);
    }

    static double wrap_index_diff_(double d, int n) noexcept {
        while (d > 0.5 * static_cast<double>(n)) {
            d -= static_cast<double>(n);
        }
        while (d < -0.5 * static_cast<double>(n)) {
            d += static_cast<double>(n);
        }
        return d;
    }

    static std::vector<double> build_mode_prior_(
        int armors_num, std::optional<int> last_id, double phase_shift, double phase_std) {
        std::vector<double> prior(armors_num, 1.0 / static_cast<double>(armors_num));
        if (!last_id.has_value() || armors_num <= 1) {
            return prior;
        }

        const double center  = static_cast<double>(*last_id) + phase_shift;
        const double sigma   = std::clamp(0.28 + 0.95 * phase_std, 0.25, 2.20);
        const double inv_var = 1.0 / (sigma * sigma);

        // MAP prior confidence follows state uncertainty: uncertain -> closer to uniform.
        const double uniform_mix = std::clamp(0.02 + 0.30 * phase_std, 0.02, 0.35);

        // Kinematic transition support in discrete armor index space.
        // One frame cannot physically skip many armor IDs unless phase uncertainty is large.
        const int max_index_step = std::clamp(
            static_cast<int>(std::ceil(std::abs(phase_shift) + 3.0 * phase_std + 0.25)), 1,
            armors_num / 2 + 1);

        double sum = 0.0;
        for (int j = 0; j < armors_num; ++j) {
            const double diff = wrap_index_diff_(static_cast<double>(j) - center, armors_num);
            if (std::abs(diff) > static_cast<double>(max_index_step)) {
                prior[j] = 0.0;
                continue;
            }
            const double g = std::exp(-0.5 * diff * diff * inv_var);
            prior[j] = (1.0 - uniform_mix) * g + uniform_mix / static_cast<double>(armors_num);
            sum += prior[j];
        }
        if (sum > 0.0) {
            for (double& p : prior) {
                p /= sum;
            }
        } else {
            // Fallback to local support if numerical underflow or all candidates were rejected.
            for (int j = 0; j < armors_num; ++j) {
                const double diff = wrap_index_diff_(static_cast<double>(j) - center, armors_num);
                if (std::abs(diff) <= static_cast<double>(max_index_step)) {
                    prior[j] = 1.0;
                    sum += 1.0;
                }
            }
            if (sum > 0.0) {
                for (double& p : prior) {
                    p /= sum;
                }
            }
        }
        return prior;
    }

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
