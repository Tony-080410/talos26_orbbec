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
#include <limits>
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

        if (!is_outpost) {
            if (const auto handled = try_robot_single_armor_center_map_update_(measurements)) {
                return *handled;
            }
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

        const bool use_robot_virtual_center_update =
            (!is_outpost && should_use_robot_virtual_center_update_(match_result.armors.size()));

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
                bool updated = false;
                if (use_robot_virtual_center_update) {
                    if (const auto virt = build_robot_virtual_center_measurement_(idx[i], z)) {
                        updated = apply_robot_virtual_center_update_(virt->first, virt->second);
                    }
                }
                if (!updated) {
                    robot_ckf_->update(z);
                }
                apply_robot_known_geometry_constraint_();
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
            P0(VZ, VZ)       = 1.0;
            P0(V_YAW, V_YAW) = 36.0;
            P0(R_0, R_0)     = 1e-6;
            P0(R_1, R_1)     = 1e-6;
            P0(H, H)         = 1e-6;
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

    [[nodiscard]] bool
        should_use_robot_virtual_center_update_(size_t matched_armors) const noexcept {
        if (!robot_ckf_ || matched_armors != 1) {
            return false;
        }
        const bool geometry_known = robot_geometry_is_known_();
        if (!geometry_known) {
            return false;
        }
        return std::isfinite(config_.robot_inekf.radius0)
            && std::isfinite(config_.robot_inekf.radius1)
            && std::isfinite(config_.robot_inekf.height);
    }

    [[nodiscard]] bool robot_geometry_is_known_() const noexcept {
        return (config_.robot.model.sigma_r0 <= 1e-6) && (config_.robot.model.sigma_r1 <= 1e-6)
            && (config_.robot.model.sigma_h <= 1e-6);
    }

    [[nodiscard]] std::optional<Eigen::Matrix<double, 4, 1>> robot_virtual_center_meas_from_raw_(
        int armor_id, const RoboCKF::VecZ& z_raw) const noexcept {
        if (armor_id < 0 || armor_id >= RobotModel::ARMORS_NUM) {
            return std::nullopt;
        }
        const bool is_high =
            (armor_id == RobotModel::kHighArmorIdA || armor_id == RobotModel::kHighArmorIdB);
        const double r0 = std::clamp(config_.robot_inekf.radius0, 0.10, 0.50);
        const double r1 = std::clamp(config_.robot_inekf.radius1, 0.10, 0.50);
        const double h  = std::clamp(config_.robot_inekf.height, -0.20, 0.20);
        const double r  = is_high ? r1 : r0;

        Eigen::Matrix<double, 4, 1> zv;
        const Eigen::Vector3d xyz   = ypd2xyz({z_raw[0], z_raw[1], std::max(0.05, z_raw[2])});
        zv[0]                       = xyz.x() + r * std::cos(z_raw[3]);
        zv[1]                       = xyz.y() + r * std::sin(z_raw[3]);
        zv[2]                       = xyz.z() - (is_high ? h : 0.0);
        constexpr double kAngleStep = 2.0 * std::numbers::pi / 4.0;
        zv[3]                       = z_raw[3] - static_cast<double>(armor_id) * kAngleStep;
        return zv;
    }

    [[nodiscard]] Eigen::Matrix<double, 4, 4>
        robot_virtual_center_raw_cov_(const RoboCKF::VecZ& z_raw) const noexcept {
        const auto& p                     = robot_model_.params;
        Eigen::Matrix<double, 4, 4> R_raw = Eigen::Matrix<double, 4, 4>::Zero();
        R_raw(0, 0)                       = std::max(1e-8, p.center_meas_yaw_var_floor);
        R_raw(1, 1)                       = std::max(1e-8, p.center_meas_pitch_var_floor);
        R_raw(2, 2)                       = std::max(1e-8, p.center_meas_dist_var_floor);
        R_raw(3, 3)                       = std::max(
            1e-8, p.center_meas_armor_yaw_var_floor
                      + p.center_meas_armor_yaw_range_k * std::log(std::abs(z_raw[2]) + 1.0));
        return R_raw;
    }

    [[nodiscard]] std::optional<std::pair<Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 4, 4>>>
        build_robot_virtual_center_measurement_from_raw_cov_(
            int armor_id, const RoboCKF::VecZ& z_raw,
            const Eigen::Matrix<double, 4, 4>& R_raw) const noexcept {
        const auto zv0 = robot_virtual_center_meas_from_raw_(armor_id, z_raw);
        if (!zv0.has_value()) {
            return std::nullopt;
        }

        const bool is_high =
            (armor_id == RobotModel::kHighArmorIdA || armor_id == RobotModel::kHighArmorIdB);
        const double r0                = std::clamp(config_.robot_inekf.radius0, 0.10, 0.50);
        const double r1                = std::clamp(config_.robot_inekf.radius1, 0.10, 0.50);
        const double r                 = is_high ? r1 : r0;
        const double yaw               = z_raw[0];
        const double pitch             = z_raw[1];
        const double dist_raw          = z_raw[2];
        const double dist_eff          = std::max(0.05, dist_raw);
        const double armor_yaw         = z_raw[3];
        const double d_dist_eff_d_dist = (dist_raw >= 0.05) ? 1.0 : 0.0;

        Eigen::Matrix<double, 4, 4> J = Eigen::Matrix<double, 4, 4>::Zero();
        Eigen::Matrix3d J_xyz_ypd     = ypd2xyz_jacobian({yaw, pitch, dist_eff});
        // The virtual center builder clamps distance at 5cm; linearization must match that
        // piecewise model exactly.
        J_xyz_ypd.col(2) *= d_dist_eff_d_dist;

        J.block<3, 3>(0, 0) = J_xyz_ypd;
        J(0, 3)             = -r * std::sin(armor_yaw);
        J(1, 3)             = r * std::cos(armor_yaw);
        J(3, 3)             = 1.0;

        const double center_cov_inflation =
            std::max(0.0, robot_model_.params.center_meas_cov_inflation);
        Eigen::Matrix<double, 4, 4> Rv = J * R_raw * J.transpose();
        Rv                             = (Rv + Rv.transpose()) * 0.5;
        // Mild inflation keeps this center-space update conservative under nonlinear
        // ypd->xyz propagation and unmodeled camera/PnP effects.
        Rv *= center_cov_inflation;
        Rv.diagonal().array() += 1e-8;
        // Yaw0 from armor_yaw is typically accurate; keep a floor for numerical robustness.
        Rv(3, 3) = std::max(Rv(3, 3), 1e-6);
        return std::pair{*zv0, Rv};
    }

    [[nodiscard]] std::optional<std::pair<Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 4, 4>>>
        build_robot_virtual_center_measurement_(
            int armor_id, const RoboCKF::VecZ& z_raw) const noexcept {
        return build_robot_virtual_center_measurement_from_raw_cov_(
            armor_id, z_raw, robot_virtual_center_raw_cov_(z_raw));
    }

    [[nodiscard]] bool apply_robot_virtual_center_update_(
        const Eigen::Matrix<double, 4, 1>& z_v, const Eigen::Matrix<double, 4, 4>& R_v) noexcept {
        if (!robot_ckf_) {
            return false;
        }
        apply_robot_known_geometry_constraint_();
        auto* x  = robot_ckf_->x_raw_ptr();
        auto* Sx = robot_ckf_->Sx_raw_ptr();
        if (x == nullptr || Sx == nullptr) {
            return false;
        }

        using MatX  = RoboCKF::MatXX;
        using Mat4  = Eigen::Matrix<double, 4, 4>;
        using Mat4X = Eigen::Matrix<double, 4, RobotModel::NX>;
        using MatX4 = Eigen::Matrix<double, RobotModel::NX, 4>;
        using Vec4  = Eigen::Matrix<double, 4, 1>;

        Mat4X H   = Mat4X::Zero();
        H(0, XC)  = 1.0;
        H(1, YC)  = 1.0;
        H(2, Z0)  = 1.0;
        H(3, YAW) = 1.0;

        MatX P = (*Sx) * Sx->transpose();
        Mat4 S = H * P * H.transpose() + R_v;
        S      = (S + S.transpose()) * 0.5;
        S.diagonal().array() += 1e-10;

        const Eigen::LDLT<Mat4> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            return false;
        }

        Vec4 nu;
        nu[0] = z_v[0] - (*x)[XC];
        nu[1] = z_v[1] - (*x)[YC];
        nu[2] = z_v[2] - (*x)[Z0];
        nu[3] = shortest_rad((*x)[YAW], z_v[3]);

        const MatX4 PHt = P * H.transpose();
        const MatX4 K   = ldlt.solve(PHt.transpose()).transpose();

        *x += K * nu;
        (*x)[YAW] = unwrap_rad(z_v[3], (*x)[YAW]);

        MatX P_post = P - K * S * K.transpose();
        P_post      = (P_post + P_post.transpose()) * 0.5;
        *Sx         = ckf::detail::chol_from_ldlt<RobotModel::NX, double>(P_post);
        return true;
    }

    [[nodiscard]] std::optional<double> compute_robot_virtual_center_match_cost_(
        const Eigen::Matrix<double, 4, 1>& z_v, const Eigen::Matrix<double, 4, 4>& R_v,
        double prior_prob) const noexcept {
        if (!robot_ckf_) {
            return std::nullopt;
        }

        using Mat4  = Eigen::Matrix<double, 4, 4>;
        using Mat4X = Eigen::Matrix<double, 4, RobotModel::NX>;
        using Vec4  = Eigen::Matrix<double, 4, 1>;

        Mat4X H   = Mat4X::Zero();
        H(0, XC)  = 1.0;
        H(1, YC)  = 1.0;
        H(2, Z0)  = 1.0;
        H(3, YAW) = 1.0;

        const auto& x          = robot_ckf_->x();
        const auto& Sx         = robot_ckf_->Sx();
        const RoboCKF::MatXX P = Sx * Sx.transpose();
        Mat4 S                 = H * P * H.transpose() + R_v;
        S                      = (S + S.transpose()) * 0.5;
        S.diagonal().array() += 1e-10;

        const Eigen::LDLT<Mat4> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            return std::nullopt;
        }

        Vec4 nu;
        nu[0] = z_v[0] - x[XC];
        nu[1] = z_v[1] - x[YC];
        nu[2] = z_v[2] - x[Z0];
        nu[3] = shortest_rad(x[YAW], z_v[3]);

        const double d2 = nu.transpose() * ldlt.solve(nu);
        if (!std::isfinite(d2) || d2 >= matcher_gate) {
            return std::nullopt;
        }

        const double p = std::clamp(prior_prob, 1e-6, 1.0);
        return d2 + (-2.0 * std::log(p));
    }

    struct RobotSingleArmorCenterMapCandidate {
        int armor_id{-1};
        RoboCKF::VecZ z_raw{};
        Eigen::Matrix<double, 4, 1> z_v{};
        Eigen::Matrix<double, 4, 4> R_v{};
        double cost{std::numeric_limits<double>::infinity()};
    };

    [[nodiscard]] std::optional<bool> try_robot_single_armor_center_map_update_(
        const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        if (!robot_ckf_ || measurements.measurements.size() != 1
            || !should_use_robot_virtual_center_update_(1)) {
            return std::nullopt;
        }

        const auto& meas = measurements.measurements.front();
        if (meas.name != target_name_) {
            state_machine(false);
            return false;
        }

        const auto ypd              = xyz2ypd(meas.transform->translation());
        const double armor_yaw_raw  = meas.yaw();
        constexpr int kArmorsNum    = RobotModel::ARMORS_NUM;
        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);

        const double phase_shift = robot_ckf_->x()[V_YAW] * last_dt_ / kAngleStep;
        const double phase_std =
            compute_phase_std_(robot_ckf_->Sx(), YAW, V_YAW, last_dt_, kAngleStep);
        const auto prior =
            build_mode_prior_(kArmorsNum, robot_last_armor_id_, phase_shift, phase_std);

        std::optional<RobotSingleArmorCenterMapCandidate> best;
        for (int armor_id = 0; armor_id < kArmorsNum; ++armor_id) {
            robot_model_.armor_id = armor_id;
            const auto z_pred     = robot_model_.h(robot_ckf_->x(), armor_id);

            RoboCKF::VecZ z;
            z[0]            = unwrap_rad(z_pred[0], ypd[0]);
            z[1]            = unwrap_rad(z_pred[1], ypd[1]);
            z[2]            = ypd[2];
            z[3]            = unwrap_rad(z_pred[3], armor_yaw_raw);
            const auto virt = build_robot_virtual_center_measurement_(armor_id, z);
            if (!virt.has_value()) {
                continue;
            }
            const auto cost = compute_robot_virtual_center_match_cost_(
                virt->first, virt->second,
                (armor_id < static_cast<int>(prior.size()) ? prior[armor_id] : 1.0));
            if (!cost.has_value()) {
                continue;
            }

            if (!best.has_value() || *cost < best->cost) {
                best = RobotSingleArmorCenterMapCandidate{
                    .armor_id = armor_id,
                    .z_raw    = z,
                    .z_v      = virt->first,
                    .R_v      = virt->second,
                    .cost     = *cost,
                };
            }
        }

        if (!best.has_value()) {
            state_machine(false);
            return false;
        }

        state_machine(true);
        robot_last_armor_id_  = best->armor_id;
        robot_model_.armor_id = best->armor_id;
        set_measurement(best->z_raw, (best->armor_id == 1 || best->armor_id == 3));

        bool updated = apply_robot_virtual_center_update_(best->z_v, best->R_v);
        if (!updated) {
            robot_ckf_->update(best->z_raw);
        }
        apply_robot_known_geometry_constraint_();
        return true;
    }

    void apply_robot_known_geometry_constraint_() noexcept {
        if (!robot_ckf_ || !robot_geometry_is_known_()) {
            return;
        }
        auto* x  = robot_ckf_->x_raw_ptr();
        auto* Sx = robot_ckf_->Sx_raw_ptr();
        if (x == nullptr || Sx == nullptr) {
            return;
        }

        const double r0 = std::clamp(config_.robot_inekf.radius0, 0.10, 0.50);
        const double r1 = std::clamp(config_.robot_inekf.radius1, 0.10, 0.50);
        const double h  = std::clamp(config_.robot_inekf.height, -0.20, 0.20);
        (*x)[R_0]       = r0;
        (*x)[R_1]       = r1;
        (*x)[H]         = h;

        using MatX                = RoboCKF::MatXX;
        MatX P                    = (*Sx) * Sx->transpose();
        constexpr double kGeomVar = 1e-10;
        for (const int idx : {R_0, R_1, H}) {
            P.row(idx).setZero();
            P.col(idx).setZero();
            P(idx, idx) = kGeomVar;
        }
        P   = (P + P.transpose()) * 0.5;
        *Sx = ckf::detail::chol_from_ldlt<RobotModel::NX, double>(P);
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
