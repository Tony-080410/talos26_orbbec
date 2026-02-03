#pragma once

#include "config.hpp"
#include "data_associator.hpp"
#include "fcs/core/types.hpp"
#include "inekf_config.hpp"
#include "jpda_matcher.hpp"
#include "motion_model.hpp"
#include "semantic_context.hpp"
#include "spdlog_hook.hpp"
#include "types.hpp"
#include "uni_kf_spaces.hpp"
#include "viterbi_smoother.hpp"

// InEKF tracker using uni_kf framework
#include "inekf_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <math/fast_tf/frame.hpp>
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
    using GimbalRef    = fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>;

    explicit Tracker(const TrackerConfig& config) noexcept
        : config_(config)
        , robot_jpda_matcher_(make_jpda_config<RobotModel>(config))
        , outpost_jpda_matcher_(make_jpda_config<OutpostModel>(config)) {
        robot_model_.params   = config_.robot.model;
        outpost_model_.params = config_.outpost.model;
    }

    /// Set configuration (lightweight: only stores config values)
    void set_config(const TrackerConfig& config) noexcept { config_ = config; }

    /// Sync configuration (heavyweight: applies config and resets tracker state)
    void sync_config() noexcept {
        SPDLOG_INFO("Tracker: syncing config, resetting to IDLE state");
        robot_model_.params   = config_.robot.model;
        outpost_model_.params = config_.outpost.model;

        // Reset tracker to IDLE state when config changes
        // JPDA matchers will use new config values on next match() call
        reset();
    }

    /// Combined predict + update step
    void step(double dt, const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        predict(dt);
        (void)update(measurements);
    }

    /// Combined predict + update step with gimbal reference
    void step(
        double dt, const AtLegacyArmorMeasurementBatch& measurements,
        const GimbalRef& gimbal) noexcept {
        predict(dt, gimbal);
        (void)update(measurements, gimbal);
    }

    /// Prediction step (time update)
    void predict(double dt) noexcept { predict_impl(dt, nullptr); }

    /// Prediction step (time update) with gimbal reference
    void predict(double dt, const GimbalRef& gimbal) noexcept { predict_impl(dt, &gimbal); }

    /// Update step (measurement update)
    /// @return true if matched successfully
    [[nodiscard]] bool update(const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        return update_impl(measurements, nullptr);
    }

    /// Update step (measurement update) with gimbal reference
    /// @return true if matched successfully
    [[nodiscard]] bool update(
        const AtLegacyArmorMeasurementBatch& measurements, const GimbalRef& gimbal) noexcept {
        return update_impl(measurements, &gimbal);
    }

private:
    void predict_impl(double dt, const GimbalRef* gimbal) noexcept {
        (void)gimbal; // InEKF runs in odom (no reference-frame adjustment).
        last_dt_ = std::max(0.0, dt);

        // Check if we've lost the target for too long
        if (status_ == TrackerStatus::TempLost && (lost_time_ + last_dt_) >= lost_threshold) {
            reset();
            return;
        }

        if (robot_ckf_.has_value()) {
            robot_ckf_->predict(dt);
            auto& e = robot_ckf_->x_mut().first.data;
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
    [[nodiscard]] bool update_impl(
        const AtLegacyArmorMeasurementBatch& measurements, const GimbalRef* gimbal) noexcept {
        // Only Outpost uses 3-armors model, all other targets use 4-armors model
        // This includes: Sentry, One, Two, Three, Four, Five, Base, BaseLarge, Invalid
        const bool is_outpost = (target_name_ == ArmorName::Outpost);
        const int armors_num  = is_outpost ? 3 : 4;

        // Validate filter state
        if (is_outpost && !outpost_ckf_) {
            return false;
        }
        if (!is_outpost && !robot_ckf_) {
            return false;
        }

        // Choose matcher based on config
        const bool use_jpda = config_.jpda_matcher.enable_jpda_matcher;
        typename DataAssociator<RobotModel>::MatchResult match_result;
        std::vector<int> idx;

        if (is_outpost) {
            if (use_jpda) {
                // Use JpdaMatcher for outpost
                semantic_context_.set_enabled(config_.jpda_matcher.enable_semantic);

                // Check if target type changed
                if (target_name_ != last_target_name_) {
                    outpost_jpda_matcher_.reset(); // Reset Viterbi state
                    last_target_name_ = target_name_;
                }

                auto result = outpost_jpda_matcher_.match(
                    measurements, outpost_ckf_->x(), outpost_ckf_->Px(), outpost_model_,
                    &semantic_context_, armors_num, matcher_gate, target_name_);

                // Direct use of JpdaMatcher result (no fallback)
                match_result.armors = result.armors;
                idx                 = result.armor_ids;
            } else {
                // Use legacy DataAssociator
                auto result = outpost_associator_.match(
                    measurements, outpost_ckf_->x(), outpost_ckf_->Px(), outpost_model_,
                    target_name_, matcher_gate, armors_num);

                for (size_t i = 0; i < result.armors.size(); ++i) {
                    match_result.armors.push_back(result.armors[i]);
                    idx.push_back(result.armor_ids[i]);
                }
            }
        } else {
            if (use_jpda) {
                // Use JpdaMatcher for robot
                semantic_context_.set_enabled(config_.jpda_matcher.enable_semantic);

                // Check if target type changed (e.g., robot → outpost)
                if (target_name_ != last_target_name_) {
                    robot_jpda_matcher_.reset(); // Reset Viterbi state
                    last_target_name_ = target_name_;
                }

                auto result = robot_jpda_matcher_.match(
                    measurements, robot_ckf_->x(), robot_ckf_->Px(), robot_model_,
                    &semantic_context_, armors_num, matcher_gate, target_name_);

                // Direct use of JpdaMatcher result (no fallback)
                match_result.armors = result.armors;
                idx                 = result.armor_ids;

                // Note: Viterbi smoothing is now handled inside robot_jpda_matcher_
                // No need to update phase_smoother_ here
            } else {
                // Use legacy DataAssociator
                auto result = robot_associator_.match(
                    measurements, robot_ckf_->x(), robot_ckf_->Px(), robot_model_, target_name_,
                    matcher_gate, armors_num);

                match_result.armors = result.armors;
                idx                 = result.armor_ids;
            }
        }

        if (match_result.armors.empty()) {
            state_machine(false);
            return false;
        }

        state_machine(true);

        // Process matched armors
        (void)gimbal; // InEKF runs in odom (no reference-frame adjustment).

        for (size_t i = 0; i < match_result.armors.size(); ++i) {
            const auto& meas = match_result.armors[i];

            const double armor_yaw_raw = meas.yaw();
            const auto& armor_pos_odom = meas.transform->translation();
            const auto z_meas          = ukf::ArmorMeas::make(armor_pos_odom, armor_yaw_raw);
            const auto ypd             = xyz2ypd(armor_pos_odom);

            if (is_outpost) {
                const int armor_id = idx[i];
                outpost_ckf_->update(z_meas, armor_id);

                if (outpost_inekf_.has_value() && config_.inekf_enabled) {
                    constexpr double kAngleStep =
                        2.0 * std::numbers::pi / static_cast<double>(OutpostModel::ARMORS_NUM);
                    const double yaw_offset = static_cast<double>(armor_id) * kAngleStep;

                    // Use spherical or Cartesian observation mode based on config
                    if (config_.outpost_inekf.use_spherical_observation) {
                        // Spherical mode: update_armor_spherical includes armor_yaw,
                        // so we skip separate update_yaw to avoid double yaw update.
                        const Eigen::Vector3d armor_ypd{ypd.x(), ypd.y(), ypd.z()};
                        outpost_inekf_->update_armor_spherical(
                            armor_ypd, armor_yaw_raw, armor_id, config_.outpost_inekf);
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
                const int armor_id = idx[i];
                robot_ckf_->update(z_meas, armor_id);

                if (robot_inekf_.has_value() && config_.inekf_enabled) {
                    constexpr double kAngleStep =
                        2.0 * std::numbers::pi / static_cast<double>(RobotModel::ARMORS_NUM);
                    const double yaw_offset = static_cast<double>(armor_id) * kAngleStep;

                    // Use spherical or Cartesian observation mode based on config
                    if (config_.robot_inekf.use_spherical_observation) {
                        // Spherical mode: update_armor_spherical includes armor_yaw,
                        // so we skip separate update_yaw to avoid double yaw update.
                        const Eigen::Vector3d armor_ypd{ypd.x(), ypd.y(), ypd.z()};
                        robot_inekf_->update_armor_spherical(
                            armor_ypd, armor_yaw_raw, armor_id, config_.robot_inekf);
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

public:
    /// Initialize tracking from first detection
    /// @return Target name if successfully initialized
    [[nodiscard]] std::optional<ArmorName>
        first_meet(const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        return first_meet_impl(measurements, nullptr);
    }

    [[nodiscard]] std::optional<ArmorName> first_meet(
        const AtLegacyArmorMeasurementBatch& measurements, const GimbalRef& gimbal) noexcept {
        return first_meet_impl(measurements, &gimbal);
    }

private:
    [[nodiscard]] std::optional<ArmorName> first_meet_impl(
        const AtLegacyArmorMeasurementBatch& measurements, const GimbalRef* gimbal) noexcept {
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
            const auto x_init = OutpostModel::make_state(x0, y0, armor_yaw, v0, z0, z0, z0);
            const auto P0     = OutpostCKF::CovX::Identity();
            outpost_ckf_.emplace(outpost_model_, x_init, P0);
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

            const auto x_init = RobotModel::make_state(
                x0, v0, y0, v0, z0, v0, armor_yaw, v0, kDefaultRadius, kDefaultRadius, 0.0);
            const auto P0 = RoboCKF::CovX::Identity();
            robot_ckf_.emplace(robot_model_, x_init, P0);
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

public:
    /// Get current tracker output
    [[nodiscard]] TrackerOutput get_output() const noexcept {
        TrackerOutput output;
        output.status      = status_;
        output.target_name = target_name_;

        // Populate state machine debug info
        output.state_machine_debug                      = StateMachineDebug{};
        output.state_machine_debug->time_in_state       = 0.0; // TODO: track time in state
        output.state_machine_debug->detecting_count     = detecting_count_;
        output.state_machine_debug->lost_time           = lost_time_;
        output.state_machine_debug->consecutive_updates = 0;   // TODO: track consecutive updates

        if (robot_ckf_) {
            // Populate filter debug info
            output.filter_debug                   = FilterDebug{};
            const auto& Px                        = robot_ckf_->Px();
            output.filter_debug->covariance_trace = Px.trace();
            // Position uncertainty (sqrt of diagonal elements for x, y, z)
            const int nx                              = RobotModel::NX;
            output.filter_debug->position_uncertainty = Eigen::Vector3d(
                std::sqrt(std::max(0.0, Px(RobotModel::XC, RobotModel::XC))),
                std::sqrt(std::max(0.0, Px(RobotModel::YC, RobotModel::YC))),
                std::sqrt(std::max(0.0, Px(RobotModel::Z0, RobotModel::Z0))));
            // Velocity uncertainty
            output.filter_debug->velocity_uncertainty = Eigen::Vector3d(
                std::sqrt(std::max(0.0, Px(RobotModel::VX, RobotModel::VX))),
                std::sqrt(std::max(0.0, Px(RobotModel::VY, RobotModel::VY))),
                std::sqrt(std::max(0.0, Px(RobotModel::VZ, RobotModel::VZ))));
            // Yaw uncertainty
            output.filter_debug->yaw_uncertainty =
                std::sqrt(std::max(0.0, Px(RobotModel::V_YAW, RobotModel::V_YAW)));

            if (robot_inekf_) {
                auto state   = robot_inekf_->get_robot_state();
                output.state = state;
            } else {
                const auto& x = robot_ckf_->x();
                const auto& e = x.first.data;
                RobotTargetState state;
                state.position   = {e(RobotModel::XC), e(RobotModel::YC), e(RobotModel::Z0)};
                state.velocity   = {e(RobotModel::VX), e(RobotModel::VY), e(RobotModel::VZ)};
                state.yaw        = ukf::YawSO3Z::angle(x.second);
                state.v_yaw      = e(RobotModel::V_YAW);
                state.radius0    = e(RobotModel::R_0);
                state.radius1    = e(RobotModel::R_1);
                state.z1         = e(RobotModel::Z0) + e(RobotModel::H);
                state.armors_num = 4;
                output.state     = state;
            }

            // Populate association debug info from JpdaMatcher
            const auto& matcher_result = robot_jpda_matcher_.get_last_result();
            if (matcher_result) {
                output.association_debug               = AssociationDebug{};
                output.association_debug->phase_probs  = matcher_result->phase_probs;
                output.association_debug->map_phase    = matcher_result->map_phase;
                output.association_debug->is_ambiguous = matcher_result->is_ambiguous;

                // Convert permutation probabilities to our format
                for (const auto& perm_prob : matcher_result->perm_probs) {
                    AssociationDebug::PermutationInfo info;
                    info.perm     = perm_prob.perm;
                    info.log_prob = perm_prob.log_prob;
                    // Compute Mahalanobis distance from log_prob (approximate)
                    // log_prob = -0.5 * (d2 + log|S|), so d2 ≈ -2 * log_prob
                    info.mahalanobis_sq = -2.0 * perm_prob.log_prob;
                    info.gate_passed =
                        info.mahalanobis_sq < matcher_gate; // Use matcher_gate as threshold
                    output.association_debug->top_permutations.push_back(info);
                }
            }
        } else if (outpost_ckf_) {
            // Populate filter debug info for outpost
            output.filter_debug                   = FilterDebug{};
            const auto& Px                        = outpost_ckf_->Px();
            output.filter_debug->covariance_trace = Px.trace();
            // Position uncertainty (only XC, YC available in OutpostModel)
            output.filter_debug->position_uncertainty = Eigen::Vector3d(
                std::sqrt(std::max(0.0, Px(OutpostModel::XC, OutpostModel::XC))),
                std::sqrt(std::max(0.0, Px(OutpostModel::YC, OutpostModel::YC))),
                std::sqrt(
                    std::max(
                        0.0, (Px(OutpostModel::Z0, OutpostModel::Z0)
                              + Px(OutpostModel::Z1, OutpostModel::Z1)
                              + Px(OutpostModel::Z2, OutpostModel::Z2))
                                 / 3.0)));
            // Velocity uncertainty - OutpostModel doesn't track Vx, Vy, Vz separately
            // Use position uncertainty as proxy approximation
            output.filter_debug->velocity_uncertainty =
                output.filter_debug->position_uncertainty * 0.1;
            // Yaw uncertainty
            output.filter_debug->yaw_uncertainty =
                std::sqrt(std::max(0.0, Px(OutpostModel::V_YAW, OutpostModel::V_YAW)));

            if (outpost_inekf_) {
                auto state   = outpost_inekf_->get_outpost_state();
                output.state = state;
            } else {
                const auto& x = outpost_ckf_->x();
                const auto& e = x.first.data;
                OutpostTargetState state;
                state.position = {e(OutpostModel::XC), e(OutpostModel::YC)};
                state.yaw      = ukf::YawSO3Z::angle(x.second);
                state.v_yaw    = e(OutpostModel::V_YAW);
                state.z        = {e(OutpostModel::Z0), e(OutpostModel::Z1), e(OutpostModel::Z2)};
                output.state   = state;
            }

            // Populate association debug info from JpdaMatcher
            const auto& matcher_result = outpost_jpda_matcher_.get_last_result();
            if (matcher_result) {
                output.association_debug               = AssociationDebug{};
                output.association_debug->phase_probs  = matcher_result->phase_probs;
                output.association_debug->map_phase    = matcher_result->map_phase;
                output.association_debug->is_ambiguous = matcher_result->is_ambiguous;

                for (const auto& perm_prob : matcher_result->perm_probs) {
                    AssociationDebug::PermutationInfo info;
                    info.perm           = perm_prob.perm;
                    info.log_prob       = perm_prob.log_prob;
                    info.mahalanobis_sq = -2.0 * perm_prob.log_prob;
                    info.gate_passed    = info.mahalanobis_sq < matcher_gate;
                    output.association_debug->top_permutations.push_back(info);
                }
            }
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
    /// Helper function to create JpdaMatcher config from TrackerConfig
    ///
    /// Problem:
    ///   JpdaMatcher needs a Config struct with specific field names and types.
    ///   We need to convert TrackerConfig::JpdaMatcherConfig to JpdaMatcher::Config.
    ///
    /// Precise:
    ///   Extracts the relevant fields from TrackerConfig and constructs JpdaMatcher::Config.
    template <typename Model>
    [[nodiscard]] static typename JpdaMatcher<Model>::Config
        make_jpda_config(const TrackerConfig& cfg) noexcept {
        return typename JpdaMatcher<Model>::Config{
            cfg.jpda_matcher.phase_ambiguous_threshold, cfg.jpda_matcher.phase_switch_lambda,
            cfg.jpda_matcher.semantic_reweight_lambda, cfg.jpda_matcher.enable_semantic,
            cfg.jpda_matcher.enable_viterbi};
    }

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

    /// Jpda matchers (advanced phase-permutation JPDA)
    /// These are optionally used based on config_.jpda_matcher.enable_jpda_matcher
    JpdaMatcher<RobotModel> robot_jpda_matcher_;
    JpdaMatcher<OutpostModel> outpost_jpda_matcher_;

    /// Semantic context for breaking symmetry
    SemanticContext semantic_context_;

    /// Viterbi smoother for phase smoothing (3-frame sliding window)
    /// Note: This is currently unused because JpdaMatcher has its own internal ViterbiSmoother.
    /// Kept for potential fallback use in the future.
    ViterbiSmoother phase_smoother_{4}; // Default to 4 phases (robot)

    /// Last tracked target name for detecting target type switches
    /// When target_name_ != last_target_name_, we reset the JpdaMatcher's Viterbi state.
    ArmorName last_target_name_{ArmorName::Invalid};

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
