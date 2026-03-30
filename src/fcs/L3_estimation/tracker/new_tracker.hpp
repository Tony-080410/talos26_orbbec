#pragma once

#include "config.hpp"
#include "core/types.hpp"
#include "extended_kalman_filter.hpp"
#include "new_motion_model.hpp"
#include "types.hpp"
#include "util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include <ceres/jet.h>

namespace fcs::L3 {

template <typename ModelT>
class EkfTargetInfo {
public:
    using Model             = ModelT;
    static constexpr int NX = Model::NX;
    static constexpr int NZ = Model::NZ;

    using VecX   = typename Model::VecX;
    using VecZ   = typename Model::VecZ;
    using MatXX  = typename Model::MatXX;
    using MatZZ  = typename Model::MatZZ;
    using Params = typename Model::Params;

    using JetX        = ceres::Jet<double, NX>;
    using PredictFunc = std::function<void(const JetX*, JetX*)>;
    using MeasureFunc = std::function<void(const JetX*, JetX*)>;
    using EKF         = ExtendedKalmanFilter<NX, NZ, PredictFunc, MeasureFunc>;

    EkfTargetInfo() = default;

    void initialize(const Params& params, const VecX& x0, const MatXX& P0) noexcept {
        model_.params = params;
        dt_           = 0.0;
        armor_id_     = 0;

        auto f = [this](const JetX* x, JetX* xp) {
            Model::template predict_state<JetX>(x, JetX(dt_), xp);
        };
        auto h = [this](const JetX* x, JetX* z) {
            Model::template measure_state<JetX>(x, armor_id_, z);
        };
        auto q = [this]() -> MatXX { return model_.Q(dt_); };
        auto r = [this](const VecZ& z) -> MatZZ { return model_.R(z); };

        ekf_ = EKF(f, h, q, r, P0);
        ekf_.setState(x0);
        x_      = x0;
        active_ = true;
    }

    void predict(double dt) noexcept {
        if (!active_) {
            return;
        }
        dt_ = std::max(0.0, dt);
        x_  = ekf_.predict();
    }

    void update(const VecZ& z, int armor_id) noexcept {
        if (!active_) {
            return;
        }
        armor_id_ = Model::clamp_armor_id(armor_id);
        x_        = ekf_.update(z);
    }

    void update(const VecZ& z, int armor_id, const MatZZ& R) noexcept {
        if (!active_) {
            return;
        }
        armor_id_ = Model::clamp_armor_id(armor_id);
        x_        = ekf_.update(z, R);
    }

    [[nodiscard]] bool active() const noexcept { return active_; }

    [[nodiscard]] const VecX& x() const noexcept { return x_; }

    [[nodiscard]] const MatXX& P() const noexcept { return ekf_.P(); }

    using MatXZ = Eigen::Matrix<double, NX, NZ>;
    [[nodiscard]] const MatXZ& K_gain() const noexcept { return ekf_.K_gain(); }
    [[nodiscard]] const MatXX& Q_mat() const noexcept { return ekf_.Q_mat(); }
    [[nodiscard]] const MatZZ& R_mat() const noexcept { return ekf_.R_mat(); }

    void set_state_unsafe(VecX x) noexcept { ekf_.setState(x); }

    [[nodiscard]] const Model& model() const noexcept { return model_; }

    [[nodiscard]] Model& model() noexcept { return model_; }

private:
    bool active_{false};
    double dt_{0.0};
    int armor_id_{0};
    Model model_{};
    EKF ekf_{};
    VecX x_{VecX::Zero()};
};

class TrackerNew {
public:
    TrackerNew() noexcept = default;

    explicit TrackerNew(const TrackerConfig& config) noexcept
        : config_(config) {}

    void predict(double dt) noexcept {
        last_dt_ = std::max(0.0, dt);

        if (status_ == TrackerStatus::TempLost && (lost_time_ + last_dt_) >= lost_threshold_) {
            reset();
            return;
        }

        if (robot_target_.has_value()) {
            robot_target_->predict(last_dt_);
        }

        if (outpost_target_.has_value()) {
            outpost_target_->predict(last_dt_);
        }
    }

    [[nodiscard]] bool update(const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        measurement_.fill(0.0);

        if (target_name_ == ArmorName::Outpost) {
            if (!outpost_target_.has_value()) {
                return false;
            }
            auto d = update_target_(
                *outpost_target_, measurements, OutpostEkfMotionModel::ARMORS_NUM,
                outpost_last_armor_id_, false);

            auto x           = outpost_target_->x();
            auto v_yaw       = x[O_VYAW];
            auto ref         = OutpostEkfMotionModel::OUTPOST_V_YAW;
            auto std         = 0.6 * OutpostEkfMotionModel::OUTPOST_V_YAW;
            constexpr auto K = 0.5; // Magic that auto stick to ground truth vyaw
            v_yaw            = OutpostEkfMotionModel::OUTPOST_V_YAW * std::tanh(K * v_yaw);
            if (v_yaw > std) {
                v_yaw = ref;
            } else if (v_yaw < -std) {
                v_yaw = -ref;
            }
            x[O_VYAW] = v_yaw;

            outpost_target_->set_state_unsafe(x);
            return d;
        }

        if (!robot_target_.has_value()) {
            return false;
        }
        return update_target_(
            *robot_target_, measurements, RobotEkfMotionModel::ARMORS_NUM, robot_last_armor_id_,
            true);
    }

    [[nodiscard]] std::optional<ArmorName>
        first_meet(const AtLegacyArmorMeasurementBatch& measurements) noexcept {
        if (measurements.empty()) {
            state_machine(false);
            return std::nullopt;
        }

        int min_idx        = -1;
        float min_distance = std::numeric_limits<float>::max();
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

        const auto& target     = measurements.measurements[static_cast<size_t>(min_idx)];
        const auto& pos        = target.transform.translation();
        const double armor_yaw = target.yaw();

        target_name_                   = target.name;
        target_color_                  = target.color;
        last_image_center_distance_px_ = target.distance_to_image_center;
        last_observation_timestamp_ns_ = target.timestamp_ns;
        const bool is_outpost          = (target_name_ == ArmorName::Outpost);

        robot_target_.reset();
        outpost_target_.reset();
        robot_last_armor_id_.reset();
        outpost_last_armor_id_.reset();
        target_jumped_ = false;

        if (is_outpost) {
            const double x0 = pos.x() + OutpostEkfMotionModel::OUTPOST_RADIUS * std::cos(armor_yaw);
            const double y0 = pos.y() + OutpostEkfMotionModel::OUTPOST_RADIUS * std::sin(armor_yaw);
            const double z0 = pos.z();

            OutpostEkfMotionModel::VecX xp0;
            xp0 << x0, y0, armor_yaw, 0.0, z0, z0, z0;
            OutpostEkfMotionModel::MatXX P0 = OutpostEkfMotionModel::MatXX::Identity();

            P0(O_XC, O_XC) = 1e-9;
            P0(O_YC, O_YC) = 1e-9;

            P0(V_YAW, V_YAW) = 1e9;

            P0(O_Z0, O_Z0) = 1e9;
            P0(O_Z1, O_Z1) = 1e9;
            P0(O_Z2, O_Z2) = 1e9;
            outpost_target_.emplace();
            outpost_target_->initialize(config_.outpost.model, xp0, P0);
            outpost_last_armor_id_ = 0;

            lost_threshold_     = config_.outpost.lost_threshold;
            tracking_threshold_ = config_.outpost.tracking_threshold;
            matcher_gate_       = config_.outpost.matcher_gate;
        } else {
            const double r0_prior = std::clamp(config_.robot_inekf.radius0, 0.10, 0.50);
            const double r1_prior = std::clamp(config_.robot_inekf.radius1, 0.10, 0.50);
            const double h_prior  = std::clamp(config_.robot_inekf.height, -0.20, 0.20);
            const double r_init   = 0.5 * (r0_prior + r1_prior);
            const double x0       = pos.x() + r_init * std::cos(armor_yaw);
            const double y0       = pos.y() + r_init * std::sin(armor_yaw);
            const double z0       = pos.z() - 0.5 * h_prior;

            RobotEkfMotionModel::VecX xp0;
            auto v0 = 0.0;
            xp0 << x0, v0, y0, v0, z0, v0, armor_yaw, v0, r0_prior, r1_prior, h_prior;
            RobotEkfMotionModel::MatXX P0 = RobotEkfMotionModel::MatXX::Identity();

            P0(XC, XC) = 1e9;
            P0(YC, YC) = 1e9;

            P0(VX, VX) = 1e9;
            P0(VY, VY) = 1e9;
            P0(VZ, VZ) = 1e9;

            P0(V_YAW, V_YAW) = 1e9;

            P0(R_0, R_0) = 1e-5;
            P0(R_1, R_1) = 1e-5;
            P0(H, H)     = 1e-5;

            robot_target_.emplace();
            robot_target_->initialize(config_.robot.model, xp0, P0);
            robot_last_armor_id_.reset();

            lost_threshold_     = config_.robot.lost_threshold;
            tracking_threshold_ = config_.robot.tracking_threshold;
            matcher_gate_       = config_.robot.matcher_gate;
        }

        state_machine(true);
        return target_name_;
    }

    [[nodiscard]] TrackerOutput get_output() const noexcept {
        TrackerOutput output;
        output.status                        = status_;
        output.target_name                   = target_name_;
        output.target_color                  = target_color_;
        output.target_jumped                 = target_jumped_;
        output.last_image_center_distance_px = last_image_center_distance_px_;
        output.last_observation_timestamp_ns = last_observation_timestamp_ns_;

        if (robot_target_.has_value()) {
            const auto& x = robot_target_->x();
            RobotTargetState state;
            state.position       = {x[XC], x[YC], x[Z0]};
            state.velocity       = {x[VX], x[VY], x[VZ]};
            state.yaw            = x[YAW];
            state.v_yaw          = x[V_YAW];
            state.radius0        = x[R_0];
            state.radius1        = x[R_1];
            state.z1             = x[Z0] + x[H];
            state.armors_num     = RobotEkfMotionModel::ARMORS_NUM;
            state.P              = robot_target_->P();
            state.K              = robot_target_->K_gain();
            state.Q              = robot_target_->Q_mat();
            state.R              = robot_target_->R_mat();
            output.state         = state;
            output.last_armor_id = robot_last_armor_id_;
        } else if (outpost_target_.has_value()) {
            const auto& x = outpost_target_->x();
            OutpostTargetState state;
            state.position       = {x[O_XC], x[O_YC]};
            state.yaw            = x[O_YAW];
            state.v_yaw          = x[O_VYAW];
            state.z              = {x[O_Z0], x[O_Z1], x[O_Z2]};
            state.P              = outpost_target_->P();
            state.K              = outpost_target_->K_gain();
            state.Q              = outpost_target_->Q_mat();
            state.R              = outpost_target_->R_mat();
            output.state         = state;
            output.last_armor_id = outpost_last_armor_id_;
        }

        return output;
    }

    void reset() noexcept {
        robot_target_.reset();
        outpost_target_.reset();
        robot_last_armor_id_.reset();
        outpost_last_armor_id_.reset();
        target_jumped_ = false;
        target_name_   = ArmorName::Invalid;
        target_color_  = ArmorColor::Neutral;
        measurement_.fill(0.0);
        last_image_center_distance_px_ = std::numeric_limits<double>::infinity();
        last_observation_timestamp_ns_ = 0;
        detecting_count_               = 0;
        last_dt_                       = 0.0;
        lost_time_                     = 0.0;
        status_                        = TrackerStatus::Idle;
    }

    [[nodiscard]] TrackerStatus status() const noexcept { return status_; }

    [[nodiscard]] ArmorName target_name() const noexcept { return target_name_; }

    [[nodiscard]] const std::array<double, RobotEkfMotionModel::NZ * 2>&
        measurement() const noexcept {
        return measurement_;
    }

private:
    template <typename TargetInfo>
    struct MatchCandidate {
        size_t obs_index{0};
        int armor_id{0};
        double cost{0.0};
        typename TargetInfo::VecZ z_update{};
        typename TargetInfo::MatZZ R_update{};
    };

    template <typename TargetInfo>
    [[nodiscard]] bool update_target_(
        TargetInfo& target, const AtLegacyArmorMeasurementBatch& measurements, int armors_num,
        std::optional<int>& last_armor_id, bool enable_robot_pair_measure) noexcept {
        using VecZ  = typename TargetInfo::VecZ;
        using MatZZ = typename TargetInfo::MatZZ;

        std::vector<size_t> obs_indices;
        obs_indices.reserve(measurements.measurements.size());
        for (size_t i = 0; i < measurements.measurements.size(); ++i) {
            if (measurements.measurements[i].name == target_name_) {
                obs_indices.push_back(i);
            }
        }

        if (obs_indices.empty()) {
            state_machine(false);
            return false;
        }

        const double angle_step = 2.0 * std::numbers::pi / static_cast<double>(armors_num);
        const double phase_shift =
            target.x()[TargetInfo::Model::kVyawIndex] * last_dt_ / angle_step;
        const double phase_std = std::clamp(std::abs(last_dt_) * 0.5, 0.05, 1.0);
        const auto prior = build_mode_prior_(armors_num, last_armor_id, phase_shift, phase_std);

        std::vector<MatchCandidate<TargetInfo>> candidates;
        candidates.reserve(obs_indices.size() * static_cast<size_t>(armors_num));

        for (const size_t obs_index : obs_indices) {
            const auto& meas = measurements.measurements[obs_index];
            const VecZ z_raw = measurement_to_z_(meas);

            // Distance-dependent R with PnP quality modulation.
            MatZZ R = target.model().R(z_raw);

            // Modulate by PnP quality: worse reprojection → larger R → less trust.
            if (meas.pnp_sigma_obs > 0.0) {
                const double q = std::clamp(
                    meas.pnp_sigma_obs / config_.pnp_quality.sigma_ref,
                    config_.pnp_quality.scale_min, config_.pnp_quality.scale_max);
                R *= q * q;
            }

            for (int id = 0; id < armors_num; ++id) {
                const VecZ z_pred = target.model().h(target.x(), id);
                const VecZ nu     = innovation_(z_pred, z_raw);
                const double d2   = mahalanobis_diag_(nu, R);
                if (!std::isfinite(d2) || d2 >= matcher_gate_) {
                    continue;
                }

                const double p = std::clamp(
                    id < static_cast<int>(prior.size()) ? prior[static_cast<size_t>(id)] : 1.0,
                    1e-6, 1.0);
                const double cost = d2 - 2.0 * std::log(p);

                MatchCandidate<TargetInfo> candidate;
                candidate.obs_index = obs_index;
                candidate.armor_id  = id;
                candidate.cost      = cost;
                candidate.z_update  = wrap_measurement_(z_pred, z_raw);
                candidate.R_update  = R;
                candidates.push_back(candidate);
            }
        }

        if (candidates.empty()) {
            state_machine(false);
            return false;
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.cost < b.cost;
        });

        std::vector<bool> used_obs(measurements.measurements.size(), false);
        std::vector<bool> used_id(static_cast<size_t>(armors_num), false);
        std::vector<MatchCandidate<TargetInfo>> selected;
        selected.reserve(static_cast<size_t>(armors_num));

        for (const auto& candidate : candidates) {
            if (used_obs[candidate.obs_index] || used_id[static_cast<size_t>(candidate.armor_id)]) {
                continue;
            }
            used_obs[candidate.obs_index]                    = true;
            used_id[static_cast<size_t>(candidate.armor_id)] = true;
            selected.push_back(candidate);
        }

        if (selected.empty()) {
            state_machine(false);
            return false;
        }

        state_machine(true);
        last_armor_id                     = selected.front().armor_id;
        const auto& primary_measurement   = measurements.measurements[selected.front().obs_index];
        last_image_center_distance_px_    = primary_measurement.distance_to_image_center;
        last_observation_timestamp_ns_    = primary_measurement.timestamp_ns;
        const bool observed_nonzero_armor = std::any_of(
            selected.begin(), selected.end(),
            [](const MatchCandidate<TargetInfo>& candidate) { return candidate.armor_id != 0; });
        if (observed_nonzero_armor) {
            target_jumped_ = true;
        }

        for (const auto& candidate : selected) {
            target.update(candidate.z_update, candidate.armor_id, candidate.R_update);
            const bool another_pair =
                enable_robot_pair_measure && (candidate.armor_id == 1 || candidate.armor_id == 3);
            set_measurement_(candidate.z_update, another_pair);
        }
        return true;
    }

    static double wrap_index_diff_(double d, int n) noexcept {
        if (n <= 0 || !std::isfinite(d)) {
            return 0.0;
        }
        const double period = static_cast<double>(n);
        double wrapped      = std::remainder(d, period); // (-period/2, period/2]
        if (!std::isfinite(wrapped)) {
            return 0.0;
        }
        // Keep historical tie behavior: map -n/2 to +n/2.
        if (wrapped <= -0.5 * period) {
            wrapped += period;
        }
        return wrapped;
    }

    static std::vector<double> build_mode_prior_(
        int armors_num, std::optional<int> last_id, double phase_shift, double phase_std) {
        std::vector<double> prior(
            static_cast<size_t>(armors_num), 1.0 / static_cast<double>(armors_num));
        if (!last_id.has_value() || armors_num <= 1) {
            return prior;
        }

        const double center      = static_cast<double>(*last_id) + phase_shift;
        const double sigma       = std::clamp(0.28 + 0.95 * phase_std, 0.25, 2.20);
        const double inv_var     = 1.0 / (sigma * sigma);
        const double uniform_mix = std::clamp(0.02 + 0.30 * phase_std, 0.02, 0.35);
        const int max_index_step = std::clamp(
            static_cast<int>(std::ceil(std::abs(phase_shift) + 3.0 * phase_std + 0.25)), 1,
            armors_num / 2 + 1);

        double sum = 0.0;
        for (int j = 0; j < armors_num; ++j) {
            const double diff = wrap_index_diff_(static_cast<double>(j) - center, armors_num);
            if (std::abs(diff) > static_cast<double>(max_index_step)) {
                prior[static_cast<size_t>(j)] = 0.0;
                continue;
            }
            const double g = std::exp(-0.5 * diff * diff * inv_var);
            prior[static_cast<size_t>(j)] =
                (1.0 - uniform_mix) * g + uniform_mix / static_cast<double>(armors_num);
            sum += prior[static_cast<size_t>(j)];
        }

        if (sum <= 0.0) {
            return std::vector<double>(static_cast<size_t>(armors_num), 1.0 / armors_num);
        }
        for (double& p : prior) {
            p /= sum;
        }
        return prior;
    }

    static RobotEkfMotionModel::VecZ
        measurement_to_z_(const AtLegacyArmorMeasurement& meas) noexcept {
        const auto ypd = xyz2ypd(meas.transform.translation());
        RobotEkfMotionModel::VecZ z;
        z[ARMOR_YAW]       = ypd[0];
        z[ARMOR_PITCH]     = ypd[1];
        z[ARMOR_DISTANCE]  = ypd[2];
        z[ARMOR_YAW_ARMOR] = meas.yaw();
        return z;
    }

    template <typename VecZ>
    static VecZ innovation_(const VecZ& z_pred, const VecZ& z_raw) noexcept {
        VecZ nu;
        nu[ARMOR_YAW]       = shortest_rad(z_pred[ARMOR_YAW], z_raw[ARMOR_YAW]);
        nu[ARMOR_PITCH]     = shortest_rad(z_pred[ARMOR_PITCH], z_raw[ARMOR_PITCH]);
        nu[ARMOR_DISTANCE]  = z_raw[ARMOR_DISTANCE] - z_pred[ARMOR_DISTANCE];
        nu[ARMOR_YAW_ARMOR] = shortest_rad(z_pred[ARMOR_YAW_ARMOR], z_raw[ARMOR_YAW_ARMOR]);
        return nu;
    }

    template <typename VecZ>
    static VecZ wrap_measurement_(const VecZ& z_pred, const VecZ& z_raw) noexcept {
        VecZ z;
        z[ARMOR_YAW]       = unwrap_rad(z_pred[ARMOR_YAW], z_raw[ARMOR_YAW]);
        z[ARMOR_PITCH]     = unwrap_rad(z_pred[ARMOR_PITCH], z_raw[ARMOR_PITCH]);
        z[ARMOR_DISTANCE]  = z_raw[ARMOR_DISTANCE];
        z[ARMOR_YAW_ARMOR] = unwrap_rad(z_pred[ARMOR_YAW_ARMOR], z_raw[ARMOR_YAW_ARMOR]);
        return z;
    }

    template <typename VecZ, typename MatZZ>
    static double mahalanobis_diag_(const VecZ& nu, const MatZZ& R) noexcept {
        double d2 = 0.0;
        for (int i = 0; i < VecZ::RowsAtCompileTime; ++i) {
            const double var = std::max(1e-9, R(i, i));
            d2 += nu[i] * nu[i] / var;
        }
        return d2;
    }

    /// Full Mahalanobis distance with LDLT decomposition for non-diagonal R.
    template <typename VecZ, typename MatZZ>
    static double mahalanobis_(const VecZ& nu, const MatZZ& R) noexcept {
        Eigen::LDLT<MatZZ> ldlt(R);
        if (ldlt.info() != Eigen::Success) {
            return 1e9;
        }
        return static_cast<double>(nu.transpose() * ldlt.solve(nu));
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
                if (detecting_count_ > static_cast<int>(tracking_threshold_)) {
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
                if (lost_time_ >= lost_threshold_) {
                    reset();
                }
            }
            break;
        }
    }

    template <typename VecZ>
    void set_measurement_(const VecZ& z, bool is_another_pair) noexcept {
        if (is_another_pair) {
            measurement_[4] = z[ARMOR_YAW];
            measurement_[5] = z[ARMOR_PITCH];
            measurement_[6] = z[ARMOR_DISTANCE];
            measurement_[7] = z[ARMOR_YAW_ARMOR];
        } else {
            measurement_[0] = z[ARMOR_YAW];
            measurement_[1] = z[ARMOR_PITCH];
            measurement_[2] = z[ARMOR_DISTANCE];
            measurement_[3] = z[ARMOR_YAW_ARMOR];
        }
    }

private:
    TrackerConfig config_{};
    TrackerStatus status_{TrackerStatus::Idle};
    ArmorName target_name_{ArmorName::Invalid};
    ArmorColor target_color_{ArmorColor::Neutral};

    std::optional<EkfTargetInfo<RobotEkfMotionModel>> robot_target_{};
    std::optional<EkfTargetInfo<OutpostEkfMotionModel>> outpost_target_{};

    double lost_threshold_{0.0};
    uint32_t tracking_threshold_{0};
    double matcher_gate_{10.0};

    int detecting_count_{0};
    double last_dt_{0.0};
    double lost_time_{0.0};

    std::array<double, RobotEkfMotionModel::NZ * 2> measurement_{};
    double last_image_center_distance_px_{std::numeric_limits<double>::infinity()};
    uint64_t last_observation_timestamp_ns_{0};
    std::optional<int> robot_last_armor_id_{};
    std::optional<int> outpost_last_armor_id_{};
    bool target_jumped_{false};
};

} // namespace fcs::L3
