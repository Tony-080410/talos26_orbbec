/// @file inekf_tracker.hpp
/// @brief InEKF-style tracker implementation for armor tracking (redact backend)
///
/// This header provides `uni_kf::TrackerInEKF_Robot` and `uni_kf::TrackerInEKF_Outpost`,
/// which are used by `fcs::L3::Tracker` when `inekf_enabled` is true.
///
/// This version uses `uni_kf::redact` as the filtering backend:
/// - Prediction and update are performed via a Cubature Kalman Filter (CKF) on a Lie group state.
/// - The state is SE_2(3) × R^k implemented as a product Lie group so sigma-point propagation
/// works.
/// - This tracker runs in the odom frame (no reference-frame adjustment).

#pragma once

#include <uni_kf/redact/covariance.hpp>
#include <uni_kf/redact/filter.hpp>
#include <uni_kf/redact/lie_ops.hpp>

// Local fcs::L3 types
#include "inekf_config.hpp"
#include "types.hpp"
#include "util.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <groups/SEn3.hpp>
#include <math/fast_tf/frame.hpp>
#include <numbers>
#include <optional>

namespace uni_kf {

using RobotInEKFConfig   = fcs::L3::RobotInEKFConfig;
using OutpostInEKFConfig = fcs::L3::OutpostInEKFConfig;
using RobotTargetState   = fcs::L3::RobotTargetState;
using OutpostTargetState = fcs::L3::OutpostTargetState;

namespace detail {

inline constexpr double kCovarianceRegularization     = 1e-12;
inline constexpr double kCoordinatedTurnOmegaEpsilon  = 1e-3;
inline constexpr double kAdaptiveProcessNoiseMaxScale = 6.0;
inline constexpr double kAdaptiveProcessNoiseDecayTau = 0.25;
inline constexpr double kRobustMaxRScale              = 100.0;
inline constexpr double kPseudoVelocityMinInterval    = 0.03;
inline constexpr double kPseudoVelocityMaxInterval    = 0.25;

[[nodiscard]] inline bool all_finite(const Eigen::Vector3d& v) noexcept { return v.allFinite(); }

[[nodiscard]] inline double sanitize_finite(double x, double fallback) noexcept {
    return std::isfinite(x) ? x : fallback;
}

[[nodiscard]] inline double chi2_gate_approx(int dof) noexcept {
    // Very small helper for robust scaling heuristics; not a strict chi2 quantile.
    // 1σ-ish: ~dof
    return static_cast<double>(dof);
}

[[nodiscard]] inline double
    adaptive_scale_from_maha(double maha, int dof, double max_scale) noexcept {
    if (!(maha > 0.0) || !std::isfinite(maha) || dof <= 0) {
        return 1.0;
    }
    const double gate = chi2_gate_approx(dof);
    if (!(maha > gate)) {
        return 1.0;
    }
    const double s = std::sqrt(maha / gate);
    return std::clamp(s, 1.0, max_scale);
}

[[nodiscard]] inline double decay_to_one(double x, double dt, double tau) noexcept {
    if (!std::isfinite(x)) {
        return 1.0;
    }
    if (!(dt > 0.0) || !std::isfinite(dt) || !(tau > 0.0) || !std::isfinite(tau)) {
        return std::max(1.0, x);
    }
    const double a = std::exp(-dt / tau);
    return 1.0 + (x - 1.0) * a;
}

[[nodiscard]] inline double
    robust_R_scale_from_maha(double maha, int dof, double max_scale) noexcept {
    if (!(maha > 0.0) || !std::isfinite(maha) || dof <= 0) {
        return 1.0;
    }
    const double gate = chi2_gate_approx(dof);
    if (!(maha > gate)) {
        return 1.0;
    }
    const double s = maha / gate;
    return std::clamp(s, 1.0, max_scale);
}

} // namespace detail

// ============================================================================
// Product Lie group: SE_2(3) × R^N
// ============================================================================

template <int ExtraDim>
struct SE23xRn {
    using SE23d              = group::SEn3<double, 2>;
    using ExtraType          = Eigen::Matrix<double, ExtraDim, 1>;
    using VectorType         = Eigen::Matrix<double, 9 + ExtraDim, 1>;
    static constexpr int Dim = 9 + ExtraDim;

    SE23d first{};
    ExtraType second = Eigen::Matrix<double, ExtraDim, 1>::Zero();

    SE23xRn() = default;
    SE23xRn(const SE23d& g, const ExtraType& e)
        : first(g)
        , second(e) {}

    [[nodiscard]] SE23xRn inv() const noexcept {
        const auto first_inv = first.inv();
        return SE23xRn(first_inv, (-second).eval());
    }

    [[nodiscard]] Eigen::MatrixXd Adjoint() const noexcept {
        Eigen::Matrix<double, Dim, Dim> Ad = Eigen::Matrix<double, Dim, Dim>::Zero();
        Ad.template block<9, 9>(0, 0)      = first.Adjoint();
        Ad.template block<ExtraDim, ExtraDim>(9, 9).setIdentity();
        return Ad;
    }

    [[nodiscard]] const Eigen::Matrix3d& R() const noexcept { return first.R(); }
    [[nodiscard]] const Eigen::Vector3d& v() const noexcept { return first.v(); }
    [[nodiscard]] const Eigen::Vector3d& p() const noexcept { return first.p(); }

    friend SE23xRn operator*(const SE23xRn& a, const SE23xRn& b) noexcept {
        return SE23xRn(a.first * b.first, (a.second + b.second).eval());
    }

    [[nodiscard]] static SE23xRn exp(const VectorType& xi) noexcept {
        const auto xi_g = xi.template head<9>().eval();
        const auto xi_e = xi.template tail<ExtraDim>().eval();
        return SE23xRn(SE23d::exp(xi_g), xi_e);
    }

    [[nodiscard]] static VectorType log(const SE23xRn& x) noexcept {
        VectorType xi;
        xi.template head<9>()        = SE23d::log(x.first);
        xi.template tail<ExtraDim>() = x.second;
        return xi;
    }
};

// ============================================================================
// Measurement Lie group: SO(2) × SO(2) × R × SO(2)
// (yaw, pitch, distance, armor_yaw)
// ============================================================================

struct SO2d {
    using Scalar             = double;
    using VectorType         = Eigen::Matrix<double, 1, 1>;
    static constexpr int Dim = 1;

    double angle_rad{0.0};

    SO2d() = default;
    explicit SO2d(double a)
        : angle_rad(fcs::L3::normalize_rad(a)) {}

    [[nodiscard]] SO2d inv() const noexcept { return SO2d(-angle_rad); }
    friend SO2d operator*(const SO2d& a, const SO2d& b) noexcept {
        return SO2d(a.angle_rad + b.angle_rad);
    }

    [[nodiscard]] static SO2d exp(const VectorType& v) noexcept { return SO2d(v(0)); }
    [[nodiscard]] static VectorType log(const SO2d& x) noexcept { return VectorType(x.angle_rad); }

    [[nodiscard]] Eigen::MatrixXd Adjoint() const noexcept {
        return Eigen::Matrix<double, 1, 1>::Identity();
    }
};

struct YpdAy {
    using VectorType         = Eigen::Matrix<double, 4, 1>;
    static constexpr int Dim = 4;

    SO2d yaw{};
    SO2d pitch{};
    double dist{0.0};
    SO2d armor_yaw{};

    YpdAy() = default;
    YpdAy(double yaw_rad, double pitch_rad, double distance, double armor_yaw_rad)
        : yaw(yaw_rad)
        , pitch(pitch_rad)
        , dist(distance)
        , armor_yaw(armor_yaw_rad) {}

    [[nodiscard]] YpdAy inv() const noexcept {
        return YpdAy(
            (yaw.inv()).angle_rad, (pitch.inv()).angle_rad, -dist, (armor_yaw.inv()).angle_rad);
    }

    friend YpdAy operator*(const YpdAy& a, const YpdAy& b) noexcept {
        return YpdAy(
            (a.yaw * b.yaw).angle_rad, (a.pitch * b.pitch).angle_rad, a.dist + b.dist,
            (a.armor_yaw * b.armor_yaw).angle_rad);
    }

    [[nodiscard]] static YpdAy exp(const VectorType& v) noexcept {
        return YpdAy(v(0), v(1), v(2), v(3));
    }

    [[nodiscard]] static VectorType log(const YpdAy& x) noexcept {
        VectorType v;
        v(0) = SO2d::log(x.yaw)(0);
        v(1) = SO2d::log(x.pitch)(0);
        v(2) = x.dist;
        v(3) = SO2d::log(x.armor_yaw)(0);
        return v;
    }

    [[nodiscard]] Eigen::MatrixXd Adjoint() const noexcept {
        return Eigen::Matrix<double, 4, 4>::Identity();
    }
};

// ============================================================================
// Process models (prediction): reuse existing FCS logic
// ============================================================================

class RobotProcessModel {
public:
    static constexpr int kExtraDim = 4;
    static constexpr int kGroupDim = 9;
    static constexpr int kStateDim = kGroupDim + kExtraDim;

    static constexpr int kIdxVYaw = 0;
    static constexpr int kIdxR0   = 1;
    static constexpr int kIdxR1   = 2;
    static constexpr int kIdxH    = 3;

    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;

    using SE23d = group::SEn3<double, 2>;
    using Point = SE23xRn<kExtraDim>;
    using MatX  = Eigen::Matrix<double, kStateDim, kStateDim>;
    using VecX  = Eigen::Matrix<double, kStateDim, 1>;
    using Mat3  = Eigen::Matrix3d;
    using Vec3  = Eigen::Vector3d;

    explicit RobotProcessModel(const RobotInEKFConfig& config, double q_scale = 1.0) noexcept
        : config_(config)
        , q_scale_(q_scale) {}

    [[nodiscard]] static Eigen::Vector3d armor_position(const Point& x, int armor_id) noexcept;

    [[nodiscard]] Point operator()(const Point& x, double dt) const noexcept {
        const double v_yaw = detail::sanitize_finite(x.second(kIdxVYaw), 0.0);
        const double yaw0  = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        const double yaw1  = yaw0 + v_yaw * dt;
        const Mat3 R1      = Eigen::AngleAxisd(yaw1, Vec3::UnitZ()).toRotationMatrix();

        Vec3 v0 = x.first.v();
        Vec3 v1 = v0;
        Vec3 p1 = x.first.p();

        if (config_.use_coordinated_turn) {
            double omega       = v_yaw;
            const double eps   = detail::kCoordinatedTurnOmegaEpsilon;
            const double theta = omega * dt;
            const double c     = std::cos(theta);
            const double s     = std::sin(theta);

            v1.x() = c * v0.x() - s * v0.y();
            v1.y() = s * v0.x() + c * v0.y();
            v1.z() = v0.z();

            if (std::abs(omega) > eps) {
                const double inv_omega = 1.0 / omega;
                p1.x() += inv_omega * (s * v0.x() - (1.0 - c) * v0.y());
                p1.y() += inv_omega * ((1.0 - c) * v0.x() + s * v0.y());
            } else {
                p1.x() += v0.x() * dt;
                p1.y() += v0.y() * dt;
            }
            p1.z() += v0.z() * dt;
        } else {
            p1 = (p1 + v0 * dt).eval();
        }

        SE23d::IsometriesType t;
        t[0] = v1;
        t[1] = p1;

        Point out;
        out.first  = SE23d(R1, t);
        out.second = x.second;
        return out;
    }

    [[nodiscard]] MatX process_noise(double dt) const noexcept {
        MatX Q = MatX::Zero();

        const double sigma_a_xy2  = config_.sigma_a_xy * config_.sigma_a_xy;
        const double sigma_a_z2   = config_.sigma_a_z * config_.sigma_a_z;
        const double sigma_a_yaw2 = config_.sigma_yaw * config_.sigma_yaw;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        auto fill_cv = [&](int idx_v, int idx_p, double q) {
            Q(idx_v, idx_v) += q * dt2;
            Q(idx_p, idx_v) += q * dt3 / 2.0;
            Q(idx_v, idx_p) += q * dt3 / 2.0;
            Q(idx_p, idx_p) += q * dt4 / 4.0;
        };

        fill_cv(3, 6, sigma_a_xy2);
        fill_cv(4, 7, sigma_a_xy2);
        fill_cv(5, 8, sigma_a_z2);

        Q(2, 2) += sigma_a_yaw2 * dt4 / 4.0;
        Q(2, kGroupDim + kIdxVYaw) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, 2) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) += sigma_a_yaw2 * dt2;

        const double sigma_r2 = config_.sigma_r * config_.sigma_r;
        const double sigma_h2 = config_.sigma_h * config_.sigma_h;
        Q(kGroupDim + kIdxR0, kGroupDim + kIdxR0) += sigma_r2 * dt;
        Q(kGroupDim + kIdxR1, kGroupDim + kIdxR1) += sigma_r2 * dt;
        Q(kGroupDim + kIdxH, kGroupDim + kIdxH) += sigma_h2 * dt;

        if (q_scale_ != 1.0) {
            Q.template block<6, 6>(3, 3) *= q_scale_;
            const int iy = 2;
            const int iw = kGroupDim + kIdxVYaw;
            Q(iy, iy) *= q_scale_;
            Q(iy, iw) *= q_scale_;
            Q(iw, iy) *= q_scale_;
            Q(iw, iw) *= q_scale_;
        }

        Q.diagonal().array() += detail::kCovarianceRegularization;
        return Q;
    }

    [[nodiscard]] static MatX jacobian(const Point& x, double dt) noexcept {
        const double v_yaw      = detail::sanitize_finite(x.second(kIdxVYaw), 0.0);
        const double omega_z_dt = v_yaw * dt;

        MatX Phi = MatX::Identity();

        Phi(0, 1) = omega_z_dt;
        Phi(1, 0) = -omega_z_dt;
        Phi(3, 4) = omega_z_dt;
        Phi(4, 3) = -omega_z_dt;
        Phi(6, 7) = omega_z_dt;
        Phi(7, 6) = -omega_z_dt;

        Phi.template block<3, 3>(6, 3) = Mat3::Identity() * dt;
        Phi(2, kGroupDim + kIdxVYaw)   = dt;

        return Phi;
    }

private:
    const RobotInEKFConfig& config_;
    double q_scale_{1.0};
};

class OutpostProcessModel {
public:
    static constexpr int kExtraDim = 3;
    static constexpr int kGroupDim = 9;
    static constexpr int kStateDim = kGroupDim + kExtraDim;

    static constexpr int kIdxVYaw = 0;
    static constexpr int kIdxDz1  = 1;
    static constexpr int kIdxDz2  = 2;

    using SE23d = group::SEn3<double, 2>;
    using Point = SE23xRn<kExtraDim>;
    using MatX  = Eigen::Matrix<double, kStateDim, kStateDim>;
    using VecX  = Eigen::Matrix<double, kStateDim, 1>;
    using Mat3  = Eigen::Matrix3d;
    using Vec3  = Eigen::Vector3d;

    explicit OutpostProcessModel(const OutpostInEKFConfig& config, double q_scale = 1.0) noexcept
        : config_(config)
        , q_scale_(q_scale) {}

    [[nodiscard]] static Eigen::Vector3d armor_position(const Point& x, int armor_id) noexcept;

    [[nodiscard]] Point operator()(const Point& x, double dt) const noexcept {
        const double v_yaw = detail::sanitize_finite(x.second(kIdxVYaw), 0.0);
        const double yaw0  = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        const double yaw1  = yaw0 + v_yaw * dt;
        const Mat3 R1      = Eigen::AngleAxisd(yaw1, Vec3::UnitZ()).toRotationMatrix();

        Vec3 v0 = x.first.v();
        Vec3 v1 = v0;
        Vec3 p1 = x.first.p();
        p1      = (p1 + v0 * dt).eval();

        SE23d::IsometriesType t;
        t[0] = v1;
        t[1] = p1;

        Point out;
        out.first  = SE23d(R1, t);
        out.second = x.second;
        return out;
    }

    [[nodiscard]] MatX process_noise(double dt) const noexcept {
        MatX Q = MatX::Zero();

        const double sigma_a_xy2  = config_.sigma_a_xy * config_.sigma_a_xy;
        const double sigma_a_z2   = config_.sigma_a_z * config_.sigma_a_z;
        const double sigma_a_yaw2 = config_.sigma_yaw * config_.sigma_yaw;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        auto fill_cv = [&](int idx_v, int idx_p, double q) {
            Q(idx_v, idx_v) += q * dt2;
            Q(idx_p, idx_v) += q * dt3 / 2.0;
            Q(idx_v, idx_p) += q * dt3 / 2.0;
            Q(idx_p, idx_p) += q * dt4 / 4.0;
        };

        fill_cv(3, 6, sigma_a_xy2);
        fill_cv(4, 7, sigma_a_xy2);
        fill_cv(5, 8, sigma_a_z2);

        Q(2, 2) += sigma_a_yaw2 * dt4 / 4.0;
        Q(2, kGroupDim + kIdxVYaw) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, 2) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) += sigma_a_yaw2 * dt2;

        const double sigma_dz2 = config_.sigma_h * config_.sigma_h;
        Q(kGroupDim + kIdxDz1, kGroupDim + kIdxDz1) += sigma_dz2 * dt;
        Q(kGroupDim + kIdxDz2, kGroupDim + kIdxDz2) += sigma_dz2 * dt;

        if (q_scale_ != 1.0) {
            Q.template block<6, 6>(3, 3) *= q_scale_;
            const int iy = 2;
            const int iw = kGroupDim + kIdxVYaw;
            Q(iy, iy) *= q_scale_;
            Q(iy, iw) *= q_scale_;
            Q(iw, iy) *= q_scale_;
            Q(iw, iw) *= q_scale_;
        }

        Q.diagonal().array() += detail::kCovarianceRegularization;
        return Q;
    }

    [[nodiscard]] static MatX jacobian(const Point& x, double dt) noexcept {
        const double v_yaw      = detail::sanitize_finite(x.second(kIdxVYaw), 0.0);
        const double omega_z_dt = v_yaw * dt;

        MatX Phi  = MatX::Identity();
        Phi(0, 1) = omega_z_dt;
        Phi(1, 0) = -omega_z_dt;
        Phi(3, 4) = omega_z_dt;
        Phi(4, 3) = -omega_z_dt;
        Phi(6, 7) = omega_z_dt;
        Phi(7, 6) = -omega_z_dt;

        Phi.template block<3, 3>(6, 3) = Mat3::Identity() * dt;
        Phi(2, kGroupDim + kIdxVYaw)   = dt;
        return Phi;
    }

private:
    const OutpostInEKFConfig& config_;
    double q_scale_{1.0};
};

// ============================================================================
// Redact Perception/Assumption adapters
// ============================================================================

template <int ExtraDim>
struct BeliefTag {
    using Flux  = SE23xRn<ExtraDim>;
    using Drift = typename Flux::VectorType;
};

struct MeasurementTag {
    using Flux  = YpdAy;
    using Drift = YpdAy::VectorType;
};

template <typename Config, typename ProcessModel, int ArmorsNum>
class SphericalPerception {
public:
    using Belief      = BeliefTag<ProcessModel::kExtraDim>;
    using Measurement = MeasurementTag;
    using Flux        = typename Belief::Flux;
    using MatrixDD    = Eigen::MatrixXd;

    enum class Mode {
        ArmorSpherical, // [yaw, pitch, dist, armor_yaw]
        ArmorYpdOnly,   // [yaw, pitch, dist] (armor_yaw ignored)
        TargetYawOnly   // [target_yaw] packed into yaw component, others ignored
    };

    void configure_armor(int armor_id, bool include_armor_yaw, const Config& cfg) noexcept {
        mode_     = include_armor_yaw ? Mode::ArmorSpherical : Mode::ArmorYpdOnly;
        armor_id_ = armor_id;
        cfg_      = &cfg;
        sigma_w_  = 0.0;
        R_.setIdentity();
    }

    void configure_target_yaw_only(double sigma_w, const Config& cfg) noexcept {
        mode_     = Mode::TargetYawOnly;
        armor_id_ = 0;
        cfg_      = &cfg;
        sigma_w_  = sigma_w;
        R_.setIdentity();
    }

    void prepare(const Flux& mean_state, const Measurement::Flux& z_meas) noexcept {
        (void)z_meas;
        if (!cfg_) {
            R_.setIdentity();
            return;
        }

        // Base R (possibly later scaled by robust logic in the caller).
        Eigen::Matrix<double, 4, 4> R = Eigen::Matrix<double, 4, 4>::Zero();

        if (mode_ == Mode::TargetYawOnly) {
            const double sw2 = sigma_w_ * sigma_w_;
            R(0, 0)          = (sw2 > 0.0 && std::isfinite(sw2)) ? sw2 : 1.0;
            R(1, 1)          = 1e12;
            R(2, 2)          = 1e12;
            R(3, 3)          = 1e12;
            R_               = R;
            return;
        }

        // For armor observations, compute view-angle dependent distance noise using prediction at
        // mean.
        const Eigen::Vector4d z_pred =
            to_vec(predict_armor_measurement(mean_state, /*include_armor_yaw=*/true));

        const double sigma_yaw   = cfg_->sigma_yaw_obs;
        const double sigma_pitch = cfg_->sigma_pitch_obs;
        const double sigma_ayaw  = cfg_->sigma_armor_yaw;

        R(0, 0) = sigma_yaw * sigma_yaw;
        R(1, 1) = sigma_pitch * sigma_pitch;

        const double center_yaw = z_pred(0);
        const double armor_yaw  = z_pred(3);
        const double delta      = fcs::L3::shortest_rad(center_yaw, armor_yaw);
        R(2, 2)                 = std::log(std::abs(delta) + 1.0) + 1.0;

        if (mode_ == Mode::ArmorSpherical) {
            const double d                = std::max(0.0, z_pred(2));
            const double sigma_armor_yaw2 = std::pow(std::log(d + 1.0) / 200.0 + sigma_ayaw, 2);
            R(3, 3)                       = sigma_armor_yaw2;
        } else {
            R(3, 3) = 1e12;
        }

        R.diagonal().array() += detail::kCovarianceRegularization;
        R_ = R;
    }

    void apply_robust_scale(double maha, int dof) noexcept {
        if (!cfg_ || !cfg_->enable_robust_update) {
            return;
        }
        const double s = detail::robust_R_scale_from_maha(maha, dof, detail::kRobustMaxRScale);
        if (s != 1.0) {
            R_ *= s;
        }
    }

    [[nodiscard]] Measurement::Flux assume(const Flux& x) const noexcept {
        if (mode_ == Mode::TargetYawOnly) {
            const double target_yaw = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
            return Measurement::Flux(target_yaw, 0.0, 0.0, 0.0);
        }
        const bool include_armor_yaw = (mode_ == Mode::ArmorSpherical);
        return predict_armor_measurement(x, include_armor_yaw);
    }

    [[nodiscard]] typename Belief::Drift
        extrude(auto& /*P*/, typename Measurement::Drift /*innovation*/) const noexcept {
        return typename Belief::Drift::Zero();
    }

    [[nodiscard]] MatrixDD measurement_noise_covariance() const noexcept { return R_; }

    [[nodiscard]] MatrixDD measurement_jacobian(const Flux& /*x*/) const noexcept {
        // Sigma-point update does not require H. Provide a safe fallback.
        MatrixDD H = MatrixDD::Zero(4, Flux::Dim);
        return H;
    }

private:
    Mode mode_{Mode::ArmorSpherical};
    int armor_id_{0};
    const Config* cfg_{nullptr};
    double sigma_w_{0.0};
    Eigen::Matrix<double, 4, 4> R_{Eigen::Matrix<double, 4, 4>::Identity()};

    [[nodiscard]] static Eigen::Vector4d to_vec(const Measurement::Flux& z) noexcept {
        return Measurement::Flux::log(z);
    }

    [[nodiscard]] Measurement::Flux
        predict_armor_measurement(const Flux& x, bool include_armor_yaw) const noexcept {
        // armor_xyz in world
        const Eigen::Vector3d armor_xyz = ProcessModel::armor_position(x, armor_id_);
        const Eigen::Vector3d ypd       = fcs::L3::xyz2ypd(armor_xyz);

        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(ArmorsNum);
        const double target_yaw     = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        const double armor_yaw =
            include_armor_yaw
                ? fcs::L3::normalize_rad(target_yaw + static_cast<double>(armor_id_) * kAngleStep)
                : 0.0;
        return Measurement::Flux(ypd.x(), ypd.y(), ypd.z(), armor_yaw);
    }
};

struct RobotPerception : public SphericalPerception<RobotInEKFConfig, RobotProcessModel, 4> {};
struct OutpostPerception
    : public SphericalPerception<OutpostInEKFConfig, OutpostProcessModel, 3> {};

struct RobotAssumption {
    using Perception = RobotPerception;
    using Flux       = typename Perception::Belief::Flux;

    void configure(const RobotInEKFConfig& cfg, double q_scale) noexcept {
        cfg_     = &cfg;
        q_scale_ = q_scale;
    }

    [[nodiscard]] Flux maneuver(auto& /*P*/, const Flux& x, double dt) const noexcept {
        if (!cfg_) {
            return x;
        }
        return RobotProcessModel(*cfg_, q_scale_)(x, dt);
    }

    [[nodiscard]] Eigen::MatrixXd process_noise_covariance(double dt) const noexcept {
        if (!cfg_) {
            return Eigen::MatrixXd::Zero(Flux::Dim, Flux::Dim);
        }
        return RobotProcessModel(*cfg_, q_scale_).process_noise(dt);
    }

    [[nodiscard]] Eigen::MatrixXd jacobian(const Flux& x, double dt) const noexcept {
        return RobotProcessModel::jacobian(x, dt);
    }

private:
    const RobotInEKFConfig* cfg_{nullptr};
    double q_scale_{1.0};
};

struct OutpostAssumption {
    using Perception = OutpostPerception;
    using Flux       = typename Perception::Belief::Flux;

    void configure(const OutpostInEKFConfig& cfg, double q_scale) noexcept {
        cfg_     = &cfg;
        q_scale_ = q_scale;
    }

    [[nodiscard]] Flux maneuver(auto& /*P*/, const Flux& x, double dt) const noexcept {
        if (!cfg_) {
            return x;
        }
        return OutpostProcessModel(*cfg_, q_scale_)(x, dt);
    }

    [[nodiscard]] Eigen::MatrixXd process_noise_covariance(double dt) const noexcept {
        if (!cfg_) {
            return Eigen::MatrixXd::Zero(Flux::Dim, Flux::Dim);
        }
        return OutpostProcessModel(*cfg_, q_scale_).process_noise(dt);
    }

    [[nodiscard]] Eigen::MatrixXd jacobian(const Flux& x, double dt) const noexcept {
        return OutpostProcessModel::jacobian(x, dt);
    }

private:
    const OutpostInEKFConfig* cfg_{nullptr};
    double q_scale_{1.0};
};

// ============================================================================
// Common tracker utilities (shared between Robot/Outpost)
// ============================================================================

template <typename FilterT, int StateDim>
class TrackerInEKF_Common {
public:
    std::optional<FilterT> filter_;

protected:
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using MatX = Eigen::Matrix<double, StateDim, StateDim>;
    using VecX = Eigen::Matrix<double, StateDim, 1>;

    double adaptive_process_noise_scale_{1.0};
    double time_since_last_predict_{0.0};

    bool has_center_ref_{false};
    Vec3 center_ref_{Vec3::Zero()};
    Vec3 center_latest_{Vec3::Zero()};
    bool has_center_latest_{false};
    Mat3 center_ref_cov_{Mat3::Identity()};
    Mat3 center_latest_cov_{Mat3::Identity()};
    double center_dt_acc_{0.0};

    void on_predict_dt(double dt) noexcept {
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            return;
        }
        time_since_last_predict_ = dt;
        center_dt_acc_ += dt;
        if (!std::isfinite(center_dt_acc_)) {
            center_dt_acc_ = 0.0;
        }
    }

    template <typename Config>
    void maybe_bump_adaptive_process_noise(double maha, int dof, const Config& config) noexcept {
        if (!config.enable_adaptive_process_noise) {
            adaptive_process_noise_scale_ = 1.0;
            return;
        }
        const double s =
            detail::adaptive_scale_from_maha(maha, dof, detail::kAdaptiveProcessNoiseMaxScale);
        adaptive_process_noise_scale_ = std::max(adaptive_process_noise_scale_, s);
    }

    template <typename Config>
    void maybe_decay_adaptive_process_noise(double dt, const Config& config) noexcept {
        if (!config.enable_adaptive_process_noise) {
            adaptive_process_noise_scale_ = 1.0;
            return;
        }
        adaptive_process_noise_scale_ = detail::decay_to_one(
            adaptive_process_noise_scale_, dt, detail::kAdaptiveProcessNoiseDecayTau);
        adaptive_process_noise_scale_ = std::clamp(adaptive_process_noise_scale_, 1.0, 1e6);
    }

    void set_center_latest_measurement(const Vec3& p_meas, const Mat3& cov_meas) noexcept {
        center_latest_     = p_meas;
        center_latest_cov_ = cov_meas;
        has_center_latest_ = true;
        if (!has_center_ref_) {
            center_ref_        = p_meas;
            center_ref_cov_    = cov_meas;
            has_center_ref_    = true;
            has_center_latest_ = false;
            center_dt_acc_     = 0.0;
        }
    }

    template <typename Config>
    [[nodiscard]] bool should_run_pseudo_velocity_update(const Config& config) const noexcept {
        if (!config.enable_pseudo_velocity_update) {
            return false;
        }
        if (!has_center_ref_ || !has_center_latest_) {
            return false;
        }
        const double dt = center_dt_acc_;
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            return false;
        }
        return true;
    }

    template <typename Config, typename Flux>
    void maybe_update_pseudo_velocity(
        const Config& config, const Flux& state, const Eigen::Matrix3d& R_world) noexcept {
        if (!should_run_pseudo_velocity_update(config) || !filter_) {
            return;
        }

        const double dt = center_dt_acc_;
        if (dt > detail::kPseudoVelocityMaxInterval) {
            center_ref_        = center_latest_;
            center_ref_cov_    = center_latest_cov_;
            has_center_latest_ = false;
            center_dt_acc_     = 0.0;
            return;
        }
        if (dt < detail::kPseudoVelocityMinInterval) {
            return;
        }

        const Vec3 v_meas = (center_latest_ - center_ref_) / dt;
        const Vec3 v_est  = state.first.v();
        const Vec3 nu     = v_meas - v_est;

        Eigen::Matrix<double, 3, StateDim> H = Eigen::Matrix<double, 3, StateDim>::Zero();
        H.template block<3, 3>(0, 3)         = R_world; // map body-frame δv to world

        Eigen::Matrix3d Rv = (center_latest_cov_ + center_ref_cov_) / (dt * dt);
        Rv.diagonal().array() += detail::kCovarianceRegularization;
        if (config.enable_robust_update) {
            const double maha = nu.dot(Rv.ldlt().solve(nu));
            const double scale =
                detail::robust_R_scale_from_maha(maha, 3, detail::kRobustMaxRScale);
            Rv *= scale;
        }

        // Apply linearized update directly on the underlying filter (KF-style).
        auto& kf = *filter_;
        auto& P  = kf.get_unc();

        const Eigen::MatrixXd K = P.compute_kalman_gain(H, Rv);
        Eigen::VectorXd dx_vec  = K * nu;
        if (dx_vec.size() != StateDim) {
            return;
        }

        // Velocity-only correction (indices 3..5) in right-invariant coordinates.
        for (int i = 0; i < dx_vec.size(); ++i) {
            if (i < 3 || i > 5) {
                dx_vec(i) = 0.0;
            }
        }
        dx_vec(0) = 0.0;
        dx_vec(1) = 0.0;

        VecX dx;
        dx = dx_vec;

        kf.state_mut() = uni_kf::redact::lie_ops::compose(kf.state(), dx);
        P.stiffen(K, H, Rv, time_since_last_predict_);

        center_ref_        = center_latest_;
        center_ref_cov_    = center_latest_cov_;
        has_center_latest_ = false;
        center_dt_acc_     = 0.0;
    }

    void zero_out_roll_pitch_covariance() noexcept {
        if (!filter_) {
            return;
        }
        auto& P           = filter_->get_unc();
        Eigen::MatrixXd M = P.mat();

        if (M.rows() >= 2) {
            // CRITICAL FIX: Do NOT zero entire rows/cols - only diagonal elements
            // Preserving cross-covariances maintains full-rank matrix for numerical stability
            M(0, 0) = detail::kCovarianceRegularization;
            M(1, 1) = detail::kCovarianceRegularization;
            // Cross-covariances with roll/pitch (rows/cols 0,1) are preserved naturally
        }

        M = (M + M.transpose()) / 2.0;
        M.diagonal().array() += detail::kCovarianceRegularization;
        P.storage().update(M);
    }
};

// ============================================================================
// Robot tracker (4 armors)
// ============================================================================

class TrackerInEKF_Robot
    : public TrackerInEKF_Common<
          uni_kf::redact::KalmanFilter<
              uni_kf::redact::Covariance<
                  SE23xRn<RobotProcessModel::kExtraDim>, uni_kf::redact::Representation::Sr,
                  uni_kf::redact::CubatureStrategy>,
              RobotPerception, RobotAssumption>,
          RobotProcessModel::kStateDim> {
    using Base = TrackerInEKF_Common<
        uni_kf::redact::KalmanFilter<
            uni_kf::redact::Covariance<
                SE23xRn<RobotProcessModel::kExtraDim>, uni_kf::redact::Representation::Sr,
                uni_kf::redact::CubatureStrategy>,
            RobotPerception, RobotAssumption>,
        RobotProcessModel::kStateDim>;

public:
    using SE23d          = group::SEn3<double, 2>;
    using Vec3           = Eigen::Vector3d;
    using Mat3           = Eigen::Matrix3d;
    using ReferenceFrame = fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>;

    static constexpr int kArmorsNum = 4;
    static constexpr int kExtraDim  = RobotProcessModel::kExtraDim;
    static constexpr int kGroupDim  = RobotProcessModel::kGroupDim;
    static constexpr int kStateDim  = RobotProcessModel::kStateDim;

    static constexpr int kIdxVYaw = RobotProcessModel::kIdxVYaw;
    static constexpr int kIdxR0   = RobotProcessModel::kIdxR0;
    static constexpr int kIdxR1   = RobotProcessModel::kIdxR1;
    static constexpr int kIdxH    = RobotProcessModel::kIdxH;

    static constexpr double kMinRadius = 0.17;
    static constexpr double kMaxRadius = 0.27;
    static constexpr double kMinHeight = -0.2;
    static constexpr double kMaxHeight = 0.2;

    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;

    TrackerInEKF_Robot() noexcept = default;

    [[nodiscard]] bool is_initialized() const noexcept { return this->filter_.has_value(); }

    void init(const Vec3& p0, double yaw0, const RobotInEKFConfig& config) noexcept {
        if (!detail::all_finite(p0) || !std::isfinite(yaw0)) {
            this->filter_.reset();
            return;
        }

        SE23d::IsometriesType t;
        t[0]         = Vec3::Zero();
        t[1]         = p0;
        const Mat3 R = Eigen::AngleAxisd(yaw0, Vec3::UnitZ()).toRotationMatrix();
        const SE23d se23(R, t);

        Eigen::Matrix<double, kExtraDim, 1> extra;
        extra << 0.0, config.radius0, config.radius1, config.height;
        clamp_extra_in_place(extra);

        SE23xRn<kExtraDim> state(se23, extra);

        Eigen::Matrix<double, kStateDim, kStateDim> P0 =
            Eigen::Matrix<double, kStateDim, kStateDim>::Zero();
        for (int i = 0; i < kGroupDim; ++i) {
            P0(i, i) = 0.1;
        }
        P0(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) = 1.0;
        P0(kGroupDim + kIdxR0, kGroupDim + kIdxR0)     = 0.01;
        P0(kGroupDim + kIdxR1, kGroupDim + kIdxR1)     = 0.01;
        P0(kGroupDim + kIdxH, kGroupDim + kIdxH)       = 0.01;

        RobotPerception perception;
        RobotAssumption assumption;
        assumption.configure(config, 1.0);

        this->filter_.emplace(state, P0, perception, assumption);
        this->adaptive_process_noise_scale_ = 1.0;
        this->time_since_last_predict_      = 0.0;
        this->has_center_ref_               = false;
        this->has_center_latest_            = false;
        this->center_dt_acc_                = 0.0;
    }

    void init(
        const Vec3& p0, double yaw0, const RobotInEKFConfig& config,
        const ReferenceFrame& ref) noexcept {
        (void)ref;
        init(p0, yaw0, config);
    }

    void predict(double dt, const RobotInEKFConfig& config) noexcept {
        if (!this->filter_ || !(dt > 0.0) || !std::isfinite(dt)) {
            return;
        }
        this->on_predict_dt(dt);

        const double q_scale =
            config.enable_adaptive_process_noise ? this->adaptive_process_noise_scale_ : 1.0;
        this->filter_->get_assumption().configure(config, q_scale);
        (void)this->filter_->predict(dt);
        this->maybe_decay_adaptive_process_noise(dt, config);

        clamp_extra_in_place(this->filter_->state_mut().second);
        this->zero_out_roll_pitch_covariance();
    }

    void predict(double dt, const RobotInEKFConfig& config, const ReferenceFrame& ref) noexcept {
        (void)ref;
        predict(dt, config);
    }

    void update_yaw(double yaw_meas, const RobotInEKFConfig& config) noexcept {
        if (!this->filter_ || !std::isfinite(yaw_meas) || !(config.sigma_w > 0.0)
            || !std::isfinite(config.sigma_w)) {
            return;
        }

        auto& kf         = *this->filter_;
        auto& perception = kf.get_perception();

        perception.configure_target_yaw_only(config.sigma_w, config);

        const auto z_meas = YpdAy(yaw_meas, 0.0, 0.0, 0.0);
        perception.prepare(kf.state(), z_meas);

        // Adaptive Q bump based on yaw innovation
        const auto z_pred       = perception.assume(kf.state());
        const Eigen::Vector4d r = YpdAy::log(z_pred.inv() * z_meas);
        const double maha       = (r(0) * r(0)) / (config.sigma_w * config.sigma_w);
        this->maybe_bump_adaptive_process_noise(maha, 1, config);

        perception.apply_robust_scale(maha, 1);
        (void)this->filter_->percept(z_meas, this->time_since_last_predict_);

        clamp_extra_in_place(kf.state_mut().second);
        this->zero_out_roll_pitch_covariance();
    }

    void update_yaw(
        double yaw_meas, const RobotInEKFConfig& config, const ReferenceFrame& ref) noexcept {
        (void)ref;
        update_yaw(yaw_meas, config);
    }

    void update_armor(const Vec3& y, int armor_id, const RobotInEKFConfig& config) noexcept {
        if (!this->filter_ || !detail::all_finite(y) || armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }
        const Eigen::Vector3d ypd = fcs::L3::xyz2ypd(y);

        // Only update yaw/pitch/distance; ignore armor_yaw component.
        update_armor_spherical_impl(
            ypd, /*armor_yaw_meas=*/0.0, armor_id, config, /*include_armor_yaw=*/false);
    }

    void update_armor(
        const Vec3& y, int armor_id, const RobotInEKFConfig& config,
        const ReferenceFrame& ref) noexcept {
        (void)ref;
        update_armor(y, armor_id, config);
    }

    void update_armor_spherical(
        const Vec3& armor_ypd, double armor_yaw_meas, int armor_id,
        const RobotInEKFConfig& config) noexcept {
        if (!this->filter_ || armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }
        if (!std::isfinite(armor_ypd.x()) || !std::isfinite(armor_ypd.y())
            || !std::isfinite(armor_ypd.z()) || !std::isfinite(armor_yaw_meas)) {
            return;
        }
        update_armor_spherical_impl(
            armor_ypd, armor_yaw_meas, armor_id, config, /*include_armor_yaw=*/true);
    }

    void update_armor_spherical(
        const Vec3& armor_ypd, double armor_yaw_meas, int armor_id, const RobotInEKFConfig& config,
        const ReferenceFrame& ref) noexcept {
        (void)ref;
        update_armor_spherical(armor_ypd, armor_yaw_meas, armor_id, config);
    }

    [[nodiscard]] Vec3 get_position() const noexcept {
        return this->filter_ ? this->filter_->state().first.p() : Vec3::Zero();
    }

    [[nodiscard]] Vec3 get_velocity() const noexcept {
        return this->filter_ ? this->filter_->state().first.v() : Vec3::Zero();
    }

    [[nodiscard]] double get_yaw() const noexcept {
        if (!this->filter_) {
            return 0.0;
        }
        const auto& R = this->filter_->state().first.R();
        return std::atan2(R(1, 0), R(0, 0));
    }

    [[nodiscard]] double get_v_yaw() const noexcept {
        return this->filter_ ? this->filter_->state().second(kIdxVYaw) : 0.0;
    }

    [[nodiscard]] RobotTargetState get_robot_state() const noexcept {
        RobotTargetState out;
        if (!this->filter_) {
            return out;
        }
        const auto& s  = this->filter_->state();
        out.position   = s.first.p();
        out.velocity   = s.first.v();
        out.yaw        = std::atan2(s.first.R()(1, 0), s.first.R()(0, 0));
        out.v_yaw      = s.second(kIdxVYaw);
        out.radius0    = s.second(kIdxR0);
        out.radius1    = s.second(kIdxR1);
        out.z1         = out.position.z() + s.second(kIdxH);
        out.armors_num = kArmorsNum;
        return out;
    }

private:
    static void clamp_extra_in_place(Eigen::Matrix<double, kExtraDim, 1>& e) noexcept {
        // e(kIdxR0) = std::clamp(e(kIdxR0), kMinRadius, kMaxRadius);
        // e(kIdxR1) = std::clamp(e(kIdxR1), kMinRadius, kMaxRadius);
        // e(kIdxH)  = std::clamp(e(kIdxH), kMinHeight, kMaxHeight);
    }

    void update_armor_spherical_impl(
        const Vec3& armor_ypd, double armor_yaw_meas, int armor_id, const RobotInEKFConfig& config,
        bool include_armor_yaw, const ReferenceFrame* ref = nullptr) noexcept {
        auto& kf         = *this->filter_;
        auto& perception = kf.get_perception();

        perception.configure_armor(armor_id, include_armor_yaw, config);

        const double armor_yaw_use = include_armor_yaw ? armor_yaw_meas : 0.0;
        const YpdAy z_meas(armor_ypd.x(), armor_ypd.y(), armor_ypd.z(), armor_yaw_use);

        perception.prepare(kf.state(), z_meas);

        const auto z_pred       = perception.assume(kf.state());
        const Eigen::Vector4d r = YpdAy::log(z_pred.inv() * z_meas);

        Eigen::Matrix4d R = perception.measurement_noise_covariance();
        const double maha = r.transpose() * R.ldlt().solve(r);
        this->maybe_bump_adaptive_process_noise(maha, include_armor_yaw ? 4 : 3, config);
        perception.apply_robust_scale(maha, include_armor_yaw ? 4 : 3);

        // Optional pseudo-velocity update: infer center position measurement from armor
        // observation.
        if (config.enable_pseudo_velocity_update) {
            const auto& x = kf.state();
            const bool is_high =
                (armor_id == RobotProcessModel::kHighArmorIdA
                 || armor_id == RobotProcessModel::kHighArmorIdB);
            const double radius   = is_high ? x.second(kIdxR1) : x.second(kIdxR0);
            const double z_offset = is_high ? x.second(kIdxH) : 0.0;

            const Eigen::Vector3d armor_xyz_meas = fcs::L3::ypd2xyz(armor_ypd);
            const double a                       = armor_yaw_meas;
            Eigen::Vector3d center_meas          = armor_xyz_meas;
            center_meas.x() += radius * std::cos(a);
            center_meas.y() += radius * std::sin(a);
            center_meas.z() -= z_offset;

            const Eigen::Matrix3d J_xyz_ypd = fcs::L3::ypd2xyz_jacobian(armor_ypd);
            Eigen::Matrix3d R_ypd           = Eigen::Matrix3d::Zero();
            R_ypd(0, 0)                     = config.sigma_yaw_obs * config.sigma_yaw_obs;
            R_ypd(1, 1)                     = config.sigma_pitch_obs * config.sigma_pitch_obs;
            R_ypd(2, 2)                     = 1.0;

            Eigen::Matrix3d P_center = J_xyz_ypd * R_ypd * J_xyz_ypd.transpose();
            P_center.diagonal().array() += detail::kCovarianceRegularization;

            this->set_center_latest_measurement(center_meas, P_center);
            this->maybe_update_pseudo_velocity(config, kf.state(), kf.state().first.R());
        }

        (void)ref;
        (void)this->filter_->percept(z_meas, this->time_since_last_predict_);
        clamp_extra_in_place(kf.state_mut().second);
        this->zero_out_roll_pitch_covariance();
    }
};

// ============================================================================
// Outpost tracker (3 armors)
// ============================================================================

class TrackerInEKF_Outpost
    : public TrackerInEKF_Common<
          uni_kf::redact::KalmanFilter<
              uni_kf::redact::Covariance<
                  SE23xRn<OutpostProcessModel::kExtraDim>, uni_kf::redact::Representation::Sr,
                  uni_kf::redact::CubatureStrategy>,
              OutpostPerception, OutpostAssumption>,
          OutpostProcessModel::kStateDim> {
    using Base = TrackerInEKF_Common<
        uni_kf::redact::KalmanFilter<
            uni_kf::redact::Covariance<
                SE23xRn<OutpostProcessModel::kExtraDim>, uni_kf::redact::Representation::Sr,
                uni_kf::redact::CubatureStrategy>,
            OutpostPerception, OutpostAssumption>,
        OutpostProcessModel::kStateDim>;

public:
    using SE23d          = group::SEn3<double, 2>;
    using Vec3           = Eigen::Vector3d;
    using Mat3           = Eigen::Matrix3d;
    using ReferenceFrame = fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>;

    static constexpr int kArmorsNum = 3;
    static constexpr int kExtraDim  = OutpostProcessModel::kExtraDim;
    static constexpr int kGroupDim  = OutpostProcessModel::kGroupDim;
    static constexpr int kStateDim  = OutpostProcessModel::kStateDim;

    static constexpr int kIdxVYaw = OutpostProcessModel::kIdxVYaw;
    static constexpr int kIdxDz1  = OutpostProcessModel::kIdxDz1;
    static constexpr int kIdxDz2  = OutpostProcessModel::kIdxDz2;

    TrackerInEKF_Outpost() noexcept = default;

    [[nodiscard]] bool is_initialized() const noexcept { return this->filter_.has_value(); }

    void init(const Vec3& p0, double yaw0, const OutpostInEKFConfig& config) noexcept {
        if (!detail::all_finite(p0) || !std::isfinite(yaw0)) {
            this->filter_.reset();
            return;
        }

        SE23d::IsometriesType t;
        t[0]         = Vec3::Zero();
        t[1]         = p0;
        const Mat3 R = Eigen::AngleAxisd(yaw0, Vec3::UnitZ()).toRotationMatrix();
        const SE23d se23(R, t);

        Eigen::Matrix<double, kExtraDim, 1> extra;
        extra << 0.0, 0.0, 0.0;
        SE23xRn<kExtraDim> state(se23, extra);

        Eigen::Matrix<double, kStateDim, kStateDim> P0 =
            Eigen::Matrix<double, kStateDim, kStateDim>::Zero();
        for (int i = 0; i < kGroupDim; ++i) {
            P0(i, i) = 0.1;
        }
        P0(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) = 1.0;
        P0(kGroupDim + kIdxDz1, kGroupDim + kIdxDz1)   = 0.01;
        P0(kGroupDim + kIdxDz2, kGroupDim + kIdxDz2)   = 0.01;

        OutpostPerception perception;
        OutpostAssumption assumption;
        assumption.configure(config, 1.0);

        this->filter_.emplace(state, P0, perception, assumption);
        this->adaptive_process_noise_scale_ = 1.0;
        this->time_since_last_predict_      = 0.0;
        this->has_center_ref_               = false;
        this->has_center_latest_            = false;
        this->center_dt_acc_                = 0.0;
    }

    void init(
        const Vec3& p0, double yaw0, const OutpostInEKFConfig& config,
        const ReferenceFrame& ref) noexcept {
        (void)ref;
        init(p0, yaw0, config);
    }

    void predict(double dt, const OutpostInEKFConfig& config) noexcept {
        if (!this->filter_ || !(dt > 0.0) || !std::isfinite(dt)) {
            return;
        }
        this->on_predict_dt(dt);
        const double q_scale =
            config.enable_adaptive_process_noise ? this->adaptive_process_noise_scale_ : 1.0;
        this->filter_->get_assumption().configure(config, q_scale);
        (void)this->filter_->predict(dt);
        this->maybe_decay_adaptive_process_noise(dt, config);
        this->zero_out_roll_pitch_covariance();
    }

    void predict(double dt, const OutpostInEKFConfig& config, const ReferenceFrame& ref) noexcept {
        (void)ref;
        predict(dt, config);
    }

    void update_yaw(double yaw_meas, const OutpostInEKFConfig& config) noexcept {
        if (!this->filter_ || !std::isfinite(yaw_meas) || !(config.sigma_w > 0.0)
            || !std::isfinite(config.sigma_w)) {
            return;
        }
        auto& kf         = *this->filter_;
        auto& perception = kf.get_perception();

        perception.configure_target_yaw_only(config.sigma_w, config);
        const auto z_meas = YpdAy(yaw_meas, 0.0, 0.0, 0.0);
        perception.prepare(kf.state(), z_meas);

        const auto z_pred       = perception.assume(kf.state());
        const Eigen::Vector4d r = YpdAy::log(z_pred.inv() * z_meas);
        const double maha       = (r(0) * r(0)) / (config.sigma_w * config.sigma_w);
        this->maybe_bump_adaptive_process_noise(maha, 1, config);
        perception.apply_robust_scale(maha, 1);

        (void)this->filter_->percept(z_meas, this->time_since_last_predict_);
        this->zero_out_roll_pitch_covariance();
    }

    void update_yaw(
        double yaw_meas, const OutpostInEKFConfig& config, const ReferenceFrame& ref) noexcept {
        (void)ref;
        update_yaw(yaw_meas, config);
    }

    void update_armor(const Vec3& y, int armor_id, const OutpostInEKFConfig& config) noexcept {
        if (!this->filter_ || !detail::all_finite(y) || armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }
        const Eigen::Vector3d ypd = fcs::L3::xyz2ypd(y);
        update_armor_spherical_impl(
            ypd, /*armor_yaw_meas=*/0.0, armor_id, config, /*include_armor_yaw=*/false);
    }

    void update_armor(
        const Vec3& y, int armor_id, const OutpostInEKFConfig& config,
        const ReferenceFrame& ref) noexcept {
        (void)ref;
        update_armor(y, armor_id, config);
    }

    void update_armor_spherical(
        const Vec3& armor_ypd, double armor_yaw_meas, int armor_id,
        const OutpostInEKFConfig& config) noexcept {
        if (!this->filter_ || armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }
        if (!std::isfinite(armor_ypd.x()) || !std::isfinite(armor_ypd.y())
            || !std::isfinite(armor_ypd.z()) || !std::isfinite(armor_yaw_meas)) {
            return;
        }
        update_armor_spherical_impl(
            armor_ypd, armor_yaw_meas, armor_id, config, /*include_armor_yaw=*/true);
    }

    void update_armor_spherical(
        const Vec3& armor_ypd, double armor_yaw_meas, int armor_id,
        const OutpostInEKFConfig& config, const ReferenceFrame& ref) noexcept {
        (void)ref;
        update_armor_spherical(armor_ypd, armor_yaw_meas, armor_id, config);
    }

    [[nodiscard]] double get_yaw() const noexcept {
        if (!this->filter_) {
            return 0.0;
        }
        const auto& R = this->filter_->state().first.R();
        return std::atan2(R(1, 0), R(0, 0));
    }

    [[nodiscard]] OutpostTargetState get_outpost_state() const noexcept {
        OutpostTargetState out;
        if (!this->filter_) {
            return out;
        }
        const auto& s = this->filter_->state();
        out.position  = {s.first.p().x(), s.first.p().y()};
        out.yaw       = std::atan2(s.first.R()(1, 0), s.first.R()(0, 0));
        out.v_yaw     = s.second(kIdxVYaw);
        out.z         = {
            s.first.p().z(), s.first.p().z() + s.second(kIdxDz1),
            s.first.p().z() + s.second(kIdxDz2)};
        return out;
    }

private:
    void update_armor_spherical_impl(
        const Vec3& armor_ypd, double armor_yaw_meas, int armor_id,
        const OutpostInEKFConfig& config, bool include_armor_yaw,
        const ReferenceFrame* ref = nullptr) noexcept {
        auto& kf         = *this->filter_;
        auto& perception = kf.get_perception();

        perception.configure_armor(armor_id, include_armor_yaw, config);
        const double armor_yaw_use = include_armor_yaw ? armor_yaw_meas : 0.0;
        const YpdAy z_meas(armor_ypd.x(), armor_ypd.y(), armor_ypd.z(), armor_yaw_use);
        perception.prepare(kf.state(), z_meas);

        const auto z_pred       = perception.assume(kf.state());
        const Eigen::Vector4d r = YpdAy::log(z_pred.inv() * z_meas);
        Eigen::Matrix4d R       = perception.measurement_noise_covariance();
        const double maha       = r.transpose() * R.ldlt().solve(r);

        this->maybe_bump_adaptive_process_noise(maha, include_armor_yaw ? 4 : 3, config);
        perception.apply_robust_scale(maha, include_armor_yaw ? 4 : 3);

        (void)ref;
        (void)this->filter_->percept(z_meas, this->time_since_last_predict_);
        this->zero_out_roll_pitch_covariance();
    }
};

// ============================================================================
// Helper: armor position computation used by Perception
// ============================================================================

inline Eigen::Vector3d
    RobotProcessModel::armor_position(const RobotProcessModel::Point& x, int armor_id) noexcept {
    constexpr double kAngleStep = 2.0 * std::numbers::pi / 4.0;
    const double target_yaw     = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
    const double armor_angle    = target_yaw + static_cast<double>(armor_id) * kAngleStep;

    const bool is_high =
        (armor_id == RobotProcessModel::kHighArmorIdA
         || armor_id == RobotProcessModel::kHighArmorIdB);
    const double radius =
        is_high ? x.second(RobotProcessModel::kIdxR1) : x.second(RobotProcessModel::kIdxR0);
    const double z_off = is_high ? x.second(RobotProcessModel::kIdxH) : 0.0;

    const double armor_x = x.first.p().x() - radius * std::cos(armor_angle);
    const double armor_y = x.first.p().y() - radius * std::sin(armor_angle);
    const double armor_z = x.first.p().z() + z_off;
    return Eigen::Vector3d{armor_x, armor_y, armor_z};
}

inline Eigen::Vector3d OutpostProcessModel::armor_position(
    const OutpostProcessModel::Point& x, int armor_id) noexcept {
    constexpr double kAngleStep     = 2.0 * std::numbers::pi / 3.0;
    constexpr double kOutpostRadius = 0.2765;
    const double target_yaw         = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
    const double armor_angle        = target_yaw + static_cast<double>(armor_id) * kAngleStep;

    double z_offset = 0.0;
    if (armor_id == 1) {
        z_offset = x.second(OutpostProcessModel::kIdxDz1);
    } else if (armor_id == 2) {
        z_offset = x.second(OutpostProcessModel::kIdxDz2);
    }

    const double armor_x = x.first.p().x() - kOutpostRadius * std::cos(armor_angle);
    const double armor_y = x.first.p().y() - kOutpostRadius * std::sin(armor_angle);
    const double armor_z = x.first.p().z() + z_offset;
    return Eigen::Vector3d{armor_x, armor_y, armor_z};
}

} // namespace uni_kf
