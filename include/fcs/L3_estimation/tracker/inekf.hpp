#pragma once

#include "inekf_config.hpp"
#include "types.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <groups/SEn3.hpp>
#include <numbers>

namespace fcs::L3 {

/// InEKF (Invariant Extended Kalman Filter) on SE₂(₃) × ℝ⁴ for ground vehicle tracking
///
/// **Mathematical Foundation:**
///
/// State space: X ∈ SE₂(₃) × ℝ⁴
///   - SE₂(₃) = (R, v, p) where R∈SO(3), v∈ℝ³, p∈ℝ³
///   - ℝ⁴ = [v_yaw, r₀, r₁, h]ᵀ (augmented Euclidean states)
///
/// **Error-State Definition (left-invariant, right-multiplicative):**
///   X = X̂ ⊕ e = X̂ · Exp(e)
/// where e = [e_θ, e_v, e_p, e_vyaw, e_r0, e_r1, e_h]ᵀ is the error state.
///
/// **Retraction (yaw-only in production):**
///   R = R̂ · exp(e_θ)
///   v = v̂ + R̂ · J_l(e_θ) · e_v
///   p = p̂ + R̂ · J_l(e_θ) · e_p
///
/// where J_r(y) = I - (1-cos||y||)/||y||² · S(y) + (||y||-sin||y||)/||y||³ · S(y)²
///
/// **Invariant Output Error (for measurement y = R·b + p):**
///   r = X̂⁻¹ · y - b = R̂ᵀ(y - p̂) - b
///
/// **Key Properties:**
///   - Error dynamics are LINEAR in error coordinates: ξ̇ = A·ξ (no higher-order terms)
///   - J_r correction in retraction preserves group structure
///   - Gain matrices converge to constants on permanent trajectories
///   - Better consistency and convergence than standard EKF
///
/// **Filter Equations:**
///   Predict:  X̂_{k|k-1} = φ(X̂_{k-1|k-1})
///   Update:   X̂_{k|k} = X̂_{k|k-1} ⊕ (K_k · E_k)  (RIEKF retraction with J_r)
///
/// **State Constraints:**
///   - Rotation is constrained to yaw-only (z-axis rotation)
///   - Pitch and roll are enforced to zero after each update
///   - Geometry parameters: r₀, r₁ ∈ [0.17, 0.27], h ∈ [-0.2, 0.2]
///
/// **References:**
///   - Wu et al., "An Invariant-EKF VINS Algorithm for Improving Consistency",
///     IROS 2017 (arXiv:1702.07920v2)
///   - Barrau & Bonnabel, "The invariant extended Kalman filter as a stable observer",
///     IEEE TAC, 2016
class InEKF {
public:
    using SE23d                    = group::SEn3<double, 2>;
    using Vec3                     = Eigen::Matrix<double, 3, 1>;
    using Vec9                     = Eigen::Matrix<double, 9, 1>;
    using Mat3                     = Eigen::Matrix<double, 3, 3>;
    using Mat9                     = Eigen::Matrix<double, 9, 9>;
    static constexpr int kGroupDim = 9;
    static constexpr int kExtraDim = 4; // [v_yaw, r0, r1, h]
    static constexpr int kStateDim = kGroupDim + kExtraDim;

    // Error-state layout: [δθ(3), δv(3), δp(3), δv_yaw, δr0, δr1, δh]
    static constexpr int kIdxVYaw = 9;
    static constexpr int kIdxR0   = 10;
    static constexpr int kIdxR1   = 11;
    static constexpr int kIdxH    = 12;

    using VecX   = Eigen::Matrix<double, kStateDim, 1>;
    using MatX   = Eigen::Matrix<double, kStateDim, kStateDim>;
    using Mat3xX = Eigen::Matrix<double, 3, kStateDim>;
    using MatXx3 = Eigen::Matrix<double, kStateDim, 3>;

    InEKF() noexcept
        : X_{}
        , P_{MatX::Zero()}
        , initialized_{false} {}

    /// Initialize filter state
    /// @param p0 Initial target center position in odom frame
    /// @param yaw0 Initial target yaw (about +Z)
    void init(const Vec3& p0, double yaw0, const InEKFConfig& config) noexcept {
        if (!all_finite(p0) || !std::isfinite(yaw0)) {
            X_     = SE23d{};
            P_     = MatX::Zero();
            v_yaw_ = 0.0;
            r0_    = config.radius0;
            r1_    = config.radius1;
            h_     = config.height;
            clamp_geometry_in_place();
            initialized_ = false;
            return;
        }
        // Initialize SE2(3) state from observation
        // Position: from first armor observation
        // Velocity: zero
        // Rotation: from yaw angle (single axis approximation)

        SE23d::IsometriesType t;
        t[0] = Vec3::Zero(); // v = 0
        t[1] = p0;           // p = target center position

                             // Rotation from yaw angle (around z-axis)
        Eigen::Matrix3d R = Eigen::AngleAxis<double>(yaw0, Vec3::UnitZ()).toRotationMatrix();
        X_                = SE23d(R, t);

        // Initialize covariance (error-state)
        P_.setZero();
        P_.block<9, 9>(0, 0)   = Mat9::Identity() * 0.1;
        P_(kIdxVYaw, kIdxVYaw) = 1.0;
        P_(kIdxR0, kIdxR0)     = 0.01;
        P_(kIdxR1, kIdxR1)     = 0.01;
        P_(kIdxH, kIdxH)       = 0.01;

        // Initialize extra states from config
        v_yaw_ = 0.0;
        r0_    = config.radius0;
        r1_    = config.radius1;
        h_     = config.height;
        clamp_geometry_in_place();
        initialized_ = true;
    }

    /// Predict step (CV motion model)
    /// @param dt Time step
    /// @param config Process noise configuration
    void predict(double dt, const InEKFConfig& config) noexcept {
        if (!initialized_) {
            return;
        }
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            return;
        }

        // ---------------------------------------------------------------------
        // Deterministic propagation (constant velocity, constant yaw rate)
        //   yaẇ = v_yaw, v̇ = 0, ṗ = v
        // ---------------------------------------------------------------------
        if (!std::isfinite(v_yaw_)) {
            v_yaw_ = 0.0;
        }
        const double dyaw        = v_yaw_ * dt;
        const double yaw0        = get_yaw();
        const double yaw1        = wrap_pi(yaw0 + (std::isfinite(dyaw) ? dyaw : 0.0));
        const Eigen::Matrix3d Rn = Eigen::AngleAxisd(yaw1, Vec3::UnitZ()).toRotationMatrix();
        SE23d::IsometriesType t;
        t[0] = X_.v();
        t[1] = (X_.p() + X_.v() * dt).eval();
        X_   = SE23d(Rn, t);

        // ---------------------------------------------------------------------
        // Covariance propagation in Lie algebra of the left-invariant error ξ
        // ξ = [δθ, δv, δp], with η = X̂⁻¹ · X
        // For z-axis only: ω = [0, 0, v_yaw]ᵀ
        //   δ̇θ = -ω×δθ + [0,0,δv_yaw]ᵀ
        //   δ̇v = -ω×δv
        //   δ̇p = -ω×δp + δv
        // ---------------------------------------------------------------------
        MatX Phi                = MatX::Identity();
        const double vyaw       = std::isfinite(v_yaw_) ? v_yaw_ : 0.0;
        const double omega_z_dt = vyaw * dt;

        // -ω× coupling for z-axis rotation
        // δ̇θ_x = +ω_z * δθ_y  =>  Φ(0,1) = +ω_z*dt
        // δ̇θ_y = -ω_z * δθ_x  =>  Φ(1,0) = -ω_z*dt
        Phi(0, 1) = omega_z_dt;  // δθ̇_x += ω_z * δθ_y
        Phi(1, 0) = -omega_z_dt; // δθ̇_y -= ω_z * δθ_x

        // Same for velocity
        Phi(3, 4) = omega_z_dt;  // δv̇_x += ω_z * δv_y
        Phi(4, 3) = -omega_z_dt; // δv̇_y -= ω_z * δv_x

        // Same for position (plus velocity coupling below)
        Phi(6, 7) = omega_z_dt;  // δṗ_x += ω_z * δp_y
        Phi(7, 6) = -omega_z_dt; // δṗ_y -= ω_z * δp_x

        // Position-velocity coupling (CV model)
        Phi.block<3, 3>(6, 3) = Mat3::Identity() * dt; // δṗ += δv

        // Yaw-yaw rate coupling
        Phi(2, kIdxVYaw) = dt; // δθ̇_z = δv_yaw

        const MatX Q = process_noise_covariance(dt, config);
        P_           = Phi * P_ * Phi.transpose() + Q;
        symmetrize_in_place(P_);
        sanitize_covariance_in_place(P_);
        clamp_geometry_in_place();
    }

                               /// Update step with armor observation
    /// @param y Armor position in odom frame
    /// @param b Armor offset in target frame (center -> armor, already rotated for armor_id)
    /// @param config Measurement noise configuration
    void update(const Vec3& y, const Vec3& b, const InEKFConfig& config) noexcept {
        if (!initialized_) {
            return;
        }
        if (!all_finite(y) || !all_finite(b)) {
            return;
        }

        // ---------------------------------------------------------------------
        // Invariant output error (paper Sec. II-C):
        //   r = ρ_{X^{-1}}(y) - ρ_{X^{-1}}(h(X)) = X^{-1} * y - b
        // For y = R b + p,  X^{-1} * y = Rᵀ (y - p)
        // ---------------------------------------------------------------------
        const Vec3 y_body = X_.R().transpose() * (y - X_.p());
        const Vec3 r      = y_body - b;

        // Linearized model for left-invariant error (η = X̂⁻¹X, X = X̂·Exp(e)):
        //   r = R̂ᵀ(y - p̂) - b ≈ δθ×b + δp = -[b]×·δθ + δp
        Mat3xX C            = Mat3xX::Zero();
        C.block<3, 3>(0, 0) = -skew(b); // Rotation coupling
        C.block<3, 3>(0, 6) = Mat3::Identity();

        const Mat3 Rm = Mat3::Identity() * (config.sigma_m * config.sigma_m);
        const Mat3 S  = (C * P_ * C.transpose() + Rm).eval();

        Eigen::LDLT<Mat3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            return;
        }

        const Mat3 S_inv = ldlt.solve(Mat3::Identity());
        const MatXx3 K   = (P_ * C.transpose() * S_inv).eval();
        VecX delta       = (K * r).eval();
        delta(0)         = 0.0;
        delta(1)         = 0.0;

        // Retraction: X = X̂ · Exp(e), with yaw-only constraint (e_θx=e_θy=0).
        Vec9 xi                   = Vec9::Zero();
        xi(2)                     = delta(2);
        xi.template segment<3>(3) = delta.template segment<3>(3);
        xi.template segment<3>(6) = delta.template segment<3>(6);
        X_                        = X_ * SE23d::exp(xi);
        v_yaw_ += delta(kIdxVYaw);
        r0_ += delta(kIdxR0);
        r1_ += delta(kIdxR1);
        h_ += delta(kIdxH);
        if (!std::isfinite(v_yaw_)) {
            v_yaw_ = 0.0;
        }
        clamp_geometry_in_place();

        const MatX I_KC = (MatX::Identity() - K * C).eval();
        P_              = I_KC * P_ * I_KC.transpose() + K * Rm * K.transpose();
        symmetrize_in_place(P_);
        sanitize_covariance_in_place(P_);

        // Synchronize covariance with enforced roll/pitch = 0 constraint
        zero_out_roll_pitch_covariance_in_place();
    }

    /// Update step with armor observation (estimates geometry like ISRCKF)
    /// @param y Armor position in odom frame
    /// @param armor_id Armor id in [0,3]
    /// @param config Measurement noise configuration
    void update_armor(const Vec3& y, int armor_id, const InEKFConfig& config) noexcept {
        if (!initialized_) {
            return;
        }
        if (!all_finite(y)) {
            return;
        }
        if (armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }

        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);
        const double yaw_offset     = static_cast<double>(armor_id) * kAngleStep;
        const Eigen::Matrix3d R_id =
            Eigen::AngleAxisd(yaw_offset, Vec3::UnitZ()).toRotationMatrix();

        const bool is_high    = (armor_id == kHighArmorIdA || armor_id == kHighArmorIdB);
        const double radius   = is_high ? r1_ : r0_;
        const double z_offset = is_high ? h_ : 0.0;

        const Vec3 b = (R_id * Vec3{-radius, 0.0, z_offset}).eval();

        const Vec3 y_body = X_.R().transpose() * (y - X_.p());
        const Vec3 r      = y_body - b;

        Mat3xX C            = Mat3xX::Zero();
        C.block<3, 3>(0, 0) = -skew(b); // Rotation coupling
        C.block<3, 3>(0, 6) = Mat3::Identity();

        // Geometry derivatives (treat radius/height as Euclidean error-states):
        // y = p + R·b(r,h)  =>  r = R̂ᵀ(y - p̂) - b̂ ≈ ... + (∂b/∂r)·δr + (∂b/∂h)·δh
        const Vec3 db_dr = (R_id * Vec3{-1.0, 0.0, 0.0}).eval();
        const Vec3 db_dh = (R_id * Vec3{0.0, 0.0, 1.0}).eval(); // base direction
        if (is_high) {
            C.block<3, 1>(0, kIdxR1) = db_dr;                   // ∂r/∂r₁
            C.block<3, 1>(0, kIdxH)  = db_dh;                   // ∂r/∂h
        } else {
            C.block<3, 1>(0, kIdxR0) = db_dr;                   // ∂r/∂r₀
        }

        const Mat3 Rm = Mat3::Identity() * (config.sigma_m * config.sigma_m);
        const Mat3 S  = (C * P_ * C.transpose() + Rm).eval();

        Eigen::LDLT<Mat3> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            return;
        }

        const Mat3 S_inv = ldlt.solve(Mat3::Identity());
        const MatXx3 K   = (P_ * C.transpose() * S_inv).eval();
        VecX delta       = (K * r).eval();
        delta(0)         = 0.0;
        delta(1)         = 0.0;

        // Retraction: X = X̂ · Exp(e), with yaw-only constraint (e_θx=e_θy=0).
        Vec9 xi                   = Vec9::Zero();
        xi(2)                     = delta(2);
        xi.template segment<3>(3) = delta.template segment<3>(3);
        xi.template segment<3>(6) = delta.template segment<3>(6);
        X_                        = X_ * SE23d::exp(xi);

        v_yaw_ += delta(kIdxVYaw);
        r0_ += delta(kIdxR0);
        r1_ += delta(kIdxR1);
        h_ += delta(kIdxH);
        if (!std::isfinite(v_yaw_)) {
            v_yaw_ = 0.0;
        }
        clamp_geometry_in_place();

        const MatX I_KC = (MatX::Identity() - K * C).eval();
        P_              = I_KC * P_ * I_KC.transpose() + K * Rm * K.transpose();
        symmetrize_in_place(P_);
        sanitize_covariance_in_place(P_);

        // Synchronize covariance with enforced roll/pitch = 0 constraint
        zero_out_roll_pitch_covariance_in_place();
    }

    /// Update yaw (about +Z) using direct yaw measurement
    /// @param yaw_meas Measured target yaw in odom frame
    void update_yaw(double yaw_meas, const InEKFConfig& config) noexcept {
        if (!initialized_) {
            return;
        }
        if (!std::isfinite(yaw_meas)) {
            return;
        }
        // Use sigma_w as yaw measurement noise (rad), consistent with existing TOML.
        if (!(config.sigma_w > 0.0) || !std::isfinite(config.sigma_w)) {
            return;
        }

        const double yaw_est = get_yaw();
        const double r       = wrap_pi(yaw_meas - yaw_est);

        Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
        H(2)                                  = 1.0; // yaw ~ δθ_z

        const double R = config.sigma_w * config.sigma_w;
        const double S = (H * P_ * H.transpose())(0, 0) + R;
        if (!(S > 0.0) || !std::isfinite(S)) {
            return;
        }

        const VecX K = (P_ * H.transpose() / S).eval();
        VecX delta   = (K * r).eval();
        delta(0)     = 0.0;
        delta(1)     = 0.0;

        // Retraction: X = X̂ · Exp(e), with yaw-only constraint (e_θx=e_θy=0).
        Vec9 xi                   = Vec9::Zero();
        xi(2)                     = delta(2);
        xi.template segment<3>(3) = delta.template segment<3>(3);
        xi.template segment<3>(6) = delta.template segment<3>(6);
        X_                        = X_ * SE23d::exp(xi);

        v_yaw_ += delta(kIdxVYaw);
        r0_ += delta(kIdxR0);
        r1_ += delta(kIdxR1);
        h_ += delta(kIdxH);
        if (!std::isfinite(v_yaw_)) {
            v_yaw_ = 0.0;
        }
        clamp_geometry_in_place();

        const MatX I_KH = (MatX::Identity() - K * H).eval();
        P_              = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
        symmetrize_in_place(P_);
        sanitize_covariance_in_place(P_);

        // Synchronize covariance with enforced roll/pitch = 0 constraint
        zero_out_roll_pitch_covariance_in_place();
    }

    /// Get current state as RobotTargetState
    [[nodiscard]] RobotTargetState get_robot_state() const noexcept {
        RobotTargetState state;

        // Position
        state.position = X_.p();

        // Velocity
        state.velocity = X_.v();

        // Yaw from rotation matrix (z-axis rotation)
        state.yaw = std::atan2(X_.R()(1, 0), X_.R()(0, 0));

        state.v_yaw = v_yaw_;

        state.radius0    = r0_;
        state.radius1    = r1_;
        state.z1         = state.position.z() + h_;
        state.armors_num = 4;

        return state;
    }

    /// Get position only
    [[nodiscard]] Vec3 get_position() const noexcept { return X_.p(); }

    /// Get velocity only
    [[nodiscard]] Vec3 get_velocity() const noexcept { return X_.v(); }

    /// Get yaw angle
    [[nodiscard]] double get_yaw() const noexcept { return std::atan2(X_.R()(1, 0), X_.R()(0, 0)); }

    [[nodiscard]] double get_v_yaw() const noexcept { return v_yaw_; }
    [[nodiscard]] double get_radius0() const noexcept { return r0_; }
    [[nodiscard]] double get_radius1() const noexcept { return r1_; }
    [[nodiscard]] double get_height() const noexcept { return h_; }

    /// Get covariance matrix (position part)
    [[nodiscard]] Mat3 get_position_covariance() const noexcept { return P_.block<3, 3>(6, 6); }

    /// Check if initialized
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// Get output as FilterOutput for comparison
    [[nodiscard]] FilterOutput get_filter_output() const noexcept {
        FilterOutput output;
        output.active              = initialized_;
        output.position            = X_.p();
        output.velocity            = X_.v();
        output.yaw                 = std::atan2(X_.R()(1, 0), X_.R()(0, 0));
        output.v_yaw               = v_yaw_;
        output.position_covariance = P_.block<3, 3>(6, 6);
        return output;
    }

private:
    SE23d X_; // State estimate (R, v, p)
    MatX P_;  // Covariance in error coordinates
    double v_yaw_{0.0};
    double r0_{0.23};
    double r1_{0.23};
    double h_{0.0};
    bool initialized_{false};

    static constexpr int kArmorsNum    = 4;
    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;
    static constexpr double kMinRadius = 0.17;
    static constexpr double kMaxRadius = 0.27;
    static constexpr double kMinHeight = -0.2;
    static constexpr double kMaxHeight = 0.2;

    /// Skew-symmetric matrix from vector
    static Mat3 skew(const Vec3& v) {
        Mat3 s;
        s << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return s;
    }

    /// Right Jacobian of SO(3) exponential map
    /// Jr(y) = I - (1-cos||y||)/||y||² · S(y) + (||y||-sin||y||)/||y||³ · S(y)²
    /// Used in RIEKF retraction: v_new = exp(e_θ) · v̂ + Jr(-e_θ) · e_v
    static Mat3 Jr(const Vec3& y) noexcept {
        const double norm_y = y.norm();
        if (norm_y < 1e-12) {
            return Mat3::Identity();
        }
        const Mat3 S_y = skew(y);
        const double a = (1.0 - std::cos(norm_y)) / (norm_y * norm_y);
        const double b = (norm_y - std::sin(norm_y)) / (norm_y * norm_y * norm_y);
        return Mat3::Identity() - a * S_y + b * S_y * S_y;
    }

    /// Left Jacobian of SO(3) exponential map
    /// Jl(y) = I + (1-cos||y||)/||y||² · S(y) + (||y||-sin||y||)/||y||³ · S(y)²
    static Mat3 Jl(const Vec3& y) noexcept {
        const double norm_y = y.norm();
        if (norm_y < 1e-12) {
            return Mat3::Identity();
        }
        const Mat3 S_y = skew(y);
        const double a = (1.0 - std::cos(norm_y)) / (norm_y * norm_y);
        const double b = (norm_y - std::sin(norm_y)) / (norm_y * norm_y * norm_y);
        return Mat3::Identity() + a * S_y + b * S_y * S_y;
    }

    [[nodiscard]] static bool all_finite(const Vec3& v) noexcept {
        return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
    }

    [[nodiscard]] static double wrap_pi(double a) noexcept {
        if (!std::isfinite(a)) {
            return a;
        }
        return std::remainder(a, 2.0 * std::numbers::pi);
    }

    void project_rotation_to_yaw_only() noexcept {
        const double yaw = get_yaw();
        SE23d::IsometriesType t;
        t[0]                    = X_.v();
        t[1]                    = X_.p();
        const Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix();
        X_                      = SE23d(R, t);
    }

    void clamp_geometry_in_place() noexcept {
        if (!std::isfinite(r0_)) {
            r0_ = 0.23;
        }
        if (!std::isfinite(r1_)) {
            r1_ = 0.23;
        }
        if (!std::isfinite(h_)) {
            h_ = 0.0;
        }
        r0_ = std::clamp(r0_, kMinRadius, kMaxRadius);
        r1_ = std::clamp(r1_, kMinRadius, kMaxRadius);
        h_  = std::clamp(h_, kMinHeight, kMaxHeight);
    }

    static void symmetrize_in_place(MatX& P) noexcept { P = 0.5 * (P + P.transpose()); }

    static void sanitize_covariance_in_place(MatX& P) noexcept {
        if (!P.allFinite()) {
            P.setZero();
        }
        constexpr double kEps = 1e-12;
        P.diagonal().array() += kEps;
    }

    void zero_out_roll_pitch_covariance_in_place() noexcept {
        // Since delta(0)=delta(1)=0 is enforced (roll/pitch are known to be zero),
        // the corresponding covariance elements should also be zero.
        P_(0, 0) = 0.0; // roll variance
        P_(1, 1) = 0.0; // pitch variance
        // Covariances with roll/pitch
        P_.row(0).setZero();
        P_.row(1).setZero();
        P_.col(0).setZero();
        P_.col(1).setZero();
        // Ensure diagonal is still positive for numerical stability
        P_(0, 0) = 1e-12;
        P_(1, 1) = 1e-12;
    }

    [[nodiscard]] static MatX
        process_noise_covariance(double dt, const InEKFConfig& config) noexcept {
        MatX Q = MatX::Zero();

        if (!(dt > 0.0) || !std::isfinite(dt)) {
            return Q;
        }

        const double sigma_a_xy2  = config.sigma_a_xy * config.sigma_a_xy;
        const double sigma_a_z2   = config.sigma_a_z * config.sigma_a_z;
        const double sigma_a_yaw2 = config.sigma_yaw * config.sigma_yaw;
        const double sigma_r2     = config.sigma_r * config.sigma_r;
        const double sigma_h2     = config.sigma_h * config.sigma_h;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        // CV model noise for [p, v] pairs (match ISRCKF discretization)
        auto fill_cv = [&](int idx_v, int idx_p, double q) {
            Q(idx_v, idx_v) += q * dt2;
            Q(idx_p, idx_v) += q * dt3 / 2.0;
            Q(idx_v, idx_p) += q * dt3 / 2.0;
            Q(idx_p, idx_p) += q * dt4 / 4.0;
        };

        fill_cv(3, 6, sigma_a_xy2); // x
        fill_cv(4, 7, sigma_a_xy2); // y
        fill_cv(5, 8, sigma_a_z2);  // z

        // YAW/VYAW: CV model noise
        Q(2, 2) += sigma_a_yaw2 * dt4 / 4.0;
        Q(2, kIdxVYaw) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kIdxVYaw, 2) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kIdxVYaw, kIdxVYaw) += sigma_a_yaw2 * dt2;

        // Random walk for geometry (radius/height drift)
        Q(kIdxR0, kIdxR0) += sigma_r2 * dt;
        Q(kIdxR1, kIdxR1) += sigma_r2 * dt;
        Q(kIdxH, kIdxH) += sigma_h2 * dt;

        constexpr double kEps = 1e-12;
        Q.diagonal().array() += kEps;
        return Q;
    }
};

} // namespace fcs::L3
