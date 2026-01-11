/// @file tracker_inekf.hpp
/// @brief InEKF wrappers for armor tracker integration
///
/// Provides drop-in replacements for fcs::L3::InEKF that use the uni_kf framework.
/// Uses ProcessModel concept for prediction, custom invariant error for updates.

#pragma once

#include "../core/filter.hpp"
#include "../core/models.hpp"
#include "../spaces/product_space.hpp"

// Include fcs::L3 config and output types directly
#include <fcs/L3_estimation/tracker/inekf_config.hpp>
#include <fcs/L3_estimation/tracker/types.hpp>

#include <Eigen/Core>
#include <cmath>
#include <groups/SEn3.hpp>
#include <numbers>

namespace uni_kf {

// Use fcs::L3 types directly
using RobotInEKFConfig   = fcs::L3::RobotInEKFConfig;
using OutpostInEKFConfig = fcs::L3::OutpostInEKFConfig;
using RobotTargetState   = fcs::L3::RobotTargetState;
using OutpostTargetState = fcs::L3::OutpostTargetState;
using FilterOutput       = fcs::L3::FilterOutput;

// ============================================================================
// Helper functions
// ============================================================================

namespace detail {

[[nodiscard]] inline double wrap_pi(double a) noexcept {
    if (!std::isfinite(a))
        return a;
    return std::remainder(a, 2.0 * std::numbers::pi);
}

[[nodiscard]] inline bool all_finite(const Eigen::Vector3d& v) noexcept {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

[[nodiscard]] inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) noexcept {
    Eigen::Matrix3d s;
    s << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return s;
}

} // namespace detail

// ============================================================================
// RobotProcessModel - Satisfies ProcessModel concept
// ============================================================================

/// @brief Process model for robot tracking on SE₂(3) × ℝ⁴
///
/// State: SE₂(3) × [v_yaw, r0, r1, h]
/// Dynamics: CV for position/velocity, constant yaw-rate, random walk for geometry
class RobotProcessModel {
public:
    static constexpr int kExtraDim = 4;
    static constexpr int kGroupDim = 9;
    static constexpr int kStateDim = kGroupDim + kExtraDim;

    // Extra state indices
    static constexpr int kIdxVYaw = 0;
    static constexpr int kIdxR0   = 1;
    static constexpr int kIdxR1   = 2;
    static constexpr int kIdxH    = 3;

    using Space      = SE23_Rn<kExtraDim>;
    using Point      = typename Space::Point;
    using Covariance = SymmetricMatrix<kStateDim, double>;
    using SE23d      = group::SEn3<double, 2>;
    using Vec3       = Eigen::Vector3d;
    using Mat3       = Eigen::Matrix3d;
    using MatX       = Eigen::Matrix<double, kStateDim, kStateDim>;

private:
    const RobotInEKFConfig& config_;

public:
    explicit RobotProcessModel(const RobotInEKFConfig& config) noexcept
        : config_(config) {}

    /// State transition: CV dynamics with yaw-rate coupling
    [[nodiscard]] Point predict(const Point& x, double dt) const noexcept {
        // Extract yaw rate
        double v_yaw = x.second(kIdxVYaw);
        if (!std::isfinite(v_yaw))
            v_yaw = 0.0;

        // Current yaw
        double yaw0 = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        double yaw1 = detail::wrap_pi(yaw0 + v_yaw * dt);

        // New rotation (yaw-only, roll/pitch = 0)
        Mat3 Rn = Eigen::AngleAxisd(yaw1, Vec3::UnitZ()).toRotationMatrix();

        // Position update: p' = p + v * dt
        typename SE23d::IsometriesType t;
        t[0] = x.first.v();                    // velocity unchanged
        t[1] = x.first.p() + x.first.v() * dt; // position integrates velocity

        // Extra state unchanged
        return {SE23d(Rn, t), x.second};
    }

    /// Process noise covariance
    [[nodiscard]] Covariance process_noise(double dt) const noexcept {
        MatX Q = MatX::Zero();

        const double sigma_a_xy2  = config_.sigma_a_xy * config_.sigma_a_xy;
        const double sigma_a_z2   = config_.sigma_a_z * config_.sigma_a_z;
        const double sigma_a_yaw2 = config_.sigma_yaw * config_.sigma_yaw;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        // CV model noise for [v, p] pairs (indices: v=3:6, p=6:9)
        auto fill_cv = [&](int idx_v, int idx_p, double q) {
            Q(idx_v, idx_v) += q * dt2;
            Q(idx_p, idx_v) += q * dt3 / 2.0;
            Q(idx_v, idx_p) += q * dt3 / 2.0;
            Q(idx_p, idx_p) += q * dt4 / 4.0;
        };

        fill_cv(3, 6, sigma_a_xy2); // x
        fill_cv(4, 7, sigma_a_xy2); // y
        fill_cv(5, 8, sigma_a_z2);  // z

        // Yaw/v_yaw CV noise (indices: yaw=2, v_yaw=9)
        Q(2, 2) += sigma_a_yaw2 * dt4 / 4.0;
        Q(2, kGroupDim + kIdxVYaw) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, 2) += sigma_a_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) += sigma_a_yaw2 * dt2;

        // Geometry random walk
        const double sigma_r2 = config_.sigma_r * config_.sigma_r;
        const double sigma_h2 = config_.sigma_h * config_.sigma_h;
        Q(kGroupDim + kIdxR0, kGroupDim + kIdxR0) += sigma_r2 * dt;
        Q(kGroupDim + kIdxR1, kGroupDim + kIdxR1) += sigma_r2 * dt;
        Q(kGroupDim + kIdxH, kGroupDim + kIdxH) += sigma_h2 * dt;

        // Regularization
        Q.diagonal().array() += 1e-12;
        return Covariance(Q);
    }

    /// Analytical Jacobian (transition matrix Φ)
    [[nodiscard]] MatX jacobian(const Point& x, double dt) const noexcept {
        double v_yaw = x.second(kIdxVYaw);
        if (!std::isfinite(v_yaw))
            v_yaw = 0.0;

        MatX Phi                = MatX::Identity();
        const double omega_z_dt = v_yaw * dt;

        // -ω× coupling for z-axis rotation (SO(3) linearization)
        Phi(0, 1) = omega_z_dt;
        Phi(1, 0) = -omega_z_dt;
        Phi(3, 4) = omega_z_dt;
        Phi(4, 3) = -omega_z_dt;
        Phi(6, 7) = omega_z_dt;
        Phi(7, 6) = -omega_z_dt;

        // Position-velocity coupling
        Phi.template block<3, 3>(6, 3) = Mat3::Identity() * dt;

        // Yaw-yaw rate coupling
        Phi(2, kGroupDim + kIdxVYaw) = dt;

        return Phi;
    }
};

// ============================================================================
// RobotArmorMeasurementModel - Satisfies InvariantMeasurementModel concept
// ============================================================================

/// @brief Invariant measurement model for armor position observations
///
/// Computes body-frame invariant error: r = R^T (y - p) - b
/// where b is the armor offset in body frame.
class RobotArmorMeasurementModel {
public:
    static constexpr int kExtraDim  = RobotProcessModel::kExtraDim;
    static constexpr int kGroupDim  = RobotProcessModel::kGroupDim;
    static constexpr int kStateDim  = RobotProcessModel::kStateDim;
    static constexpr int kArmorsNum = 4;

    // High armor IDs
    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;

    // Extra state indices
    static constexpr int kIdxVYaw = RobotProcessModel::kIdxVYaw;
    static constexpr int kIdxR0   = RobotProcessModel::kIdxR0;
    static constexpr int kIdxR1   = RobotProcessModel::kIdxR1;
    static constexpr int kIdxH    = RobotProcessModel::kIdxH;

    using Space            = SE23_Rn<kExtraDim>;
    using Point            = typename Space::Point;
    using MeasurementSpace = Euclidean<3, double>;
    using MeasPoint        = typename MeasurementSpace::Point;
    using MeasFiber        = typename MeasurementSpace::Fiber;
    using Covariance       = SymmetricMatrix<3, double>;

    using SE23d  = group::SEn3<double, 2>;
    using Vec3   = Eigen::Vector3d;
    using Mat3   = Eigen::Matrix3d;
    using Mat3xX = Eigen::Matrix<double, 3, kStateDim>;

private:
    int armor_id_;
    double sigma_m_;

public:
    RobotArmorMeasurementModel(int armor_id, double sigma_m) noexcept
        : armor_id_(armor_id)
        , sigma_m_(sigma_m) {}

    /// Compute body-frame invariant error: r = R^T (y - p) - b
    [[nodiscard]] MeasFiber invariant_error(const Point& x, const MeasPoint& y) const noexcept {
        // Armor offset in body frame
        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);
        const double yaw_offset     = static_cast<double>(armor_id_) * kAngleStep;
        const Mat3 R_id = Eigen::AngleAxisd(yaw_offset, Vec3::UnitZ()).toRotationMatrix();

        const bool is_high    = (armor_id_ == kHighArmorIdA || armor_id_ == kHighArmorIdB);
        const double radius   = is_high ? x.second(kIdxR1) : x.second(kIdxR0);
        const double z_offset = is_high ? x.second(kIdxH) : 0.0;
        const Vec3 b          = R_id * Vec3{-radius, 0.0, z_offset};

        // Invariant error: r = R^T (y - p) - b
        const Vec3 y_body = x.first.R().transpose() * (y.data - x.first.p());
        const Vec3 r      = y_body - b;

        return MeasFiber{r};
    }

    /// Error Jacobian C (∂r/∂δ)
    [[nodiscard]] Mat3xX error_jacobian(const Point& x) const noexcept {
        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);
        const double yaw_offset     = static_cast<double>(armor_id_) * kAngleStep;
        const Mat3 R_id = Eigen::AngleAxisd(yaw_offset, Vec3::UnitZ()).toRotationMatrix();

        const bool is_high    = (armor_id_ == kHighArmorIdA || armor_id_ == kHighArmorIdB);
        const double radius   = is_high ? x.second(kIdxR1) : x.second(kIdxR0);
        const double z_offset = is_high ? x.second(kIdxH) : 0.0;
        const Vec3 b          = R_id * Vec3{-radius, 0.0, z_offset};

        Mat3xX C = Mat3xX::Zero();

        // Rotation coupling: -[b]×
        C.template block<3, 3>(0, 0) = -detail::skew(b);

        // Position: I
        C.template block<3, 3>(0, 6) = Mat3::Identity();

        // Geometry Jacobian
        const Vec3 db_dr = R_id * Vec3{-1.0, 0.0, 0.0};
        const Vec3 db_dh = R_id * Vec3{0.0, 0.0, 1.0};
        if (is_high) {
            C.template block<3, 1>(0, kGroupDim + kIdxR1) = db_dr;
            C.template block<3, 1>(0, kGroupDim + kIdxH)  = db_dh;
        } else {
            C.template block<3, 1>(0, kGroupDim + kIdxR0) = db_dr;
        }

        return C;
    }

    /// Measurement noise covariance
    [[nodiscard]] Covariance measurement_noise() const noexcept {
        return Covariance::ScaledIdentity(sigma_m_ * sigma_m_);
    }
};

// ============================================================================
// RobotYawMeasurementModel - Satisfies InvariantMeasurementModel concept
// ============================================================================

/// @brief Invariant measurement model for yaw angle observations
///
/// Simple scalar measurement: yaw error = yaw_meas - yaw_est
class RobotYawMeasurementModel {
public:
    static constexpr int kExtraDim = RobotProcessModel::kExtraDim;
    static constexpr int kGroupDim = RobotProcessModel::kGroupDim;
    static constexpr int kStateDim = RobotProcessModel::kStateDim;

    using Space            = SE23_Rn<kExtraDim>;
    using Point            = typename Space::Point;
    using MeasurementSpace = Euclidean<1, double>;
    using MeasPoint        = typename MeasurementSpace::Point;
    using MeasFiber        = typename MeasurementSpace::Fiber;
    using Covariance       = SymmetricMatrix<1, double>;

    using Mat1xX = Eigen::Matrix<double, 1, kStateDim>;

private:
    double sigma_w_;

public:
    explicit RobotYawMeasurementModel(double sigma_w) noexcept
        : sigma_w_(sigma_w) {}

    /// Compute yaw error (wrapped to [-π, π])
    [[nodiscard]] MeasFiber
        invariant_error(const Point& x, const MeasPoint& yaw_meas) const noexcept {
        const double yaw_est = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        const double r       = detail::wrap_pi(yaw_meas.data(0) - yaw_est);
        return MeasFiber{Eigen::Matrix<double, 1, 1>{r}};
    }

    /// Error Jacobian H (yaw is index 2 in the error state)
    [[nodiscard]] Mat1xX error_jacobian(const Point& /*x*/) const noexcept {
        Mat1xX H = Mat1xX::Zero();
        H(0, 2)  = 1.0; // yaw is at index 2 in SO(3) error
        return H;
    }

    /// Measurement noise covariance
    [[nodiscard]] Covariance measurement_noise() const noexcept {
        return Covariance::ScaledIdentity(sigma_w_ * sigma_w_);
    }
};

// ============================================================================
// TrackerInEKF - Robot (4-armor) variant
// ============================================================================

/// @brief InEKF for robot tracking using uni_kf framework
///
/// State: SE₂(3) × ℝ⁴ where extra = [v_yaw, r0, r1, h]
/// Uses RobotProcessModel for prediction (via filter.predict with ProcessModel concept)
/// Uses invariant error for measurement updates (custom implementation)
///
/// This is a drop-in replacement for fcs::L3::InEKF<RobotModel>
class TrackerInEKF_Robot {
public:
    using SE23d = group::SEn3<double, 2>;
    using Vec3  = Eigen::Vector3d;
    using Mat3  = Eigen::Matrix3d;

    static constexpr int kArmorsNum = 4;
    static constexpr int kExtraDim  = RobotProcessModel::kExtraDim;
    static constexpr int kGroupDim  = RobotProcessModel::kGroupDim;
    static constexpr int kStateDim  = RobotProcessModel::kStateDim;

    // Extra state indices (forwarded from model)
    static constexpr int kIdxVYaw = RobotProcessModel::kIdxVYaw;
    static constexpr int kIdxR0   = RobotProcessModel::kIdxR0;
    static constexpr int kIdxR1   = RobotProcessModel::kIdxR1;
    static constexpr int kIdxH    = RobotProcessModel::kIdxH;

    // Geometry constraints
    static constexpr double kMinRadius = 0.17;
    static constexpr double kMaxRadius = 0.27;
    static constexpr double kMinHeight = -0.2;
    static constexpr double kMaxHeight = 0.2;

    // High armor IDs (same as RobotModel)
    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;

private:
    using Space      = SE23_Rn<kExtraDim>;
    using Point      = typename Space::Point;
    using Covariance = SymmetricMatrix<kStateDim, double>;
    using MatX       = Eigen::Matrix<double, kStateDim, kStateDim>;
    using VecX       = Eigen::Matrix<double, kStateDim, 1>;
    using Mat3xX     = Eigen::Matrix<double, 3, kStateDim>;
    using MatXx3     = Eigen::Matrix<double, kStateDim, 3>;

    InCKF_SE23_Augmented<kExtraDim> filter_;

public:
    TrackerInEKF_Robot() noexcept = default;

    /// Initialize filter state
    void init(const Vec3& p0, double yaw0, const RobotInEKFConfig& config) noexcept {
        if (!detail::all_finite(p0) || !std::isfinite(yaw0)) {
            filter_.reset();
            return;
        }

        // Create SE2(3) element
        SE23d::IsometriesType t;
        t[0] = Vec3::Zero(); // velocity
        t[1] = p0;           // position

        Mat3 R = Eigen::AngleAxisd(yaw0, Vec3::UnitZ()).toRotationMatrix();
        SE23d se23(R, t);

        // Extra state: [v_yaw, r0, r1, h]
        VecN<kExtraDim> extra;
        extra.data << 0.0, config.radius0, config.radius1, config.height;
        clamp_extra_in_place(extra.data);

        Point state{se23, extra};

        // Initialize covariance
        Covariance P = Covariance::Zero();
        // SE2(3) part
        for (int i = 0; i < kGroupDim; ++i) {
            P(i, i) = 0.1;
        }
        // v_yaw
        P(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) = 1.0;
        // Geometry
        P(kGroupDim + kIdxR0, kGroupDim + kIdxR0) = 0.01;
        P(kGroupDim + kIdxR1, kGroupDim + kIdxR1) = 0.01;
        P(kGroupDim + kIdxH, kGroupDim + kIdxH)   = 0.01;

        filter_.init(state, P);
    }

    /// Predict step using ProcessModel concept
    void predict(double dt, const RobotInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || dt <= 0.0 || !std::isfinite(dt)) {
            return;
        }

        // Use RobotProcessModel which satisfies ProcessModel concept
        RobotProcessModel model(config);
        filter_.predict(model, dt);

        // Post-processing: clamp geometry and enforce yaw-only rotation
        clamp_extra_in_place(filter_.belief().mean.second.data);
        zero_out_roll_pitch_covariance();
    }

    /// Update with armor observation using InvariantMeasurementModel
    ///
    /// Uses RobotArmorMeasurementModel for invariant error calculation.
    /// Applies roll/pitch constraint to the update.
    void update_armor(const Vec3& y, int armor_id, const RobotInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !detail::all_finite(y)) {
            return;
        }
        if (armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }

        // Create measurement model
        RobotArmorMeasurementModel meas_model(armor_id, config.sigma_m);

        // Use framework's update_invariant with roll/pitch constraint
        update_invariant_yaw_only(RobotArmorMeasurementModel::MeasPoint{y}, meas_model);
    }

    /// Update yaw using InvariantMeasurementModel
    ///
    /// Uses RobotYawMeasurementModel for yaw error calculation.
    void update_yaw(double yaw_meas, const RobotInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !std::isfinite(yaw_meas)) {
            return;
        }
        if (!(config.sigma_w > 0.0) || !std::isfinite(config.sigma_w)) {
            return;
        }

        // Create measurement model
        RobotYawMeasurementModel meas_model(config.sigma_w);

        // Use framework's update_invariant with roll/pitch constraint
        update_invariant_yaw_only(
            RobotYawMeasurementModel::MeasPoint{Eigen::Matrix<double, 1, 1>{yaw_meas}}, meas_model);
    }

    // ------------------------------------------------------------------------
    // State accessors
    // ------------------------------------------------------------------------

    [[nodiscard]] RobotTargetState get_robot_state() const noexcept {
        RobotTargetState state_out;
        if (!filter_.is_initialized()) {
            return state_out;
        }

        const auto& state    = filter_.mean();
        state_out.position   = state.first.p();
        state_out.velocity   = state.first.v();
        state_out.yaw        = std::atan2(state.first.R()(1, 0), state.first.R()(0, 0));
        state_out.v_yaw      = state.second(kIdxVYaw);
        state_out.radius0    = state.second(kIdxR0);
        state_out.radius1    = state.second(kIdxR1);
        state_out.z1         = state_out.position.z() + state.second(kIdxH);
        state_out.armors_num = kArmorsNum;
        return state_out;
    }

    [[nodiscard]] Vec3 get_position() const noexcept {
        return filter_.is_initialized() ? filter_.mean().first.p() : Vec3::Zero();
    }

    [[nodiscard]] Vec3 get_velocity() const noexcept {
        return filter_.is_initialized() ? filter_.mean().first.v() : Vec3::Zero();
    }

    [[nodiscard]] double get_yaw() const noexcept {
        if (!filter_.is_initialized())
            return 0.0;
        const auto& R = filter_.mean().first.R();
        return std::atan2(R(1, 0), R(0, 0));
    }

    [[nodiscard]] double get_v_yaw() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(kIdxVYaw) : 0.0;
    }

    [[nodiscard]] double get_radius0() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(kIdxR0) : 0.22;
    }

    [[nodiscard]] double get_radius1() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(kIdxR1) : 0.22;
    }

    [[nodiscard]] double get_height() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(kIdxH) : 0.0;
    }

    [[nodiscard]] Mat3 get_position_covariance() const noexcept {
        if (!filter_.is_initialized())
            return Mat3::Zero();
        return filter_.covariance().matrix().template block<3, 3>(6, 6);
    }

    [[nodiscard]] bool is_initialized() const noexcept { return filter_.is_initialized(); }

    [[nodiscard]] FilterOutput get_filter_output() const noexcept {
        FilterOutput output;
        output.active = filter_.is_initialized();
        if (output.active) {
            const auto& state          = filter_.mean();
            output.position            = state.first.p();
            output.velocity            = state.first.v();
            output.yaw                 = get_yaw();
            output.v_yaw               = state.second(kIdxVYaw);
            output.position_covariance = get_position_covariance();
        }
        return output;
    }

private:
    /// Update using InvariantMeasurementModel with roll/pitch constraint
    ///
    /// This is the core update implementation that:
    /// 1. Gets invariant error and Jacobian from the model
    /// 2. Computes Kalman gain
    /// 3. Zeros out roll/pitch in the update delta
    /// 4. Applies the update with yaw-only rotation constraint
    template <typename Model>
    requires InvariantMeasurementModel<Model, Space> void update_invariant_yaw_only(
        const typename Model::MeasurementSpace::Point& z, const Model& model) noexcept {

        using M               = typename Model::MeasurementSpace;
        constexpr int MeasDim = M::FiberDim;

        auto& belief = filter_.belief();
        auto& state  = belief.mean;
        auto& P      = belief.covariance;

        // Get invariant error from model
        auto r = model.invariant_error(state, z);

        // Get error Jacobian from model
        auto C = model.error_jacobian(state);

        // Get measurement noise from model
        auto R = model.measurement_noise();

        // Innovation covariance: S = C P C^T + R
        Eigen::Matrix<double, MeasDim, MeasDim> S_mat = C * P.matrix() * C.transpose() + R.matrix();

        // Solve for Kalman gain
        Eigen::LDLT<Eigen::Matrix<double, MeasDim, MeasDim>> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            return;
        }

        Eigen::Matrix<double, kStateDim, MeasDim> K =
            P.matrix() * C.transpose()
            * ldlt.solve(Eigen::Matrix<double, MeasDim, MeasDim>::Identity());

        // Compute update delta
        VecX delta = K * r.data;

        // Zero out roll/pitch updates (yaw-only rotation constraint)
        delta(0) = 0.0;
        delta(1) = 0.0;

        // Apply update with yaw-only retraction
        apply_update(delta);

        // Covariance update (Joseph form)
        MatX I_KC  = MatX::Identity() - K * C;
        auto P_new = I_KC * P.matrix() * I_KC.transpose() + K * R.matrix() * K.transpose();
        P          = Covariance(P_new);

        // Post-processing
        belief.symmetrize();
        belief.regularize();
        clamp_extra_in_place(state.second.data);
        zero_out_roll_pitch_covariance();
    }

    static void clamp_extra_in_place(Eigen::Vector4d& extra) noexcept {
        if (!std::isfinite(extra(kIdxVYaw)))
            extra(kIdxVYaw) = 0.0;
        if (!std::isfinite(extra(kIdxR0)))
            extra(kIdxR0) = 0.23;
        if (!std::isfinite(extra(kIdxR1)))
            extra(kIdxR1) = 0.23;
        if (!std::isfinite(extra(kIdxH)))
            extra(kIdxH) = 0.0;
        extra(kIdxR0) = std::clamp(extra(kIdxR0), kMinRadius, kMaxRadius);
        extra(kIdxR1) = std::clamp(extra(kIdxR1), kMinRadius, kMaxRadius);
        extra(kIdxH)  = std::clamp(extra(kIdxH), kMinHeight, kMaxHeight);
    }

    void apply_update(const VecX& delta) noexcept {
        auto& state = filter_.belief().mean;

        // SE2(3) retraction with yaw-only constraint
        Eigen::Matrix<double, 9, 1> xi = Eigen::Matrix<double, 9, 1>::Zero();
        xi(2)                          = delta(2);                     // yaw only
        xi.template segment<3>(3)      = delta.template segment<3>(3); // velocity
        xi.template segment<3>(6)      = delta.template segment<3>(6); // position
        state.first                    = state.first * SE23d::exp(xi);

        // Extra state update
        state.second.data += delta.template segment<kExtraDim>(kGroupDim);
        if (!std::isfinite(state.second(kIdxVYaw))) {
            state.second(kIdxVYaw) = 0.0;
        }
    }

    void zero_out_roll_pitch_covariance() noexcept {
        auto& P    = filter_.belief().covariance;
        auto P_mat = P.matrix();

        P_mat.row(0).setZero();
        P_mat.row(1).setZero();
        P_mat.col(0).setZero();
        P_mat.col(1).setZero();
        P_mat(0, 0) = 1e-12;
        P_mat(1, 1) = 1e-12;

        P = Covariance(P_mat);
    }
};

// ============================================================================
// TrackerInEKF - Outpost (3-armor) variant
// ============================================================================

/// @brief InEKF for outpost tracking using uni_kf framework
///
/// State: SE₂(3) × ℝ³ where extra = [v_yaw, dz1, dz2]
/// This is a drop-in replacement for fcs::L3::InEKF<OutpostModel>
class TrackerInEKF_Outpost {
public:
    using SE23d = group::SEn3<double, 2>;
    using Vec3  = Eigen::Vector3d;
    using Mat3  = Eigen::Matrix3d;

    static constexpr int kArmorsNum = 3;
    static constexpr int kExtraDim  = 3;
    static constexpr int kGroupDim  = 9;
    static constexpr int kStateDim  = kGroupDim + kExtraDim;

    // Extra state indices
    static constexpr int kIdxVYaw = 0;
    static constexpr int kIdxDz1  = 1;
    static constexpr int kIdxDz2  = 2;

    static constexpr double kOutpostRadius = 0.2765;

private:
    using Space      = SE23_Rn<kExtraDim>;
    using Point      = typename Space::Point;
    using Covariance = SymmetricMatrix<kStateDim, double>;
    using MatX       = Eigen::Matrix<double, kStateDim, kStateDim>;
    using VecX       = Eigen::Matrix<double, kStateDim, 1>;
    using Mat3xX     = Eigen::Matrix<double, 3, kStateDim>;
    using MatXx3     = Eigen::Matrix<double, kStateDim, 3>;

    InCKF_SE23_Augmented<kExtraDim> filter_;

public:
    TrackerInEKF_Outpost() noexcept = default;

    void init(const Vec3& p0, double yaw0, const OutpostInEKFConfig& config) noexcept {
        if (!detail::all_finite(p0) || !std::isfinite(yaw0)) {
            filter_.reset();
            return;
        }

        SE23d::IsometriesType t;
        t[0] = Vec3::Zero();
        t[1] = p0;

        Mat3 R = Eigen::AngleAxisd(yaw0, Vec3::UnitZ()).toRotationMatrix();
        SE23d se23(R, t);

        VecN<kExtraDim> extra;
        extra.data << 0.0, 0.0, 0.0;

        Point state{se23, extra};

        Covariance P = Covariance::Zero();
        for (int i = 0; i < kGroupDim; ++i) {
            P(i, i) = 0.1;
        }
        P(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) = 1.0;
        P(kGroupDim + kIdxDz1, kGroupDim + kIdxDz1)   = 0.01;
        P(kGroupDim + kIdxDz2, kGroupDim + kIdxDz2)   = 0.01;

        filter_.init(state, P);
    }

    void predict(double dt, const OutpostInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || dt <= 0.0 || !std::isfinite(dt)) {
            return;
        }

        auto& belief = filter_.belief();
        auto& state  = belief.mean;

        double v_yaw = state.second(kIdxVYaw);
        if (!std::isfinite(v_yaw))
            v_yaw = 0.0;

        const double yaw0 = get_yaw();
        const double yaw1 = detail::wrap_pi(yaw0 + v_yaw * dt);
        const Mat3 Rn     = Eigen::AngleAxisd(yaw1, Vec3::UnitZ()).toRotationMatrix();

        SE23d::IsometriesType t;
        t[0]        = state.first.v();
        t[1]        = state.first.p() + state.first.v() * dt;
        state.first = SE23d(Rn, t);

        MatX Phi                = MatX::Identity();
        const double omega_z_dt = v_yaw * dt;

        Phi(0, 1)                      = omega_z_dt;
        Phi(1, 0)                      = -omega_z_dt;
        Phi(3, 4)                      = omega_z_dt;
        Phi(4, 3)                      = -omega_z_dt;
        Phi(6, 7)                      = omega_z_dt;
        Phi(7, 6)                      = -omega_z_dt;
        Phi.template block<3, 3>(6, 3) = Mat3::Identity() * dt;
        Phi(2, kGroupDim + kIdxVYaw)   = dt;

        MatX Q = build_process_noise(dt, config);

        auto& P    = belief.covariance;
        auto P_mat = Phi * P.matrix() * Phi.transpose() + Q;
        P          = Covariance(P_mat);

        belief.symmetrize();
        belief.regularize();
        clamp_extra_in_place(state.second.data);
        zero_out_roll_pitch_covariance();
    }

    void update_armor(const Vec3& y, int armor_id, const OutpostInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !detail::all_finite(y))
            return;
        if (armor_id < 0 || armor_id >= kArmorsNum)
            return;

        auto& belief = filter_.belief();
        auto& state  = belief.mean;
        auto& P      = belief.covariance;

        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);
        const double yaw_offset     = static_cast<double>(armor_id) * kAngleStep;
        const Mat3 R_id = Eigen::AngleAxisd(yaw_offset, Vec3::UnitZ()).toRotationMatrix();

        double z_offset = 0.0;
        if (armor_id == 1)
            z_offset = state.second(kIdxDz1);
        else if (armor_id == 2)
            z_offset = state.second(kIdxDz2);

        const Vec3 b      = R_id * Vec3{-kOutpostRadius, 0.0, z_offset};
        const Vec3 y_body = state.first.R().transpose() * (y - state.first.p());
        const Vec3 r      = y_body - b;

        Mat3xX C                     = Mat3xX::Zero();
        C.template block<3, 3>(0, 0) = -detail::skew(b);
        C.template block<3, 3>(0, 6) = Mat3::Identity();

        const Vec3 db_dz = R_id * Vec3{0.0, 0.0, 1.0};
        if (armor_id == 1) {
            C.template block<3, 1>(0, kGroupDim + kIdxDz1) = db_dz;
        } else if (armor_id == 2) {
            C.template block<3, 1>(0, kGroupDim + kIdxDz2) = db_dz;
        }

        const Mat3 Rm = Mat3::Identity() * (config.sigma_m * config.sigma_m);
        const Mat3 S  = C * P.matrix() * C.transpose() + Rm;

        Eigen::LDLT<Mat3> ldlt(S);
        if (ldlt.info() != Eigen::Success)
            return;

        const Mat3 S_inv = ldlt.solve(Mat3::Identity());
        const MatXx3 K   = P.matrix() * C.transpose() * S_inv;
        VecX delta       = K * r;
        delta(0)         = 0.0;
        delta(1)         = 0.0;

        apply_update(delta);

        const MatX I_KC = MatX::Identity() - K * C;
        auto P_new      = I_KC * P.matrix() * I_KC.transpose() + K * Rm * K.transpose();
        P               = Covariance(P_new);

        belief.symmetrize();
        belief.regularize();
        clamp_extra_in_place(state.second.data);
        zero_out_roll_pitch_covariance();
    }

    void update_yaw(double yaw_meas, const OutpostInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !std::isfinite(yaw_meas))
            return;
        if (!(config.sigma_w > 0.0) || !std::isfinite(config.sigma_w))
            return;

        auto& belief = filter_.belief();
        auto& state  = belief.mean;
        auto& P      = belief.covariance;

        const double yaw_est = get_yaw();
        const double r       = detail::wrap_pi(yaw_meas - yaw_est);

        Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
        H(2)                                  = 1.0;

        const double R = config.sigma_w * config.sigma_w;
        const double S = (H * P.matrix() * H.transpose())(0, 0) + R;
        if (!(S > 0.0) || !std::isfinite(S))
            return;

        const VecX K = P.matrix() * H.transpose() / S;
        VecX delta   = K * r;
        delta(0)     = 0.0;
        delta(1)     = 0.0;

        apply_update(delta);

        const MatX I_KH = MatX::Identity() - K * H;
        auto P_new      = I_KH * P.matrix() * I_KH.transpose() + K * R * K.transpose();
        P               = Covariance(P_new);

        belief.symmetrize();
        belief.regularize();
        clamp_extra_in_place(state.second.data);
        zero_out_roll_pitch_covariance();
    }

    [[nodiscard]] OutpostTargetState get_outpost_state() const noexcept {
        OutpostTargetState state_out;
        if (!filter_.is_initialized())
            return state_out;

        const auto& state  = filter_.mean();
        state_out.position = {state.first.p().x(), state.first.p().y()};
        state_out.yaw      = std::atan2(state.first.R()(1, 0), state.first.R()(0, 0));
        state_out.v_yaw    = state.second(kIdxVYaw);
        state_out.z        = {
            state.first.p().z(), state.first.p().z() + state.second(kIdxDz1),
            state.first.p().z() + state.second(kIdxDz2)};
        return state_out;
    }

    [[nodiscard]] Vec3 get_position() const noexcept {
        return filter_.is_initialized() ? filter_.mean().first.p() : Vec3::Zero();
    }

    [[nodiscard]] double get_yaw() const noexcept {
        if (!filter_.is_initialized())
            return 0.0;
        const auto& R = filter_.mean().first.R();
        return std::atan2(R(1, 0), R(0, 0));
    }

    [[nodiscard]] bool is_initialized() const noexcept { return filter_.is_initialized(); }

    [[nodiscard]] FilterOutput get_filter_output() const noexcept {
        FilterOutput output;
        output.active = filter_.is_initialized();
        if (output.active) {
            const auto& state          = filter_.mean();
            output.position            = state.first.p();
            output.velocity            = state.first.v();
            output.yaw                 = get_yaw();
            output.v_yaw               = state.second(kIdxVYaw);
            output.position_covariance = filter_.covariance().matrix().template block<3, 3>(6, 6);
        }
        return output;
    }

private:
    static void clamp_extra_in_place(Eigen::Vector3d& extra) noexcept {
        if (!std::isfinite(extra(kIdxVYaw)))
            extra(kIdxVYaw) = 0.0;
        if (!std::isfinite(extra(kIdxDz1)))
            extra(kIdxDz1) = 0.0;
        if (!std::isfinite(extra(kIdxDz2)))
            extra(kIdxDz2) = 0.0;
    }

    void apply_update(const VecX& delta) noexcept {
        auto& state = filter_.belief().mean;

        Eigen::Matrix<double, 9, 1> xi = Eigen::Matrix<double, 9, 1>::Zero();
        xi(2)                          = delta(2);
        xi.template segment<3>(3)      = delta.template segment<3>(3);
        xi.template segment<3>(6)      = delta.template segment<3>(6);
        state.first                    = state.first * SE23d::exp(xi);

        state.second.data += delta.template segment<kExtraDim>(kGroupDim);
        if (!std::isfinite(state.second(kIdxVYaw))) {
            state.second(kIdxVYaw) = 0.0;
        }
    }

    void zero_out_roll_pitch_covariance() noexcept {
        auto& P    = filter_.belief().covariance;
        auto P_mat = P.matrix();

        P_mat.row(0).setZero();
        P_mat.row(1).setZero();
        P_mat.col(0).setZero();
        P_mat.col(1).setZero();
        P_mat(0, 0) = 1e-12;
        P_mat(1, 1) = 1e-12;

        P = Covariance(P_mat);
    }

    [[nodiscard]] static MatX
        build_process_noise(double dt, const OutpostInEKFConfig& config) noexcept {
        MatX Q = MatX::Zero();

        const double sigma_xy2  = config.sigma_a_xy * config.sigma_a_xy;
        const double sigma_z2   = config.sigma_a_z * config.sigma_a_z;
        const double sigma_yaw2 = config.sigma_yaw * config.sigma_yaw;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        // XY drift
        Q(6, 6) = sigma_xy2 * dt2;
        Q(7, 7) = sigma_xy2 * dt2;

        // Z drift (applied to base position)
        Q(8, 8) = sigma_z2 * dt2;

        // Yaw/v_yaw CV noise
        Q(2, 2) += sigma_yaw2 * dt4 / 4.0;
        Q(2, kGroupDim + kIdxVYaw) += sigma_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, 2) += sigma_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) += sigma_yaw2 * dt2;

        // dz1/dz2 drift
        const double sigma_h2 = config.sigma_h * config.sigma_h;
        Q(kGroupDim + kIdxDz1, kGroupDim + kIdxDz1) += sigma_h2 * dt;
        Q(kGroupDim + kIdxDz2, kGroupDim + kIdxDz2) += sigma_h2 * dt;

        Q.diagonal().array() += 1e-12;
        return Q;
    }
};

} // namespace uni_kf
