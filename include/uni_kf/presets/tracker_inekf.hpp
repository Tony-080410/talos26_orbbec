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

// Constants for regularization and constraints
inline constexpr double kCovarianceRegularization = 1e-12;

// ============================================================================
// Anisotropic Measurement Noise (ported from motion_model.hpp insights)
// ============================================================================

/// @brief Compute dynamic measurement noise covariance in body frame
///
/// === Problem ===
/// Isotropic noise R = σ²I ignores the anisotropic nature of visual measurements:
/// - Depth (radial) uncertainty dominates when viewing armor from the side
/// - Angular (tangential) uncertainty grows with distance due to PnP instability
///
/// === Intuition ===
/// Visual measurement uncertainty has a "cone" shape along the sight line:
/// - Radial (depth): unreliable when armor surface is nearly parallel to sight line
/// - Tangential (angular): unreliable when target is far (small pixels → large position error)
///
/// === Analogy ===
/// Like estimating the distance to a wall by looking at it:
/// - Looking straight at the wall: depth is clear from texture/stereo
/// - Looking at grazing angle: depth becomes ambiguous
///
/// === Precise (from motion_model.hpp) ===
/// Constructs anisotropic covariance R in body frame:
///
///   R = U × Λ × U^T
///
/// where U = [e_r, e_h, e_v] (radial/tangential basis in body frame)
///       Λ = diag(σ_r², σ_t², σ_t²)
///
/// Radial variance (depth uncertainty):
///   σ_r² = σ_base² × (1 + k_r × log(sin(view_angle) + 1))
///   - view_angle: angle between sight line and armor normal
///   - k_r: radial sensitivity coefficient
///
/// Tangential variance (angular uncertainty):
///   σ_t² = σ_base² × (1 + k_t × log(distance + 1))
///   - k_t: tangential sensitivity coefficient
///
/// @param y_body Observation in body frame: R^T(y - p)
/// @param armor_normal_body Armor plate outward normal in body frame (unit vector)
/// @param sigma_base Base measurement noise std-dev (m)
/// @param k_radial Radial noise sensitivity to view angle (default 1.0)
/// @param k_tangent Tangential noise sensitivity to distance (default 0.1)
/// @return 3×3 SPD covariance matrix in body frame
[[nodiscard]] inline Eigen::Matrix3d compute_anisotropic_R_body(
    const Eigen::Vector3d& y_body, const Eigen::Vector3d& armor_normal_body, double sigma_base,
    double k_radial = 1.0, double k_tangent = 0.1) noexcept {
    const double sigma_base2 = sigma_base * sigma_base;
    const double distance    = y_body.norm();

    // Degenerate case: very close observation
    if (distance < 1e-6) {
        return Eigen::Matrix3d::Identity() * sigma_base2;
    }

    // === Step 1: Compute sight-line direction in body frame ===
    const Eigen::Vector3d e_r = y_body / distance; // radial (sight line)

    // === Step 2: Compute view angle (angle between sight line and armor normal) ===
    // armor_normal points outward from robot center
    // When viewing head-on: e_r ≈ -armor_normal → cos ≈ -1 → |cos| ≈ 1
    // When viewing from side: e_r ⊥ armor_normal → cos ≈ 0
    const double cos_view = std::abs(armor_normal_body.dot(e_r));
    const double sin_view = std::sqrt(1.0 - cos_view * cos_view);

    // === Step 3: Dynamic variance scaling (configurable) ===
    // Radial: depth uncertainty increases when viewing from side
    const double radial_scale = 1.0 + k_radial * std::log(sin_view + 1.0);
    const double sigma_r2     = sigma_base2 * radial_scale;

    // Tangential: angular uncertainty increases with distance
    const double tangent_scale = 1.0 + k_tangent * std::log(distance + 1.0);
    const double sigma_t2      = sigma_base2 * tangent_scale;

    // === Step 4: Build orthonormal basis {e_r, e_h, e_v} in body frame ===
    // e_h: horizontal tangent (perpendicular to e_r, in horizontal plane)
    Eigen::Vector3d e_h = Eigen::Vector3d::UnitZ().cross(e_r);
    double e_h_norm     = e_h.norm();
    if (e_h_norm < 1e-6) {
        // Looking straight up/down: use x-axis as fallback
        e_h = Eigen::Vector3d::UnitX();
    } else {
        e_h /= e_h_norm;
    }

    // e_v: vertical tangent (perpendicular to both e_r and e_h)
    const Eigen::Vector3d e_v = e_r.cross(e_h);

    // === Step 5: Assemble anisotropic covariance ===
    // R = U × diag(σ_r², σ_t², σ_t²) × U^T
    Eigen::Matrix3d U;
    U.col(0) = e_r;
    U.col(1) = e_h;
    U.col(2) = e_v;

    const Eigen::Vector3d sigma_diag(sigma_r2, sigma_t2, sigma_t2);
    return U * sigma_diag.asDiagonal() * U.transpose();
}

/// @brief Compute armor normal direction in body frame
/// @param armor_id Armor index (0, 1, 2, 3 for robot; 0, 1, 2 for outpost)
/// @param armors_num Total number of armors
/// @return Unit normal vector pointing outward from robot center
[[nodiscard]] inline Eigen::Vector3d armor_normal_body(int armor_id, int armors_num) noexcept {
    const double angle_step  = 2.0 * std::numbers::pi / static_cast<double>(armors_num);
    const double armor_angle = static_cast<double>(armor_id) * angle_step;
    // Normal points outward: opposite to the direction from center to armor
    return Eigen::Vector3d{-std::cos(armor_angle), -std::sin(armor_angle), 0.0};
}

[[nodiscard]] inline double wrap_pi(double a) noexcept {
    if (!std::isfinite(a)) {
        return a;
    }
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

[[nodiscard]] inline double sanitize_finite(double value, double fallback) noexcept {
    return std::isfinite(value) ? value : fallback;
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
        const double v_yaw = detail::sanitize_finite(x.second(kIdxVYaw), 0.0);

        const double yaw0 = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        const double yaw1 = detail::wrap_pi(yaw0 + v_yaw * dt);

        const Mat3 Rn = Eigen::AngleAxisd(yaw1, Vec3::UnitZ()).toRotationMatrix();

        typename SE23d::IsometriesType t;
        t[0] = x.first.v();
        t[1] = x.first.p() + x.first.v() * dt;

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

        Q.diagonal().array() += detail::kCovarianceRegularization;
        return Covariance(Q);
    }

    /// Analytical Jacobian (transition matrix Φ)
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
};

// ============================================================================
// OutpostProcessModel - Satisfies ProcessModel concept
// ============================================================================

/// @brief Process model for outpost tracking on SE₂(3) × ℝ³
///
/// State: SE₂(3) × [v_yaw, dz1, dz2]
/// Dynamics: CV for position/velocity, constant yaw-rate, random walk for z-offsets
class OutpostProcessModel {
public:
    static constexpr int kExtraDim = 3;
    static constexpr int kGroupDim = 9;
    static constexpr int kStateDim = kGroupDim + kExtraDim;

    // Extra state indices
    static constexpr int kIdxVYaw = 0;
    static constexpr int kIdxDz1  = 1;
    static constexpr int kIdxDz2  = 2;

    using Space      = SE23_Rn<kExtraDim>;
    using Point      = typename Space::Point;
    using Covariance = SymmetricMatrix<kStateDim, double>;
    using SE23d      = group::SEn3<double, 2>;
    using Vec3       = Eigen::Vector3d;
    using Mat3       = Eigen::Matrix3d;
    using MatX       = Eigen::Matrix<double, kStateDim, kStateDim>;

private:
    const OutpostInEKFConfig& config_;

public:
    explicit OutpostProcessModel(const OutpostInEKFConfig& config) noexcept
        : config_(config) {}

    /// State transition: CV dynamics with yaw-rate coupling
    [[nodiscard]] Point predict(const Point& x, double dt) const noexcept {
        const double v_yaw = detail::sanitize_finite(x.second(kIdxVYaw), 0.0);

        const double yaw0 = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        const double yaw1 = detail::wrap_pi(yaw0 + v_yaw * dt);

        const Mat3 Rn = Eigen::AngleAxisd(yaw1, Vec3::UnitZ()).toRotationMatrix();

        typename SE23d::IsometriesType t;
        t[0] = x.first.v();
        t[1] = x.first.p() + x.first.v() * dt;

        return {SE23d(Rn, t), x.second};
    }

    /// Process noise covariance
    [[nodiscard]] Covariance process_noise(double dt) const noexcept {
        MatX Q = MatX::Zero();

        const double sigma_xy2  = config_.sigma_a_xy * config_.sigma_a_xy;
        const double sigma_z2   = config_.sigma_a_z * config_.sigma_a_z;
        const double sigma_yaw2 = config_.sigma_yaw * config_.sigma_yaw;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        // CV model noise for [v, p] pairs (same as RobotProcessModel)
        auto fill_cv = [&](int idx_v, int idx_p, double q) {
            Q(idx_v, idx_v) += q * dt2;
            Q(idx_p, idx_v) += q * dt3 / 2.0;
            Q(idx_v, idx_p) += q * dt3 / 2.0;
            Q(idx_p, idx_p) += q * dt4 / 4.0;
        };

        fill_cv(3, 6, sigma_xy2); // x
        fill_cv(4, 7, sigma_xy2); // y
        fill_cv(5, 8, sigma_z2);  // z

        // Yaw/v_yaw CV noise
        Q(2, 2) += sigma_yaw2 * dt4 / 4.0;
        Q(2, kGroupDim + kIdxVYaw) += sigma_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, 2) += sigma_yaw2 * dt3 / 2.0;
        Q(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) += sigma_yaw2 * dt2;

        // dz1/dz2 drift
        const double sigma_h2 = config_.sigma_h * config_.sigma_h;
        Q(kGroupDim + kIdxDz1, kGroupDim + kIdxDz1) += sigma_h2 * dt;
        Q(kGroupDim + kIdxDz2, kGroupDim + kIdxDz2) += sigma_h2 * dt;

        Q.diagonal().array() += detail::kCovarianceRegularization;
        return Covariance(Q);
    }

    /// Analytical Jacobian (transition matrix Φ)
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
};

// ============================================================================
// Shared Yaw Measurement Model
// ============================================================================

/// @brief Invariant measurement model for yaw angle observations
///
/// Simple scalar measurement: yaw error = yaw_meas - yaw_est
/// Template parameter allows usage with both Robot and Outpost state dimensions
template <int ExtraDim>
class YawMeasurementModel {
public:
    static constexpr int kGroupDim = 9;
    static constexpr int kStateDim = kGroupDim + ExtraDim;

    using Space            = SE23_Rn<ExtraDim>;
    using Point            = typename Space::Point;
    using MeasurementSpace = Euclidean<1, double>;
    using MeasPoint        = typename MeasurementSpace::Point;
    using MeasFiber        = typename MeasurementSpace::Fiber;
    using Covariance       = SymmetricMatrix<1, double>;

    using Mat1xX = Eigen::Matrix<double, 1, kStateDim>;

private:
    double sigma_w_;

public:
    explicit YawMeasurementModel(double sigma_w) noexcept
        : sigma_w_(sigma_w) {}

    /// Compute yaw error (wrapped to [-π, π])
    [[nodiscard]] static MeasFiber
        invariant_error(const Point& x, const MeasPoint& yaw_meas) noexcept {
        const double yaw_est = std::atan2(x.first.R()(1, 0), x.first.R()(0, 0));
        const double r       = detail::wrap_pi(yaw_meas.data(0) - yaw_est);
        return MeasFiber{Eigen::Matrix<double, 1, 1>{r}};
    }

    /// Error Jacobian H (yaw is index 2 in the error state)
    [[nodiscard]] static Mat1xX error_jacobian(const Point& /*x*/) noexcept {
        Mat1xX H = Mat1xX::Zero();
        H(0, 2)  = 1.0;
        return H;
    }

    /// Measurement noise covariance
    [[nodiscard]] Covariance measurement_noise() const noexcept {
        return Covariance::ScaledIdentity(sigma_w_ * sigma_w_);
    }
};

// ============================================================================
// RobotArmorMeasurementModel - Satisfies InvariantMeasurementModel concept
// ============================================================================

/// @brief Invariant measurement model for armor position observations
///
/// Computes body-frame invariant error: r = R^T (y - p) - b
/// where b is the armor offset in body frame.
///
/// === Noise Modes ===
/// - Static (default): R = σ²I - traditional InEKF
/// - Dynamic: R = f(geometry) - anisotropic noise adapted from motion_model.hpp
///
/// === Dynamic Noise Rationale (from motion_model.hpp) ===
/// Visual measurements have direction-dependent uncertainty:
/// - Depth (radial): unreliable when viewing armor from the side
/// - Angular (tangential): unreliable at long range due to PnP instability
///
/// The dynamic mode preserves InEKF's invariant error structure while
/// incorporating geometry-aware noise modeling.
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
    bool use_dynamic_noise_;
    double k_radial_;
    double k_tangent_;
    mutable Vec3 cached_y_body_; // Cached from invariant_error for noise computation

public:
    /// @brief Construct with static isotropic noise (default, backward compatible)
    RobotArmorMeasurementModel(int armor_id, double sigma_m) noexcept
        : armor_id_(armor_id)
        , sigma_m_(sigma_m)
        , use_dynamic_noise_(false)
        , k_radial_(1.0)
        , k_tangent_(0.1)
        , cached_y_body_(Vec3::Zero()) {}

    /// @brief Construct with configurable dynamic anisotropic noise
    /// @param armor_id Armor index (0-3)
    /// @param sigma_m Base measurement noise std-dev (m)
    /// @param use_dynamic Enable dynamic noise based on viewing geometry
    /// @param k_radial Radial noise sensitivity (default 1.0)
    /// @param k_tangent Tangential noise sensitivity (default 0.1)
    RobotArmorMeasurementModel(
        int armor_id, double sigma_m, bool use_dynamic, double k_radial = 1.0,
        double k_tangent = 0.1) noexcept
        : armor_id_(armor_id)
        , sigma_m_(sigma_m)
        , use_dynamic_noise_(use_dynamic)
        , k_radial_(k_radial)
        , k_tangent_(k_tangent)
        , cached_y_body_(Vec3::Zero()) {}

    /// Compute body-frame invariant error: r = R^T (y - p) - b
    /// Also caches y_body for dynamic noise computation
    [[nodiscard]] MeasFiber invariant_error(const Point& x, const MeasPoint& y) const noexcept {
        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);
        const double yaw_offset     = static_cast<double>(armor_id_) * kAngleStep;
        const Mat3 R_id = Eigen::AngleAxisd(yaw_offset, Vec3::UnitZ()).toRotationMatrix();

        const bool is_high    = (armor_id_ == kHighArmorIdA || armor_id_ == kHighArmorIdB);
        const double radius   = is_high ? x.second(kIdxR1) : x.second(kIdxR0);
        const double z_offset = is_high ? x.second(kIdxH) : 0.0;
        const Vec3 b          = R_id * Vec3{-radius, 0.0, z_offset};

        const Vec3 y_body = x.first.R().transpose() * (y.data - x.first.p());
        const Vec3 r      = y_body - b;

        // Cache y_body for dynamic noise computation (called after invariant_error)
        cached_y_body_ = y_body;

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

        C.template block<3, 3>(0, 0) = -detail::skew(b);
        C.template block<3, 3>(0, 6) = Mat3::Identity();

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
    ///
    /// Returns either:
    /// - Static: R = σ²I (isotropic, traditional InEKF)
    /// - Dynamic: R = f(y_body, armor_normal, k_r, k_t) (anisotropic, configurable)
    [[nodiscard]] Covariance measurement_noise() const noexcept {
        if (!use_dynamic_noise_) {
            return Covariance::ScaledIdentity(sigma_m_ * sigma_m_);
        }

        // Dynamic mode: compute anisotropic noise based on viewing geometry
        const Vec3 n_armor = detail::armor_normal_body(armor_id_, kArmorsNum);
        const Mat3 R_mat   = detail::compute_anisotropic_R_body(
            cached_y_body_, n_armor, sigma_m_, k_radial_, k_tangent_);
        return Covariance(R_mat);
    }
};

// ============================================================================
// OutpostArmorMeasurementModel - Satisfies InvariantMeasurementModel concept
// ============================================================================

/// @brief Invariant measurement model for outpost armor position observations
///
/// Same dynamic noise capability as RobotArmorMeasurementModel.
class OutpostArmorMeasurementModel {
public:
    static constexpr int kExtraDim         = OutpostProcessModel::kExtraDim;
    static constexpr int kGroupDim         = OutpostProcessModel::kGroupDim;
    static constexpr int kStateDim         = OutpostProcessModel::kStateDim;
    static constexpr int kArmorsNum        = 3;
    static constexpr double kOutpostRadius = 0.2765;

    // Extra state indices
    static constexpr int kIdxVYaw = OutpostProcessModel::kIdxVYaw;
    static constexpr int kIdxDz1  = OutpostProcessModel::kIdxDz1;
    static constexpr int kIdxDz2  = OutpostProcessModel::kIdxDz2;

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
    bool use_dynamic_noise_;
    double k_radial_;
    double k_tangent_;
    mutable Vec3 cached_y_body_;

public:
    /// @brief Construct with static isotropic noise (default)
    OutpostArmorMeasurementModel(int armor_id, double sigma_m) noexcept
        : armor_id_(armor_id)
        , sigma_m_(sigma_m)
        , use_dynamic_noise_(false)
        , k_radial_(1.0)
        , k_tangent_(0.1)
        , cached_y_body_(Vec3::Zero()) {}

    /// @brief Construct with configurable dynamic anisotropic noise
    OutpostArmorMeasurementModel(
        int armor_id, double sigma_m, bool use_dynamic, double k_radial = 1.0,
        double k_tangent = 0.1) noexcept
        : armor_id_(armor_id)
        , sigma_m_(sigma_m)
        , use_dynamic_noise_(use_dynamic)
        , k_radial_(k_radial)
        , k_tangent_(k_tangent)
        , cached_y_body_(Vec3::Zero()) {}

    /// Compute body-frame invariant error: r = R^T (y - p) - b
    [[nodiscard]] MeasFiber invariant_error(const Point& x, const MeasPoint& y) const noexcept {
        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);
        const double yaw_offset     = static_cast<double>(armor_id_) * kAngleStep;
        const Mat3 R_id = Eigen::AngleAxisd(yaw_offset, Vec3::UnitZ()).toRotationMatrix();

        double z_offset = 0.0;
        if (armor_id_ == 1) {
            z_offset = x.second(kIdxDz1);
        } else if (armor_id_ == 2) {
            z_offset = x.second(kIdxDz2);
        }

        const Vec3 b      = R_id * Vec3{-kOutpostRadius, 0.0, z_offset};
        const Vec3 y_body = x.first.R().transpose() * (y.data - x.first.p());
        const Vec3 r      = y_body - b;

        // Cache y_body for dynamic noise computation
        cached_y_body_ = y_body;

        return MeasFiber{r};
    }

    /// Error Jacobian C (∂r/∂δ)
    [[nodiscard]] Mat3xX error_jacobian(const Point& x) const noexcept {
        constexpr double kAngleStep = 2.0 * std::numbers::pi / static_cast<double>(kArmorsNum);
        const double yaw_offset     = static_cast<double>(armor_id_) * kAngleStep;
        const Mat3 R_id = Eigen::AngleAxisd(yaw_offset, Vec3::UnitZ()).toRotationMatrix();

        double z_offset = 0.0;
        if (armor_id_ == 1) {
            z_offset = x.second(kIdxDz1);
        } else if (armor_id_ == 2) {
            z_offset = x.second(kIdxDz2);
        }

        const Vec3 b = R_id * Vec3{-kOutpostRadius, 0.0, z_offset};

        Mat3xX C                     = Mat3xX::Zero();
        C.template block<3, 3>(0, 0) = -detail::skew(b);
        C.template block<3, 3>(0, 6) = Mat3::Identity();

        const Vec3 db_dz = R_id * Vec3{0.0, 0.0, 1.0};
        if (armor_id_ == 1) {
            C.template block<3, 1>(0, kGroupDim + kIdxDz1) = db_dz;
        } else if (armor_id_ == 2) {
            C.template block<3, 1>(0, kGroupDim + kIdxDz2) = db_dz;
        }

        return C;
    }

    /// Measurement noise covariance (static or dynamic)
    [[nodiscard]] Covariance measurement_noise() const noexcept {
        if (!use_dynamic_noise_) {
            return Covariance::ScaledIdentity(sigma_m_ * sigma_m_);
        }

        const Vec3 n_armor = detail::armor_normal_body(armor_id_, kArmorsNum);
        const Mat3 R_mat   = detail::compute_anisotropic_R_body(
            cached_y_body_, n_armor, sigma_m_, k_radial_, k_tangent_);
        return Covariance(R_mat);
    }
};

// ============================================================================
// TrackerInEKF_Base - CRTP base for shared implementation
// ============================================================================

/// @brief Base template for TrackerInEKF using CRTP pattern
///
/// Extracts common implementation shared between Robot and Outpost variants.
/// Template parameters allow compile-time polymorphism without runtime overhead.
template <typename Derived, int ExtraDim, typename ExtraVec>
class TrackerInEKF_Base {
protected:
    using SE23d = group::SEn3<double, 2>;
    using Vec3  = Eigen::Vector3d;
    using Mat3  = Eigen::Matrix3d;

    static constexpr int kGroupDim = 9;
    static constexpr int kStateDim = kGroupDim + ExtraDim;

    using Space      = SE23_Rn<ExtraDim>;
    using Point      = typename Space::Point;
    using Covariance = SymmetricMatrix<kStateDim, double>;
    using MatX       = Eigen::Matrix<double, kStateDim, kStateDim>;
    using VecX       = Eigen::Matrix<double, kStateDim, 1>;

    InCKF_SE23_Augmented<ExtraDim> filter_;

public:
    TrackerInEKF_Base() noexcept = default;

    [[nodiscard]] bool is_initialized() const noexcept { return filter_.is_initialized(); }

    [[nodiscard]] Vec3 get_position() const noexcept {
        return filter_.is_initialized() ? filter_.mean().first.p() : Vec3::Zero();
    }

    [[nodiscard]] Vec3 get_velocity() const noexcept {
        return filter_.is_initialized() ? filter_.mean().first.v() : Vec3::Zero();
    }

    [[nodiscard]] double get_yaw() const noexcept {
        if (!filter_.is_initialized()) {
            return 0.0;
        }
        const auto& R = filter_.mean().first.R();
        return std::atan2(R(1, 0), R(0, 0));
    }

    [[nodiscard]] double get_v_yaw() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(0) : 0.0;
    }

    [[nodiscard]] Mat3 get_position_covariance() const noexcept {
        if (!filter_.is_initialized()) {
            return Mat3::Zero();
        }
        return filter_.covariance().matrix().template block<3, 3>(6, 6);
    }

    [[nodiscard]] FilterOutput get_filter_output() const noexcept {
        FilterOutput output;
        output.active = filter_.is_initialized();
        if (output.active) {
            const auto& state          = filter_.mean();
            output.position            = state.first.p();
            output.velocity            = state.first.v();
            output.yaw                 = get_yaw();
            output.v_yaw               = state.second(0);
            output.position_covariance = get_position_covariance();
        }
        return output;
    }

protected:
    /// Update using InvariantMeasurementModel with roll/pitch constraint
    template <typename Model>
    requires InvariantMeasurementModel<Model, Space> void update_invariant_yaw_only(
        const typename Model::MeasurementSpace::Point& z, const Model& model) noexcept {
        using M               = typename Model::MeasurementSpace;
        constexpr int MeasDim = M::FiberDim;

        auto& belief = filter_.belief();
        auto& state  = belief.mean;
        auto& P      = belief.covariance;

        const auto r = model.invariant_error(state, z);
        const auto C = model.error_jacobian(state);
        const auto R = model.measurement_noise();

        const Eigen::Matrix<double, MeasDim, MeasDim> S_mat =
            C * P.matrix() * C.transpose() + R.matrix();

        Eigen::LDLT<Eigen::Matrix<double, MeasDim, MeasDim>> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            return;
        }

        const Eigen::Matrix<double, kStateDim, MeasDim> K =
            P.matrix() * C.transpose()
            * ldlt.solve(Eigen::Matrix<double, MeasDim, MeasDim>::Identity());

        VecX delta = K * r.data;
        delta(0)   = 0.0;
        delta(1)   = 0.0;

        apply_update(delta);

        const MatX I_KC  = MatX::Identity() - K * C;
        const auto P_new = I_KC * P.matrix() * I_KC.transpose() + K * R.matrix() * K.transpose();
        P                = Covariance(P_new);

        belief.symmetrize();
        belief.regularize();
        derived().clamp_extra_in_place(state.second.data);
        zero_out_roll_pitch_covariance();
    }

    void apply_update(const VecX& delta) noexcept {
        auto& state = filter_.belief().mean;

        Eigen::Matrix<double, 9, 1> xi = Eigen::Matrix<double, 9, 1>::Zero();
        xi(2)                          = delta(2);
        xi.template segment<3>(3)      = delta.template segment<3>(3);
        xi.template segment<3>(6)      = delta.template segment<3>(6);
        state.first                    = state.first * SE23d::exp(xi);

        state.second.data += delta.template segment<ExtraDim>(kGroupDim);
        state.second(0) = detail::sanitize_finite(state.second(0), 0.0);
    }

    void zero_out_roll_pitch_covariance() noexcept {
        auto& P    = filter_.belief().covariance;
        auto P_mat = P.matrix();

        P_mat.row(0).setZero();
        P_mat.row(1).setZero();
        P_mat.col(0).setZero();
        P_mat.col(1).setZero();
        P_mat(0, 0) = detail::kCovarianceRegularization;
        P_mat(1, 1) = detail::kCovarianceRegularization;

        P = Covariance(P_mat);
    }

private:
    Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }
};

// ============================================================================
// TrackerInEKF_Robot - Robot (4-armor) variant
// ============================================================================

/// @brief InEKF for robot tracking using uni_kf framework
///
/// State: SE₂(3) × ℝ⁴ where extra = [v_yaw, r0, r1, h]
/// Uses RobotProcessModel for prediction (via filter.predict with ProcessModel concept)
/// Uses invariant error for measurement updates (custom implementation)
///
/// This is a drop-in replacement for fcs::L3::InEKF<RobotModel>
class TrackerInEKF_Robot : public TrackerInEKF_Base<TrackerInEKF_Robot, 4, Eigen::Vector4d> {
    using Base = TrackerInEKF_Base<TrackerInEKF_Robot, 4, Eigen::Vector4d>;
    friend Base;

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

    TrackerInEKF_Robot() noexcept = default;

    /// Initialize filter state
    void init(const Vec3& p0, double yaw0, const RobotInEKFConfig& config) noexcept {
        if (!detail::all_finite(p0) || !std::isfinite(yaw0)) {
            filter_.reset();
            return;
        }

        SE23d::IsometriesType t;
        t[0] = Vec3::Zero();
        t[1] = p0;

        const Mat3 R = Eigen::AngleAxisd(yaw0, Vec3::UnitZ()).toRotationMatrix();
        const SE23d se23(R, t);

        VecN<kExtraDim> extra;
        extra.data << 0.0, config.radius0, config.radius1, config.height;
        clamp_extra_in_place(extra.data);

        const Point state{se23, extra};

        Covariance P = Covariance::Zero();
        for (int i = 0; i < kGroupDim; ++i) {
            P(i, i) = 0.1;
        }
        P(kGroupDim + kIdxVYaw, kGroupDim + kIdxVYaw) = 1.0;
        P(kGroupDim + kIdxR0, kGroupDim + kIdxR0)     = 0.01;
        P(kGroupDim + kIdxR1, kGroupDim + kIdxR1)     = 0.01;
        P(kGroupDim + kIdxH, kGroupDim + kIdxH)       = 0.01;

        filter_.init(state, P);
    }

    /// Predict step using ProcessModel concept
    void predict(double dt, const RobotInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || dt <= 0.0 || !std::isfinite(dt)) {
            return;
        }

        const RobotProcessModel model(config);
        filter_.predict(model, dt);

        clamp_extra_in_place(filter_.belief().mean.second.data);
        zero_out_roll_pitch_covariance();
    }

    /// Update with armor observation using InvariantMeasurementModel
    /// @param y Observed armor position in world frame
    /// @param armor_id Armor index (0-3)
    /// @param config Filter configuration (including use_dynamic_noise flag)
    void update_armor(const Vec3& y, int armor_id, const RobotInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !detail::all_finite(y)) {
            return;
        }
        if (armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }

        const RobotArmorMeasurementModel meas_model(
            armor_id, config.sigma_m, config.use_dynamic_noise, config.dynamic_noise_k_radial,
            config.dynamic_noise_k_tangent);
        update_invariant_yaw_only(RobotArmorMeasurementModel::MeasPoint{y}, meas_model);
    }

    /// Update yaw using InvariantMeasurementModel
    void update_yaw(double yaw_meas, const RobotInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !std::isfinite(yaw_meas)) {
            return;
        }
        if (!(config.sigma_w > 0.0) || !std::isfinite(config.sigma_w)) {
            return;
        }

        const YawMeasurementModel<kExtraDim> meas_model(config.sigma_w);
        using MeasPoint = typename YawMeasurementModel<kExtraDim>::MeasPoint;
        update_invariant_yaw_only(MeasPoint{Eigen::Matrix<double, 1, 1>{yaw_meas}}, meas_model);
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

    [[nodiscard]] double get_radius0() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(kIdxR0) : 0.22;
    }

    [[nodiscard]] double get_radius1() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(kIdxR1) : 0.22;
    }

    [[nodiscard]] double get_height() const noexcept {
        return filter_.is_initialized() ? filter_.mean().second(kIdxH) : 0.0;
    }

private:
    static void clamp_extra_in_place(Eigen::Vector4d& extra) noexcept {
        extra(kIdxVYaw) = detail::sanitize_finite(extra(kIdxVYaw), 0.0);
        extra(kIdxR0)   = detail::sanitize_finite(extra(kIdxR0), 0.23);
        extra(kIdxR1)   = detail::sanitize_finite(extra(kIdxR1), 0.23);
        extra(kIdxH)    = detail::sanitize_finite(extra(kIdxH), 0.0);

        extra(kIdxR0) = std::clamp(extra(kIdxR0), kMinRadius, kMaxRadius);
        extra(kIdxR1) = std::clamp(extra(kIdxR1), kMinRadius, kMaxRadius);
        extra(kIdxH)  = std::clamp(extra(kIdxH), kMinHeight, kMaxHeight);
    }
};

// ============================================================================
// TrackerInEKF_Outpost - Outpost (3-armor) variant
// ============================================================================

/// @brief InEKF for outpost tracking using uni_kf framework
///
/// State: SE₂(3) × ℝ³ where extra = [v_yaw, dz1, dz2]
/// This is a drop-in replacement for fcs::L3::InEKF<OutpostModel>
class TrackerInEKF_Outpost : public TrackerInEKF_Base<TrackerInEKF_Outpost, 3, Eigen::Vector3d> {
    using Base = TrackerInEKF_Base<TrackerInEKF_Outpost, 3, Eigen::Vector3d>;
    friend Base;

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

    /// Predict step using ProcessModel concept
    void predict(double dt, const OutpostInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || dt <= 0.0 || !std::isfinite(dt)) {
            return;
        }

        const OutpostProcessModel model(config);
        filter_.predict(model, dt);

        clamp_extra_in_place(filter_.belief().mean.second.data);
        zero_out_roll_pitch_covariance();
    }

    void update_armor(const Vec3& y, int armor_id, const OutpostInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !detail::all_finite(y)) {
            return;
        }
        if (armor_id < 0 || armor_id >= kArmorsNum) {
            return;
        }

        const OutpostArmorMeasurementModel meas_model(
            armor_id, config.sigma_m, config.use_dynamic_noise, config.dynamic_noise_k_radial,
            config.dynamic_noise_k_tangent);
        update_invariant_yaw_only(OutpostArmorMeasurementModel::MeasPoint{y}, meas_model);
    }

    void update_yaw(double yaw_meas, const OutpostInEKFConfig& config) noexcept {
        if (!filter_.is_initialized() || !std::isfinite(yaw_meas)) {
            return;
        }
        if (!(config.sigma_w > 0.0) || !std::isfinite(config.sigma_w)) {
            return;
        }

        const YawMeasurementModel<kExtraDim> meas_model(config.sigma_w);
        using MeasPoint = typename YawMeasurementModel<kExtraDim>::MeasPoint;
        update_invariant_yaw_only(MeasPoint{Eigen::Matrix<double, 1, 1>{yaw_meas}}, meas_model);
    }

    [[nodiscard]] OutpostTargetState get_outpost_state() const noexcept {
        OutpostTargetState state_out;
        if (!filter_.is_initialized()) {
            return state_out;
        }

        const auto& state  = filter_.mean();
        state_out.position = {state.first.p().x(), state.first.p().y()};
        state_out.velocity = state.first.v();
        state_out.yaw      = std::atan2(state.first.R()(1, 0), state.first.R()(0, 0));
        state_out.v_yaw    = state.second(kIdxVYaw);
        state_out.z        = {
            state.first.p().z(), state.first.p().z() + state.second(kIdxDz1),
            state.first.p().z() + state.second(kIdxDz2)};
        return state_out;
    }

private:
    static void clamp_extra_in_place(Eigen::Vector3d& extra) noexcept {
        extra(kIdxVYaw) = detail::sanitize_finite(extra(kIdxVYaw), 0.0);
        extra(kIdxDz1)  = detail::sanitize_finite(extra(kIdxDz1), 0.0);
        extra(kIdxDz2)  = detail::sanitize_finite(extra(kIdxDz2), 0.0);
    }
};

} // namespace uni_kf
