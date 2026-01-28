#pragma once

#include "motion_model_params.hpp"
#include "uni_kf_spaces.hpp"
#include "util.hpp"

#include <uni_kf/concepts/quadratic_form.hpp>
#include <uni_kf/core/filter.hpp>

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <expected>
#include <numbers>

// Forward declaration to avoid including toml++ in algorithm headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L3 {

// ============================================================================
// RobotModel: 4-armor Robot Target (state on R^10 × SO3_z)
// ============================================================================

struct RobotModel {
    using Scalar = double;

    static constexpr int ARMORS_NUM = 4;

    // Euclidean component of the state: [xc, vx, yc, vy, z0, vz, v_yaw, r0, r1, h]
    enum EuclidIndex : int {
        XC         = 0,
        VX         = 1,
        YC         = 2,
        VY         = 3,
        Z0         = 4,
        VZ         = 5,
        V_YAW      = 6,
        R_0        = 7,
        R_1        = 8,
        H          = 9,
        EUCLID_DIM = 10,
    };

    using EuclidSpace = uni_kf::Euclidean<EUCLID_DIM, Scalar>;
    using YawSpace    = ukf::YawSO3Z;
    using StateSpace  = uni_kf::ProductSpace<EuclidSpace, YawSpace>;
    using MeasSpace   = ukf::ArmorMeasSpace;

    using StatePoint = StateSpace::Point;
    using MeasPoint  = MeasSpace::Point;

    static constexpr int NX = StateSpace::FiberDim; // 11
    static constexpr int NZ = MeasSpace::FiberDim;  // 4

    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;

    RobotModelParams params;

    [[nodiscard]] static auto make_state(
        double xc, double vx, double yc, double vy, double z0, double vz, double yaw, double v_yaw,
        double r0, double r1, double h) noexcept -> StatePoint {
        typename EuclidSpace::Point::EigenType e;
        e << xc, vx, yc, vy, z0, vz, v_yaw, r0, r1, h;
        return {EuclidSpace::Point(e), YawSpace::from_angle(yaw)};
    }

    [[nodiscard]] auto predict(const StatePoint& x, double dt) const noexcept -> StatePoint {
        auto e = x.first.data;
        e(XC) += e(VX) * dt;
        e(YC) += e(VY) * dt;
        e(Z0) += e(VZ) * dt;

        const double d_yaw = e(V_YAW) * dt;
        const auto yaw_new = YawSpace::act(x.second, YawSpace::fiber_from_angle(d_yaw));

        return {EuclidSpace::Point(e), yaw_new};
    }

    [[nodiscard]] auto process_noise(double dt) const noexcept
        -> uni_kf::SymmetricMatrix<NX, Scalar> {
        Eigen::Matrix<Scalar, NX, NX> Q = Eigen::Matrix<Scalar, NX, NX>::Zero();
        if (dt <= Scalar(0)) {
            return uni_kf::SymmetricMatrix<NX, Scalar>(Q);
        }

        const Scalar sigma_a_xy2  = params.sigma_a_xy * params.sigma_a_xy;
        const Scalar sigma_a_z2   = params.sigma_a_z * params.sigma_a_z;
        const Scalar sigma_a_yaw2 = params.sigma_a_yaw * params.sigma_a_yaw;

        const Scalar dt2 = dt * dt;
        const Scalar dt3 = dt2 * dt;
        const Scalar dt4 = dt2 * dt2;

        auto fill_cv = [&](int idx_p, int idx_v, Scalar q) {
            Q(idx_p, idx_p) = q * dt4 / 4.0;
            Q(idx_p, idx_v) = q * dt3 / 2.0;
            Q(idx_v, idx_p) = Q(idx_p, idx_v);
            Q(idx_v, idx_v) = q * dt2;
        };

        // CV blocks
        fill_cv(XC, VX, sigma_a_xy2);
        fill_cv(YC, VY, sigma_a_xy2);
        fill_cv(Z0, VZ, sigma_a_z2);
        // yaw is the last fiber component of StateSpace (Euclid first, then yaw)
        constexpr int kYawFiber = EUCLID_DIM;
        fill_cv(kYawFiber, V_YAW, sigma_a_yaw2);

        // Random walk noise for R0, R1, H
        Q(R_0, R_0) += (params.sigma_r0 * params.sigma_r0) * dt;
        Q(R_1, R_1) += (params.sigma_r1 * params.sigma_r1) * dt;
        Q(H, H) += (params.sigma_h * params.sigma_h) * dt;

        return uni_kf::SymmetricMatrix<NX, Scalar>(Q);
    }

    [[nodiscard]] auto measure(const StatePoint& x, int id) const noexcept -> MeasPoint {
        assert(id >= 0 && id <= ARMORS_NUM - 1);

        const auto& e = x.first.data;

        constexpr Scalar angle_step = 2.0 * std::numbers::pi / static_cast<Scalar>(ARMORS_NUM);

        const Scalar yaw0    = YawSpace::angle(x.second);
        const Scalar tmp_yaw = yaw0 + static_cast<Scalar>(id) * angle_step;

        Scalar radius;
        Scalar armor_z;
        if (id == kHighArmorIdA || id == kHighArmorIdB) {
            armor_z = e(Z0) + e(H);
            radius  = e(R_1);
        } else {
            armor_z = e(Z0);
            radius  = e(R_0);
        }

        const Scalar armor_x = e(XC) - radius * std::cos(tmp_yaw);
        const Scalar armor_y = e(YC) - radius * std::sin(tmp_yaw);
        return ukf::ArmorMeas::make(Eigen::Vector3d{armor_x, armor_y, armor_z}, tmp_yaw);
    }

    [[nodiscard]] auto measurement_noise(const MeasPoint& z) const noexcept
        -> uni_kf::SymmetricMatrix<NZ, Scalar> {
        const Eigen::Vector3d dir = ukf::ArmorMeas::direction(z);
        const Scalar bearing_yaw  = std::atan2(dir.y(), dir.x());
        const auto delta =
            ukf::YawSO3Z::diff(ukf::YawSO3Z::from_angle(bearing_yaw), z.second.second);
        const Scalar delta_angle = std::abs(delta.data(0));
        const Scalar distance    = ukf::ArmorMeas::distance(z);

        Eigen::Matrix<Scalar, NZ, 1> diag;
        diag << 4e-3, 4e-3, std::log(delta_angle + 1.0) + 1.0,
            std::log(std::abs(distance) + 1.0) / 200.0 + 9e-2;
        return uni_kf::SymmetricMatrix<NZ, Scalar>::Diagonal(diag);
    }

    /// Get the expected radius for a given armor ID.
    ///
    /// Problem:
    ///   For geometric consistency constraints, we need to know the expected radius
    ///   for each armor based on the current state estimate.
    ///
    /// Precise:
    ///   Robot armors 0,2 use r0; armors 1,3 use r1.
    [[nodiscard]] Scalar expected_radius(const StatePoint& x, int armor_id) const noexcept {
        const auto& e = x.first.data;
        if (armor_id == kHighArmorIdA || armor_id == kHighArmorIdB) {
            return e(R_1);
        }
        return e(R_0);
    }

    /// Get the expected height (z-coordinate) for a given armor ID.
    ///
    /// Problem:
    ///   For geometric consistency constraints, we need to know the expected height
    ///   for each armor based on the current state estimate.
    ///
    /// Precise:
    ///   Robot armors 0,2 use z0; armors 1,3 use z0 + h.
    [[nodiscard]] Scalar expected_height(const StatePoint& x, int armor_id) const noexcept {
        const auto& e = x.first.data;
        if (armor_id == kHighArmorIdA || armor_id == kHighArmorIdB) {
            return e(Z0) + e(H);
        }
        return e(Z0);
    }

    /// Get the center XY coordinates from state.
    [[nodiscard]] Eigen::Vector2d center_xy(const StatePoint& x) const noexcept {
        const auto& e = x.first.data;
        return {e(XC), e(YC)};
    }
};

// ============================================================================
// OutpostModel: 3-armor Outpost Target (state on R^6 × SO3_z)
// ============================================================================

struct OutpostModel {
    using Scalar = double;

    static constexpr Scalar OUTPOST_RADIUS = 0.2765;
    static constexpr int ARMORS_NUM        = 3;

    // Euclidean component of the state: [xc, yc, v_yaw, z0, z1, z2]
    enum EuclidIndex : int {
        XC         = 0,
        YC         = 1,
        V_YAW      = 2,
        Z0         = 3,
        Z1         = 4,
        Z2         = 5,
        EUCLID_DIM = 6,
    };

    using EuclidSpace = uni_kf::Euclidean<EUCLID_DIM, Scalar>;
    using YawSpace    = ukf::YawSO3Z;
    using StateSpace  = uni_kf::ProductSpace<EuclidSpace, YawSpace>;
    using MeasSpace   = ukf::ArmorMeasSpace;

    using StatePoint = StateSpace::Point;
    using MeasPoint  = MeasSpace::Point;

    static constexpr int NX = StateSpace::FiberDim; // 7
    static constexpr int NZ = MeasSpace::FiberDim;  // 4

    OutpostModelParams params;

    [[nodiscard]] static auto make_state(
        double xc, double yc, double yaw, double v_yaw, double z0, double z1, double z2) noexcept
        -> StatePoint {
        typename EuclidSpace::Point::EigenType e;
        e << xc, yc, v_yaw, z0, z1, z2;
        return {EuclidSpace::Point(e), YawSpace::from_angle(yaw)};
    }

    [[nodiscard]] auto predict(const StatePoint& x, double dt) const noexcept -> StatePoint {
        const auto& e      = x.first.data;
        const double d_yaw = e(V_YAW) * dt;
        const auto yaw_new = YawSpace::act(x.second, YawSpace::fiber_from_angle(d_yaw));
        return {x.first, yaw_new};
    }

    [[nodiscard]] auto process_noise(double dt) const noexcept
        -> uni_kf::SymmetricMatrix<NX, Scalar> {
        Eigen::Matrix<Scalar, NX, NX> Q = Eigen::Matrix<Scalar, NX, NX>::Zero();
        if (dt <= Scalar(0)) {
            return uni_kf::SymmetricMatrix<NX, Scalar>(Q);
        }

        const Scalar sigma_xy2  = params.sigma_q_xy * params.sigma_q_xy;
        const Scalar sigma_z2   = params.sigma_q_z * params.sigma_q_z;
        const Scalar sigma_yaw2 = params.sigma_a_yaw * params.sigma_a_yaw;

        const Scalar dt2 = dt * dt;
        const Scalar dt3 = dt2 * dt;
        const Scalar dt4 = dt2 * dt2;

        // X, Y: drift-like random walk (Q ∝ dt^2)
        Q(XC, XC) = sigma_xy2 * dt2;
        Q(YC, YC) = sigma_xy2 * dt2;

        // YAW/VYAW: CV model (yaw is last fiber component)
        constexpr int kYawFiber = EUCLID_DIM;
        Q(kYawFiber, kYawFiber) = sigma_yaw2 * dt4 / 4.0;
        Q(kYawFiber, V_YAW)     = sigma_yaw2 * dt3 / 2.0;
        Q(V_YAW, kYawFiber)     = Q(kYawFiber, V_YAW);
        Q(V_YAW, V_YAW)         = sigma_yaw2 * dt2;

        // Z0, Z1, Z2: drift-like random walk (Q ∝ dt^2)
        Q(Z0, Z0) = sigma_z2 * dt2;
        Q(Z1, Z1) = sigma_z2 * dt2;
        Q(Z2, Z2) = sigma_z2 * dt2;

        return uni_kf::SymmetricMatrix<NX, Scalar>(Q);
    }

    [[nodiscard]] auto measure(const StatePoint& x, int id) const noexcept -> MeasPoint {
        assert(id >= 0 && id <= ARMORS_NUM - 1);

        const auto& e = x.first.data;

        constexpr Scalar angle_step = 2.0 * std::numbers::pi / static_cast<Scalar>(ARMORS_NUM);
        const Scalar yaw0           = YawSpace::angle(x.second);
        const Scalar tmp_yaw        = yaw0 + angle_step * static_cast<Scalar>(id);

        const Scalar armor_x = e(XC) - OUTPOST_RADIUS * std::cos(tmp_yaw);
        const Scalar armor_y = e(YC) - OUTPOST_RADIUS * std::sin(tmp_yaw);
        const Scalar armor_z = (id == 0) ? e(Z0) : (id == 1 ? e(Z1) : e(Z2));
        return ukf::ArmorMeas::make(Eigen::Vector3d{armor_x, armor_y, armor_z}, tmp_yaw);
    }

    [[nodiscard]] auto measurement_noise(const MeasPoint& z) const noexcept
        -> uni_kf::SymmetricMatrix<NZ, Scalar> {
        const Eigen::Vector3d dir = ukf::ArmorMeas::direction(z);
        const Scalar bearing_yaw  = std::atan2(dir.y(), dir.x());
        const auto delta =
            ukf::YawSO3Z::diff(ukf::YawSO3Z::from_angle(bearing_yaw), z.second.second);
        const Scalar delta_angle = std::abs(delta.data(0));
        const Scalar distance    = ukf::ArmorMeas::distance(z);

        Eigen::Matrix<Scalar, NZ, 1> diag;
        diag << 4e-3, 4e-3, std::log(delta_angle + 1.0) + 1.0,
            std::log(std::abs(distance) + 1.0) / 200.0 + 9e-2;
        return uni_kf::SymmetricMatrix<NZ, Scalar>::Diagonal(diag);
    }

    /// Get the expected radius for a given armor ID.
    ///
    /// Problem:
    ///   For geometric consistency constraints, we need to know the expected radius
    ///   for each armor based on the current state estimate.
    ///
    /// Precise:
    ///   Outpost has fixed radius for all armors.
    [[nodiscard]] static constexpr Scalar expected_radius(
        [[maybe_unused]] const StatePoint& x, [[maybe_unused]] int armor_id) noexcept {
        return OUTPOST_RADIUS;
    }

    /// Get the expected height (z-coordinate) for a given armor ID.
    ///
    /// Problem:
    ///   For geometric consistency constraints, we need to know the expected height
    ///   for each armor based on the current state estimate.
    ///
    /// Precise:
    ///   Outpost armors have independent heights: z0, z1, z2.
    [[nodiscard]] Scalar expected_height(const StatePoint& x, int armor_id) const noexcept {
        const auto& e = x.first.data;
        switch (armor_id) {
        case 0: return e(Z0);
        case 1: return e(Z1);
        case 2: return e(Z2);
        default: return e(Z0);
        }
    }

    /// Get the center XY coordinates from state.
    [[nodiscard]] Eigen::Vector2d center_xy(const StatePoint& x) const noexcept {
        const auto& e = x.first.data;
        return {e(XC), e(YC)};
    }
};

// ============================================================================
// Unified tracker SRCKF wrapper (KF + motion/observation model)
// ============================================================================

template <typename Model>
class UniTrackerSRCKF {
public:
    using StateSpace        = typename Model::StateSpace;
    using StatePoint        = typename Model::StatePoint;
    using MeasSpace         = typename Model::MeasSpace;
    using MeasPoint         = typename Model::MeasPoint;
    static constexpr int NX = Model::NX;
    static constexpr int NZ = Model::NZ;

    using Filter = uni_kf::UnifiedKalmanFilter<
        StateSpace, uni_kf::CubaturePointsTag, uni_kf::CovarianceTag, double>;
    using CovX = uni_kf::SymmetricMatrix<NX, double>;
    using CovZ = uni_kf::SymmetricMatrix<NZ, double>;

    struct ArmorMeasurementModel {
        using MeasurementSpace = MeasSpace;

        const Model* motion_model{nullptr};
        int armor_id{0};
        MeasPoint z_meas{};

        [[nodiscard]] MeasPoint predict(const StatePoint& x) const noexcept {
            return motion_model->measure(x, armor_id);
        }

        [[nodiscard]] CovZ measurement_noise() const noexcept {
            return motion_model->measurement_noise(z_meas);
        }

        [[nodiscard]] MeasPoint operator()(const StatePoint& x) const noexcept {
            return predict(x);
        }
    };

    UniTrackerSRCKF() noexcept = default;

    UniTrackerSRCKF(Model& model, const StatePoint& x0, const CovX& P0) noexcept
        : model_(&model) {
        filter_.init(x0, P0);
    }

    [[nodiscard]] bool initialized() const noexcept { return filter_.is_initialized(); }

    void predict(double dt) noexcept {
        if (!model_) {
            return;
        }
        filter_.predict(*model_, dt);
    }

    void update(const MeasPoint& z, int armor_id) noexcept {
        if (!model_) {
            return;
        }
        ArmorMeasurementModel meas_model{model_, armor_id, z};
        filter_.update(z, meas_model);
    }

    [[nodiscard]] const StatePoint& x() const noexcept { return filter_.mean(); }
    [[nodiscard]] StatePoint& x_mut() noexcept { return filter_.belief().mean; }

    [[nodiscard]] auto Px() const noexcept { return filter_.belief().covariance(); }

    [[nodiscard]] const Filter& filter() const noexcept { return filter_; }

private:
    Model* model_{nullptr};
    Filter filter_;
};

using RoboCKF    = UniTrackerSRCKF<RobotModel>;
using OutpostCKF = UniTrackerSRCKF<OutpostModel>;

} // namespace fcs::L3
