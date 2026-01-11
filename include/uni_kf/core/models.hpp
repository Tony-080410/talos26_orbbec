/// @file models.hpp
/// @brief Process and measurement model concepts for Kalman filtering
///
/// Defines the interface for dynamics (process) and measurement models
/// that work with the unified Kalman filter framework.

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/quadratic_form.hpp"
#include "../spaces/euclidean.hpp"

#include <concepts>

namespace uni_kf {

// ============================================================================
// ProcessModel Concept
// ============================================================================

/// @brief Concept for state transition (process) models
///
/// A ProcessModel defines how the state evolves over time:
/// - Dynamics function: x' = f(x, dt)
/// - Process noise covariance: Q(dt)
/// - Optional: analytical Jacobian F = ∂f/∂x
template <typename M, typename D>
concept ProcessModel = DifferentialSpace<D> && requires(M m, typename D::Point x, double dt) {
    /// State transition function
    { m.predict(x, dt) } -> std::same_as<typename D::Point>;

    /// Process noise covariance
    { m.process_noise(dt) } -> std::same_as<SymmetricMatrix<D::FiberDim, typename D::Scalar>>;
};

/// @brief Extended ProcessModel with analytical Jacobian
template <typename M, typename D>
concept ProcessModelWithJacobian =
    ProcessModel<M, D> && requires(M m, typename D::Point x, double dt) {
        /// Analytical Jacobian of dynamics
        {
            m.jacobian(x, dt)
        } -> std::same_as<Eigen::Matrix<typename D::Scalar, D::FiberDim, D::FiberDim>>;
    };

// ============================================================================
// MeasurementModel Concept
// ============================================================================

/// @brief Concept for measurement (observation) models
///
/// A MeasurementModel defines how measurements relate to state:
/// - Measurement function: z = h(x)
/// - Measurement noise covariance: R
/// - Optional: analytical Jacobian H = ∂h/∂x
template <typename M, typename D>
concept MeasurementModel = DifferentialSpace<D> && requires {
    /// Measurement space (could be different DifferentialSpace)
    typename M::MeasurementSpace;
} && requires(M m, typename D::Point x, typename M::MeasurementSpace::Point z) {
    /// Measurement function
    { m.predict(x) } -> std::same_as<typename M::MeasurementSpace::Point>;

    /// Measurement noise covariance
    {
        m.measurement_noise()
    } -> std::same_as<SymmetricMatrix<M::MeasurementSpace::FiberDim, typename D::Scalar>>;
};

/// @brief Extended MeasurementModel with analytical Jacobian
template <typename M, typename D>
concept MeasurementModelWithJacobian =
    MeasurementModel<M, D> && requires(M m, typename D::Point x) {
        /// Analytical Jacobian of measurement function
        {
            m.jacobian(x)
        } -> std::same_as<
            Eigen::Matrix<typename D::Scalar, M::MeasurementSpace::FiberDim, D::FiberDim>>;
    };

// ============================================================================
// InvariantMeasurementModel Concept (for InEKF)
// ============================================================================

/// @brief Concept for invariant measurement models (used in InEKF)
///
/// InEKF uses body-frame invariant error instead of standard innovation:
/// - Standard EKF: innovation = z - h(x)
/// - InEKF: r = R^T (y - p) - b (body-frame error)
///
/// The Jacobian C relates the error to the error state, not the measurement to state.
template <typename M, typename D>
concept InvariantMeasurementModel = DifferentialSpace<D> && requires {
    typename M::MeasurementSpace;
} && requires(M m, typename D::Point x, typename M::MeasurementSpace::Point z) {
    /// Compute body-frame invariant error: r = R^T (y - p) - b
    { m.invariant_error(x, z) } -> std::same_as<typename M::MeasurementSpace::Fiber>;

    /// Error Jacobian C (∂r/∂δ where δ is the error state)
    {
        m.error_jacobian(x)
    } -> std::convertible_to<
        Eigen::Matrix<typename D::Scalar, M::MeasurementSpace::FiberDim, D::FiberDim>>;

    /// Measurement noise covariance
    {
        m.measurement_noise()
    } -> std::same_as<SymmetricMatrix<M::MeasurementSpace::FiberDim, typename D::Scalar>>;
};

// ============================================================================
// Simple model wrappers
// ============================================================================

/// @brief Wrapper for lambda-based process model
template <DifferentialSpace D, typename DynFunc, typename NoiseFunc>
class LambdaProcessModel {
    DynFunc f_;
    NoiseFunc q_;

public:
    using Space = D;

    LambdaProcessModel(DynFunc f, NoiseFunc q) noexcept
        : f_(std::move(f))
        , q_(std::move(q)) {}

    [[nodiscard]] auto predict(const typename D::Point& x, double dt) const noexcept ->
        typename D::Point {
        return f_(x, dt);
    }

    [[nodiscard]] auto process_noise(double dt) const noexcept
        -> SymmetricMatrix<D::FiberDim, typename D::Scalar> {
        return q_(dt);
    }
};

/// @brief Helper to create lambda process model with type deduction
template <DifferentialSpace D, typename DynFunc, typename NoiseFunc>
[[nodiscard]] auto make_process_model(DynFunc&& f, NoiseFunc&& q) noexcept
    -> LambdaProcessModel<D, std::decay_t<DynFunc>, std::decay_t<NoiseFunc>> {
    return {std::forward<DynFunc>(f), std::forward<NoiseFunc>(q)};
}

/// @brief Wrapper for lambda-based measurement model
template <DifferentialSpace D, DifferentialSpace M, typename MeasFunc, typename NoiseFunc>
class LambdaMeasurementModel {
    MeasFunc h_;
    NoiseFunc r_;

public:
    using StateSpace       = D;
    using MeasurementSpace = M;

    LambdaMeasurementModel(MeasFunc h, NoiseFunc r) noexcept
        : h_(std::move(h))
        , r_(std::move(r)) {}

    [[nodiscard]] auto predict(const typename D::Point& x) const noexcept -> typename M::Point {
        return h_(x);
    }

    [[nodiscard]] auto measurement_noise() const noexcept
        -> SymmetricMatrix<M::FiberDim, typename D::Scalar> {
        return r_();
    }
};

/// @brief Helper to create lambda measurement model with type deduction
template <DifferentialSpace D, DifferentialSpace M, typename MeasFunc, typename NoiseFunc>
[[nodiscard]] auto make_measurement_model(MeasFunc&& h, NoiseFunc&& r) noexcept
    -> LambdaMeasurementModel<D, M, std::decay_t<MeasFunc>, std::decay_t<NoiseFunc>> {
    return {std::forward<MeasFunc>(h), std::forward<NoiseFunc>(r)};
}

// ============================================================================
// Common model templates
// ============================================================================

/// @brief Constant velocity process model for Euclidean space
///
/// State: [position, velocity]
/// Dynamics: position' = position + velocity * dt, velocity' = velocity
template <int PosDim, typename Scalar = double>
class ConstantVelocityModel {
public:
    static constexpr int StateDim = 2 * PosDim;
    using Space                   = Euclidean<StateDim, Scalar>;
    using Point                   = typename Space::Point;
    using Covariance              = SymmetricMatrix<StateDim, Scalar>;

private:
    Scalar sigma_a_; // Acceleration noise std dev

public:
    explicit ConstantVelocityModel(Scalar sigma_a) noexcept
        : sigma_a_(sigma_a) {}

    [[nodiscard]] Point predict(const Point& x, double dt) const noexcept {
        typename Point::EigenType x_new;
        // Position update: p' = p + v * dt
        x_new.template head<PosDim>() =
            x.data.template head<PosDim>() + x.data.template tail<PosDim>() * dt;
        // Velocity unchanged
        x_new.template tail<PosDim>() = x.data.template tail<PosDim>();
        return Point(x_new);
    }

    [[nodiscard]] Covariance process_noise(double dt) const noexcept {
        // Discrete-time CV noise covariance
        // Q = [dt^4/4*I,  dt^3/2*I] * sigma_a^2
        //     [dt^3/2*I,  dt^2*I  ]
        typename Covariance::MatrixType Q = Covariance::MatrixType::Zero();

        Scalar q   = sigma_a_ * sigma_a_;
        Scalar dt2 = dt * dt;
        Scalar dt3 = dt2 * dt;
        Scalar dt4 = dt2 * dt2;

        for (int i = 0; i < PosDim; ++i) {
            // Position-position
            Q(i, i) = q * dt4 / Scalar{4};
            // Position-velocity
            Q(i, PosDim + i) = q * dt3 / Scalar{2};
            Q(PosDim + i, i) = q * dt3 / Scalar{2};
            // Velocity-velocity
            Q(PosDim + i, PosDim + i) = q * dt2;
        }

        return Covariance(Q);
    }

    [[nodiscard]] auto jacobian(const Point& /*x*/, double dt) const noexcept
        -> Eigen::Matrix<Scalar, StateDim, StateDim> {
        // F = [I, dt*I]
        //     [0,   I ]
        Eigen::Matrix<Scalar, StateDim, StateDim> F =
            Eigen::Matrix<Scalar, StateDim, StateDim>::Identity();

        for (int i = 0; i < PosDim; ++i) {
            F(i, PosDim + i) = dt;
        }

        return F;
    }
};

/// @brief Position measurement model
///
/// Measures only position from a position-velocity state
template <int PosDim, typename Scalar = double>
class PositionMeasurementModel {
public:
    static constexpr int StateDim = 2 * PosDim;
    static constexpr int MeasDim  = PosDim;

    using StateSpace       = Euclidean<StateDim, Scalar>;
    using MeasurementSpace = Euclidean<MeasDim, Scalar>;
    using StatePoint       = typename StateSpace::Point;
    using MeasPoint        = typename MeasurementSpace::Point;
    using MeasCovariance   = SymmetricMatrix<MeasDim, Scalar>;

private:
    MeasCovariance R_; // Measurement noise covariance

public:
    explicit PositionMeasurementModel(Scalar sigma_m) noexcept
        : R_(MeasCovariance::ScaledIdentity(sigma_m * sigma_m)) {}

    explicit PositionMeasurementModel(const MeasCovariance& R) noexcept
        : R_(R) {}

    [[nodiscard]] MeasPoint predict(const StatePoint& x) const noexcept {
        return MeasPoint(x.data.template head<PosDim>());
    }

    [[nodiscard]] MeasCovariance measurement_noise() const noexcept { return R_; }

    [[nodiscard]] auto jacobian(const StatePoint& /*x*/) const noexcept
        -> Eigen::Matrix<Scalar, MeasDim, StateDim> {
        // H = [I, 0]
        Eigen::Matrix<Scalar, MeasDim, StateDim> H =
            Eigen::Matrix<Scalar, MeasDim, StateDim>::Zero();
        H.template block<MeasDim, MeasDim>(0, 0).setIdentity();
        return H;
    }
};

// Type aliases for common configurations
using CV1D = ConstantVelocityModel<1>;
using CV2D = ConstantVelocityModel<2>;
using CV3D = ConstantVelocityModel<3>;

using PosMeas1D = PositionMeasurementModel<1>;
using PosMeas2D = PositionMeasurementModel<2>;
using PosMeas3D = PositionMeasurementModel<3>;

} // namespace uni_kf
