/// @file filter.hpp
/// @brief Unified Kalman Filter - the main filter class
///
/// UnifiedKalmanFilter<D, Strategy> provides a generic Kalman filter that works
/// on any DifferentialSpace with pluggable propagation strategies.

#pragma once

#include "../spaces/lie_manifold.hpp"
#include "../spaces/product_space.hpp"
#include "../strategies/cubature_points.hpp"
#include "../strategies/first_order.hpp"
#include "belief.hpp"
#include "models.hpp"

namespace uni_kf {

// ============================================================================
// PropagationStrategy tag types
// ============================================================================

/// @brief Tag for first-order (EKF-style) propagation
struct FirstOrderTag {};

/// @brief Tag for sigma-point (UKF-style) propagation
struct SigmaPointsTag {};

/// @brief Tag for cubature-point (CKF-style) propagation
struct CubaturePointsTag {};

// ============================================================================
// UnifiedKalmanFilter
// ============================================================================

/// @brief Unified Kalman Filter on DifferentialSpaces
///
/// This is the core filter class that works on any DifferentialSpace.
/// Different filter variants (EKF, UKF, InEKF) are achieved by:
/// - D: The state space (Euclidean for EKF, LieManifold for InEKF)
/// - Strategy: Propagation strategy (FirstOrder or SigmaPoints)
///
/// @tparam D DifferentialSpace for the state
/// @tparam StrategyTag Tag selecting propagation strategy
/// @tparam Scalar_ Floating point type
template <DifferentialSpace D, typename StrategyTag = FirstOrderTag, typename Scalar_ = double>
class UnifiedKalmanFilter {
public:
    using Space      = D;
    using Scalar     = Scalar_;
    using Point      = typename D::Point;
    using Fiber      = typename D::Fiber;
    using BeliefType = Belief<D, Scalar>;
    using Covariance = typename BeliefType::Covariance;

    static constexpr int Dim = D::FiberDim;

private:
    BeliefType belief_;
    bool initialized_ = false;

    // Numerical differentiation step size
    Scalar eps_ = Scalar{1e-7};

public:
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    UnifiedKalmanFilter() noexcept = default;

    explicit UnifiedKalmanFilter(const BeliefType& belief) noexcept
        : belief_(belief)
        , initialized_(true) {}

    UnifiedKalmanFilter(const Point& mean, const Covariance& cov) noexcept
        : belief_(mean, cov)
        , initialized_(true) {}

    // ------------------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------------------

    /// @brief Initialize filter with mean and covariance
    void init(const Point& mean, const Covariance& cov) noexcept {
        belief_      = BeliefType{mean, cov};
        initialized_ = true;
    }

    /// @brief Initialize at identity with given covariance
    void init_at_identity(const Covariance& cov) noexcept {
        belief_      = BeliefType{D::identity(), cov};
        initialized_ = true;
    }

    /// @brief Initialize with belief
    void init(const BeliefType& belief) noexcept {
        belief_      = belief;
        initialized_ = true;
    }

    /// @brief Reset filter to uninitialized state
    void reset() noexcept {
        belief_      = BeliefType{};
        initialized_ = false;
    }

    // ------------------------------------------------------------------------
    // State access
    // ------------------------------------------------------------------------

    [[nodiscard]] const BeliefType& belief() const noexcept { return belief_; }

    [[nodiscard]] BeliefType& belief() noexcept { return belief_; }

    [[nodiscard]] const Point& mean() const noexcept { return belief_.mean; }

    [[nodiscard]] const Covariance& covariance() const noexcept { return belief_.covariance; }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    // ------------------------------------------------------------------------
    // Prediction step
    // ------------------------------------------------------------------------

    /// @brief Predict step with dynamics function and process noise
    ///
    /// @param dynamics State transition function x' = f(x, dt)
    /// @param Q Process noise covariance
    /// @param dt Time step (passed to dynamics)
    template <typename Dynamics>
    void predict(Dynamics&& dynamics, const Covariance& Q, double dt) noexcept {
        if (!initialized_) {
            return;
        }

        // Create dynamics wrapper that captures dt
        auto f = [&dynamics, dt](const Point& x) { return dynamics(x, dt); };

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            // First-order (EKF) prediction
            auto [x_new, P_new] =
                FirstOrder::predict<D>(belief_.mean, belief_.covariance, f, Q, eps_);
            belief_.mean       = x_new;
            belief_.covariance = P_new;
        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            // Cubature points (CKF) prediction
            auto [x_new, P_new] =
                CubaturePoints::predict<D>(belief_.mean, belief_.covariance, f, Q);
            belief_.mean       = x_new;
            belief_.covariance = P_new;
        } else {
            // Sigma points (UKF) prediction - would use SigmaPoints strategy
            // TODO: Implement when SigmaPointsTag is used
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag and CubaturePointsTag implemented");
        }

        belief_.symmetrize();
        belief_.regularize();
    }

    /// @brief Predict step with ProcessModel
    template <typename Model>
    requires ProcessModel<Model, D> void predict(const Model& model, double dt) noexcept {
        if (!initialized_) {
            return;
        }

        auto Q = model.process_noise(dt);

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            // First-order (EKF) strategy
            if constexpr (ProcessModelWithJacobian<Model, D>) {
                // Use analytical Jacobian if available
                auto x_new = model.predict(belief_.mean, dt);
                auto F     = model.jacobian(belief_.mean, dt);
                auto P_new = FirstOrder::propagate_covariance<D>(belief_.covariance, F, Q);

                belief_.mean       = x_new;
                belief_.covariance = P_new;
            } else {
                // Numerical Jacobian
                auto f = [&model, dt](const Point& x) { return model.predict(x, dt); };
                auto [x_new, P_new] =
                    FirstOrder::predict<D>(belief_.mean, belief_.covariance, f, Q, eps_);
                belief_.mean       = x_new;
                belief_.covariance = P_new;
            }
        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            // Cubature points (CKF) strategy - always uses sampling
            auto f = [&model, dt](const Point& x) { return model.predict(x, dt); };
            auto [x_new, P_new] =
                CubaturePoints::predict<D>(belief_.mean, belief_.covariance, f, Q);
            belief_.mean       = x_new;
            belief_.covariance = P_new;
        } else {
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag and CubaturePointsTag implemented");
        }

        belief_.symmetrize();
        belief_.regularize();
    }

    // ------------------------------------------------------------------------
    // Update step
    // ------------------------------------------------------------------------

    /// @brief Update step with measurement and measurement function
    ///
    /// @tparam M Measurement DifferentialSpace
    /// @param z Actual measurement
    /// @param h Measurement function x → z
    /// @param R Measurement noise covariance
    template <DifferentialSpace M, typename MeasFunc>
    void update(
        const typename M::Point& z, MeasFunc&& h,
        const SymmetricMatrix<M::FiberDim, Scalar>& R) noexcept {
        if (!initialized_) {
            return;
        }

        constexpr int MeasDim = M::FiberDim;

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            // EKF update using Jacobian
            // Predicted measurement
            auto z_pred = h(belief_.mean);

            // Measurement Jacobian (numerical)
            auto H = FirstOrder::jacobian<D, M>(belief_.mean, std::forward<MeasFunc>(h), eps_);

            // Innovation
            auto innovation = M::diff(z_pred, z);

            // Innovation covariance: S = H P H^T + R
            typename Covariance::MatrixType P             = belief_.covariance.matrix();
            Eigen::Matrix<Scalar, MeasDim, MeasDim> S_mat = H * P * H.transpose() + R.matrix();

            // Solve for Kalman gain: K = P H^T S^{-1}
            Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(S_mat);
            if (ldlt.info() != Eigen::Success) {
                return; // Skip update if innovation covariance is singular
            }

            Eigen::Matrix<Scalar, Dim, MeasDim> K =
                P * H.transpose() * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

            // State update in tangent space
            typename Fiber::EigenType delta = K * innovation.data;
            belief_.mean                    = D::act(belief_.mean, Fiber(delta));

            // Covariance update (Joseph form for numerical stability)
            typename Covariance::MatrixType I_KH = Covariance::MatrixType::Identity() - K * H;
            typename Covariance::MatrixType P_new =
                I_KH * P * I_KH.transpose() + K * R.matrix() * K.transpose();

            belief_.covariance = Covariance(P_new);
        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            // CKF update using cubature points
            auto [H_dummy, Pxy, Pzz] = CubaturePoints::measurement_covariances<D, M>(
                belief_.mean, belief_.covariance, std::forward<MeasFunc>(h), R);

            // Solve for Kalman gain: K = Pxy Pzz^{-1}
            Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(Pzz);
            if (ldlt.info() != Eigen::Success) {
                return;
            }

            Eigen::Matrix<Scalar, Dim, MeasDim> K =
                Pxy * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

            // Predicted measurement for innovation
            auto z_pred     = h(belief_.mean);
            auto innovation = M::diff(z_pred, z);

            // State update
            typename Fiber::EigenType delta = K * innovation.data;
            belief_.mean                    = D::act(belief_.mean, Fiber(delta));

            // Covariance update (simplified form for CKF)
            typename Covariance::MatrixType P     = belief_.covariance.matrix();
            typename Covariance::MatrixType P_new = P - K * Pzz * K.transpose();

            belief_.covariance = Covariance(P_new);
        } else {
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag and CubaturePointsTag implemented");
        }

        belief_.symmetrize();
        belief_.regularize();
    }

    /// @brief Update step with MeasurementModel
    template <typename Model>
    requires MeasurementModel<Model, D>
    void update(const typename Model::MeasurementSpace::Point& z, const Model& model) noexcept {
        if (!initialized_) {
            return;
        }

        using M               = typename Model::MeasurementSpace;
        constexpr int MeasDim = M::FiberDim;

        // Measurement noise
        auto R = model.measurement_noise();

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            // EKF update using Jacobian
            // Predicted measurement
            auto z_pred = model.predict(belief_.mean);

            // Measurement Jacobian
            Eigen::Matrix<Scalar, MeasDim, Dim> H;
            if constexpr (MeasurementModelWithJacobian<Model, D>) {
                H = model.jacobian(belief_.mean);
            } else {
                auto h = [&model](const Point& x) { return model.predict(x); };
                H      = FirstOrder::jacobian<D, M>(belief_.mean, h, eps_);
            }

            // Innovation
            auto innovation = M::diff(z_pred, z);

            // Innovation covariance: S = H P H^T + R
            typename Covariance::MatrixType P             = belief_.covariance.matrix();
            Eigen::Matrix<Scalar, MeasDim, MeasDim> S_mat = H * P * H.transpose() + R.matrix();

            // Solve for Kalman gain
            Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(S_mat);
            if (ldlt.info() != Eigen::Success) {
                return;
            }

            Eigen::Matrix<Scalar, Dim, MeasDim> K =
                P * H.transpose() * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

            // State update
            typename Fiber::EigenType delta = K * innovation.data;
            belief_.mean                    = D::act(belief_.mean, Fiber(delta));

            // Covariance update (Joseph form)
            typename Covariance::MatrixType I_KH = Covariance::MatrixType::Identity() - K * H;
            typename Covariance::MatrixType P_new =
                I_KH * P * I_KH.transpose() + K * R.matrix() * K.transpose();

            belief_.covariance = Covariance(P_new);
        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            // CKF update using cubature points
            auto h                   = [&model](const Point& x) { return model.predict(x); };
            auto [H_dummy, Pxy, Pzz] = CubaturePoints::measurement_covariances<D, M>(
                belief_.mean, belief_.covariance, h, R);

            // Solve for Kalman gain: K = Pxy Pzz^{-1}
            Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(Pzz);
            if (ldlt.info() != Eigen::Success) {
                return;
            }

            Eigen::Matrix<Scalar, Dim, MeasDim> K =
                Pxy * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

            // Predicted measurement for innovation
            auto z_pred     = model.predict(belief_.mean);
            auto innovation = M::diff(z_pred, z);

            // State update
            typename Fiber::EigenType delta = K * innovation.data;
            belief_.mean                    = D::act(belief_.mean, Fiber(delta));

            // Covariance update
            typename Covariance::MatrixType P     = belief_.covariance.matrix();
            typename Covariance::MatrixType P_new = P - K * Pzz * K.transpose();

            belief_.covariance = Covariance(P_new);
        } else {
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag and CubaturePointsTag implemented");
        }

        belief_.symmetrize();
        belief_.regularize();
    }

    /// @brief Direct measurement update (for Euclidean measurement spaces)
    ///
    /// When measurement space equals state space or a simple projection,
    /// this provides a more efficient interface.
    template <int MeasDim>
    void update_direct(
        const Eigen::Matrix<Scalar, MeasDim, 1>& z, const Eigen::Matrix<Scalar, MeasDim, Dim>& H,
        const SymmetricMatrix<MeasDim, Scalar>& R) noexcept {
        if (!initialized_) {
            return;
        }

        // Predicted measurement
        Eigen::Matrix<Scalar, MeasDim, 1> z_pred = H * belief_.mean.data;

        // Innovation
        Eigen::Matrix<Scalar, MeasDim, 1> innovation = z - z_pred;

        // Innovation covariance
        typename Covariance::MatrixType P         = belief_.covariance.matrix();
        Eigen::Matrix<Scalar, MeasDim, MeasDim> S = H * P * H.transpose() + R.matrix();

        // Kalman gain
        Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            return;
        }

        Eigen::Matrix<Scalar, Dim, MeasDim> K =
            P * H.transpose() * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

        // Update
        typename Fiber::EigenType delta = K * innovation;
        belief_.mean                    = D::act(belief_.mean, Fiber(delta));

        typename Covariance::MatrixType I_KH = Covariance::MatrixType::Identity() - K * H;
        typename Covariance::MatrixType P_new =
            I_KH * P * I_KH.transpose() + K * R.matrix() * K.transpose();

        belief_.covariance = Covariance(P_new);
        belief_.symmetrize();
        belief_.regularize();
    }

    // ------------------------------------------------------------------------
    // Invariant update step (for InEKF)
    // ------------------------------------------------------------------------

    /// @brief Invariant measurement update (for InEKF on Lie groups)
    ///
    /// InEKF uses body-frame invariant error instead of standard innovation:
    /// - Standard EKF: innovation = z - h(x)
    /// - InEKF: r = R^T (y - p) - b (body-frame error)
    ///
    /// The model provides:
    /// - invariant_error(x, z): computes body-frame error r
    /// - error_jacobian(x): Jacobian C of error w.r.t. error state
    /// - measurement_noise(): R matrix
    ///
    /// @tparam Model InvariantMeasurementModel
    /// @param z Measurement (world-frame position)
    /// @param model Model providing invariant error and Jacobian
    template <typename Model>
    requires InvariantMeasurementModel<Model, D> void update_invariant(
        const typename Model::MeasurementSpace::Point& z, const Model& model) noexcept {
        if (!initialized_) {
            return;
        }

        using M               = typename Model::MeasurementSpace;
        constexpr int MeasDim = M::FiberDim;

        // Compute invariant error (body-frame)
        auto r = model.invariant_error(belief_.mean, z);

        // Error Jacobian C
        auto C = model.error_jacobian(belief_.mean);

        // Measurement noise
        auto R = model.measurement_noise();

        // Innovation covariance: S = C P C^T + R
        typename Covariance::MatrixType P             = belief_.covariance.matrix();
        Eigen::Matrix<Scalar, MeasDim, MeasDim> S_mat = C * P * C.transpose() + R.matrix();

        // Solve for Kalman gain: K = P C^T S^{-1}
        Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            return; // Skip update if S is singular
        }

        Eigen::Matrix<Scalar, Dim, MeasDim> K =
            P * C.transpose() * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

        // State update: δ = K r, then retract
        typename Fiber::EigenType delta = K * r.data;
        belief_.mean                    = D::act(belief_.mean, Fiber(delta));

        // Covariance update (Joseph form for numerical stability)
        typename Covariance::MatrixType I_KC = Covariance::MatrixType::Identity() - K * C;
        typename Covariance::MatrixType P_new =
            I_KC * P * I_KC.transpose() + K * R.matrix() * K.transpose();

        belief_.covariance = Covariance(P_new);
        belief_.symmetrize();
        belief_.regularize();
    }

    /// @brief Invariant update with post-processing callback
    ///
    /// Useful when you need to apply constraints after update (e.g., zero roll/pitch)
    template <typename Model, typename PostProcess>
    requires InvariantMeasurementModel<Model, D> void update_invariant(
        const typename Model::MeasurementSpace::Point& z, const Model& model,
        PostProcess&& post_process) noexcept {
        update_invariant(z, model);
        if (initialized_) {
            post_process(belief_);
        }
    }

    // ------------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------------

    /// @brief Set numerical differentiation step size
    void set_eps(Scalar eps) noexcept { eps_ = eps; }

    [[nodiscard]] Scalar eps() const noexcept { return eps_; }
};

// ============================================================================
// Type aliases for common configurations
// ============================================================================

/// @brief Extended Kalman Filter on Euclidean space
template <int N, typename Scalar = double>
using EKF = UnifiedKalmanFilter<Euclidean<N, Scalar>, FirstOrderTag, Scalar>;

/// @brief Extended Kalman Filter on a Lie group
template <typename G, LieSide Side = LieSide::Right, typename Scalar = double>
using LieEKF = UnifiedKalmanFilter<LieManifold<G, Side>, FirstOrderTag, Scalar>;

/// @brief Invariant EKF on SE₂(3)
template <LieSide Side = LieSide::Right, typename Scalar = double>
using InEKF_SE23 =
    UnifiedKalmanFilter<LieManifold<group::SEn3<Scalar, 2>, Side>, FirstOrderTag, Scalar>;

/// @brief Invariant CKF on SE₂(3) × ℝⁿ (augmented with extra states)
template <int ExtraDim, LieSide Side = LieSide::Right, typename Scalar = double>
using InCKF_SE23_Augmented =
    UnifiedKalmanFilter<SE23_Rn<ExtraDim, Scalar, Side>, CubaturePointsTag, Scalar>;

} // namespace uni_kf
