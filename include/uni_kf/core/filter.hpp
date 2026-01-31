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
#include "../strategies/sigma_points.hpp"
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
/// Different filter variants are achieved by template composition:
/// - D: The state space (Euclidean, LieManifold, ProductSpace)
/// - StrategyTag: Propagation strategy (FirstOrder, SigmaPoints, CubaturePoints)
/// - ReprTag: Uncertainty representation (Covariance, Information, SquareRoot)
///
/// Examples:
/// - EKF: FirstOrder × Covariance
/// - UKF: SigmaPoints × Covariance
/// - SRUKF: SigmaPoints × SquareRoot
/// - Information Filter: FirstOrder × Information
///
/// @tparam D DifferentialSpace for the state
/// @tparam StrategyTag Tag selecting propagation strategy
/// @tparam ReprTag Tag selecting uncertainty representation
/// @tparam Scalar_ Floating point type
template <
    DifferentialSpace D, typename StrategyTag = FirstOrderTag, typename ReprTag = CovarianceTag,
    typename Scalar_ = double>
class UnifiedKalmanFilter {
public:
    using Space      = D;
    using Scalar     = Scalar_;
    using Point      = typename D::Point;
    using Fiber      = typename D::Fiber;
    using BeliefType = Belief<D, ReprTag, Scalar>;
    using Covariance = SymmetricMatrix<D::FiberDim, Scalar>;

    static constexpr int Dim = D::FiberDim;

private:
    BeliefType belief_;
    bool initialized_ = false;

    // Numerical differentiation step size
    Scalar eps_ = Scalar{1e-7};

    // ------------------------------------------------------------------------
    // Helper: Compute Kalman gain and apply Joseph form update
    // ------------------------------------------------------------------------

    template <int MeasDim>
    [[nodiscard]] auto compute_kalman_gain(
        const Eigen::Matrix<Scalar, MeasDim, Dim>& H,
        const SymmetricMatrix<MeasDim, Scalar>& R) const noexcept
        -> std::expected<Eigen::Matrix<Scalar, Dim, MeasDim>, std::string> {
        auto P                                        = belief_.covariance();
        Eigen::Matrix<Scalar, MeasDim, MeasDim> S_mat = H * P.matrix() * H.transpose() + R.matrix();

        Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(S_mat);
        if (ldlt.info() != Eigen::Success) {
            return std::unexpected("Innovation covariance is singular");
        }

        Eigen::Matrix<Scalar, Dim, MeasDim> K =
            P.matrix() * H.transpose()
            * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());
        return K;
    }

    template <int MeasDim>
    void apply_joseph_update(
        const Eigen::Matrix<Scalar, Dim, MeasDim>& K, const Eigen::Matrix<Scalar, MeasDim, Dim>& H,
        const SymmetricMatrix<MeasDim, Scalar>& R,
        const typename Fiber::EigenType& correction) noexcept {
        // State update
        belief_.mean = D::act(belief_.mean, Fiber(correction));

        // Covariance update (Joseph form for numerical stability)
        auto P                               = belief_.covariance();
        typename Covariance::MatrixType I_KH = Covariance::MatrixType::Identity() - K * H;
        typename Covariance::MatrixType P_new =
            I_KH * P.matrix() * I_KH.transpose() + K * R.matrix() * K.transpose();

        belief_.uncertainty = typename BeliefType::Repr(Covariance(P_new));
    }

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

    [[nodiscard]] auto covariance() const noexcept { return belief_.covariance(); }

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

        // Wrapper that captures dt for dynamics function
        auto f = [&dynamics, dt](const Point& x) { return dynamics(x, dt); };

        auto P = belief_.covariance();

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            auto [x_new, P_new] = FirstOrder::predict<D>(belief_.mean, P, f, Q, eps_);
            belief_.mean        = x_new;
            belief_.uncertainty = typename BeliefType::Repr(P_new);
        } else if constexpr (std::is_same_v<StrategyTag, SigmaPointsTag>) {
            auto [x_new, P_new] = StandardUKF::predict<D>(belief_.mean, P, f, Q);
            belief_.mean        = x_new;
            belief_.uncertainty = typename BeliefType::Repr(P_new);
        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            auto [x_new, P_new] = CubaturePoints::predict<D>(belief_.mean, P, f, Q);
            belief_.mean        = x_new;
            belief_.uncertainty = typename BeliefType::Repr(P_new);
        } else {
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, SigmaPointsTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag, SigmaPointsTag, and CubaturePointsTag implemented");
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
        auto P = belief_.covariance();

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            if constexpr (ProcessModelWithJacobian<Model, D>) {
                // Use analytical Jacobian
                auto x_new = model.predict(belief_.mean, dt);
                auto F     = model.jacobian(belief_.mean, dt);
                auto P_new = FirstOrder::propagate_covariance<D>(P, F, Q);

                belief_.mean        = x_new;
                belief_.uncertainty = typename BeliefType::Repr(P_new);
            } else {
                // Use numerical Jacobian
                auto f              = [&model, dt](const Point& x) { return model.predict(x, dt); };
                auto [x_new, P_new] = FirstOrder::predict<D>(belief_.mean, P, f, Q, eps_);
                belief_.mean        = x_new;
                belief_.uncertainty = typename BeliefType::Repr(P_new);
            }
        } else if constexpr (std::is_same_v<StrategyTag, SigmaPointsTag>) {
            auto f              = [&model, dt](const Point& x) { return model.predict(x, dt); };
            auto [x_new, P_new] = SigmaPoints<>::predict<D>(belief_.mean, P, f, Q);
            belief_.mean        = x_new;
            belief_.uncertainty = typename BeliefType::Repr(P_new);
        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            auto f              = [&model, dt](const Point& x) { return model.predict(x, dt); };
            auto [x_new, P_new] = CubaturePoints::predict<D>(belief_.mean, P, f, Q);
            belief_.mean        = x_new;
            belief_.uncertainty = typename BeliefType::Repr(P_new);
        } else {
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, SigmaPointsTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag, SigmaPointsTag, and CubaturePointsTag implemented");
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
        auto P                = belief_.covariance();

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            // Compute predicted measurement and Jacobian
            auto z_pred = h(belief_.mean);
            auto H      = FirstOrder::jacobian<D, M>(belief_.mean, std::forward<MeasFunc>(h), eps_);
            auto innovation = M::diff(z_pred, z);

            // Compute Kalman gain and apply update
            auto K_result = compute_kalman_gain<MeasDim>(H, R);
            if (!K_result) {
                return;
            }

            typename Fiber::EigenType correction = (*K_result) * innovation.data;
            apply_joseph_update<MeasDim>(*K_result, H, R, correction);

        } else if constexpr (std::is_same_v<StrategyTag, SigmaPointsTag>) {
            auto [x_new, P_new] = StandardUKF::update<D, M>(belief_.mean, P, z, h, R);
            belief_.mean        = x_new;
            belief_.uncertainty = typename BeliefType::Repr(P_new);

        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            auto [z_pred, H_dummy, Pxy, Pzz] = CubaturePoints::measurement_covariances<D, M>(
                belief_.mean, P, std::forward<MeasFunc>(h), R);

            // Solve for Kalman gain
            Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(Pzz);
            if (ldlt.info() != Eigen::Success) {
                return;
            }

            Eigen::Matrix<Scalar, Dim, MeasDim> K =
                Pxy * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

            // Compute innovation and apply update
            auto innovation                      = M::diff(z_pred, z);
            typename Fiber::EigenType correction = K * innovation.data;
            belief_.mean                         = D::act(belief_.mean, Fiber(correction));

            // Simplified covariance update for CKF
            typename Covariance::MatrixType P_new = P.matrix() - K * Pzz * K.transpose();
            belief_.uncertainty                   = typename BeliefType::Repr(Covariance(P_new));

        } else {
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, SigmaPointsTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag, SigmaPointsTag, and CubaturePointsTag implemented");
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

        auto R = model.measurement_noise();
        auto P = belief_.covariance();

        if constexpr (std::is_same_v<StrategyTag, FirstOrderTag>) {
            // Compute predicted measurement
            auto z_pred = model.predict(belief_.mean);

            // Get Jacobian (analytical or numerical)
            Eigen::Matrix<Scalar, MeasDim, Dim> H;
            if constexpr (MeasurementModelWithJacobian<Model, D>) {
                H = model.jacobian(belief_.mean);
            } else {
                auto h = [&model](const Point& x) { return model.predict(x); };
                H      = FirstOrder::jacobian<D, M>(belief_.mean, h, eps_);
            }

            // Compute innovation and apply update
            auto innovation = M::diff(z_pred, z);
            auto K_result   = compute_kalman_gain<MeasDim>(H, R);
            if (!K_result) {
                return;
            }

            typename Fiber::EigenType correction = (*K_result) * innovation.data;
            apply_joseph_update<MeasDim>(*K_result, H, R, correction);

        } else if constexpr (std::is_same_v<StrategyTag, SigmaPointsTag>) {
            auto h              = [&model](const Point& x) { return model.predict(x); };
            auto [x_new, P_new] = StandardUKF::update<D, M>(belief_.mean, P, z, h, R);
            belief_.mean        = x_new;
            belief_.uncertainty = typename BeliefType::Repr(P_new);

        } else if constexpr (std::is_same_v<StrategyTag, CubaturePointsTag>) {
            auto h = [&model](const Point& x) { return model.predict(x); };
            auto [z_pred, H_dummy, Pxy, Pzz] =
                CubaturePoints::measurement_covariances<D, M>(belief_.mean, P, h, R);

            // Solve for Kalman gain
            Eigen::LDLT<Eigen::Matrix<Scalar, MeasDim, MeasDim>> ldlt(Pzz);
            if (ldlt.info() != Eigen::Success) {
                return;
            }

            Eigen::Matrix<Scalar, Dim, MeasDim> K =
                Pxy * ldlt.solve(Eigen::Matrix<Scalar, MeasDim, MeasDim>::Identity());

            // Compute innovation and apply update
            auto innovation                      = M::diff(z_pred, z);
            typename Fiber::EigenType correction = K * innovation.data;
            belief_.mean                         = D::act(belief_.mean, Fiber(correction));

            // Simplified covariance update for CKF
            typename Covariance::MatrixType P_new = P.matrix() - K * Pzz * K.transpose();
            belief_.uncertainty                   = typename BeliefType::Repr(Covariance(P_new));

        } else {
            static_assert(
                std::is_same_v<StrategyTag, FirstOrderTag>
                    || std::is_same_v<StrategyTag, SigmaPointsTag>
                    || std::is_same_v<StrategyTag, CubaturePointsTag>,
                "Only FirstOrderTag, SigmaPointsTag, and CubaturePointsTag implemented");
        }

        belief_.symmetrize();
        belief_.regularize();
    }

    /// @brief Direct measurement update for Euclidean measurement spaces
    ///
    /// Provides efficient interface when measurement space equals state space
    /// or uses a simple projection.
    template <int MeasDim>
    void update_direct(
        const Eigen::Matrix<Scalar, MeasDim, 1>& z, const Eigen::Matrix<Scalar, MeasDim, Dim>& H,
        const SymmetricMatrix<MeasDim, Scalar>& R) noexcept {
        if (!initialized_) {
            return;
        }

        // Predicted measurement and innovation
        Eigen::Matrix<Scalar, MeasDim, 1> z_pred     = H * belief_.mean.data;
        Eigen::Matrix<Scalar, MeasDim, 1> innovation = z - z_pred;

        // Compute Kalman gain and apply update
        auto K_result = compute_kalman_gain<MeasDim>(H, R);
        if (!K_result) {
            return;
        }

        apply_joseph_update<MeasDim>(*K_result, H, R, (*K_result) * innovation);
        belief_.symmetrize();
        belief_.regularize();
    }

    // ------------------------------------------------------------------------
    // Invariant update step (for InEKF)
    // ------------------------------------------------------------------------

    /// @brief Invariant measurement update for InEKF on Lie groups
    ///
    /// Uses body-frame invariant error instead of standard innovation:
    /// - Standard EKF: innovation = z - h(x)
    /// - InEKF: r = R^T (y - p) - b (body-frame error)
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

        // Compute invariant error and Jacobian
        auto r = model.invariant_error(belief_.mean, z);
        auto C = model.error_jacobian(belief_.mean);
        auto R = model.measurement_noise();

        // Compute Kalman gain and apply update
        auto K_result = compute_kalman_gain<MeasDim>(C, R);
        if (!K_result) {
            return;
        }

        typename Fiber::EigenType correction = (*K_result) * r.data;
        apply_joseph_update<MeasDim>(*K_result, C, R, correction);

        belief_.symmetrize();
        belief_.regularize();
    }

    /// @brief Invariant update with correction modifier
    ///
    /// Useful when you need to modify correction before applying (e.g., zero roll/pitch)
    template <typename Model, typename CorrectionModifier>
    requires InvariantMeasurementModel<Model, D> void update_invariant(
        const typename Model::MeasurementSpace::Point& z, const Model& model,
        CorrectionModifier&& modifier) noexcept {
        if (!initialized_) {
            return;
        }

        using M               = typename Model::MeasurementSpace;
        constexpr int MeasDim = M::FiberDim;

        // Compute invariant error and Jacobian
        auto r = model.invariant_error(belief_.mean, z);
        auto C = model.error_jacobian(belief_.mean);
        auto R = model.measurement_noise();

        // Compute Kalman gain
        auto K_result = compute_kalman_gain<MeasDim>(C, R);
        if (!K_result) {
            return;
        }

        // Compute correction and apply modifier
        typename Fiber::EigenType correction = (*K_result) * r.data;
        modifier(correction);

        // Apply update
        apply_joseph_update<MeasDim>(*K_result, C, R, correction);
        belief_.symmetrize();
        belief_.regularize();
    }

    // ------------------------------------------------------------------------
    // Linear update with correction modifier
    // ------------------------------------------------------------------------

    /// @brief Linear measurement update with pre-computed innovation and correction modifier
    ///
    /// Useful for spherical observation models where innovation is computed externally
    /// and correction needs to be modified (e.g., zero out roll/pitch for yaw-only tracking).
    ///
    /// @tparam MeasDim Measurement dimension
    /// @tparam CorrectionModifier Callable that modifies correction vector
    /// @param H Measurement Jacobian (MeasDim x Dim)
    /// @param nu Innovation vector (pre-computed: z - h(x))
    /// @param R Measurement noise covariance
    /// @param modifier Function to modify correction before applying (e.g., zero roll/pitch)
    template <int MeasDim, typename CorrectionModifier>
    void update_linear(
        const Eigen::Matrix<Scalar, MeasDim, Dim>& H, const Eigen::Matrix<Scalar, MeasDim, 1>& nu,
        const SymmetricMatrix<MeasDim, Scalar>& R, CorrectionModifier&& modifier) noexcept {
        if (!initialized_) {
            return;
        }

        // Compute Kalman gain
        auto K_result = compute_kalman_gain<MeasDim>(H, R);
        if (!K_result) {
            return;
        }

        // Compute correction and apply modifier
        typename Fiber::EigenType correction = (*K_result) * nu;
        modifier(correction);

        // Apply update
        apply_joseph_update<MeasDim>(*K_result, H, R, correction);
        belief_.symmetrize();
        belief_.regularize();
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
/// @tparam N State dimension
/// @tparam ReprTag Uncertainty representation (default: Covariance)
/// @tparam Scalar Floating point type
template <int N, typename ReprTag = CovarianceTag, typename Scalar = double>
using EKF = UnifiedKalmanFilter<Euclidean<N, Scalar>, FirstOrderTag, ReprTag, Scalar>;

/// @brief Extended Kalman Filter on a Lie group
template <
    typename G, LieSide Side = LieSide::Right, typename ReprTag = CovarianceTag,
    typename Scalar = double>
using LieEKF = UnifiedKalmanFilter<LieManifold<G, Side>, FirstOrderTag, ReprTag, Scalar>;

/// @brief Invariant EKF on SE₂(3)
template <LieSide Side = LieSide::Right, typename ReprTag = CovarianceTag, typename Scalar = double>
using InEKF_SE23 =
    UnifiedKalmanFilter<LieManifold<group::SEn3<Scalar, 2>, Side>, FirstOrderTag, ReprTag, Scalar>;

/// @brief Invariant CKF on SE₂(3) × ℝⁿ (augmented with extra states)
template <
    int ExtraDim, LieSide Side = LieSide::Right, typename ReprTag = CovarianceTag,
    typename Scalar = double>
using InUKF_SE23_Augmented =
    UnifiedKalmanFilter<SE23_Rn<ExtraDim, Scalar, Side>, SigmaPointsTag, ReprTag, Scalar>;

/// @brief Invariant CKF on SE₂(3) × ℝⁿ (augmented with extra states)
template <
    int ExtraDim, LieSide Side = LieSide::Right, typename ReprTag = CovarianceTag,
    typename Scalar = double>
using InCKF_SE23_Augmented =
    UnifiedKalmanFilter<SE23_Rn<ExtraDim, Scalar, Side>, CubaturePointsTag, ReprTag, Scalar>;

} // namespace uni_kf
