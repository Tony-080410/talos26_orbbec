#pragma once
#include "covariance.hpp"
#include "space.hpp"
#include <concepts>
#include <expected>
#include <type_traits>

namespace uni_kf::redact {

// ============================================================================
// Perception Error Enum
// ============================================================================

enum class PerceptError {
    R_not_spd,          // Measurement noise covariance R is not positive-definite
    sr_downdate_failed, // Rank-1 downdate failed (SR factor became indefinite)
    udu_update_failed   // Scalar update failed (UDU diagonal became non-positive)
};

// ============================================================================
// Perception Concept
// ============================================================================

/// @brief Perception: measurement model h(x) and Kalman gain computation
///
/// A Perception defines how measurements are related to state and provides
/// the Kalman gain for updating the state estimate.
template <typename T, typename Cov>
concept Perception =
    requires(T& t, typename T::Belief::Flux x, typename T::Measurement::Flux z, Cov& P) {
        typename T::Belief;
        typename T::Measurement;
        requires Space<typename T::Belief>;
        requires Space<typename T::Measurement>;

        /// Observation: z_pred = h(x)
        { t.assume(x) } -> std::same_as<typename T::Measurement::Flux>;

        /// Compute correction from innovation: Δx = K * innovation
        /// Internally calls P.compute_kalman_gain() to get K
        {
            t.extrude(P, std::declval<typename T::Measurement::Drift>())
        } -> std::same_as<typename T::Belief::Drift>;

        /// Get measurement noise covariance R
        { t.measurement_noise_covariance() } -> std::convertible_to<Eigen::MatrixXd>;

        /// Get measurement Jacobian H or J_h (for EKF/KF)
        /// For UKF/CKF, returns a linear approximation or empty
        { t.measurement_jacobian(x) } -> std::convertible_to<Eigen::MatrixXd>;
    };

// ============================================================================
// Assumption Concept
// ============================================================================

/// @brief Assumption: dynamics model f(x, dt) for state prediction
///
/// An Assumption defines how the state evolves over time.
template <typename T, typename Cov>
concept Assumption = requires(T& t, typename T::Perception::Belief::Flux x, double dt, Cov& P) {
    typename T::Perception;
    requires Perception<typename T::Perception, Cov>;

    /// State prediction: x_next = f(x, dt)
    { t.maneuver(P, x, dt) } -> std::same_as<typename T::Perception::Belief::Flux>;

    /// Get process noise covariance Q
    { t.process_noise_covariance() } -> std::convertible_to<Eigen::MatrixXd>;
};

// ============================================================================
// KalmanFilter
// ============================================================================

/// @brief Unified Kalman Filter supporting KF/EKF/UKF/CKF variants
///
/// The specific filter variant is determined by the Covariance type, which
/// encapsulates both the storage strategy (Direct/Joseph/SR/UDU) and the
/// propagation strategy (Linear/Linearized/Unscented/Cubature).
///
/// @tparam Covariance The covariance type (e.g., Covariance<Belief, Repr, Strategy>)
/// @tparam P The Perception type (measurement model)
/// @tparam A The Assumption type (dynamics model)
template <typename Covariance, Perception<Covariance> P, Assumption<Covariance> A>
class KalmanFilter {
public:
    using Belief      = typename P::Belief;
    using Measurement = typename P::Measurement;
    using MatrixDD    = Eigen::MatrixXd;

    static constexpr bool kSigmaPointStrategy =
        std::is_same_v<
            typename Covariance::StrategyType,
            UnscentedStrategy<Belief, typename Covariance::Storage>>
        || std::is_same_v<
            typename Covariance::StrategyType,
            CubatureStrategy<Belief, typename Covariance::Storage>>;

    static constexpr bool kCubatureStrategy = std::is_same_v<
        typename Covariance::StrategyType, CubatureStrategy<Belief, typename Covariance::Storage>>;

    /// @brief Default constructor
    KalmanFilter() = default;

    /// @brief Constructor with initial belief
    explicit KalmanFilter(
        typename Belief::Flux initial_belief, P perception = P{}, A assumption = A{})
        : belief(std::move(initial_belief))
        , unc()
        , perception(std::move(perception))
        , assumption(std::move(assumption)) {
        // Trigger covariance initialization by calling a dummy predict
        // This ensures the internal storage is properly sized
        [[maybe_unused]] auto dummy = predict(0.001);
    }

    /// @brief Constructor with initial belief and initial covariance
    explicit KalmanFilter(
        typename Belief::Flux initial_belief, MatrixDD initial_covariance, P perception = P{},
        A assumption = A{})
        : belief(std::move(initial_belief))
        , unc(Covariance(initial_covariance))
        , perception(std::move(perception))
        , assumption(std::move(assumption)) {}

    /// @brief Prediction step: x' = f(x, dt), P' = propagate(P, x, Q, dt)
    [[nodiscard]] typename Belief::Flux predict(double dt) noexcept {
        // Save old state before update (needed for adjoint transformation on SE(3))
        const typename Belief::Flux old_belief = belief;

        // Create dynamics lambda that captures assumption
        // This allows UKF/CKF to propagate sigma points through the dynamics model
        const auto dynamics_lambda = [&](const typename Belief::Flux& state, double delta_t) {
            return assumption.maneuver(unc, state, delta_t);
        };

        // State prediction: x' = f(x, dt)
        belief = assumption.maneuver(unc, belief, dt);

        // Covariance propagation: P' = propagate(P, x_old, x_new, dynamics, Q, dt)
        const MatrixDD Q = assumption.process_noise_covariance();
        unc.propagate(old_belief, belief, dynamics_lambda, Q, dt);

        return belief;
    }

    /// @brief Measurement update step
    /// @param meas The measurement z
    /// @param dt Time step (for covariance update)
    /// @return std::expected<void, PerceptError> success if update succeeded
    std::expected<void, PerceptError> percept(typename Measurement::Flux meas, double dt) {
        // Compute innovation: Δz = z - h(x)
        // diff(x, y) = y - x, so diff(predicted, meas) gives meas - predicted.
        const auto predicted_measurement = perception.assume(belief);
        const auto innovation            = Measurement::diff(predicted_measurement, meas);

        Eigen::VectorXd innovation_vec;
        if constexpr (requires { innovation.data; }) {
            innovation_vec = innovation.data;
        } else {
            innovation_vec = innovation;
        }

        const MatrixDD H = perception.measurement_jacobian(belief);
        const MatrixDD R = perception.measurement_noise_covariance();

        // --------------------------------------------------------------------
        // Compile-time dispatch by filter variant
        // --------------------------------------------------------------------
        if constexpr (kSigmaPointStrategy) {
            // Sigma-point filters (UKF/CKF): use sigma-point measurement transform.
            MatrixDD innovation_cov;
            const MatrixDD K =
                compute_sigma_point_kalman_gain(belief, predicted_measurement, R, &innovation_cov);

            typename Belief::Drift correction;
            if constexpr (requires { correction.data; }) {
                correction.data = K * innovation_vec;
            } else {
                correction = K * innovation_vec;
            }

            belief = Belief::compose(belief, correction);
            unc.stiffen_innovation(K, innovation_cov);
            return std::expected<void, PerceptError>{};
        } else if constexpr (
            Covariance::Storage::repr == Representation::Sr
            || Covariance::Storage::repr == Representation::Udu) {
            // True factorized measurement update for SR/UDU (KF/EKF-style)

            // Decorrelate measurements: L*L^T = R, so z' = L^{-1} z, H' = L^{-1} H, R' = I.
            Eigen::LLT<MatrixDD> lltR(R);
            if (lltR.info() != Eigen::Success) {
                // Measurement noise covariance R is not positive-definite
                return std::unexpected(PerceptError::R_not_spd);
            } else {
                const MatrixDD L = lltR.matrixL();
                const Eigen::VectorXd innov_decorr =
                    L.triangularView<Eigen::Lower>().solve(innovation_vec);
                const MatrixDD H_decorr = L.triangularView<Eigen::Lower>().solve(H);

                const int m                 = static_cast<int>(innov_decorr.size());
                const int n                 = unc.dimension();
                Eigen::VectorXd drift_total = Eigen::VectorXd::Zero(n);
                for (int i = 0; i < m; ++i) {
                    const Eigen::VectorXd h = H_decorr.row(i).transpose();

                    // Sequential scalar processing uses the residual w.r.t. the current
                    // accumulated linearized correction.
                    const double dz = innov_decorr(i) - h.dot(drift_total);

                    Eigen::VectorXd dx_vec(n);

                    if constexpr (Covariance::Storage::repr == Representation::Sr) {
                        auto& S = unc.storage().sr_factor();

                        // a = S^T * h
                        const Eigen::VectorXd a = S.transpose() * h;
                        const double s          = a.squaredNorm() + 1.0;

                        const Eigen::VectorXd u    = S * a;
                        const Eigen::VectorXd Kvec = u / s;

                        dx_vec = Kvec * dz;

                        const Eigen::VectorXd v = u / std::sqrt(s);
                        if (!unc.storage().sr_rank1_downdate(v)) {
                            // Rank-1 downdate failed - SR factor became indefinite
                            return std::unexpected(PerceptError::sr_downdate_failed);
                        }
                    } else {
                        // UDU
                        if (!unc.storage().udu_update_scalar(dz, 1.0, h, dx_vec)) {
                            // Scalar update failed - UDU diagonal became non-positive
                            return std::unexpected(PerceptError::udu_update_failed);
                        }
                    }

                    drift_total += dx_vec;
                }

                // Apply the total correction once (consistent with EKF linearization).
                typename Belief::Drift dx;
                if constexpr (requires { dx.data; }) {
                    dx.data = drift_total;
                } else {
                    dx = drift_total;
                }
                belief = Belief::compose(belief, dx);
                return std::expected<void, PerceptError>{};
            }
        } else {
            // Standard KF/EKF path (Direct storage, updater-controlled)
            const MatrixDD K = unc.compute_kalman_gain(H, R);

            typename Belief::Drift correction;
            if constexpr (requires { correction.data; }) {
                correction.data = K * innovation_vec;
            } else {
                correction = K * innovation_vec;
            }

            // For right-invariant error on Lie groups, the state update changes the
            // tangent space at which the covariance is defined.
            // For small corrections (typical in well-behaved systems), Ad ≈ I
            // So we use standard EKF update which assumes the tangent space doesn't change
            // significantly
            unc.stiffen(K, H, R, dt);
            belief = Belief::compose(belief, correction);
            return std::expected<void, PerceptError>{};
        }
    }

private:
    /// @brief Compute Kalman gain with sigma-point measurement transform for Sampleable spaces
    ///
    /// For UKF/CKF with Sampleable Belief, uses sigma-point transform in measurement space:
    /// - Transform state sigma points to measurement space
    /// - Compute measurement mean and covariance from transformed points
    /// - Compute cross-covariance between state and measurement
    ///
    /// @param state Current state estimate
    /// @param predicted_measurement Predicted measurement z_pred = h(x)
    /// @param H Measurement Jacobian (for non-sampleable fallback)
    /// @param R Measurement noise covariance
    /// @return Kalman gain K
    [[nodiscard]] MatrixDD compute_kalman_gain_impl(
        const typename Belief::Flux& state, const typename Measurement::Flux& predicted_measurement,
        const MatrixDD& H, const MatrixDD& R) const noexcept {

        if constexpr (kSigmaPointStrategy) {
            return compute_sigma_point_kalman_gain(state, predicted_measurement, R);
        }
        return unc.compute_kalman_gain(H, R);
    }

    /// @brief Sigma-point Kalman gain computation for UKF/CKF measurement update
    ///
    /// Computes K = P_xy * P_yy^-1 where:
    /// - P_yy = measurement covariance from sigma points
    /// - P_xy = cross-covariance between state and measurement
    ///
    /// **UKF Formulas** (Julier & Uhlmann 2004, Sec. III):
    ///   Lambda: λ = α²(n + κ) - n
    ///   Sigma points: χ₀ = x̄, χᵢ = x̄ ± √(n+λ)·S[:, i] for i=1..n
    ///   Weights:
    ///     w₀^m = λ/(n+λ)           (mean weight for center point)
    ///     w₀^c = w₀^m + (1-α²+β) (covariance weight for center point)
    ///     wᵢ = 1/(2(n+λ))          (weights for remaining points)
    ///
    /// **UKF Reference**:
    /// - Julier, S. J., & Uhlmann, J. K. (2004). "Unscented Filtering and Nonlinear Estimation."
    ///   *Proceedings of the IEEE*, 92(3), 401-422.
    ///   [PDF](https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf)
    ///   - Sec. III: The Unscented Transform
    ///   - Recommended: α=1e-3 (low dim), κ=0 (scaled UKF), β=2 (Gaussian)
    ///   - For n≥5: use α=1.0 to prevent negative lambda
    ///
    /// **CKF Reference**:
    /// - Arasaratnam, I., & Haykin, S. (2009). "Cubature Kalman Filters."
    ///   *IEEE Transactions on Automatic Control*, 54(6), 1254-1269.
    ///   [ResearchGate](https://www.researchgate.net/publication/224471882_Cubature_Kalman_Filters)
    ///   - Third-degree spherical-radial rule
    ///   - 2n points at distance ±√n from center
    ///   - Equal weights: w = 1/(2n)
    ///
    /// **Karcher Mean Reference** (for Lie group measurement spaces):
    /// - Lou, Y., et al. (2020). "Differentiating through the Fréchet Mean." ICML 2020.
    ///   [PDF](http://proceedings.mlr.press/v119/lou20a/lou20a-supp.pdf)
    ///   - Iteration: μ_{k+1} = μ_k · Exp(Σ w_i * Log(μ_k⁻¹ · z_i))
    [[nodiscard]] MatrixDD compute_sigma_point_kalman_gain(
        const typename Belief::Flux& state, const typename Measurement::Flux& predicted_measurement,
        const MatrixDD& R, MatrixDD* innovation_cov_out = nullptr) const noexcept {

        // Get dimensions from covariance matrix
        const MatrixDD current_cov = unc.mat();
        const int n                = current_cov.rows(); // State dimension
        const int m                = R.rows();           // Measurement dimension

        // Compute Cholesky of P (or scaled P for UKF)
        // For UnscentedStrategy, access UKF parameters from strategy object.
        // For CubatureStrategy, uses fixed spherical-radial weights (no alpha/beta/kappa).
        double alpha, beta, kappa;
        if constexpr (kCubatureStrategy) {
            alpha = 1.0;
            beta  = 0.0;
            kappa = 0.0;
        } else {
            alpha = unc.strategy().alpha;
            beta  = unc.strategy().beta;
            kappa = unc.strategy().kappa;
        }
        const double lambda = alpha * alpha * (n + kappa) - n;

        const double scale = kCubatureStrategy ? 1.0 : (n + lambda);
        Eigen::LLT<MatrixDD> llt(scale * current_cov);
        if (llt.info() != Eigen::Success) {
            // Compute standard K = P * H' * (H * P * H' + R)^-1
            const MatrixDD H              = perception.measurement_jacobian(state);
            const MatrixDD innovation_cov = H * current_cov * H.transpose() + R;
            const MatrixDD PHt            = current_cov * H.transpose();
            const auto ldlt               = innovation_cov.ldlt();
            if (ldlt.info() == Eigen::Success) {
                return ldlt.solve(PHt.transpose()).transpose();
            }
            return innovation_cov.completeOrthogonalDecomposition()
                .solve(PHt.transpose())
                .transpose();
        }
        const MatrixDD S = llt.matrixL();

        // Generate state sigma points
        std::vector<typename Belief::Flux> state_points;
        std::vector<double> weights_mean;
        std::vector<double> weights_cov;

        if constexpr (kCubatureStrategy) {
            const double w = 1.0 / (2.0 * static_cast<double>(n));
            const double s = std::sqrt(static_cast<double>(n));
            for (int i = 0; i < n; ++i) {
                typename Belief::Drift xi = Belief::zero_drift();
                if constexpr (requires { xi.data; }) {
                    xi.data = s * S.col(i);
                } else {
                    xi = s * S.col(i);
                }
                state_points.push_back(Belief::compose(state, xi));
                weights_mean.push_back(w);
                weights_cov.push_back(w);

                if constexpr (requires { xi.data; }) {
                    xi.data = -xi.data;
                } else {
                    xi = -xi;
                }
                state_points.push_back(Belief::compose(state, xi));
                weights_mean.push_back(w);
                weights_cov.push_back(w);
            }
        } else {
            // UKF weights (Julier & Uhlmann 2004, Sec. III)
            const double w0_m = lambda / (n + lambda);
            const double w0_c = w0_m + (1 - alpha * alpha + beta);
            const double wi   = 1.0 / (2 * (n + lambda));

            // Point 0: mean state
            state_points.push_back(state);
            weights_mean.push_back(w0_m);
            weights_cov.push_back(w0_c);

            // Points ± columns of S
            for (int i = 0; i < n; ++i) {
                typename Belief::Drift xi = Belief::zero_drift();
                if constexpr (requires { xi.data; }) {
                    xi.data = S.col(i);
                } else {
                    xi = S.col(i);
                }

                state_points.push_back(Belief::compose(state, xi));
                weights_mean.push_back(wi);
                weights_cov.push_back(wi);

                if constexpr (requires { xi.data; }) {
                    xi.data = -S.col(i);
                } else {
                    xi = -S.col(i);
                }
                state_points.push_back(Belief::compose(state, xi));
                weights_mean.push_back(wi);
                weights_cov.push_back(wi);
            }
        }

        // Transform sigma points to measurement space through perception model
        std::vector<typename Measurement::Flux> meas_points;
        meas_points.reserve(state_points.size());
        for (const auto& sp : state_points) {
            meas_points.push_back(perception.assume(sp));
        }

        // Compute measurement mean from transformed sigma points
        typename Measurement::Flux y_pred;

        // Constraint: measurement must be either Euclidean or Lie-group-like
        static_assert(
            requires { y_pred.data; } ||
            (requires { Measurement::identity(); } &&
             requires { Measurement::compose(y_pred, Measurement::zero_drift()); }),
            "Sigma-point measurement mean requires either Euclidean space (Flux has .data) "
            "or Lie-group space (supports identity, compose, zero_drift, and diff).");

        if constexpr (requires { y_pred.data; }) {
            // Measurement space is Euclidean: use closed-form weighted average
            y_pred.data = weights_mean[0] * meas_points[0].data;
            for (size_t i = 1; i < meas_points.size(); ++i) {
                y_pred.data += weights_mean[i] * meas_points[i].data;
            }
        } else if constexpr (
            requires { Measurement::identity(); }
            && requires { Measurement::compose(y_pred, Measurement::zero_drift()); }) {
            // Measurement space is LieGroup: use iterative logarithmic averaging (Karcher mean)
            // This is the proper Riemannian mean on manifolds
            //
            // Reference:
            // - Lou, Y., et al. (2020). "Differentiating through the Fréchet Mean."
            //   *ICML 2020*. [PDF](http://proceedings.mlr.press/v119/lou20a/lou20a-supp.pdf)
            // - Tron, R., et al. (2011). "Fast and scalable Karcher means."
            //   *EUSIPCO 2011*.
            // Iteration: μ_{k+1} = μ_k · Exp(Σ w_i · Log(μ_k⁻¹ · z_i))
            // 2-3 iterations sufficient for most applications
            y_pred = meas_points[0];

            for (size_t iter = 0; iter < 3; ++iter) {
                // Compute weighted average of logarithms in tangent space
                typename Measurement::Drift step = Measurement::zero_drift();
                for (size_t i = 0; i < meas_points.size(); ++i) {
                    const auto delta = Measurement::diff(y_pred, meas_points[i]);
                    if constexpr (requires { step.data; }) {
                        step.data += static_cast<double>(weights_mean[i]) * delta.data;
                    } else {
                        step += static_cast<double>(weights_mean[i]) * delta;
                    }
                }

                // Update: mu <- mu ⊕ step
                y_pred = Measurement::compose(y_pred, step);
            }
        }

        // Compute measurement covariance P_yy and cross-covariance P_xy
        MatrixDD P_yy = MatrixDD::Zero(m, m);
        MatrixDD P_xy = MatrixDD::Zero(n, m);

        for (size_t i = 0; i < state_points.size(); ++i) {
            // Measurement deviation: z_i - ẑ
            typename Measurement::Drift z_diff = Measurement::diff(y_pred, meas_points[i]);

            // State deviation: x_i - x̂ (in tangent space)
            typename Belief::Drift x_diff = Belief::diff(state, state_points[i]);

            // Extract vectors from Drift types
            Eigen::VectorXd z_vec;
            if constexpr (requires { z_diff.data; }) {
                z_vec = z_diff.data;
            } else {
                z_vec = z_diff;
            }

            Eigen::VectorXd x_vec;
            if constexpr (requires { x_diff.data; }) {
                x_vec = x_diff.data;
            } else {
                x_vec = x_diff;
            }

            P_yy += weights_cov[i] * (z_vec * z_vec.transpose());
            P_xy += weights_cov[i] * (x_vec * z_vec.transpose());
        }

        // Add measurement noise
        P_yy += R;

        if (innovation_cov_out) {
            *innovation_cov_out = P_yy;
        }

        // Compute Kalman gain: K = P_xy * P_yy^-1
        // Avoid explicit inverse: solve P_yy * X = P_xy^T, then K = X^T
        const auto ldlt = P_yy.ldlt();
        if (ldlt.info() == Eigen::Success) {
            return ldlt.solve(P_xy.transpose()).transpose();
        }
        return P_yy.completeOrthogonalDecomposition().solve(P_xy.transpose()).transpose();
    }

public:
    /// @brief Get current belief (state estimate)
    [[nodiscard]] const typename Belief::Flux& state() const noexcept { return belief; }

    /// @brief Get current covariance matrix
    [[nodiscard]] MatrixDD covariance() const noexcept { return unc.mat(); }

    /// @brief Get covariance object (for advanced use)
    [[nodiscard]] const Covariance& get_unc() const noexcept { return unc; }
    [[nodiscard]] Covariance& get_unc() noexcept { return unc; }

    /// @brief Get perception (measurement model)
    [[nodiscard]] const P& get_perception() const noexcept { return perception; }
    [[nodiscard]] P& get_perception() noexcept { return perception; }

    /// @brief Get assumption (dynamics model)
    [[nodiscard]] const A& get_assumption() const noexcept { return assumption; }
    [[nodiscard]] A& get_assumption() noexcept { return assumption; }

private:
    typename Belief::Flux belief;
    Covariance unc;
    P perception;
    A assumption;
};

} // namespace uni_kf::redact
