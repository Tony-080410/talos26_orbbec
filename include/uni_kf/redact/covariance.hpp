#pragma once
#include "factorization/cholupdate.hpp"
#include "lie_ops.hpp"
#include <Eigen/Dense>
#include <concepts>
#include <functional>
#include <type_traits>
#include <unsupported/Eigen/MatrixFunctions>

namespace uni_kf::redact {

namespace detail {

template <typename...>
inline constexpr bool kAlwaysFalseV = false;

template <typename LieGroup>
concept HasDiscreteDynamicsJacobian = requires(const typename LieGroup::VectorType& x, double dt) {
    { LieGroup::dynamics_jacobian_discrete(x, dt) } -> std::convertible_to<Eigen::MatrixXd>;
};

template <typename LieGroup>
concept HasContinuousDynamicsJacobian = requires(const typename LieGroup::VectorType& x) {
    { LieGroup::dynamics_jacobian_continuous(x) } -> std::convertible_to<Eigen::MatrixXd>;
};

// Back-compat: legacy ambiguous API. If you still use this, it is interpreted as DISCRETE Φ.
template <typename LieGroup>
concept HasLegacyDynamicsJacobian = requires(const typename LieGroup::VectorType& x, double dt) {
    { LieGroup::dynamics_jacobian(x, dt) } -> std::convertible_to<Eigen::MatrixXd>;
};

template <typename LieGroup>
[[nodiscard]] inline Eigen::MatrixXd
    discrete_transition_jacobian(const typename LieGroup::VectorType& x, double dt) {
    if constexpr (HasDiscreteDynamicsJacobian<LieGroup>) {
        return LieGroup::dynamics_jacobian_discrete(x, dt);
    } else if constexpr (HasContinuousDynamicsJacobian<LieGroup>) {
        const Eigen::MatrixXd A = LieGroup::dynamics_jacobian_continuous(x);
        if (dt == 0.0) {
            return Eigen::MatrixXd::Identity(A.rows(), A.cols());
        }
        // Aggressive/strict discretization: Φ = expm(A·dt).
        return (A * dt).exp();
    } else {
        static_assert(
            HasLegacyDynamicsJacobian<LieGroup>,
            "LieGroup must provide `dynamics_jacobian_discrete(x, dt)` or "
            "`dynamics_jacobian_continuous(x)` (or legacy `dynamics_jacobian(x, dt)`).");
        return LieGroup::dynamics_jacobian(x, dt);
    }
}

} // namespace detail

/// @file covariance.hpp
/// @brief Covariance propagation and update strategies for the unified Kalman filter
///
/// ## Mathematical Formulation
///
/// This file implements covariance propagation strategies for various Kalman filter
/// variants: KF, EKF, UKF, CKF, InKF, and InEKF.
///
/// ### Standard Kalman Filter Formulation
///
/// **Prediction**:
///   x' = F · x
///   P' = F · P · F^T + Q
///
/// **Measurement Update**:
///   K = P · H^T · (H · P · H^T + R)^(-1)
///   x' = x + K · (z - H · x)
///   P' = (I - K · H) · P
///
/// ### Invariant Extended Kalman Filter (InEKF) Formulation
///
/// For right-invariant error definition η_R = Log(X̂⁻¹X) on Lie groups:
///
/// **Prediction** (Hartley 2019 form for general nonlinear dynamics):
///   X̂' = f(X̂, dt)                // State prediction on manifold
///   Φ = expm(A_t·dt)                // Discretized state transition
///   P' = Φ · P · Φ^T + Q̂_d         // Covariance propagation (error coordinates)
///
/// Continuous-time Riccati equation (Hartley RSS 2019, Eq. 7):
///   d/dt P_t = A_t P_t + P_t A_t^T + Q̂_t
///
/// Discrete-time solution:
///   Φ = expm(A_t·Δt)
///   P_{k+1} = Φ P_k Φ^T + Q̂_d
///
/// Note: For group-affine systems (Bonnabel 2017 Eq. 20b): P' = F·P·F^T + Q (no Adjoint)
/// For general nonlinear systems (Hartley 2019): P' = Φ·P·Φ^T + Q (Adjoint only affects Q, not P)
///
/// **Measurement Update**:
///   r = Log(ẑ⁻¹ · z)              // Residual in tangent space
///   K = P · H^T · (H · P · H^T + R)^(-1)
///   X̂ = X̂ ⊕ Exp(K · r)           // State update (retraction)
///   P' = (I - K · H) · P
///
/// where:
///   - F is the dynamics Jacobian in tangent space
///   - H is the measurement Jacobian
///   - ⊕ denotes compose (state retraction)
///   - Log denotes logarithm map (manifold → tangent)
///   - Exp denotes exponential map (tangent → manifold)
///   - Ad_X is the Adjoint operator at X
///
/// ### References
///
/// 1. **Barrau, A., & Bonnabel, S.** (2017).
///    "The Invariant Extended Kalman Filter as a Stable Observer."
///    *IEEE Transactions on Automatic Control*, 62(5), 1797-1812.
///    DOI: [10.1109/TAC.2016.2594083](https://doi.org/10.1109/TAC.2016.2594083)
///    arXiv: [1410.1465](https://arxiv.org/pdf/1410.1465)
///    - Defines group-affine systems and proves stability
///    - Eq. (20b): Covariance propagation for group-affine systems
///
/// 2. **Barfoot, T. D.** (2024).
///    *State Estimation for Robotics* (2nd ed.).
///    Cambridge University Press.
///    [PDF](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf)
///    - Chapter 9: The Batch Estimator (for EKF derivation)
///    - Chapter 10: The Kalman Filter
///    - Section 11.3: InEKF on manifolds
///    - Appendix E: The Matrix Determinant Lemma
///
/// 3. **Hartley, J., Ghaffari, M., Eustice, R. M., & Grizzle, J. W.** (2019).
///    "Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State Estimation."
///    *Robotics: Science and Systems* (RSS 2019).
///    [PDF](https://www.roboticsproceedings.org/rss14/p50.pdf)
///    - Eq. (7): Continuous-time Riccati equation: d/dt P_t = A_t P_t + P_t A_t^T + Q̂_t
///    - Discrete-time approximation: P_{k+1} = Φ P_k Φ^T + Q̂_k where Φ = exp_m(A_t Δt) ≈ I + A_t Δt
///    - Implementation: [GitHub](https://github.com/UMich-BipedLab/Contact-Aided-Invariant-EKF)
///
/// 4. **Julier, S. J., & Uhlmann, J. K.** (2004).
///    "Unscented Filtering and Nonlinear Estimation."
///    *Proceedings of the IEEE*, 92(3), 401-422.
///    [PDF](https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf)
///    - UKF parameter selection: alpha=1e-3 (low dim), kappa=3-n, beta=2 (Gaussian)
///    - For high-dimensional systems (n≥5), use alpha=1.0
///
/// 5. **Arasaratnam, I., & Haykin, S.** (2009).
///    "Cubature Kalman Filters."
///    *IEEE Transactions on Automatic Control*, 54(6), 1254-1269.
///    [ResearchGate](https://www.researchgate.net/publication/224471882_Cubature_Kalman_Filters)
///    - Derives third-degree spherical-radial cubature rule
///    - Cubature points: ±√n scaled by covariance
///
/// 6. **Bierman, G. J.** (1976).
///    "Measurement updating using the U-D factorization."
///    *Automatica*, 12(4), 375-382.
///    [ScienceDirect](https://www.sciencedirect.com/science/article/pii/0005109876900583)
///    - Scalar measurement update algorithm
///    - Modified weighted Gram-Schmidt (MWGS) predictor
///
/// 7. **Tracy, K.** (2022).
///    "A Square-Root Kalman Filter Using Only QR Decompositions."
///    arXiv:[2208.06452](https://arxiv.org/pdf/2208.06452)
///    - QR-based square root filter
///    - qr_r([√A; √B]) = R for P' = A + B
///
/// 8. **Gill, P. E., Golub, G. H., Murray, W., & Saunders, M. A.** (1974).
///    "Methods for modifying matrix factorizations."
///    *Mathematics of Computation*, 28(126), 505-535.
///    - Cholesky rank-1 update/downdate algorithms
///    - Numerical stability analysis
///
/// 9. **Lou, Y., et al.** (2020).
///    "Differentiating through the Fréchet Mean."
///    *ICML 2020*.
///    [PDF](http://proceedings.mlr.press/v119/lou20a/lou20a-supp.pdf)
///    - Karcher mean iteration: μ_{k+1} = μ_k · Exp(Σ w_i * Log(μ_k⁻¹ · p_i))
///    - 2-3 iterations sufficient for most applications
///
/// ### Storage Representations
///
/// - **Direct**: Full covariance matrix P (O(n²) storage, O(n²) predict)
/// - **SR**: Square root S where P = S·S^T (O(n²) storage, O(n²) predict via QR)
/// - **UDU**: Bierman-Thornton UDU^T factorization (O(n²) storage, O(n²) predict)
///
/// ### Numerical Stability Considerations
///
/// 1. **UDU epsilon**: Use relative threshold (ε * max(D)) instead of absolute
/// 2. **UKF alpha**: For high-dimensional states, alpha=1.0 prevents negative lambda
/// 3. **SR QR**: Supports both HouseholderQR (fast) and ColPivQR (stable) variants
/// 4. **Symmetrization**: P ← (P + P^T) / 2 after updates to maintain SPD property
///

// ============================================================================
// Storage Representation Enum
// ============================================================================

/// @brief Covariance matrix storage representation
///
/// Only describes how P is stored/propagated (Direct/SR/UDU). The measurement
/// update formula (e.g. Joseph) is selected via an Updater policy.
enum class Representation { Direct, Sr, Udu };

/// @brief QR decomposition variant for square-root prediction
///
/// - Householder: Fast, standard QR (Eigen::HouseholderQR)
/// - ColPiv: Rank-revealing QR with column pivoting (Eigen::ColPivHouseholderQR)
///            More numerically stable for ill-conditioned matrices
///
/// Reference: Tracy 2022, arXiv:2208.06452
enum class QRVariant { Householder, ColPiv };

// ============================================================================
// CovarianceStorage - Storage Layer
// ============================================================================

/// @brief Storage layer for covariance matrices
///
/// Handles how the covariance matrix P is stored and retrieved, independent
/// of the propagation strategy (KF/EKF/UKF/CKF).
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Repr The storage representation
/// @tparam QRV The QR variant for SR prediction (only used when Repr == Sr)
template <
    typename LieGroup, Representation Repr = Representation::Direct,
    QRVariant QRVariant = QRVariant::Householder>
class CovarianceStorage {
public:
    static constexpr Representation repr = Repr;
    using MatrixDD                       = Eigen::MatrixXd;
    using Flux                           = LieGroup;

    /// @brief Default constructor - initializes to small identity
    /// @param dim State dimension (must be > 0 for proper initialization)
    explicit CovarianceStorage(int dim = 1)
        : dim_(dim) {
        if constexpr (Repr == Representation::Sr) {
            storage_ = MatrixDD::Identity(dim, dim) * 0.1;
        } else if constexpr (Repr == Representation::Udu) {
            // UDU storage: separate U and D
            u_ = MatrixDD::Identity(dim, dim);
            d_ = Eigen::VectorXd::Ones(dim) * 0.01;
        } else {
            // Direct: full matrix
            storage_ = MatrixDD::Identity(dim, dim) * 0.01;
        }
    }

    /// @brief Constructor from initial covariance matrix
    explicit CovarianceStorage(const MatrixDD& init_cov)
        : dim_(init_cov.rows()) {
        if constexpr (Repr == Representation::Sr) {
            // Compute Cholesky decomposition: P = SS^T
            Eigen::LLT<MatrixDD> llt(init_cov);
            if (llt.info() == Eigen::Success) {
                storage_ = llt.matrixL();
            } else {
                // Fallback: use sqrt of eigenvalues
                Eigen::SelfAdjointEigenSolver<MatrixDD> es(init_cov);
                storage_ = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
            }
        } else if constexpr (Repr == Representation::Udu) {
            // Compute UDU decomposition: P = UDU^T
            compute_udu(init_cov);
        } else {
            storage_ = init_cov;
        }
    }

    /// @brief Retrieve full covariance matrix P
    [[nodiscard]] MatrixDD mat() const noexcept {
        if constexpr (Repr == Representation::Sr) {
            // P = SS^T
            return storage_ * storage_.transpose();
        } else if constexpr (Repr == Representation::Udu) {
            // P = UDU^T
            return u_ * d_.asDiagonal() * u_.transpose();
        } else {
            // Direct
            return storage_;
        }
    }

    /// @brief Update covariance with new matrix
    void update(const MatrixDD& new_cov) noexcept {
        if constexpr (Repr == Representation::Sr) {
            // Compute Cholesky of new covariance
            Eigen::LLT<MatrixDD> llt(new_cov);
            if (llt.info() == Eigen::Success) {
                storage_ = llt.matrixL();
            } else {
                // Fallback for numerical issues
                Eigen::SelfAdjointEigenSolver<MatrixDD> es(new_cov);
                MatrixDD eigenvalues = es.eigenvalues().cwiseMax(0.0); // Ensure non-negative
                storage_             = es.eigenvectors() * eigenvalues.cwiseSqrt().asDiagonal();
            }
        } else if constexpr (Repr == Representation::Udu) {
            compute_udu(new_cov);
        } else {
            storage_ = new_cov;
        }
    }

    /// @brief Get dimension
    [[nodiscard]] int dimension() const noexcept { return dim_; }

    // ========================================================================
    // Representation-specific accessors
    // ========================================================================

    [[nodiscard]] const MatrixDD& sr_factor() const noexcept requires(Repr == Representation::Sr) {
        return storage_;
    }

    [[nodiscard]] MatrixDD& sr_factor() noexcept requires(Repr == Representation::Sr) {
        return storage_;
    }

    [[nodiscard]] const MatrixDD& udu_u() const noexcept requires(Repr == Representation::Udu) {
        return u_;
    }

    [[nodiscard]] const Eigen::VectorXd& udu_d() const noexcept
        requires(Repr == Representation::Udu) {
        return d_;
    }

    // ========================================================================
    // SR rank-1 update/downdate (O(n^2))
    // ========================================================================

    [[nodiscard]] bool sr_rank1_update(const Eigen::VectorXd& x) noexcept
        requires(Repr == Representation::Sr) {
        // NOTE: cholupdate is in-place and may partially modify the factor before failing.
        // Preserve strong exception-safety (and keep `storage_` valid) by applying to a copy.
        MatrixDD candidate = storage_;
        if (!factorization::cholupdate_lower_inplace(candidate, x, +1)) {
            return false;
        }
        storage_ = std::move(candidate);
        return true;
    }

    [[nodiscard]] bool sr_rank1_downdate(const Eigen::VectorXd& x) noexcept
        requires(Repr == Representation::Sr) {
        // NOTE: downdates can fail (matrix becomes non-SPD). Never leave `storage_` corrupted.
        MatrixDD candidate = storage_;
        if (!factorization::cholupdate_lower_inplace(candidate, x, -1)) {
            return false;
        }
        storage_ = std::move(candidate);
        return true;
    }

    // Square-root predict:
    //   P = S S^T
    //   P' = Phi P Phi^T + Q
    // Factorize Q = G G^T, let A = [Phi*S, G], then P' = A A^T.
    // QR on A^T gives A^T = Qr * R, so P' = R^T R and S' = R^T.
    //
    // Reference: Tracy 2022, arXiv:2208.06452, Section 2
    // qr_r([√A; √B]) returns upper triangular R such that P' = R^T * R
    [[nodiscard]] bool sr_predict(const MatrixDD& Phi, const MatrixDD& Q) noexcept
        requires(Repr == Representation::Sr) {
        const int n = dim_;
        if (Phi.rows() != n || Phi.cols() != n || Q.rows() != n || Q.cols() != n) {
            return false;
        }

        MatrixDD G;
        Eigen::LLT<MatrixDD> lltQ(Q);
        if (lltQ.info() == Eigen::Success) {
            G = lltQ.matrixL();
        } else {
            Eigen::SelfAdjointEigenSolver<MatrixDD> es(Q);
            if (es.info() != Eigen::Success) {
                return false;
            }
            const Eigen::VectorXd q = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
            G                       = es.eigenvectors() * q.asDiagonal();
        }

        const MatrixDD FS = Phi * storage_;
        MatrixDD A_t(2 * n, n);
        A_t.topRows(n)    = FS.transpose();
        A_t.bottomRows(n) = G.transpose();

        // Select QR variant based on template parameter
        if constexpr (QRVariant == QRVariant::ColPiv) {
            // Rank-revealing QR with column pivoting (more stable for ill-conditioned)
            Eigen::ColPivHouseholderQR<MatrixDD> qr(A_t);
            qr.setThreshold(std::numeric_limits<double>::epsilon());
            const MatrixDD R =
                qr.matrixQR().topLeftCorner(n, n).template triangularView<Eigen::Upper>();
            storage_ = R.transpose();
            storage_.triangularView<Eigen::StrictlyUpper>().setZero();
            return storage_.allFinite() && qr.rank() == n;
        } else {
            // Standard Householder QR (faster, no column pivoting)
            Eigen::HouseholderQR<MatrixDD> qr(A_t);
            const MatrixDD R =
                qr.matrixQR().topLeftCorner(n, n).template triangularView<Eigen::Upper>();
            storage_ = R.transpose();
            storage_.triangularView<Eigen::StrictlyUpper>().setZero();
            return storage_.allFinite();
        }
    }

    // ========================================================================
    // UDU scalar update + Thornton MWGS predict (O(n^2))
    // ========================================================================

    // Bierman scalar measurement update (KFCore-style). Updates U and d in-place.
    // Returns dx (state correction in tangent space) for this scalar measurement.
    [[nodiscard]] bool udu_update_scalar(
        double dz, double R_scalar, const Eigen::VectorXd& H_line, Eigen::VectorXd& dx_out) noexcept
        requires(Repr == Representation::Udu) {
        const int n = dim_;
        if (H_line.size() != n) {
            return false;
        }
        if (R_scalar <= 0.0) {
            return false;
        }

        // a = U^T * H_line
        Eigen::VectorXd a = u_.transpose() * H_line;
        Eigen::VectorXd b = d_.array() * a.array();

        double alpha = R_scalar;
        double gamma = 1.0 / alpha;

        for (int j = 0; j < n; ++j) {
            double beta = alpha;
            alpha += a(j) * b(j);

            const double lambda = -a(j) * gamma;
            gamma               = 1.0 / alpha;

            d_(j) *= beta * gamma;
            for (int i = 0; i < j; ++i) {
                const double uij = u_(i, j);
                u_(i, j)         = uij + b(i) * lambda;
                b(i) += b(j) * uij;
            }
        }

        dx_out = gamma * dz * b;
        return dx_out.allFinite() && u_.allFinite() && d_.allFinite();
    }

    // Thornton MWGS predictor update.
    // Supports general SPD/PSD Q by factoring Q = G G^T and running MWGS with (G, diag(1)).
    //
    // Reference:
    // - Thornton, C. L. (1976). "Triangular Covariance Factorizations for Kalman Filtering."
    //   NASA Technical Report. [NASA
    //   TR](https://ntrs.nasa.gov/api/citations/19770005172/downloads/19770005172.pdf)
    //   - Section 5: Modified Weighted Gram-Schmidt (MWGS) algorithm
    // - Bierman & Thornton: U-D filter implementation (cited by 137+ papers)
    // - Also see: "A Summary on the UD Kalman Filter"
    // [arXiv:2203.06105](https://arxiv.org/pdf/2203.06105)
    [[nodiscard]] bool udu_predict(const MatrixDD& Phi, const MatrixDD& Q) noexcept
        requires(Repr == Representation::Udu) {
        const int n = dim_;
        if (Phi.rows() != n || Phi.cols() != n || Q.rows() != n || Q.cols() != n) {
            return false;
        }

        MatrixDD G_tmp;
        Eigen::LLT<MatrixDD> lltQ(Q);
        if (lltQ.info() == Eigen::Success) {
            G_tmp = lltQ.matrixL();
        } else {
            Eigen::SelfAdjointEigenSolver<MatrixDD> es(Q);
            if (es.info() != Eigen::Success) {
                return false;
            }
            const Eigen::VectorXd q = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
            G_tmp                   = es.eigenvectors() * q.asDiagonal();
        }

        const Eigen::VectorXd Qdiag = Eigen::VectorXd::Ones(n);
        Eigen::MatrixXd PhiU        = Phi * u_;

        const Eigen::VectorXd din = d_;
        u_.setIdentity(n, n);

        for (int i = n - 1; i >= 0; --i) {
            double sigma = 0.0;
            for (int j = 0; j < n; ++j) {
                const double pij = PhiU(i, j);
                sigma += pij * pij * din(j);

                const double gij = G_tmp(i, j);
                sigma += gij * gij * Qdiag(j);
            }
            d_(i) = sigma;

            for (int j = 0; j < i; ++j) {
                sigma = 0.0;
                for (int k = 0; k < n; ++k) {
                    sigma += PhiU(i, k) * din(k) * PhiU(j, k);
                }
                for (int k = 0; k < n; ++k) {
                    sigma += G_tmp(i, k) * Qdiag(k) * G_tmp(j, k);
                }

                const double uji = sigma / d_(i);
                u_(j, i)         = uji;

                PhiU.row(j) -= uji * PhiU.row(i);
                G_tmp.row(j) -= uji * G_tmp.row(i);
            }
        }

        return u_.allFinite() && d_.allFinite();
    }

private:
    /// @brief Compute UDU decomposition: P = UDU^T
    /// Uses modified Cholesky decomposition where U is unit upper triangular
    void compute_udu(const MatrixDD& P) noexcept {
        int n = static_cast<int>(P.rows());
        u_    = MatrixDD::Identity(n, n);
        d_    = Eigen::VectorXd::Zero(n);

        // Use hybrid absolute/relative epsilon for numerical stability
        // For diagonal elements near 1.0, use absolute threshold (epsilon)
        // For very small/large elements, scale relative to maximum diagonal
        constexpr double epsilon = 1e-12;
        const double max_diag    = P.diagonal().cwiseAbs().maxCoeff();
        const double threshold   = epsilon * std::max(1.0, max_diag);

        // Modified Cholesky (UDU^T) decomposition
        for (int j = n - 1; j >= 0; --j) {
            d_(j) = P(j, j);
            for (int k = j + 1; k < n; ++k) {
                d_(j) -= u_(j, k) * u_(j, k) * d_(k);
            }
            // Ensure diagonal is positive using hybrid threshold.
            // IMPORTANT: threshold must not depend on uninitialized parts of d_.
            if (d_(j) <= threshold) {
                d_(j) = threshold;
            }
            for (int i = j - 1; i >= 0; --i) {
                double sum = P(i, j);
                for (int k = j + 1; k < n; ++k) {
                    sum -= u_(i, k) * u_(j, k) * d_(k);
                }
                u_(i, j) = sum / d_(j);
            }
        }
    }

    int dim_;
    MatrixDD storage_;
    MatrixDD u_;
    Eigen::VectorXd d_;
};

// ============================================================================
// LinearizedStrategy - Extended Kalman Filter
// ============================================================================

/// @brief Linearized covariance propagation (EKF)
///
/// Uses Jacobians computed at current state: J_f = df/dx, J_h = dh/dx
template <typename LieGroup, typename Storage>
class LinearizedStrategy {
public:
    using MatrixDD = Eigen::MatrixXd;

    /// @brief Propagate: P' = Φ*P*Φ^T + Q
    /// @param transition_jacobian Discrete transition Jacobian Φ (provided by the process model)
    void propagate(
        Storage& P, const LieGroup& old_state, const LieGroup& new_state,
        const std::function<LieGroup(const LieGroup&, double)>& dynamics,
        const MatrixDD& transition_jacobian, const MatrixDD& process_noise, double dt) noexcept {
        (void)old_state;
        (void)new_state;
        (void)dynamics;
        (void)dt;
        const MatrixDD& Phi = transition_jacobian;
        if constexpr (Storage::repr == Representation::Sr) {
            if (!P.sr_predict(Phi, process_noise)) {
                const MatrixDD current_P = P.mat();
                P.update(Phi * current_P * Phi.transpose() + process_noise);
            }
        } else if constexpr (Storage::repr == Representation::Udu) {
            if (!P.udu_predict(Phi, process_noise)) {
                const MatrixDD current_P = P.mat();
                P.update(Phi * current_P * Phi.transpose() + process_noise);
            }
        } else {
            const MatrixDD current_P = P.mat();
            P.update(Phi * current_P * Phi.transpose() + process_noise);
        }
    }

    /// @brief Compute Kalman gain: K = P*J_h^T * (J_h*P*J_h^T + R)^-1
    [[nodiscard]] MatrixDD compute_kalman_gain(
        Storage& P, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise) const noexcept {
        const MatrixDD current_P = P.mat();
        const MatrixDD innovation_cov =
            measurement_jacobian * current_P * measurement_jacobian.transpose() + measurement_noise;
        const MatrixDD PHt = current_P * measurement_jacobian.transpose();
        const auto ldlt    = innovation_cov.ldlt();
        if (ldlt.info() == Eigen::Success) {
            return ldlt.solve(PHt.transpose()).transpose();
        }
        return innovation_cov.completeOrthogonalDecomposition().solve(PHt.transpose()).transpose();
    }

    /// @brief Update: P' = (I - K*J_h) * P
    /// Note: measurement_noise is not used here (already incorporated in K)
    void stiffen(
        Storage& P, const MatrixDD& kalman_gain, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise, double dt) noexcept {
        (void)measurement_noise; // Noise already incorporated in Kalman gain K
        (void)dt;
        const MatrixDD I         = MatrixDD::Identity(P.dimension(), P.dimension());
        const MatrixDD current_P = P.mat();
        P.update((I - kalman_gain * measurement_jacobian) * current_P);
    }
};

// ============================================================================
// UnscentedStrategy - Unscented Kalman Filter
// ============================================================================

/// @brief Unscented covariance propagation (UKF)
///
/// Uses sigma-point transform: generates points, propagates through
/// nonlinear function, recovers mean and covariance.
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Storage The covariance storage type
/// @tparam Alpha UKF alpha parameter (spread of sigma points, default: 1e-3)
/// @tparam Beta UKF beta parameter (optimal for Gaussian, default: 2.0)
/// @tparam Kappa UKF kappa parameter (secondary scaling, default: 0.0)
///
/// UKF parameter reference:
/// - Julier & Uhlmann 2004: alpha=1e-3 for low-dim, alpha=1.0 for high-dim (n≥5)
/// - beta=2.0 is optimal for Gaussian distributions
/// - kappa=0.0 for scaled UKF, kappa=3-n for classic formulation
template <
    typename LieGroup, typename Storage, double Alpha = 1e-3, double Beta = 2.0, double Kappa = 0.0>
requires requires(const LieGroup& x, const LieGroup& y) {
    { x * y } -> std::convertible_to<LieGroup>;
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
    { LieGroup::exp(LieGroup::VectorType::Zero()) } -> std::convertible_to<LieGroup>;
} class UnscentedStrategy {
public:
    using MatrixDD = Eigen::MatrixXd;

    static constexpr double alpha = Alpha;
    static constexpr double beta  = Beta;
    static constexpr double kappa = Kappa;

    /// @brief Propagate using sigma-point transform
    ///
    /// Generates sigma points, propagates them through dynamics,
    /// recovers mean and covariance from transformed points.
    ///
    /// Uses template parameters Alpha, Beta, Kappa for sigma-point computation.
    /// @param old_state State before propagation (used for sigma-point generation)
    /// @param new_state State after propagation (not used, computed from sigma points)
    /// @param dynamics Function to propagate state: x' = f(x, dt)
    void propagate(
        Storage& P, const LieGroup& old_state, const LieGroup& new_state,
        const std::function<LieGroup(const LieGroup&, double)>& dynamics,
        const MatrixDD& process_noise, double dt) noexcept {
        (void)new_state; // Not used, computed from sigma points

        const MatrixDD current_P = P.mat();
        const int n              = static_cast<int>(current_P.rows());

        // UKF weights (Julier & Uhlmann 2004, Sec. III)
        const double lambda = alpha * alpha * (n + kappa) - n;
        const double w0_m   = lambda / (n + lambda);
        const double w0_c   = w0_m + (1 - alpha * alpha + beta);
        const double wi     = 1.0 / (2 * (n + lambda));

        // Compute Cholesky: (n+λ) * P = SS^T
        Eigen::LLT<MatrixDD> llt((n + lambda) * current_P);
        if (llt.info() != Eigen::Success) {
            return;
        }
        const MatrixDD S = llt.matrixL();

        std::vector<LieGroup> states;
        std::vector<typename LieGroup::VectorType> drifts;
        std::vector<double> weights_mean;
        std::vector<double> weights_cov;

        // Point 0: mean state
        states.push_back(old_state);
        drifts.push_back(LieGroup::VectorType::Zero());
        weights_mean.push_back(w0_m);
        weights_cov.push_back(w0_c);

        // Points ± columns of S
        for (int i = 0; i < n; ++i) {
            typename LieGroup::VectorType xi = S.col(i);

            states.push_back(lie_ops::compose(old_state, xi));
            drifts.push_back(xi);
            weights_mean.push_back(wi);
            weights_cov.push_back(wi);

            xi = -S.col(i);
            states.push_back(lie_ops::compose(old_state, xi));
            drifts.push_back(xi);
            weights_mean.push_back(wi);
            weights_cov.push_back(wi);
        }

        // Propagate sigma points
        std::vector<LieGroup> propagated;
        std::vector<typename LieGroup::VectorType> propagated_drifts;
        propagated.reserve(states.size());
        propagated_drifts.reserve(states.size());
        for (size_t i = 0; i < states.size(); ++i) {
            const auto x_i = dynamics(states[i], dt);
            const auto xi  = lie_ops::diff(old_state, x_i);
            propagated.push_back(x_i);
            propagated_drifts.push_back(xi);
        }

        // Compute weighted mean (Karcher mean on Lie group)
        LieGroup mean_state = propagated[0];
        for (int iter = 0; iter < 3; ++iter) {
            typename LieGroup::VectorType step = LieGroup::VectorType::Zero();
            for (size_t i = 0; i < propagated.size(); ++i) {
                const auto delta = lie_ops::diff(mean_state, propagated[i]);
                step += weights_mean[i] * delta;
            }
            mean_state = lie_ops::compose(mean_state, step);
        }

        // Compute weighted covariance
        MatrixDD P_new = MatrixDD::Zero(n, n);
        for (size_t i = 0; i < propagated.size(); ++i) {
            const auto diff = lie_ops::diff(mean_state, propagated[i]);
            P_new += weights_cov[i] * (diff * diff.transpose());
        }

        P.update(P_new + process_noise);
    }

    /// @brief Compute Kalman gain using linear approximation
    ///
    /// Note: For proper sigma-point measurement transform, use the Filter layer
    /// implementation which transforms sigma points through the measurement model.
    /// This linear approximation is used as fallback when the measurement model
    /// is not accessible at the Strategy layer.
    [[nodiscard]] MatrixDD compute_kalman_gain(
        Storage& P, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise) const noexcept {
        const MatrixDD current_P = P.mat();
        const MatrixDD innovation_cov =
            measurement_jacobian * current_P * measurement_jacobian.transpose() + measurement_noise;
        const MatrixDD PHt = current_P * measurement_jacobian.transpose();
        const auto ldlt    = innovation_cov.ldlt();
        if (ldlt.info() == Eigen::Success) {
            return ldlt.solve(PHt.transpose()).transpose();
        }
        return innovation_cov.completeOrthogonalDecomposition().solve(PHt.transpose()).transpose();
    }

    /// @brief Update using simple form: P' = (I - KH) * P
    /// Note: measurement_noise is not used here (already incorporated in K)
    void stiffen(
        Storage& P, const MatrixDD& kalman_gain, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise, double dt) noexcept {
        (void)measurement_noise; // Noise already incorporated in Kalman gain K
        (void)dt;
        const int n              = P.dimension();
        const MatrixDD I         = MatrixDD::Identity(n, n);
        const MatrixDD current_P = P.mat();
        P.update((I - kalman_gain * measurement_jacobian) * current_P);
    }
};

// ============================================================================
// CubatureStrategy - Cubature Kalman Filter
// ============================================================================

/// @brief Cubature covariance propagation (CKF)
///
/// Uses spherical-radial cubature rule for sigma points.
/// Similar to UKF but uses different sigma point selection.
template <typename LieGroup, typename Storage>
requires requires(const LieGroup& x, const LieGroup& y) {
    { x * y } -> std::convertible_to<LieGroup>;
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
    { LieGroup::exp(LieGroup::VectorType::Zero()) } -> std::convertible_to<LieGroup>;
} class CubatureStrategy {
public:
    using MatrixDD = Eigen::MatrixXd;

    /// @brief Propagate using cubature transform
    ///
    /// Uses spherical-radial cubature rule for sigma points.
    /// Generates 2n points at distance sqrt(n) from center.
    /// @param old_state State before propagation (used for sigma-point generation)
    /// @param new_state State after propagation (not used, computed from sigma points)
    /// @param dynamics Function to propagate state: x' = f(x, dt)
    void propagate(
        Storage& P, const LieGroup& old_state, const LieGroup& new_state,
        const std::function<LieGroup(const LieGroup&, double)>& dynamics,
        const MatrixDD& process_noise, double dt) noexcept {
        (void)new_state; // Not used, computed from sigma points
        const MatrixDD current_P = P.mat();
        const int n              = static_cast<int>(current_P.rows());
        if (n <= 0 || current_P.cols() != n) {
            return;
        }
        const double weight = 1.0 / (2.0 * static_cast<double>(n));

        MatrixDD S;
        Eigen::LLT<MatrixDD> llt(current_P);
        if (llt.info() == Eigen::Success) {
            S = llt.matrixL();
        } else {
            // PSD fallback: eigen-decompose and clamp eigenvalues
            Eigen::SelfAdjointEigenSolver<MatrixDD> es(current_P);
            if (es.info() != Eigen::Success) {
                return;
            }
            const Eigen::VectorXd evals = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
            S                           = es.eigenvectors() * evals.asDiagonal();
        }

        // Spherical-radial cubature points: x_i = x ⊕ (± sqrt(n) * S.col(i))
        std::vector<LieGroup> states;
        std::vector<double> weights;

        const double scale = std::sqrt(static_cast<double>(n));
        for (int i = 0; i < n; ++i) {
            typename LieGroup::VectorType xi = scale * S.col(i);
            states.push_back(lie_ops::compose(old_state, xi));
            weights.push_back(weight);

            xi = -scale * S.col(i);
            states.push_back(lie_ops::compose(old_state, xi));
            weights.push_back(weight);
        }

        // Propagate sigma points
        std::vector<LieGroup> propagated;
        propagated.reserve(states.size());
        for (const auto& sp : states) {
            propagated.push_back(dynamics(sp, dt));
        }

        // Compute weighted mean (Karcher mean on Lie group)
        LieGroup mean_state = propagated[0];
        for (int iter = 0; iter < 3; ++iter) {
            typename LieGroup::VectorType step = LieGroup::VectorType::Zero();
            for (size_t i = 0; i < propagated.size(); ++i) {
                const auto delta = lie_ops::diff(mean_state, propagated[i]);
                step += weights[i] * delta;
            }
            mean_state = lie_ops::compose(mean_state, step);
        }

        // Compute weighted covariance
        MatrixDD P_new = MatrixDD::Zero(n, n);
        for (size_t i = 0; i < propagated.size(); ++i) {
            const auto diff = lie_ops::diff(mean_state, propagated[i]);
            P_new += weights[i] * (diff * diff.transpose());
        }

        P.update(P_new + process_noise);
    }

    /// @brief Compute Kalman gain using linear approximation
    ///
    /// Note: For proper sigma-point measurement transform, use the Filter layer
    /// implementation which transforms sigma points through the measurement model.
    /// This linear approximation is used as fallback when the measurement model
    /// is not accessible at the Strategy layer.
    [[nodiscard]] MatrixDD compute_kalman_gain(
        Storage& P, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise) const noexcept {
        const MatrixDD current_P = P.mat();
        const MatrixDD innovation_cov =
            measurement_jacobian * current_P * measurement_jacobian.transpose() + measurement_noise;
        const MatrixDD PHt = current_P * measurement_jacobian.transpose();
        const auto ldlt    = innovation_cov.ldlt();
        if (ldlt.info() == Eigen::Success) {
            return ldlt.solve(PHt.transpose()).transpose();
        }
        return innovation_cov.completeOrthogonalDecomposition().solve(PHt.transpose()).transpose();
    }

    /// @brief Update using simple form: P' = (I - KH) * P
    /// Note: measurement_noise is not used here (already incorporated in K)
    void stiffen(
        Storage& P, const MatrixDD& kalman_gain, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise, double dt) noexcept {
        (void)measurement_noise; // Noise already incorporated in Kalman gain K
        (void)dt;
        const int n              = P.dimension();
        const MatrixDD I         = MatrixDD::Identity(n, n);
        const MatrixDD current_P = P.mat();
        P.update((I - kalman_gain * measurement_jacobian) * current_P);
    }
};

// ============================================================================
// Invariant Kalman Filter Strategies (for Lie Groups)
// ============================================================================

/// @brief Invariant Linear Strategy (InKF) - For PURE KINEMATIC systems only
///
/// For PURE KINEMATIC systems on Lie groups (F ≈ I), uses right-invariant error.
///
/// **IMPORTANT**: This strategy assumes F = I (pure kinematics). For general
/// nonlinear dynamics with F ≠ I, use InvariantLinearizedStrategy instead.
///
/// Mathematical background (Bonnabel 2017):
/// For right-invariant error ξ_R = Log(X̂⁻¹X) on PURE KINEMATIC systems:
///   P' = P + Q  (F = I, no Adjoint needed)
///
/// For general dynamics, use InvariantLinearizedStrategy which includes:
///   P' = Φ · P · Φ^T + Q
///
/// Reference:
/// - Bonnabel 2017: [arXiv:1410.1465](https://arxiv.org/pdf/1410.1465), Section III.B
/// - Barfoot 2024: [PDF](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf), Section 11.3
///
/// Requirements:
/// - LieGroup must support inv() and log() (lieplusplus native methods)
/// - Suitable for PURE KINEMATIC systems (constant velocity, no control input)
template <typename LieGroup, typename Storage>
requires requires(const LieGroup& x) {
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
} class InvariantLinearStrategy {
public:
    using MatrixDD = Eigen::MatrixXd;

    /// @brief Propagate covariance for pure kinematic systems (F = I)
    ///
    /// For PURE KINEMATIC systems (F ≈ I):
    ///   P' = P + Q
    ///
    /// This is ONLY valid when the dynamics Jacobian is identity (F = I).
    /// For general dynamics with F ≠ I, use InvariantLinearizedStrategy.
    ///
    /// Reference: Bonnabel 2017, Section III.B; Barfoot 2024, Section 11.3
    ///
    /// @param P Covariance storage
    /// @param old_state State before propagation (X_old)
    /// @param new_state State after propagation (X_new = X_old * exp(twist*dt))
    /// @param dynamics Function to propagate state: x' = f(x, dt) (not used here)
    void propagate(
        Storage& P, const LieGroup& old_state, const LieGroup& new_state,
        const std::function<LieGroup(const LieGroup&, double)>& dynamics,
        const MatrixDD& process_noise, double dt) noexcept {
        (void)dynamics; // Dynamics not needed for group-affine covariance propagation
        (void)old_state;
        (void)new_state;
        (void)dt;       // dt not needed (Q is already discretized)

        const MatrixDD current_P = P.mat();
        P.update(current_P + process_noise);
    }

    /// @brief Compute Kalman gain (same as standard KF)
    [[nodiscard]] MatrixDD compute_kalman_gain(
        Storage& P, const MatrixDD& measurement_matrix,
        const MatrixDD& measurement_noise) const noexcept {
        const MatrixDD current_P = P.mat();
        const MatrixDD innovation_cov =
            measurement_matrix * current_P * measurement_matrix.transpose() + measurement_noise;
        const MatrixDD PHt = current_P * measurement_matrix.transpose();
        const auto ldlt    = innovation_cov.ldlt();
        if (ldlt.info() == Eigen::Success) {
            return ldlt.solve(PHt.transpose()).transpose();
        }
        return innovation_cov.completeOrthogonalDecomposition().solve(PHt.transpose()).transpose();
    }

    /// @brief Update: P' = (I - KH) * P
    void stiffen(
        Storage& P, const MatrixDD& kalman_gain, const MatrixDD& measurement_matrix,
        const MatrixDD& measurement_noise, double dt) noexcept {
        (void)dt;
        const int n              = P.dimension();
        const MatrixDD I         = MatrixDD::Identity(n, n);
        const MatrixDD current_P = P.mat();

        (void)measurement_noise;
        const MatrixDD I_KH = I - kalman_gain * measurement_matrix;
        P.update(I_KH * current_P);
    }
};

/// @brief Invariant Linearized Strategy (InEKF)
///
/// Extended version of InKF that uses Jacobians for nonlinear dynamics
/// on Lie groups. Combines the benefits of:
/// - Invariant error definition (better numerical properties)
/// - Jacobian linearization (handles nonlinear dynamics)
///
/// Uses the right-invariant error convention:
///   ξ = Log(X̂⁻¹X), X = X̂ · Exp(ξ)
/// Covariance P is expressed in the (right-invariant) error coordinates.
/// This implementation does not apply any state-dependent Adjoint to P.
/// If your process noise is specified in another frame, map it into the same
/// error coordinates when constructing `process_noise`.
template <typename LieGroup, typename Storage>
class InvariantLinearizedStrategy {
public:
    using MatrixDD = Eigen::MatrixXd;

    /// @brief Propagate covariance using linearized dynamics (InEKF prediction)
    ///
    /// Mathematical background (Bonnabel 2017, Hartley RSS 2019):
    /// For right-invariant error ξ_R = Log(X̂⁻¹X) on Lie groups:
    ///
    /// **Continuous-time Riccati equation** (Hartley RSS 2019, Eq. 7):
    ///   d/dt P_t = A_t P_t + P_t A_t^T + Q̂_t
    ///
    /// **Discrete-time approximation** (first-order Euler integration):
    ///   Φ = exp_m(A_t Δt)  (discretized state transition)
    ///   P' = Φ · P · Φ^T + Q̂
    ///
    /// where Q̂ is the process noise covariance expressed in the same error coordinates.
    ///
    /// For GROUP-AFFINE systems (Bonnabel 2017, Eq. (20b)):
    ///   P' = F · P · F^T + Q  (no Adjoint needed for group-affine)
    ///
    /// Reference:
    /// - Bonnabel 2017: [arXiv:1410.1465](https://arxiv.org/pdf/1410.1465), Eq. (20b)
    /// - Hartley et al. 2019: "Contact-Aided InEKF", RSS 2019
    ///   [PDF](https://www.roboticsproceedings.org/rss14/p50.pdf), Eq. (7)
    ///
    /// @param transition_jacobian Discrete transition Jacobian Φ in the invariant error coordinates
    void propagate(
        Storage& P, const LieGroup& old_state, const LieGroup& new_state,
        const std::function<LieGroup(const LieGroup&, double)>& dynamics,
        const MatrixDD& transition_jacobian, const MatrixDD& process_noise, double dt) noexcept {
        (void)old_state;
        (void)new_state;
        (void)dynamics;
        (void)dt;
        const MatrixDD current_P = P.mat();
        const MatrixDD& Phi      = transition_jacobian;
        const MatrixDD P_pred    = Phi * current_P * Phi.transpose();

        P.update(P_pred + process_noise);
    }

    /// @brief Compute Kalman gain using measurement Jacobian
    [[nodiscard]] MatrixDD compute_kalman_gain(
        Storage& P, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise) const noexcept {
        const MatrixDD current_P = P.mat();
        const MatrixDD innovation_cov =
            measurement_jacobian * current_P * measurement_jacobian.transpose() + measurement_noise;
        const MatrixDD PHt = current_P * measurement_jacobian.transpose();
        const auto ldlt    = innovation_cov.ldlt();
        if (ldlt.info() == Eigen::Success) {
            return ldlt.solve(PHt.transpose()).transpose();
        }
        return innovation_cov.completeOrthogonalDecomposition().solve(PHt.transpose()).transpose();
    }

    /// @brief Update: P' = (I - KH) * P
    void stiffen(
        Storage& P, const MatrixDD& kalman_gain, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise, double dt) noexcept {
        (void)dt;
        const int n              = P.dimension();
        const MatrixDD I         = MatrixDD::Identity(n, n);
        const MatrixDD current_P = P.mat();

        (void)measurement_noise;
        const MatrixDD I_KH = I - kalman_gain * measurement_jacobian;
        P.update(I_KH * current_P);
    }
};

// ============================================================================
// Combined Covariance Class
// ============================================================================

// ============================================================================
// Update Policies (measurement covariance update)
// ============================================================================

struct DirectSimpleUpdater {
    using MatrixDD = Eigen::MatrixXd;

    template <typename Storage>
    static void
        update(Storage& P, const MatrixDD& K, const MatrixDD& H, const MatrixDD& R) noexcept {
        // Simple form: P+ = (I - K H) P-
        // Note: We symmetricize after update to maintain numerical stability
        (void)R;
        const int n              = P.dimension();
        const MatrixDD I         = MatrixDD::Identity(n, n);
        const MatrixDD current_P = P.mat();
        MatrixDD Pnew            = (I - K * H) * current_P;
        Pnew                     = (Pnew + Pnew.transpose()) * 0.5;
        P.update(Pnew);
    }

    template <typename Storage>
    static void update_innovation(Storage& P, const MatrixDD& K, const MatrixDD& S) noexcept {
        MatrixDD Pnew = P.mat() - K * S * K.transpose();
        Pnew          = (Pnew + Pnew.transpose()) * 0.5;
        P.update(Pnew);
    }
};

struct DirectJosephUpdater {
    using MatrixDD = Eigen::MatrixXd;

    template <typename Storage>
    static void
        update(Storage& P, const MatrixDD& K, const MatrixDD& H, const MatrixDD& R) noexcept {
        const int n              = P.dimension();
        const MatrixDD I         = MatrixDD::Identity(n, n);
        const MatrixDD current_P = P.mat();
        const MatrixDD I_KH      = I - K * H;
        MatrixDD Pnew            = I_KH * current_P * I_KH.transpose() + K * R * K.transpose();
        Pnew                     = (Pnew + Pnew.transpose()) * 0.5;
        P.update(Pnew);
    }
};

struct SrUpdater {
    using MatrixDD = Eigen::MatrixXd;

    template <typename Storage>
    static void
        update(Storage& P, const MatrixDD& K, const MatrixDD& H, const MatrixDD& R) noexcept {
        const int n              = P.dimension();
        const MatrixDD I         = MatrixDD::Identity(n, n);
        const MatrixDD current_P = P.mat();
        const MatrixDD I_KH      = I - K * H;
        MatrixDD Pnew            = I_KH * current_P * I_KH.transpose() + K * R * K.transpose();
        Pnew                     = (Pnew + Pnew.transpose()) * 0.5;
        P.update(Pnew);
    }

    template <typename Storage>
    static void update_innovation(Storage& P, const MatrixDD& K, const MatrixDD& S) noexcept {
        // SR: P+ = P- - W W^T, where W = K * chol(S)
        Eigen::LLT<MatrixDD> llt(S);
        if (llt.info() == Eigen::Success) {
            const MatrixDD L = llt.matrixL();
            const MatrixDD W = K * L;
            for (int i = 0; i < W.cols(); ++i) {
                if (!P.sr_rank1_downdate(W.col(i))) {
                    // Fallback to full update
                    MatrixDD Pnew = P.mat() - K * S * K.transpose();
                    Pnew          = (Pnew + Pnew.transpose()) * 0.5;
                    P.update(Pnew);
                    return;
                }
            }
            return;
        }

        // Fallback
        MatrixDD Pnew = P.mat() - K * S * K.transpose();
        Pnew          = (Pnew + Pnew.transpose()) * 0.5;
        P.update(Pnew);
    }
};

struct UduUpdater {
    using MatrixDD = Eigen::MatrixXd;
};

template <Representation Repr, typename Updater>
struct ValidUpdater : std::false_type {};
template <>
struct ValidUpdater<Representation::Direct, DirectSimpleUpdater> : std::true_type {};
template <>
struct ValidUpdater<Representation::Direct, DirectJosephUpdater> : std::true_type {};
template <>
struct ValidUpdater<Representation::Sr, SrUpdater> : std::true_type {};
template <>
struct ValidUpdater<Representation::Udu, UduUpdater> : std::true_type {};

template <Representation Repr>
struct DefaultUpdater;
template <>
struct DefaultUpdater<Representation::Direct> {
    using type = DirectSimpleUpdater;
};
template <>
struct DefaultUpdater<Representation::Sr> {
    using type = SrUpdater;
};
template <>
struct DefaultUpdater<Representation::Udu> {
    using type = UduUpdater;
};
template <Representation Repr>
using DefaultUpdaterT = typename DefaultUpdater<Repr>::type;

namespace detail {
template <template <typename, typename> class Strategy>
struct IsSigmaPointStrategyTemplate : std::false_type {};

template <>
struct IsSigmaPointStrategyTemplate<UnscentedStrategy> : std::true_type {};

template <>
struct IsSigmaPointStrategyTemplate<CubatureStrategy> : std::true_type {};
} // namespace detail

/// @brief Unified covariance with pluggable storage and strategy
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Repr Storage representation (Direct/SR/UDU)
/// @tparam Strategy Propagation strategy class
/// @tparam Updater Measurement update policy
template <
    typename LieGroup, Representation Repr, template <typename, typename> class Strategy,
    typename Updater = DefaultUpdaterT<Repr>>
requires ValidUpdater<Repr, Updater>::value
      && (
             // Direct storage must have a measurement update formula.
             (Repr != Representation::Direct)
             || requires(
                 CovarianceStorage<LieGroup, Repr>& s, const Eigen::MatrixXd& K,
                 const Eigen::MatrixXd& H,
                 const Eigen::MatrixXd& R) { Updater::update(s, K, H, R); })
      && (
             // Sigma-point strategies require innovation-covariance update support.
             (!detail::IsSigmaPointStrategyTemplate<Strategy>::value)
             || requires(
                 CovarianceStorage<LieGroup, Repr>& s, const Eigen::MatrixXd& K,
                 const Eigen::MatrixXd& S) { Updater::update_innovation(s, K, S); })
class Covariance {
public:
    using Storage      = CovarianceStorage<LieGroup, Repr>;
    using StrategyType = Strategy<LieGroup, Storage>;
    using MatrixDD     = Eigen::MatrixXd;

    /// @brief Get dimension from LieGroup type (compile-time if available)
    [[nodiscard]] static constexpr int get_dimension() { return LieGroup::Dim; }

    /// @brief Default constructor - initializes with dimension from LieGroup type
    explicit Covariance()
        : storage_(get_dimension())
        , strategy_() {}

    /// @brief Constructor with explicit dimension
    explicit Covariance(int dim)
        : storage_(dim)
        , strategy_() {}

    /// @brief Constructor from initial covariance
    explicit Covariance(const MatrixDD& init_cov)
        : storage_(init_cov)
        , strategy_() {}

    /// @brief Propagate covariance through dynamics
    ///
    /// This overload is intended for strategies that do not require an explicit
    /// transition Jacobian (e.g., UKF/CKF, pure additive models).
    void propagate(
        const LieGroup& old_state, const LieGroup& new_state,
        const std::function<LieGroup(const LieGroup&, double)>& dynamics,
        const MatrixDD& process_noise, double dt) noexcept {
        if constexpr (requires(
                          StrategyType& s, Storage& P, const LieGroup& a, const LieGroup& b,
                          const std::function<LieGroup(const LieGroup&, double)>& f,
                          const MatrixDD& Q, double h) { s.propagate(P, a, b, f, Q, h); }) {
            strategy_.propagate(storage_, old_state, new_state, dynamics, process_noise, dt);
        } else {
            static_assert(
                detail::kAlwaysFalseV<StrategyType>,
                "This covariance strategy requires a transition Jacobian Φ. Call propagate(..., "
                "Phi, Q, dt).");
        }
    }

    /// @brief Propagate covariance through dynamics with explicit transition Jacobian Φ
    ///
    /// Intended for EKF/InEKF linearized propagation:
    ///   P' = Φ P Φᵀ + Q
    void propagate(
        const LieGroup& old_state, const LieGroup& new_state,
        const std::function<LieGroup(const LieGroup&, double)>& dynamics, const MatrixDD& Phi,
        const MatrixDD& process_noise, double dt) noexcept {
        if constexpr (requires(
                          StrategyType& s, Storage& P, const LieGroup& a, const LieGroup& b,
                          const std::function<LieGroup(const LieGroup&, double)>& f,
                          const MatrixDD& F, const MatrixDD& Q,
                          double h) { s.propagate(P, a, b, f, F, Q, h); }) {
            strategy_.propagate(storage_, old_state, new_state, dynamics, Phi, process_noise, dt);
        } else if constexpr (requires(
                                 StrategyType& s, Storage& P, const LieGroup& a, const LieGroup& b,
                                 const std::function<LieGroup(const LieGroup&, double)>& f,
                                 const MatrixDD& Q, double h) { s.propagate(P, a, b, f, Q, h); }) {
            // Strategy does not use Φ; ignore it.
            strategy_.propagate(storage_, old_state, new_state, dynamics, process_noise, dt);
        } else {
            static_assert(
                detail::kAlwaysFalseV<StrategyType>,
                "Unsupported covariance strategy propagate() signature.");
        }
    }

    /// @brief Compute Kalman gain
    [[nodiscard]] MatrixDD compute_kalman_gain(
        const MatrixDD& measurement_jacobian, const MatrixDD& measurement_noise) noexcept {
        return strategy_.compute_kalman_gain(storage_, measurement_jacobian, measurement_noise);
    }

    /// @brief Update covariance after measurement
    void stiffen(
        const MatrixDD& kalman_gain, const MatrixDD& measurement_jacobian,
        const MatrixDD& measurement_noise, double dt) noexcept
        requires requires(Storage& s, const MatrixDD& K, const MatrixDD& H, const MatrixDD& R) {
            Updater::update(s, K, H, R);
        } {
        (void)dt;
        Updater::update(storage_, kalman_gain, measurement_jacobian, measurement_noise);
    }

    /// @brief Update covariance from innovation covariance (sigma-point filters)
    ///
    /// For UKF/CKF, the correct update is:
    ///   P+ = P- - K * S * K^T
    /// where S is the innovation covariance (includes R).
    void stiffen_innovation(const MatrixDD& kalman_gain, const MatrixDD& innovation_cov) noexcept
        requires requires(Storage& s, const MatrixDD& K, const MatrixDD& S) {
            Updater::update_innovation(s, K, S);
        } {
        Updater::update_innovation(storage_, kalman_gain, innovation_cov);
    }

    /// @brief Get full covariance matrix
    [[nodiscard]] MatrixDD mat() const noexcept { return storage_.mat(); }

    /// @brief Get dimension
    [[nodiscard]] int dimension() const noexcept { return storage_.dimension(); }

    [[nodiscard]] Storage& storage() noexcept { return storage_; }
    [[nodiscard]] const Storage& storage() const noexcept { return storage_; }

    /// @brief Get strategy object (for accessing strategy-specific parameters like UKF
    /// alpha/beta/kappa)
    [[nodiscard]] const StrategyType& strategy() const noexcept { return strategy_; }
    [[nodiscard]] StrategyType& strategy() noexcept { return strategy_; }

    // ========================================================================
    // Numerical Stability Methods
    // ========================================================================

    /// @brief Symmetrize covariance matrix: P ← (P + P^T) / 2
    void symmetrize() noexcept {
        MatrixDD P = mat();
        P          = (P + P.transpose()) / 2.0;
        storage_.update(P);
    }

    /// @brief Add small regularization to diagonal for numerical stability
    void regularize(double eps = 1e-12) noexcept {
        MatrixDD P = mat();
        P.diagonal().array() += eps;
        storage_.update(P);
    }

    /// @brief Check if all covariance values are finite
    [[nodiscard]] bool is_finite() const noexcept { return mat().allFinite(); }

private:
    Storage storage_;
    StrategyType strategy_;
};

} // namespace uni_kf::redact
