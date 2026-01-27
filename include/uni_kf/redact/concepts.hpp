#pragma once
#include "space.hpp"
#include <Eigen/Dense>
#include <concepts>

namespace uni_kf::redact {

// ============================================================================
// Lie Group Concept
// ============================================================================

/// @brief Concept for Lie groups with exponential/logarithm maps
///
/// LieGroup extends Space with:
/// - exp: Tangent → Manifold (exponential map)
/// - log: Manifold → Tangent (logarithm map)
/// - adjoint_matrix: Adjoint operator Ad_X
/// - inv_adjoint_matrix: Inverse Adjoint operator Ad_X⁻¹
/// - inverse: Group inverse X⁻¹
///
/// ## Mathematical Background
///
/// For a Lie group G with Lie algebra g:
/// - exp: g → G maps tangent vectors to group elements
/// - log: G → g maps group elements to tangent vectors
/// - Ad_X: g → g is the adjoint operator at X ∈ G
///
/// ## Error Definition
///
/// For right-invariant error (used by InKF/InEKF):
///   η_R = Log(X̂⁻¹ · X)  (error in tangent space)
///
/// Key property: exp(diff(x, y)) = x⁻¹ · y by definition for Lie groups.
/// This is NOT a derived property, but the DEFINITION of diff.
template <typename T>
concept LieGroup = Space<T> && requires(typename T::Flux a, typename T::Drift xi) {
    /// Exponential map: exp(ξ) = X (tangent → manifold)
    /// For SO(3): exp(ω) = R (Rodrigues formula)
    /// For SE(3): exp(ξ) = T (matrix exponential)
    { T::exp(xi) } -> std::same_as<typename T::Flux>;

    /// Logarithm map: log(X) = ξ (manifold → tangent)
    /// For SO(3): log(R) = ω (axis-angle)
    /// For SE(3): log(T) = ξ (twist coordinates)
    { T::log(a) } -> std::same_as<typename T::Drift>;

    /// Adjoint operator: Ad_X : g → g
    /// For SO(3): Ad_R(ω) = R · ω (rotation of angular velocity)
    /// For SE(3): Ad_T(ξ) = [R, t; 0, 1] · [ω; v] (spatial transform)
    { T::adjoint_matrix(a) } -> std::convertible_to<Eigen::MatrixXd>;

    /// Inverse Adjoint operator: Ad_X⁻¹ : g → g
    { T::inv_adjoint_matrix(a) } -> std::convertible_to<Eigen::MatrixXd>;

    /// Group inverse: X⁻¹
    /// For SO(3): R⁻¹ = R^T
    /// For SE(3): T⁻¹ = [R^T, -R^T·t; 0, 1]
    { T::inverse(a) } -> std::same_as<typename T::Flux>;
};

/// @brief Alias for LieGroup (used by InKF/InEKF strategies)
///
/// InvariantGroup = LieGroup for right-invariant error definitions.
/// The "invariant" refers to the error definition η_R = Log(X̂⁻¹X) which
/// transforms predictably under group actions.
template <typename T>
concept InvariantGroup = LieGroup<T>;

// ============================================================================
// Linear Dynamics Concept
// ============================================================================

/// @brief Concept for spaces with constant state transition matrix
///
/// LinearDynamics requires:
/// - state_transition_matrix(): Returns constant F matrix
/// - Suitable for systems with time-invariant linear dynamics
template <typename T>
concept LinearDynamics = Space<T> && requires {
    /// State transition matrix F (constant, time-invariant)
    /// For discrete-time LTI systems: x_{k+1} = F · x_k
    { T::state_transition_matrix() } -> std::convertible_to<Eigen::MatrixXd>;
};

// ============================================================================
// Dynamics Linearizable Concept (for EKF/InEKF prediction)
// ============================================================================

/// @brief Concept for spaces that can be linearized for dynamics prediction
///
/// DynamicsLinearizable requires:
/// - dynamics_jacobian(state, dt): Returns Jacobian J_f = ∂f/∂x at current state
/// - Suitable for EKF and InEKF prediction steps
/// - This is the MINIMAL requirement for EKF-style prediction
template <typename T>
concept DynamicsLinearizable = Space<T> && requires(typename T::Flux state, double dt) {
    /// Dynamics Jacobian: J_f = df/dx at current state
    /// For nonlinear dynamics x' = f(x, dt), computes ∂f/∂x at current state
    { T::dynamics_jacobian(state, dt) } -> std::convertible_to<Eigen::MatrixXd>;
};

// ============================================================================
// Measurement Linearizable Concept (for EKF update)
// ============================================================================

/// @brief Concept for spaces that can be linearized for measurement update
///
/// MeasurementLinearizable requires:
/// - measurement_jacobian(state): Returns Jacobian J_h = ∂h/∂x at current state
/// - Suitable for EKF measurement update steps
template <typename T>
concept MeasurementLinearizable = Space<T> && requires(typename T::Flux state) {
    /// Measurement Jacobian: J_h = dh/dx at current state
    /// For measurement model z = h(x), computes ∂h/∂x at current state
    { T::measurement_jacobian(state) } -> std::convertible_to<Eigen::MatrixXd>;
};

// ============================================================================
// Combined Linearizable Concept (for full EKF)
// ============================================================================

/// @brief Concept for spaces with both dynamics and measurement linearization
///
/// Linearizable = DynamicsLinearizable + MeasurementLinearizable
/// This is the traditional EKF requirement where both prediction and update
/// use Jacobian linearization.
template <typename T>
concept Linearizable = DynamicsLinearizable<T> && MeasurementLinearizable<T>;

// ============================================================================
// Invariant Linearizable Concept (for InEKF)
// ============================================================================

/// @brief Concept for Lie groups with linearizable dynamics
///
/// InvariantLinearizableGroup requires:
/// - LieGroup structure (exp, log, adjoint)
/// - dynamics_jacobian for nonlinear dynamics on Lie groups
/// - Suitable for InEKF (Invariant Extended Kalman Filter)
///
/// NOTE: Does NOT require measurement_jacobian because InEKF can use
/// invariant measurement definitions that don't require Jacobians.
template <typename T>
concept InvariantLinearizableGroup = InvariantGroup<T> && DynamicsLinearizable<T>;

// ============================================================================
// Sampleable Concept (for UKF/CKF)
// ============================================================================

/// @brief Concept for spaces that support sigma-point sampling
///
/// Sampleable requires:
/// - sigma_points(mean, cov, alpha, beta, kappa): Generate weighted sigma points
/// - mean(points): Compute weighted mean from sigma points
/// - covariance(points, mean): Compute weighted covariance from sigma points
/// - Suitable for UKF and CKF
template <typename T>
concept Sampleable =
    Space<T>
    && requires(
        typename T::Flux mean, Eigen::MatrixXd cov, double alpha, double beta, double kappa) {
           /// Generate weighted sigma points around mean with given covariance
           /// Returns vector of (point, weight_mean, weight_cov) tuples
           /// UKF parameters (Julier & Uhlmann 2004):
           ///   alpha: Spread of sigma points (1e-3 for low-dim, 1.0 for high-dim)
           ///   beta: 2.0 for Gaussian distributions
           ///   kappa: Secondary scaling (0 or 3-n)
           {
               T::sigma_points(mean, cov, alpha, beta, kappa)
           } -> std::same_as<std::vector<typename T::WeightedPoint>>;

           /// Compute weighted mean from sigma points
           /// Uses weighted mean formula: μ = Σ w_i · x_i
           {
               T::mean(std::declval<std::vector<typename T::WeightedPoint>>())
           } -> std::same_as<typename T::Flux>;

           /// Compute weighted covariance from sigma points around mean
           /// Uses weighted covariance formula: P = Σ w_i · (x_i - μ) · (x_i - μ)^T
           {
               T::covariance(std::declval<std::vector<typename T::WeightedPoint>>(), mean)
           } -> std::same_as<Eigen::MatrixXd>;
       };

// ============================================================================
// Combined Concepts
// ============================================================================

/// @brief Spaces that are both linear and linearizable
template <typename T>
concept LinearAndLinearizable = LinearDynamics<T> && Linearizable<T>;

} // namespace uni_kf::redact
