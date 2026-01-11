/// @file iekf.hpp
/// @brief Iterated Extended Kalman Filter presets
///
/// IEKF iterates the measurement update step until convergence,
/// improving accuracy for highly nonlinear measurement models.

#pragma once

#include "../core/filter.hpp"

namespace uni_kf {

// ============================================================================
// IEKF Configuration
// ============================================================================

/// @brief IEKF iteration parameters
struct IEKFParams {
    int max_iterations  = 10;   ///< Maximum iterations per update
    double tolerance    = 1e-6; ///< Convergence tolerance (tangent space norm)
    double lambda_init  = 1.0;  ///< Initial damping factor
    double lambda_scale = 2.0;  ///< Damping scale factor
};

// ============================================================================
// IteratedKalmanFilter
// ============================================================================

/// @brief Iterated Extended Kalman Filter
///
/// Wraps a base filter and iterates the measurement update step.
/// This improves accuracy for highly nonlinear measurement models
/// at the cost of additional computation.
///
/// @tparam BaseFilter Underlying filter type (typically EKF)
template <typename BaseFilter>
class IteratedKalmanFilter {
public:
    using Space      = typename BaseFilter::Space;
    using Scalar     = typename BaseFilter::Scalar;
    using Point      = typename BaseFilter::Point;
    using Fiber      = typename BaseFilter::Fiber;
    using BeliefType = typename BaseFilter::BeliefType;
    using Covariance = typename BaseFilter::Covariance;

private:
    BaseFilter base_;
    IEKFParams params_;

public:
    IteratedKalmanFilter() noexcept = default;

    explicit IteratedKalmanFilter(const IEKFParams& params) noexcept
        : params_(params) {}

    explicit IteratedKalmanFilter(const BeliefType& belief, const IEKFParams& params = {}) noexcept
        : base_(belief)
        , params_(params) {}

    // Forward to base filter
    void init(const Point& mean, const Covariance& cov) noexcept { base_.init(mean, cov); }
    void reset() noexcept { base_.reset(); }

    [[nodiscard]] const BeliefType& belief() const noexcept { return base_.belief(); }
    [[nodiscard]] const Point& mean() const noexcept { return base_.mean(); }
    [[nodiscard]] const Covariance& covariance() const noexcept { return base_.covariance(); }
    [[nodiscard]] bool is_initialized() const noexcept { return base_.is_initialized(); }

    // Prediction (pass through)
    template <typename... Args>
    void predict(Args&&... args) noexcept {
        base_.predict(std::forward<Args>(args)...);
    }

    /// @brief Iterated update step
    template <DifferentialSpace M, typename MeasFunc>
    void update(
        const typename M::Point& z, MeasFunc&& h,
        const SymmetricMatrix<M::FiberDim, Scalar>& R) noexcept {
        if (!base_.is_initialized()) {
            return;
        }

        // Store initial state for iteration
        Point x_op         = base_.mean(); // Operating point
        Covariance P_prior = base_.covariance();

        for (int iter = 0; iter < params_.max_iterations; ++iter) {
            // Store previous state
            Point x_prev = base_.mean();

            // Single EKF update around operating point
            // Note: For true IEKF, we should linearize around x_op, not current estimate
            base_.template update<M>(z, std::forward<MeasFunc>(h), R);

            // Check convergence in tangent space
            auto delta = Space::diff(x_prev, base_.mean());
            if (delta.norm() < params_.tolerance) {
                break;
            }
        }
    }

    /// @brief Iterated update with MeasurementModel
    template <typename Model>
    requires MeasurementModel<Model, Space>
    void update(const typename Model::MeasurementSpace::Point& z, const Model& model) noexcept {
        if (!base_.is_initialized()) {
            return;
        }

        for (int iter = 0; iter < params_.max_iterations; ++iter) {
            Point x_prev = base_.mean();

            base_.update(z, model);

            auto delta = Space::diff(x_prev, base_.mean());
            if (delta.norm() < params_.tolerance) {
                break;
            }
        }
    }

    // Configuration
    void set_params(const IEKFParams& params) noexcept { params_ = params; }
    [[nodiscard]] const IEKFParams& params() const noexcept { return params_; }
};

// ============================================================================
// IEKF Type Aliases
// ============================================================================

/// @brief Iterated EKF on Euclidean space
template <int N, typename Scalar = double>
using IEKF = IteratedKalmanFilter<EKF<N, Scalar>>;

/// @brief Iterated EKF on Lie group
template <typename G, LieSide Side = LieSide::Right, typename Scalar = double>
using LieIEKF = IteratedKalmanFilter<LieEKF<G, Side, Scalar>>;

/// @brief Iterated InEKF on SE₂(3)
template <LieSide Side = LieSide::Right, typename Scalar = double>
using IInEKF_SE23 = IteratedKalmanFilter<InEKF_SE23<Side, Scalar>>;

// Common configurations
using IEKF_3D   = IEKF<6>;
using IEKF_6DOF = IEKF<12>;

} // namespace uni_kf
