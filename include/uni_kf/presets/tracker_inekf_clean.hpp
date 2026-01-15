/// @file tracker_inekf_clean.hpp
/// @brief Clean InEKF tracker using real uni_kf UnifiedKalmanFilter
///
/// Demonstrates how to properly use UnifiedKalmanFilter for InEKF tracking.
/// This is a MINIMAL WORKING EXAMPLE - extend as needed for production use.

#pragma once

#include "../core/filter.hpp"
#include "../core/models.hpp"
#include "../presets/inekf.hpp"
#include "../spaces/product_space.hpp"

#include <Eigen/Core>
#include <groups/SEn3.hpp>

namespace uni_kf {

// ============================================================================
// Clean InEKF Tracker Example
// ============================================================================

/// @brief Minimal InEKF tracker using real UnifiedKalmanFilter
///
/// State: SE₂(3) × ℝ⁴ = (R, v, p, [v_yaw, r0, r1, h])
///
/// Key insight: This is just a thin wrapper around UnifiedKalmanFilter.
/// All Kalman math is handled by the filter. We only need to:
/// 1. Define the process model
/// 2. Call filter.predict(model, dt)
/// 3. Call filter.update<MeasSpace>(z, h, R)
///
/// Compare this to the old tracker_inekf.hpp which manually implements
/// all the Kalman equations. This version is ~10x shorter and clearer.
///
class MinimalInEKFTracker {
public:
    // State space: SE₂(3) × ℝ⁴
    using SE23       = group::SEn3<double, 2>;
    using LiePart    = LieManifold<SE23, LieSide::Right>;
    using ExtraPart  = Euclidean<4>;
    using StateSpace = ProductSpace<LiePart, ExtraPart>;

    // Filter type: Choose your fighter!
    // - InEKF: FirstOrderTag (EKF on manifold)
    // - InUKF: SigmaPointsTag (UKF on manifold)
    // - InCKF: CubaturePointsTag (CKF on manifold) ← recommended for stability
    using Filter = UnifiedKalmanFilter<StateSpace, CubaturePointsTag, CovarianceTag>;

    // Simple constant-velocity process model
    struct ProcessModel {
        using StateSpace = MinimalInEKFTracker::StateSpace;

        [[nodiscard]] auto predict(const typename StateSpace::Point& x, double dt) const noexcept ->
            typename StateSpace::Point {
            const auto& se23   = x.first;  // SE₂(3) part
            const auto& extras = x.second; // ℝ⁴ part [v_yaw, r0, r1, h]

            // Constant velocity + rotation prediction
            double v_yaw               = extras.data(0);
            Eigen::Vector3d omega_body = Eigen::Vector3d::UnitZ() * v_yaw;

            auto R_new = se23.rotation() * SE23::SO3Type::exp(omega_body * dt);
            auto v_new = se23.velocity(); // Constant velocity
            auto p_new = se23.position() + se23.velocity() * dt;

            return StateSpace::Point{
                SE23{R_new, v_new, p_new},
                extras  // Extras remain constant
            };
        }

        [[nodiscard]] auto process_noise(double dt) const noexcept
            -> SymmetricMatrix<StateSpace::FiberDim, double> {
            // Process noise: 9 (SE23 tangent) + 4 (extras) = 13 dim
            Eigen::Matrix<double, 13, 1> noise_std;
            noise_std.segment<3>(0) = Eigen::Vector3d::Constant(0.01); // rotation
            noise_std.segment<3>(3) = Eigen::Vector3d::Constant(0.1);  // velocity
            noise_std.segment<3>(6) = Eigen::Vector3d::Constant(0.05); // position
            noise_std.segment<4>(9) = Eigen::Vector4d::Constant(0.01); // extras

            return SymmetricMatrix<13, double>::Diagonal(noise_std.array().square() * dt);
        }
    };

    static_assert(uni_kf::ProcessModel<ProcessModel, StateSpace>);

private:
    Filter filter_;
    ProcessModel process_model_;
    bool initialized_ = false;

public:
    MinimalInEKFTracker() noexcept = default;

    /// @brief Initialize filter with initial state
    void init(
        const Eigen::Vector3d& position, double yaw, const Eigen::Vector3d& velocity, double v_yaw,
        const Eigen::Vector4d& extras_init // [v_yaw, r0, r1, h]
        ) noexcept {
        // Build SE₂(3) element
        auto R = SE23::SO3Type::from_euler_zyx(yaw, 0, 0);
        SE23 se23{R, velocity, position};

        // Build extras
        auto extras = ExtraPart::Point{extras_init};

        // Combined state
        typename StateSpace::Point state{se23, extras};

        // Initial covariance (product space: 9 + 4 = 13)
        auto cov = SymmetricMatrix<13, double>::Identity() * 0.01;

        filter_.init(state, cov);
        initialized_ = true;
    }

    /// @brief Predict step - just forwards to filter!
    void predict(double dt) noexcept {
        if (!initialized_)
            return;
        filter_.predict(process_model_, dt);
    }

    /// @brief Update with position measurement (simplified)
    ///
    /// Full InEKF uses invariant measurement: z = R^T(y - p)
    /// This is simplified for demonstration.
    void update_position(
        const Eigen::Vector3d& measured_pos, const Eigen::Matrix3d& R_meas) noexcept {
        if (!initialized_)
            return;

        using MeasSpace = Euclidean<3>;

        // Measurement function: h(x) = position
        auto measurement_fn = [](const typename StateSpace::Point& x) ->
            typename MeasSpace::Point { return MeasSpace::Point{x.first.position()}; };

        typename MeasSpace::Point z{measured_pos};
        filter_.update<MeasSpace>(z, measurement_fn, R_meas);
    }

    /// @brief Get current state estimate
    [[nodiscard]] auto get_state() const noexcept {
        const auto& belief = filter_.belief();
        const auto& state  = belief.mean;

        struct Output {
            Eigen::Vector3d position;
            Eigen::Vector3d velocity;
            double yaw;
            double v_yaw;
            Eigen::Vector4d extras;
        };

        Output out;
        out.position = state.first.position();
        out.velocity = state.first.velocity();
        out.yaw      = std::atan2(state.first.rotation()(1, 0), state.first.rotation()(0, 0));
        out.v_yaw    = state.second.data(0);
        out.extras   = state.second.data;

        return out;
    }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] const Filter& filter() const noexcept { return filter_; }
};

// ============================================================================
// Key Takeaways
// ============================================================================

/// This tracker is ~150 lines vs 1000+ in the old version because:
///
/// 1. **No manual Kalman math**: UnifiedKalmanFilter handles everything
/// 2. **ProcessModel concept**: Just define predict() and process_noise()
/// 3. **Strategy composability**: Change CubaturePointsTag to SigmaPointsTag for UKF
/// 4. **Representation composability**: Change CovarianceTag to SquareRootTag for SRUKF
///
/// To extend this:
/// - Add proper invariant measurement (R^T(y - p) form)
/// - Add multiple measurement types (armor, IMU, etc.)
/// - Add outlier rejection / innovation gating
/// - Integrate with actual fcs::L3 config types
///
/// But the core Kalman filter logic? That's all handled by UnifiedKalmanFilter.

} // namespace uni_kf
