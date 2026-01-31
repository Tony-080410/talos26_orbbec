#pragma once

#include "filter.hpp"
#include <Eigen/Dense>
#include <concepts>
#include <groups/SEn3.hpp>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni_kf::redact {

// ============================================================================
// Detail Namespace - Helper Concepts and Functions
// ============================================================================

namespace detail {

/// @brief Concept: Does the LieGroup support adjoint transformations?
///
/// This is required for frame adjustment on Lie groups, where the covariance
/// transforms as: P' = Ad_C * P * Ad_C^T
template <typename LieGroup>
concept HasAdjoint = requires(const LieGroup& x) {
    { x.Adjoint() } -> std::convertible_to<Eigen::MatrixXd>;
};

/// @brief Concept: Can we extract SE(3) from ReferenceFrame?
///
/// This enables frame adjustment for reference frames that provide an SE(3)
/// transformation (e.g., fast_tf::FuckTransform, group::SE3d).
template <typename RefFrame>
concept HasSE3 = requires(const RefFrame& frame) {
    { frame->se3() } -> std::same_as<const group::SE3d&>;
};

/// @brief Concept: Does Filter expose mutable state and covariance?
///
/// This is required for in-place frame adjustment for filters that store mean
/// and covariance separately (e.g., redact::KalmanFilter).
template <typename Filter>
concept HasMutableStateAndCovariance = requires(Filter& f) {
    { f.state_mut() } -> std::same_as<typename Filter::Belief::Flux&>;
    { f.get_unc() };
};

template <typename Flux>
concept HasSe23Isometries = requires { typename Flux::IsometriesType; };

template <typename Flux>
concept HasFirstSe23Isometries = requires {
    typename std::remove_cvref_t<decltype(std::declval<Flux>().first)>::IsometriesType;
};

template <typename Flux>
concept HasSecondVector = requires(Flux x) { x.second.setZero(); };

template <typename Flux>
[[nodiscard]] inline Flux lift_reference_transform_se23_pure_rotation(const group::SE3d& C23) {
    // Lift an SE(3) reference drift into an SE_2(3)-like state using a pure rotation
    // (translation/velocity offsets are set to 0).
    if constexpr (HasSe23Isometries<Flux>) {
        typename Flux::IsometriesType se23;
        se23[0] = Eigen::Vector3d::Zero();
        se23[1] = Eigen::Vector3d::Zero();
        return Flux(C23.R(), se23);
    } else if constexpr (HasFirstSe23Isometries<Flux> && HasSecondVector<Flux>) {
        using Se23 = std::remove_cvref_t<decltype(std::declval<Flux>().first)>;
        typename Se23::IsometriesType se23;
        se23[0] = Eigen::Vector3d::Zero();
        se23[1] = Eigen::Vector3d::Zero();
        Flux out{};
        out.first = Se23(C23.R(), se23);
        out.second.setZero();
        return out;
    } else if constexpr (std::is_same_v<Flux, group::SE3d>) {
        return C23;
    } else {
        static_assert(
            sizeof(Flux) == 0,
            "Cannot lift group::SE3d reference transform into this state type. "
            "Provide a state type that is SE(3)/SE_2(3) like, or a product with .first/.second.");
    }
}

/// @brief Compute drift transformation C23 = fresh.inv() * stale
///
/// This computes the coordinate transformation from the stale (old) reference
/// frame to the fresh (new) reference frame.
///
/// ## Mathematical Background
///
/// For state X represented in frame T_stale:
///   X_stale = T_stale^{-1} * X_world
///
/// When frame moves to T_fresh, we need:
///   X_fresh = T_fresh^{-1} * X_world = (T_fresh^{-1} * T_stale) * X_stale
///
/// The drift transformation is:
///   C23 = T_fresh^{-1} * T_stale = fresh.inv() * stale
///
/// ## Parameters
/// - stale: Old reference frame (T_stale)
/// - fresh: New reference frame (T_fresh)
///
/// ## Returns
/// - SE(3) transformation from stale to fresh frame (returns actual SE(3) type, not LieManifold
/// wrapper)
template <typename RefFrame>
[[nodiscard]] auto compute_drift_transform(const RefFrame& stale, const RefFrame& fresh) {
    if constexpr (HasSE3<RefFrame>) {
        // For fast_tf::FuckTransform and similar types
        return fresh->se3().inv() * stale->se3();
    } else {
        // For direct SE(3) types
        return fresh.inv() * stale;
    }
}

} // namespace detail

// ============================================================================
// Adjusted<> Wrapper
// ============================================================================

/// @brief Wrapper that adds frame adjustment capability to any KalmanFilter-like type
///
/// The Adjusted wrapper uses composition + perfect forwarding to add automatic
/// frame adjustment to any filter type. When the reference frame changes, the
/// filter's state and covariance are adjusted to maintain consistency.
///
/// ## Syntax
/// ```cpp
/// Adjusted<KalmanFilter<Cov, P, A>, ReferenceFrame> tracker;
/// ```
///
/// ## Key Features
/// 1. **No modifications to existing code** - works with any filter type
/// 2. **Perfect forwarding constructors** - accepts any arguments the wrapped filter accepts
/// 3. **Transparent interface** - forwards all methods from wrapped filter
/// 4. **Optional auto-adjust** - `predict()` and `percept()` accept optional reference frame
///
/// ## Frame Adjustment
///
/// When the reference frame changes (e.g., gimbal moves), the state must be adjusted:
///   X' = C23 * X
///   P' = Ad_C23 * P * Ad_C23^T
///
/// where C23 = T_fresh^{-1} * T_stale is the drift transformation.
///
/// For SE_2(3) (9D state with velocity), we use a pure rotation (velocity=0, position=0)
/// for the frame change. This ensures the Adjoint matrix is block-diagonal, correctly
/// rotating all covariance components:
///   - Position covariance: P_pp' = R * P_pp * R^T
///   - Velocity covariance: P_vv' = R * P_vv * R^T
///   - Angular velocity covariance: P_ωω' = R * P_ωω * R^T
///
/// The position offset from the reference frame change does NOT affect the target's
/// velocity, so we use pure rotation rather than including the translation.
///
/// ## Example Usage
/// ```cpp
/// // Define types
/// using MyLieGroup = group::SE3<double>;
/// using MyBelief = Belief<MyLieGroup>;
/// using MyCovariance = Covariance<MyLieGroup, Representation::Direct,
/// InvariantLinearStrategy<MyLieGroup>>; using MyFilter = KalmanFilter<MyCovariance, MyPerception,
/// MyAssumption>; using AdjustedFilter = Adjusted<MyFilter, FuckTransform<odom>>;
///
/// // Create tracker
/// AdjustedFilter tracker(initial_belief, initial_cov, perception, assumption);
///
/// // Initialize with reference frame
/// FuckTransform<odom> gimbal_pose = /* ... */;
/// tracker.init(gimbal_pose, initial_state, initial_cov);
///
/// // Use with auto-adjust
/// FuckTransform<odom> new_gimbal_pose = /* ... */;
/// tracker.predict(dt, new_gimbal_pose);           // Auto-adjusts before predict
/// tracker.percept(measurement, dt, new_gimbal_pose); // Auto-adjusts before percept
///
/// // Or manual adjust
/// tracker.adjust(new_gimbal_pose);
/// tracker.predict(dt);
/// ```
///
/// ## Template Parameters
/// @tparam Filter The filter type to wrap (must satisfy HasMutableBelief)
/// @tparam ReferenceFrame The reference frame type (e.g., fast_tf::FuckTransform<Frame>)
template <typename Filter, typename ReferenceFrame>
class Adjusted {
private:
    using Belief      = typename Filter::Belief;
    using Measurement = typename Filter::Measurement;
    using MatrixDD    = typename Filter::MatrixDD;

    Filter filter_;                                ///< Wrapped filter instance
    std::optional<ReferenceFrame> last_reference_; ///< Last reference frame for drift computation
    bool initialized_ = false;                     ///< Has the filter been initialized?

public:
    // ========================================================================
    // Type Aliases
    // ========================================================================

    using BeliefType      = Belief;
    using MeasurementType = Measurement;

    // ========================================================================
    // Constructors (Perfect Forwarding)
    // ========================================================================

    /// @brief Default constructor
    Adjusted() = default;

    /// @brief Forward all arguments to wrapped filter constructor
    ///
    /// This constructor uses perfect forwarding to accept any arguments that
    /// the wrapped filter accepts. This enables flexible construction:
    /// ```cpp
    /// Adjusted<KalmanFilter<...>, RefFrame> tracker(
    ///     initial_belief,           // Forwarded to KalmanFilter
    ///     initial_covariance,
    ///     perception_model,
    ///     assumption_model
    /// );
    /// ```
    template <typename... Args>
    explicit Adjusted(Args&&... args)
        : filter_(std::forward<Args>(args)...)
        , initialized_(false) {}

    // ========================================================================
    // Core: Frame Adjustment
    // ========================================================================

    /// @brief Adjust filter state to new reference frame
    ///
    /// This method updates the filter's state and covariance to account for
    /// a change in the reference frame. The adjustment is a deterministic
    /// coordinate transformation.
    ///
    /// ## Adjustment Logic
    ///
    /// 1. **No previous reference**: Store reference, mark as uninitialized
    /// 2. **Not initialized**: Update reference only, no transformation
    /// 3. **Has mutable belief + Lie group support**: Apply frame adjustment
    /// 4. **No mutable belief or no adjoint**: No transformation possible
    ///
    /// ## Frame Transformation
    ///
    /// For SE(3) and SO(3):
    ///   X' = C23 * X
    ///   P' = Ad_C23 * P * Ad_C23^T
    ///
    /// For SE_2(3) with SE(3) frame:
    ///   We use pure rotation (no position offset) for frame changes:
    ///   ```cpp
    ///   se23[0] = {0, 0, 0};      // Zero velocity
    ///   se23[1] = {0, 0, 0};      // Zero position (pure rotation)
    ///   C23_se23 = SE23(C23.R(), se23);
    ///   ```
    ///   This ensures the Adjoint is block-diagonal, correctly rotating all
    ///   covariance components without introducing spurious cross-terms.
    ///
    /// ## Parameters
    /// @param new_reference The new reference frame
    void adjust(const ReferenceFrame& new_reference) noexcept {
        // Case 1: No previous reference
        if (!last_reference_) {
            last_reference_ = new_reference;
            initialized_    = false;
            return;
        }

        // Case 2: Not initialized
        if (!initialized_) {
            last_reference_ = new_reference;
            return;
        }

        // Case 3: Filter exposes mutable state + covariance + Adjoint
        if constexpr (
            detail::HasMutableStateAndCovariance<Filter>
            && detail::HasAdjoint<typename Belief::Flux>) {
            const auto C23 = detail::compute_drift_transform(*last_reference_, new_reference);

            // Lift SE(3) drift into the filter's state space (SE_2(3) uses pure rotation).
            const typename Belief::Flux C =
                detail::lift_reference_transform_se23_pure_rotation<typename Belief::Flux>(C23);

            // State coordinate change (left action): X' = C * X
            filter_.state_mut() = C * filter_.state();

            // Covariance coordinate change: P' = Ad_C * P * Ad_C^T
            const Eigen::MatrixXd Ad = C.Adjoint();
            const Eigen::MatrixXd P  = filter_.get_unc().mat();
            Eigen::MatrixXd result   = Ad * P * Ad.transpose();
            result                   = (result + result.transpose()) / 2.0;
            result.diagonal().array() += 1e-12;
            filter_.get_unc().storage().update(result);
        }
        // Case 4: No mutable belief access or no adjoint - no transformation possible

        last_reference_ = new_reference;
    }

    // ========================================================================
    // Initialization
    // ========================================================================

    /// @brief Initialize filter with reference frame, initial state, and covariance
    ///
    /// @param reference The reference frame (e.g., gimbal pose)
    /// @param initial_belief The initial state estimate
    /// @param initial_cov The initial covariance matrix
    void init(
        const ReferenceFrame& reference, typename Belief::Flux initial_belief,
        MatrixDD initial_cov) {
        (void)initial_belief;
        (void)initial_cov;
        last_reference_ = reference;
        initialized_    = true;
        // Note: Wrapped filter must be initialized separately or via constructor
        // This just sets up the reference frame tracking
    }

    /// @brief Initialize filter from a measurement
    ///
    /// This initializes the filter state from a measurement, using the reference
    /// frame to transform the measurement into the filter's state space.
    ///
    /// @param reference The reference frame (e.g., gimbal pose)
    /// @param measurement The measurement to initialize from
    template <typename Meas>
    void init_from_measurement(const ReferenceFrame& reference, Meas measurement) {
        (void)measurement;
        last_reference_ = reference;
        initialized_    = true;
        // Note: Actual initialization depends on the specific filter type
        // This just sets up the reference frame tracking
    }

    // ========================================================================
    // Auto-adjusting predict/percept
    // ========================================================================

    /// @brief Predict step with optional auto-adjust
    ///
    /// If a reference frame is provided, adjusts the filter to the new frame
    /// before prediction. This enables seamless tracking when the reference
    /// frame (e.g., gimbal) is moving.
    ///
    /// @param dt Time step
    /// @param reference Optional new reference frame (triggers auto-adjust)
    /// @return Predicted state
    auto predict(double dt, const std::optional<ReferenceFrame>& reference = std::nullopt) {
        if (reference) {
            adjust(*reference);
        }
        return filter_.predict(dt);
    }

    /// @brief Measurement update with optional auto-adjust
    ///
    /// If a reference frame is provided, adjusts the filter to the new frame
    /// before the measurement update.
    ///
    /// @param meas The measurement
    /// @param dt Time step
    /// @param reference Optional new reference frame (triggers auto-adjust)
    /// @return Result of percept (expected<void, PerceptError> for KalmanFilter)
    auto percept(
        typename Measurement::Flux meas, double dt,
        const std::optional<ReferenceFrame>& reference = std::nullopt) {
        if (reference) {
            adjust(*reference);
        }
        return filter_.percept(std::move(meas), dt);
    }

    // ========================================================================
    // Forwarded Accessors
    // ========================================================================

    /// @brief Get current state estimate
    [[nodiscard]] const typename Belief::Flux& state() const noexcept { return filter_.state(); }

    /// @brief Get current covariance matrix
    [[nodiscard]] MatrixDD covariance() const noexcept { return filter_.covariance(); }

    /// @brief Get const reference to wrapped filter
    [[nodiscard]] const Filter& get_filter() const noexcept { return filter_; }

    /// @brief Get mutable reference to wrapped filter
    [[nodiscard]] Filter& get_filter() noexcept { return filter_; }

    /// @brief Get perception (measurement model) if available
    ///
    /// This method is only available if the wrapped filter provides a
    /// get_perception() method.
    template <typename F = Filter>
    [[nodiscard]] auto& get_perception() noexcept
        requires requires { std::declval<F&>().get_perception(); } {
        return filter_.get_perception();
    }

    /// @brief Get assumption (dynamics model) if available
    ///
    /// This method is only available if the wrapped filter provides a
    /// get_assumption() method.
    template <typename F = Filter>
    [[nodiscard]] auto& get_assumption() noexcept
        requires requires { std::declval<F&>().get_assumption(); } {
        return filter_.get_assumption();
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /// @brief Check if filter has been initialized
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// @brief Reset filter to initial state
    ///
    /// Clears the reference frame tracking and marks the filter as uninitialized.
    void reset() noexcept {
        last_reference_.reset();
        initialized_ = false;
    }
};

// ============================================================================
// Type Traits - Compile-time Verification
// ============================================================================

namespace detail {

/// @brief Type trait to verify Adjusted works with KalmanFilter
template <typename Cov, Perception<Cov> P, Assumption<Cov> A, typename RefFrame>
inline constexpr bool AdjustedKalmanFilterValid_v =
    requires(Adjusted<KalmanFilter<Cov, P, A>, RefFrame> adj) {
        {
            adj.state()
        } -> std::convertible_to<const typename KalmanFilter<Cov, P, A>::Belief::Flux&>;
        { adj.covariance() } -> std::convertible_to<Eigen::MatrixXd>;
        { adj.get_filter() } -> std::same_as<KalmanFilter<Cov, P, A>&>;
    };

} // namespace detail

} // namespace uni_kf::redact
