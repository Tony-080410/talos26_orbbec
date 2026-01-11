/// @file uni_kf.hpp
/// @brief Unified Kalman Filter Framework - Single include header
///
/// This header includes all components of the uni_kf framework.
/// For selective imports, include specific headers from subdirectories.
///
/// ## Quick Start
///
/// ```cpp
/// #include <uni_kf/uni_kf.hpp>
///
/// // Create a 3D constant-velocity EKF
/// uni_kf::EKF_3D filter;
/// filter.init(initial_state, initial_covariance);
///
/// // Predict
/// filter.predict(cv_model, dt);
///
/// // Update
/// filter.update(measurement, measurement_model);
/// ```
///
/// ## Framework Overview
///
/// ### DifferentialSpace Concept
/// The core abstraction unifying vector spaces and Lie groups:
/// - `Euclidean<N>` - R^n vector spaces
/// - `LieManifold<G, Side>` - Lie groups (SO3, SE3, SE₂(3))
/// - `ProductSpace<D1, D2>` - Cartesian products
///
/// ### Filter Types
/// - `EKF<N>` - Extended Kalman Filter on R^n
/// - `UKF<N>` - Unscented Kalman Filter on R^n
/// - `LieEKF<G>` - EKF on Lie groups
/// - `InEKF_SE23` - Invariant EKF on SE₂(3)
/// - `IEKF<N>` - Iterated EKF
///
/// ### Uncertainty Representations
/// - `CovarianceRepr` - Standard covariance matrix P
/// - `InformationRepr` - Information matrix Y = P^{-1}
/// - `SquareRootRepr` - Square root S where P = SS^T
///
/// ## Example: Robot Tracking with InEKF
///
/// ```cpp
/// #include <uni_kf/uni_kf.hpp>
///
/// // State: SE₂(3) × R⁴ (pose + [v_yaw, r0, r1, h])
/// uni_kf::InEKF_Robot filter = uni_kf::make_robot_inekf(
///     initial_position, initial_yaw);
///
/// // Predict with constant velocity model
/// filter.predict([](auto x, double dt) {
///     // ... dynamics
/// }, process_noise, dt);
///
/// // Update with armor observation
/// filter.update<uni_kf::R3>(armor_position, measurement_model);
/// ```

#pragma once

// ============================================================================
// Concepts
// ============================================================================

#include "concepts/differential_space.hpp"
#include "concepts/linear.hpp"
#include "concepts/quadratic_form.hpp"

// ============================================================================
// Spaces
// ============================================================================

#include "spaces/euclidean.hpp"
#include "spaces/lie_manifold.hpp"
#include "spaces/product_space.hpp"

// ============================================================================
// Uncertainty Representations
// ============================================================================

#include "representations/covariance.hpp"
#include "representations/information.hpp"
#include "representations/square_root.hpp"

// ============================================================================
// Propagation Strategies
// ============================================================================

#include "strategies/cubature_points.hpp"
#include "strategies/first_order.hpp"
#include "strategies/sigma_points.hpp"

// ============================================================================
// Core Components
// ============================================================================

#include "core/belief.hpp"
#include "core/filter.hpp"
#include "core/models.hpp"
#include "core/types.hpp"

// ============================================================================
// Presets
// ============================================================================

#include "presets/ckf.hpp"
#include "presets/ekf.hpp"
#include "presets/iekf.hpp"
#include "presets/inekf.hpp"
#include "presets/tracker_inekf.hpp"
#include "presets/tracker_srckf.hpp"
#include "presets/ukf.hpp"

// ============================================================================
// Version Information
// ============================================================================

namespace uni_kf {

/// Framework version
inline constexpr struct {
    int major = 0;
    int minor = 1;
    int patch = 0;
} version;

/// Framework name
inline constexpr const char* name = "uni_kf";

/// Framework description
inline constexpr const char* description =
    "Unified Kalman Filter Framework - Type-safe filters on DifferentialSpaces";

} // namespace uni_kf
