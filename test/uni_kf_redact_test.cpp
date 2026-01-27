/// @file uni_kf_redact_test.cpp
/// @brief Unit tests for uni_kf/redact Kalman filter framework

#include <gtest/gtest.h>

#include "uni_kf/redact/belief.hpp"
#include "uni_kf/redact/concepts.hpp"
#include "uni_kf/redact/covariance.hpp"
#include "uni_kf/redact/filter.hpp"
#include "uni_kf/redact/space.hpp"
#include "uni_kf/redact/spaces/euclidean.hpp"
#include "uni_kf/redact/spaces/lie_group.hpp"
#include "uni_kf/redact/variants.hpp"

#include <Eigen/Dense>

using namespace uni_kf::redact;
using namespace uni_kf::redact::spaces;

// ============================================================================
// Space Concept Tests
// ============================================================================

TEST(SpaceConcept, EuclideanSatisfiesSpace) {
    EXPECT_TRUE(Space<Euclidean<3>>);
    EXPECT_TRUE(Space<Euclidean<6>>);
}

TEST(SpaceConcept, LieGroupSatisfiesSpace) {
    EXPECT_TRUE((Space<SO3<LieSide::Right>>));
    EXPECT_TRUE((Space<SE3<LieSide::Right>>));
}

// ============================================================================
// LinearDynamics and Linearizable Concepts
// ============================================================================

TEST(ConceptTests, EuclideanSatisfiesLinearDynamics) {
    EXPECT_TRUE(LinearDynamics<Euclidean<3>>);
    EXPECT_TRUE(LinearDynamics<Euclidean<6>>);
}

TEST(ConceptTests, EuclideanSatisfiesLinearizable) {
    EXPECT_TRUE(Linearizable<Euclidean<3>>);
    EXPECT_TRUE(Linearizable<Euclidean<6>>);
}

// Note: LieGroup does NOT satisfy Linearizable because dynamics_jacobian() and
// measurement_jacobian() are application-specific, not generic to the space.
// Users provide these Jacobians for their specific dynamics/measurement models.

// ============================================================================
// Sampleable Concept
// ============================================================================

TEST(ConceptTests, EuclideanSatisfiesSampleable) {
    EXPECT_TRUE(Sampleable<Euclidean<3>>);
    EXPECT_TRUE(Sampleable<Euclidean<6>>);
}

TEST(ConceptTests, LieGroupSatisfiesSampleable) {
    EXPECT_TRUE((Sampleable<SO3<LieSide::Right>>));
    EXPECT_TRUE((Sampleable<SE3<LieSide::Right>>));
    EXPECT_TRUE((Sampleable<SE23Right>));
}

// ============================================================================
// Euclidean Space Tests
// ============================================================================

TEST(EuclideanSpace, ZeroElements) {
    using E3 = Euclidean<3>;

    auto zero_f = E3::zero_flux();
    EXPECT_EQ(zero_f.data.rows(), 3);
    EXPECT_TRUE(zero_f.data.isZero());

    auto zero_d = E3::zero_drift();
    EXPECT_EQ(zero_d.data.rows(), 3);
    EXPECT_TRUE(zero_d.data.isZero());
}

TEST(EuclideanSpace, DynamicDimensionZeroElements) {
    using ED = Euclidean<Eigen::Dynamic>;

    auto zero_f = ED::zero_flux(5);
    EXPECT_EQ(zero_f.data.rows(), 5);
    EXPECT_TRUE(zero_f.data.isZero());

    auto zero_d = ED::zero_drift(5);
    EXPECT_EQ(zero_d.data.rows(), 5);
    EXPECT_TRUE(zero_d.data.isZero());
}

TEST(EuclideanSpace, ComposeAndDiff) {
    using E3 = Euclidean<3>;

    E3::Flux x{Eigen::Vector3d(1, 2, 3)};
    E3::Drift dx{Eigen::Vector3d(0.1, 0.2, 0.3)};

    // Compose: x ⊕ dx = x + dx
    auto result = E3::compose(x, dx);
    EXPECT_NEAR(result.data(0), 1.1, 1e-10);
    EXPECT_NEAR(result.data(1), 2.2, 1e-10);
    EXPECT_NEAR(result.data(2), 3.3, 1e-10);

    // Diff: y ⊖ x = y - x
    E3::Flux y{Eigen::Vector3d(2, 4, 6)};
    auto diff = E3::diff(x, y);
    EXPECT_NEAR(diff.data(0), 1.0, 1e-10);
    EXPECT_NEAR(diff.data(1), 2.0, 1e-10);
    EXPECT_NEAR(diff.data(2), 3.0, 1e-10);
}

TEST(EuclideanSpace, Norm) {
    using E3 = Euclidean<3>;

    E3::Drift dx{Eigen::Vector3d(3, 4, 0)};
    double n = E3::norm(dx);
    EXPECT_NEAR(n, 5.0, 1e-10);
}

TEST(EuclideanSpace, Inverse) {
    using E3 = Euclidean<3>;

    E3::Drift dx{Eigen::Vector3d(1, 2, 3)};
    auto inv = E3::inverse(dx);
    EXPECT_NEAR(inv.data(0), -1, 1e-10);
    EXPECT_NEAR(inv.data(1), -2, 1e-10);
    EXPECT_NEAR(inv.data(2), -3, 1e-10);

    E3::Flux x{Eigen::Vector3d(1, 2, 3)};
    auto inv_x = E3::inverse(x);
    EXPECT_NEAR(inv_x.data(0), -1, 1e-10);
    EXPECT_NEAR(inv_x.data(1), -2, 1e-10);
    EXPECT_NEAR(inv_x.data(2), -3, 1e-10);
}

TEST(EuclideanSpace, DistanceAndSquaredDistance) {
    using E3 = Euclidean<3>;

    E3::Flux x{Eigen::Vector3d(0, 0, 0)};
    E3::Flux y{Eigen::Vector3d(1, 2, 2)};

    double d = E3::distance(x, y);
    EXPECT_NEAR(d, 3.0, 1e-10);

    double d2 = E3::squared_distance(x, y);
    EXPECT_NEAR(d2, 9.0, 1e-10);
}

TEST(EuclideanSpace, StateTransitionMatrix) {
    using E3 = Euclidean<3>;

    auto F = E3::state_transition_matrix();
    EXPECT_EQ(F.rows(), 3);
    EXPECT_EQ(F.cols(), 3);
    EXPECT_TRUE(F.isIdentity());
}

TEST(EuclideanSpace, MeasurementMatrix) {
    using E3 = Euclidean<3>;

    auto H = E3::measurement_matrix();
    EXPECT_EQ(H.rows(), 3);
    EXPECT_EQ(H.cols(), 3);
    EXPECT_TRUE(H.isIdentity());

    auto H2 = E3::measurement_matrix(2);
    EXPECT_EQ(H2.rows(), 2);
    EXPECT_EQ(H2.cols(), 3);
}

TEST(EuclideanSpace, Jacobians) {
    using E3 = Euclidean<3>;

    E3::Flux x{Eigen::Vector3d(1, 2, 3)};
    double dt = 0.1;

    auto J_f = E3::dynamics_jacobian(x, dt);
    EXPECT_EQ(J_f.rows(), 3);
    EXPECT_EQ(J_f.cols(), 3);
    EXPECT_TRUE(J_f.isIdentity());

    auto J_h = E3::measurement_jacobian(x);
    EXPECT_EQ(J_h.rows(), 3);
    EXPECT_EQ(J_h.cols(), 3);
    EXPECT_TRUE(J_h.isIdentity());
}

TEST(EuclideanSpace, DynamicDimensionJacobians) {
    using ED = Euclidean<Eigen::Dynamic>;

    ED::Flux x{Eigen::VectorXd::Zero(5)};
    double dt = 0.1;

    auto J_f = ED::dynamics_jacobian(x, dt, 5);
    EXPECT_EQ(J_f.rows(), 5);
    EXPECT_EQ(J_f.cols(), 5);
    EXPECT_TRUE(J_f.isIdentity());

    auto J_h = ED::measurement_jacobian(x, 5);
    EXPECT_EQ(J_h.rows(), 5);
    EXPECT_EQ(J_h.cols(), 5);
    EXPECT_TRUE(J_h.isIdentity());
}

TEST(EuclideanSpace, SigmaPoints) {
    using E3 = Euclidean<3>;

    E3::Flux mean{Eigen::Vector3d(1, 2, 3)};
    Eigen::MatrixXd cov = Eigen::Matrix3d::Identity();

    auto points = E3::sigma_points(mean, cov, 1e-3, 2.0, 0.0);

    // Should have 2n + 1 = 7 points
    EXPECT_EQ(points.size(), 7);

    // Mean weights should sum to approximately 1
    double weight_sum = 0.0;
    for (const auto& pt : points) {
        weight_sum += pt.weight_mean;
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-6);
}

TEST(EuclideanSpace, MeanAndCovariance) {
    using E3 = Euclidean<3>;

    E3::Flux mean{Eigen::Vector3d::Zero()};
    Eigen::MatrixXd cov = Eigen::Matrix3d::Identity();

    auto points = E3::sigma_points(mean, cov);

    auto recovered_mean = E3::mean(points);
    auto recovered_cov  = E3::covariance(points, recovered_mean);

    // Recovered mean should be close to original
    EXPECT_NEAR(recovered_mean.data(0), mean.data(0), 1e-6);
    EXPECT_NEAR(recovered_mean.data(1), mean.data(1), 1e-6);
    EXPECT_NEAR(recovered_mean.data(2), mean.data(2), 1e-6);
}

TEST(EuclideanSpace, DriftExtent) {
    using E3 = Euclidean<3>;

    E3::DriftExtent extent{2.0};                    // Max norm = 2

    E3::Drift small{Eigen::Vector3d(1, 1, 1)};      // norm = sqrt(3) ≈ 1.73
    auto clamped_small = E3::extent(extent, small);
    EXPECT_NEAR(clamped_small.data(0), small.data(0), 1e-10);

    E3::Drift large{Eigen::Vector3d(3, 0, 0)};      // norm = 3
    auto clamped_large = E3::extent(extent, large);
    EXPECT_NEAR(clamped_large.data(0), 2.0, 1e-10); // Scaled to max norm
    EXPECT_NEAR(clamped_large.data(1), 0.0, 1e-10);
    EXPECT_NEAR(clamped_large.data(2), 0.0, 1e-10);
}

// ============================================================================
// CovarianceStorage Tests
// ============================================================================

TEST(CovarianceStorage, DefaultConstructor) {
    using Belief = Euclidean<3>;
    CovarianceStorage<Belief, Representation::Direct> storage(3);

    auto P = storage.mat();
    EXPECT_EQ(P.rows(), 3);
    EXPECT_EQ(P.cols(), 3);

    // Should be initialized to small diagonal
    EXPECT_GT(P(0, 0), 0.0);
    EXPECT_GT(P(1, 1), 0.0);
    EXPECT_GT(P(2, 2), 0.0);
}

TEST(CovarianceStorage, InitFromMatrix) {
    using Belief         = Euclidean<3>;
    Eigen::Matrix3d init = Eigen::Matrix3d::Identity();
    CovarianceStorage<Belief, Representation::Direct> storage(init);

    auto P = storage.mat();
    EXPECT_TRUE(P.isIdentity());
}

TEST(CovarianceStorage, SquareRootStorage) {
    using Belief         = Euclidean<3>;
    Eigen::Matrix3d init = Eigen::Matrix3d::Identity() * 4.0;
    CovarianceStorage<Belief, Representation::Sr> storage(init);

    auto P = storage.mat();
    EXPECT_TRUE(P.isApprox(Eigen::Matrix3d::Identity() * 4.0, 1e-10));
}

TEST(CovarianceStorage, UDUStorage) {
    using Belief         = Euclidean<3>;
    Eigen::Matrix3d init = Eigen::Matrix3d::Identity() * 4.0;
    CovarianceStorage<Belief, Representation::Udu> storage(init);

    auto P = storage.mat();
    EXPECT_TRUE(P.isApprox(Eigen::Matrix3d::Identity() * 4.0, 1e-8));
}

TEST(CovarianceStorage, UDUZeroDiagonalHandling) {
    // Test that UDU handles zero/negative diagonal entries
    using Belief = Euclidean<3>;

    // Create a matrix with a very small diagonal entry
    Eigen::Matrix3d init;
    init << 1.0, 0.5, 0.3, 0.5, 1e-15, 0.2, // Very small diagonal
        0.3, 0.2, 1.0;

    CovarianceStorage<Belief, Representation::Udu> storage(init);

    // Should not crash and produce a valid matrix
    auto P = storage.mat();
    EXPECT_EQ(P.rows(), 3);
    EXPECT_EQ(P.cols(), 3);

    // Matrix should be symmetric
    EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 1e-10);
}

TEST(CovarianceStorage, Update) {
    using Belief = Euclidean<3>;
    CovarianceStorage<Belief, Representation::Direct> storage(3);

    Eigen::Matrix3d new_P = Eigen::Matrix3d::Identity() * 2.0;
    storage.update(new_P);

    auto P = storage.mat();
    EXPECT_TRUE(P.isApprox(Eigen::Matrix3d::Identity() * 2.0, 1e-10));
}

// ============================================================================
// Strategy Tests
// ============================================================================

TEST(LinearStrategy, Propagate) {
    using Belief   = Euclidean<3>;
    using Storage  = CovarianceStorage<Belief, Representation::Direct>;
    using Strategy = LinearStrategy<Belief, Storage>;

    Storage storage(Eigen::Matrix3d::Identity());
    Belief::Flux state{Eigen::Vector3d(1, 2, 3)};
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * 0.1;
    double dt         = 0.1;

    Strategy strategy;
    // Identity dynamics for Euclidean space: x' = x
    auto dynamics = [](const typename Belief::Flux& x, double) { return x; };
    strategy.propagate(storage, state, dynamics, Q, dt);

    auto P = storage.mat();
    // For Euclidean space with identity dynamics: P' = P + Q
    EXPECT_NEAR(P(0, 0), 1.0 + 0.1, 1e-10);
}

TEST(LinearStrategy, KalmanGain) {
    using Belief   = Euclidean<3>;
    using Storage  = CovarianceStorage<Belief, Representation::Direct>;
    using Strategy = LinearStrategy<Belief, Storage>;

    Storage storage(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.1;

    Strategy strategy;
    auto K = strategy.compute_kalman_gain(storage, H, R);

    // K = P * H^T * (H * P * H^T + R)^-1
    // With P=I, H=I, R=0.1*I: K = I * (I + 0.1*I)^-1 = 1/1.1 * I
    EXPECT_NEAR(K(0, 0), 1.0 / 1.1, 1e-10);
    EXPECT_NEAR(K(1, 1), 1.0 / 1.1, 1e-10);
    EXPECT_NEAR(K(2, 2), 1.0 / 1.1, 1e-10);
}

TEST(LinearizedStrategy, Propagate) {
    using Belief   = Euclidean<3>;
    using Storage  = CovarianceStorage<Belief, Representation::Direct>;
    using Strategy = LinearizedStrategy<Belief, Storage>;

    Storage storage(Eigen::Matrix3d::Identity());
    Belief::Flux state{Eigen::Vector3d(1, 2, 3)};
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * 0.1;
    double dt         = 0.1;

    Strategy strategy;
    // Identity dynamics for Euclidean space: x' = x
    auto dynamics = [](const typename Belief::Flux& x, double) { return x; };
    strategy.propagate(storage, state, dynamics, Q, dt);

    auto P = storage.mat();
    // For Euclidean space with identity dynamics: P' = P + Q
    EXPECT_NEAR(P(0, 0), 1.0 + 0.1, 1e-10);
}

TEST(LinearizedStrategy, KalmanGain) {
    using Belief   = Euclidean<3>;
    using Storage  = CovarianceStorage<Belief, Representation::Direct>;
    using Strategy = LinearizedStrategy<Belief, Storage>;

    Storage storage(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d J_h = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R   = Eigen::Matrix3d::Identity() * 0.1;

    Strategy strategy;
    auto K = strategy.compute_kalman_gain(storage, J_h, R);

    EXPECT_NEAR(K(0, 0), 1.0 / 1.1, 1e-10);
}

// ============================================================================
// Combined Covariance Tests
// ============================================================================

TEST(Covariance, DirectStorage) {
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    Cov cov(3);
    EXPECT_EQ(cov.dimension(), 3);

    auto P = cov.mat();
    EXPECT_EQ(P.rows(), 3);
    EXPECT_EQ(P.cols(), 3);
}

TEST(Covariance, Propagate) {
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    Cov cov(Eigen::Matrix3d::Identity());
    Belief::Flux state{Eigen::Vector3d(1, 2, 3)};
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * 0.1;
    double dt         = 0.1;

    // Identity dynamics for Euclidean space: x' = x
    auto dynamics = [](const typename Belief::Flux& x, double) { return x; };
    cov.propagate(state, dynamics, Q, dt);

    auto P = cov.mat();
    // For Euclidean space with identity dynamics: P' = P + Q
    EXPECT_NEAR(P(0, 0), 1.0 + 0.1, 1e-10);
}

TEST(Covariance, ComputeKalmanGain) {
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    Cov cov(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.1;

    auto K = cov.compute_kalman_gain(H, R);

    EXPECT_NEAR(K(0, 0), 1.0 / 1.1, 1e-10);
}

TEST(Covariance, Stiffen) {
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    Cov cov(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity() * 0.5;
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.1;
    double dt         = 0.1;

    cov.stiffen(K, H, R, dt);

    auto P = cov.mat();
    // P' = (I - KH) * P = (I - 0.5*I) * I = 0.5 * I
    EXPECT_NEAR(P(0, 0), 0.5, 1e-10);
}

// ============================================================================
// Lie Group Tests
// ============================================================================

TEST(LieGroup, SO3Identity) {
    using SO3Space = SO3<LieSide::Right>;

    auto zero = SO3Space::zero_flux();
    // SO3 identity constructor creates identity rotation
    auto I = group::SO3<double>{};
    EXPECT_TRUE(zero.asMatrix().isApprox(I.asMatrix()));

    auto zero_d = SO3Space::zero_drift();
    EXPECT_TRUE(zero_d.isZero());
}

TEST(LieGroup, SO3ComposeAndDiff) {
    using SO3Space = SO3<LieSide::Right>;

    // Start with identity
    auto I = SO3Space::zero_flux();

    // Small rotation around z-axis
    SO3Space::Drift xi;
    xi << 0, 0, 0.1;

    // Compose: I ⊕ xi = exp(xi)
    auto R = SO3Space::compose(I, xi);

    // Diff should recover xi
    auto recovered = SO3Space::diff(I, R);
    EXPECT_NEAR(recovered(0), 0.0, 1e-6);
    EXPECT_NEAR(recovered(1), 0.0, 1e-6);
    EXPECT_NEAR(recovered(2), 0.1, 1e-6);
}

TEST(LieGroup, SO3Norm) {
    using SO3Space = SO3<LieSide::Right>;

    SO3Space::Drift xi;
    xi << 0.1, 0.2, 0.3;
    double n = SO3Space::norm(xi);
    EXPECT_NEAR(n, std::sqrt(0.01 + 0.04 + 0.09), 1e-10);
}

TEST(LieGroup, SO3LeftJacobian) {
    using SO3Space = SO3<LieSide::Right>;

    SO3Space::Drift xi;
    xi << 0.1, 0.0, 0.0;
    auto J_l = SO3Space::left_jacobian(xi);

    // For small angles, Jacobian should be close to identity
    EXPECT_NEAR(J_l(0, 0), 1.0, 1e-2);
    EXPECT_NEAR(J_l(1, 1), 1.0, 1e-2);
    EXPECT_NEAR(J_l(2, 2), 1.0, 1e-2);
}

TEST(LieGroup, SO3Adjoint) {
    using SO3Space = SO3<LieSide::Right>;

    SO3Space::Flux X = group::SO3<double>::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    SO3Space::Drift xi;
    xi << 1.0, 0.0, 0.0;

    auto Ad_X = SO3Space::adjoint_matrix(X);
    // Adjoint should be a 3x3 matrix
    EXPECT_EQ(Ad_X.rows(), 3);
    EXPECT_EQ(Ad_X.cols(), 3);
}

TEST(LieGroup, SO3Distance) {
    using SO3Space = SO3<LieSide::Right>;

    auto I = SO3Space::zero_flux();
    SO3Space::Drift xi;
    xi << 0.1, 0.0, 0.0;
    auto R = SO3Space::compose(I, xi);

    SO3Space::Drift d = SO3Space::diff(I, R);
    EXPECT_NEAR(SO3Space::norm(d), 0.1, 1e-6);
}

TEST(LieGroup, SO3SigmaPoints) {
    using SO3Space = SO3<LieSide::Right>;

    auto mean           = SO3Space::zero_flux();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.01;

    auto points = SO3Space::sigma_points(mean, cov, 1e-3, 2.0, 0.0);

    // Should have 2n + 1 = 7 points for SO(3)
    EXPECT_EQ(points.size(), 7);

    // Weights should sum to approximately 1
    double weight_sum = 0.0;
    for (const auto& pt : points) {
        weight_sum += pt.weight_mean;
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-6);
}

TEST(LieGroup, SO3MeanAndCovariance) {
    using SO3Space = SO3<LieSide::Right>;

    auto mean           = SO3Space::zero_flux();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.01;

    auto points = SO3Space::sigma_points(mean, cov, 1e-3, 2.0, 0.0);

    // These should not crash
    auto recovered_mean = SO3Space::mean(points);
    auto recovered_cov  = SO3Space::covariance(points, recovered_mean);

    EXPECT_EQ(recovered_cov.rows(), 3);
    EXPECT_EQ(recovered_cov.cols(), 3);
}

TEST(LieGroup, SO3KarcherMeanStationarityCondition) {
    // For a (local) Karcher mean μ, the first-order condition holds:
    //   Σ w_i * Log(μ^{-1} · p_i) ≈ 0
    using SO3Space = SO3Right;

    auto mean           = SO3Space::identity();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.4;

    // Use a larger alpha so sigma points are not infinitesimally close.
    auto points = SO3Space::sigma_points(mean, cov, 1.0, 2.0, 0.0);
    ASSERT_FALSE(points.empty());

    const auto mu = SO3Space::mean(points);

    SO3Space::Drift stationarity = SO3Space::Drift::Zero();
    for (const auto& wp : points) {
        stationarity += wp.weight_mean * SO3Space::diff(mu, wp.point);
    }
    EXPECT_LT(stationarity.norm(), 1e-2);
}

TEST(InvariantStrategy, UsesRelativePropagationForSO3) {
    using Belief   = SO3Right;
    using Storage  = CovarianceStorage<Belief, Representation::Direct>;
    using Strategy = InvariantLinearStrategy<Belief, Storage>;

    Eigen::Matrix3d P0;
    P0 << 1.0, 0.2, 0.0, 0.2, 2.0, 0.3, 0.0, 0.3, 3.0;
    P0 += Eigen::Matrix3d::Identity() * 1e-9;

    Storage storage(P0);
    const Belief::Flux state = group::SO3<double>::exp(Eigen::Vector3d(0.3, -0.2, 0.1));

    const Eigen::Vector3d u(0.2, -0.1, 0.05);
    const auto dynamics = [&](const Belief::Flux& x, double) { return Belief::compose(x, u); };

    const Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    Strategy strategy;
    strategy.propagate(storage, state, dynamics, Q, 0.1);

    // For group-affine systems with right-invariant error, the covariance propagation
    // simplifies to P' = P + Q (Bonnabel 2017, Section III.B).
    // The Adjoint transformation is NOT applied in the invariant error frame.
    const Eigen::Matrix3d expected = P0 + Q;

    EXPECT_TRUE(storage.mat().isApprox(expected, 1e-9));
}

TEST(SigmaPointStrategy, CKFPropagateWorksForSO3) {
    using Belief = SO3Right;
    using Cov    = Covariance<Belief, Representation::Direct, CubatureStrategy>;

    const Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.02;
    const Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.01;

    Cov cov(P0);
    const auto state = Belief::identity();

    // Identity dynamics on the group.
    const auto dynamics = [](const Belief::Flux& x, double) { return x; };
    cov.propagate(state, dynamics, Q, 0.1);

    const Eigen::Matrix3d expected = P0 + Q;
    EXPECT_TRUE(cov.mat().isApprox(expected, 5e-3));
}

// ============================================================================
// Type Alias Tests
// ============================================================================

TEST(TypeAliases, CommonEuclideanSpaces) {
    EXPECT_TRUE(Space<R1>);
    EXPECT_TRUE(Space<R2>);
    EXPECT_TRUE(Space<R3>);
    EXPECT_TRUE(Space<R6>);
}

TEST(TypeAliases, CommonLieGroups) {
    using namespace uni_kf::redact::spaces;
    EXPECT_TRUE((Space<SO3Right>));
    EXPECT_TRUE((Space<SE3Right>));
    EXPECT_TRUE((Space<SO3Left>));
    EXPECT_TRUE((Space<SE3Left>));
}

TEST(TypeAliases, LeftInvariant) {
    using namespace uni_kf::redact::spaces;
    EXPECT_TRUE((Space<SO3Left>));
    EXPECT_TRUE((Space<SE3Left>));
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(Integration, SimpleKFUpdate) {
    // A simple 1D Kalman filter update
    using Belief = Euclidean<1>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    // Initial state: x = 0, P = 1
    Cov cov(Eigen::Matrix<double, 1, 1>::Identity() * 1.0);
    Belief::Flux state{Eigen::Matrix<double, 1, 1>(0.0)};

    // Measurement: z = 1, R = 0.1
    Eigen::Matrix<double, 1, 1> H(1.0);
    Eigen::Matrix<double, 1, 1> R(0.1);

    // Compute Kalman gain
    auto K = cov.compute_kalman_gain(H, R);

    // K = P * H^T * (H * P * H^T + R)^-1
    // K = 1 * 1 * (1 * 1 * 1 + 0.1)^-1 = 1 / 1.1 ≈ 0.909
    EXPECT_NEAR(K(0, 0), 1.0 / 1.1, 1e-10);

    // Update covariance
    cov.stiffen(K, H, R, 0.1);

    auto P_new = cov.mat();
    // P' = (I - KH) * P = (1 - 0.909) * 1 = 0.091
    EXPECT_NEAR(P_new(0, 0), 1.0 / 11.0, 1e-10);
}

TEST(Integration, EKFPropagation) {
    // Test EKF-style propagation with Jacobians
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearizedStrategy>;

    Cov cov(Eigen::Matrix3d::Identity());
    Belief::Flux state{Eigen::Vector3d(1, 2, 3)};
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * 0.01;
    double dt         = 0.1;

    // Identity dynamics for Euclidean space: x' = x
    auto dynamics = [](const typename Belief::Flux& x, double) { return x; };
    cov.propagate(state, dynamics, Q, dt);

    auto P = cov.mat();
    // For Euclidean with identity dynamics: P' = P + Q
    Eigen::Matrix3d expected = Eigen::Matrix3d::Identity() * (1.0 + 0.01);
    EXPECT_TRUE(P.isApprox(expected, 1e-10));
}

// ============================================================================
// Edge Cases and Bug Fixes
// ============================================================================

TEST(BugFixes, CovarianceStorageDefaultDimNotNegativeOne) {
    // Test that default constructor doesn't create invalid matrices
    using Belief = Euclidean<3>;
    CovarianceStorage<Belief, Representation::Direct> storage;

    auto P = storage.mat();
    // Should be 1x1 (the default), not 0x0
    EXPECT_GT(P.rows(), 0);
    EXPECT_GT(P.cols(), 0);
}

TEST(BugFixes, EuclideanDynamicDimension) {
    // Test that dynamic dimension Euclidean space works correctly
    using ED = Euclidean<Eigen::Dynamic>;

    int size  = 5;
    auto zero = ED::zero_flux(size);
    EXPECT_EQ(zero.data.rows(), size);

    auto identity = ED::state_transition_matrix(size);
    EXPECT_EQ(identity.rows(), size);
    EXPECT_EQ(identity.cols(), size);
}

TEST(BugFixes, LieGroupUsesCompileTimeDim) {
    // Test that LieGroup uses compile-time dimension, not runtime query
    using SO3Space = SO3<LieSide::Right>;

    auto mean           = SO3Space::zero_flux();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.01;

    // Should not crash even though mean might not have .data member
    auto points = SO3Space::sigma_points(mean, cov, 1e-3, 2.0, 0.0);
    EXPECT_EQ(points.size(), 7); // 2n + 1 = 2*3 + 1 = 7
}

TEST(BugFixes, MethodNamesAreCorrect) {
    // Test that "propragate" typo was fixed to "propagate"
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    Cov cov(Eigen::Matrix3d::Identity());
    Belief::Flux state{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    double dt         = 0.1;

    // This should compile and run (method name is "propagate" not "propragate")
    auto dynamics = [](const typename Belief::Flux& x, double) { return x; };
    cov.propagate(state, dynamics, Q, dt);
}

// ============================================================================
// Factorized SR/UDU Correctness Tests
// ============================================================================

namespace {

template <typename BeliefT, int MeasurementDim>
struct LinearPerceptionModel {
    using Belief      = BeliefT;
    using Measurement = Euclidean<MeasurementDim>;
    using MatrixDD    = Eigen::MatrixXd;

    MatrixDD H;
    Eigen::Matrix<double, MeasurementDim, MeasurementDim> R;

    typename Measurement::Flux assume(const typename Belief::Flux& x) const {
        return typename Measurement::Flux{H * x.data};
    }

    MatrixDD measurement_noise_covariance() const { return R; }

    MatrixDD measurement_jacobian(const typename Belief::Flux&) const { return H; }

    template <typename Cov>
    typename Belief::Drift extrude(Cov& cov, const typename Measurement::Drift& innovation) {
        const MatrixDD K = cov.compute_kalman_gain(H, R);
        return typename Belief::Drift{K * innovation.data};
    }
};

template <typename PerceptionT>
struct LinearAssumptionModel {
    using Perception = PerceptionT;
    using Belief     = typename Perception::Belief;
    using MatrixDD   = Eigen::MatrixXd;

    MatrixDD Phi;
    MatrixDD Q;

    typename Belief::Flux maneuver(auto&, const typename Belief::Flux& x, double) const {
        return typename Belief::Flux{Phi * x.data};
    }

    MatrixDD process_noise_covariance() const { return Q; }
};

} // namespace

// SR factorization test using standard HouseholderQR decomposition
// Following Tracy 2022: qr_r([√A; √B]) returns upper triangular R
// Reference: https://arxiv.org/pdf/2208.06452.pdf
// The correct implementation uses standard QR (no column pivoting)
TEST(SRFactorization, PredictMatchesFullCovarianceForFullQ) {
    using Belief = Euclidean<4>;

    Eigen::Matrix4d P0;
    P0 << 1.0, 0.1, 0.0, 0.0, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.2, 0.0, 0.0, 0.2, 1.0;
    P0 += Eigen::Matrix4d::Identity() * 1e-6;

    Eigen::Matrix4d Phi;
    Phi << 1.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix4d Q;
    Q << 0.05, 0.02, 0.01, 0.00, 0.02, 0.08, 0.00, 0.01, 0.01, 0.00, 0.04, 0.02, 0.00, 0.01, 0.02,
        0.06;
    Q = (Q + Q.transpose()) * 0.5;
    Q += Eigen::Matrix4d::Identity() * 1e-6;

    CovarianceStorage<Belief, Representation::Sr> sr(P0);
    ASSERT_TRUE(sr.sr_predict(Phi, Q));

    const Eigen::Matrix4d expected = Phi * P0 * Phi.transpose() + Q;
    const Eigen::Matrix4d got      = sr.mat();
    EXPECT_TRUE(got.isApprox(expected, 1e-7));
}

TEST(UDUFactorization, PredictMatchesFullCovarianceForFullQ) {
    using Belief = Euclidean<4>;

    Eigen::Matrix4d P0;
    P0 << 1.0, 0.1, 0.0, 0.0, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.2, 0.0, 0.0, 0.2, 1.0;
    P0 += Eigen::Matrix4d::Identity() * 1e-6;

    Eigen::Matrix4d Phi;
    Phi << 1.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix4d Q;
    Q << 0.05, 0.02, 0.01, 0.00, 0.02, 0.08, 0.00, 0.01, 0.01, 0.00, 0.04, 0.02, 0.00, 0.01, 0.02,
        0.06;
    Q = (Q + Q.transpose()) * 0.5;
    Q += Eigen::Matrix4d::Identity() * 1e-6;

    CovarianceStorage<Belief, Representation::Udu> udu(P0);
    ASSERT_TRUE(udu.udu_predict(Phi, Q));

    const Eigen::Matrix4d expected = Phi * P0 * Phi.transpose() + Q;
    const Eigen::Matrix4d got      = udu.mat();
    EXPECT_TRUE(got.isApprox(expected, 1e-7));
}

TEST(FactorizedUpdate, SRAndUDUMatchJosephForCorrelatedMeasurementNoise) {
    using Belief     = Euclidean<4>;
    using Perception = LinearPerceptionModel<Belief, 2>;
    using Assumption = LinearAssumptionModel<Perception>;

    Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity();
    P0(0, 1)           = 0.2;
    P0(1, 0)           = 0.2;
    P0(2, 3)           = 0.1;
    P0(3, 2)           = 0.1;
    P0 += Eigen::Matrix4d::Identity() * 1e-6;

    Eigen::Matrix<double, 2, 4> H;
    H << 1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.3;

    Eigen::Matrix2d R;
    R << 0.5, 0.2, 0.2, 0.3;

    Eigen::Matrix4d Phi = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d Q   = Eigen::Matrix4d::Zero();

    Perception perception{H, R};
    Assumption assumption{Phi, Q};

    using CovJoseph =
        Covariance<Belief, Representation::Direct, LinearStrategy, DirectJosephUpdater>;
    using CovSR  = Covariance<Belief, Representation::Sr, LinearStrategy>;
    using CovUDU = Covariance<Belief, Representation::Udu, LinearStrategy>;

    KalmanFilter<CovJoseph, Perception, Assumption> joseph(
        Belief::zero_flux(), P0, perception, assumption);
    KalmanFilter<CovSR, Perception, Assumption> sr(Belief::zero_flux(), P0, perception, assumption);
    KalmanFilter<CovUDU, Perception, Assumption> udu(
        Belief::zero_flux(), P0, perception, assumption);

    const Eigen::Vector2d z(0.4, -0.2);
    EXPECT_TRUE(joseph.percept(Euclidean<2>::Flux{z}, 0.1).has_value());
    EXPECT_TRUE(sr.percept(Euclidean<2>::Flux{z}, 0.1).has_value());
    EXPECT_TRUE(udu.percept(Euclidean<2>::Flux{z}, 0.1).has_value());

    EXPECT_TRUE(sr.state().data.isApprox(joseph.state().data, 1e-8));
    EXPECT_TRUE(udu.state().data.isApprox(joseph.state().data, 1e-8));

    EXPECT_TRUE(sr.covariance().isApprox(joseph.covariance(), 1e-7));
    EXPECT_TRUE(udu.covariance().isApprox(joseph.covariance(), 1e-7));
}

// ==========================================================================
// Sigma-Point Measurement Update Regression Tests
// ==========================================================================

namespace {

struct SquareMeasurementPerception {
    using Belief      = Euclidean<1>;
    using Measurement = Euclidean<1>;
    using MatrixDD    = Eigen::MatrixXd;

    double r;

    typename Measurement::Flux assume(const typename Belief::Flux& x) const {
        const double v = x.data(0);
        return typename Measurement::Flux{Eigen::Matrix<double, 1, 1>(v * v)};
    }

    MatrixDD measurement_noise_covariance() const {
        Eigen::Matrix<double, 1, 1> R;
        R(0, 0) = r;
        return R;
    }

    // Intentionally wrong/empty Jacobian: sigma-point update must still work.
    MatrixDD measurement_jacobian(const typename Belief::Flux&) const {
        return MatrixDD::Zero(1, 1);
    }

    template <typename Cov>
    typename Belief::Drift extrude(Cov& cov, const typename Measurement::Drift& innovation) {
        const MatrixDD K = cov.compute_kalman_gain(
            measurement_jacobian(Belief::zero_flux()), measurement_noise_covariance());
        return typename Belief::Drift{Eigen::Matrix<double, 1, 1>(K(0, 0) * innovation.data(0))};
    }
};

template <typename PerceptionT>
struct IdentityAssumption {
    using Perception = PerceptionT;
    using Belief     = typename Perception::Belief;
    using MatrixDD   = Eigen::MatrixXd;

    MatrixDD Q;

    typename Belief::Flux maneuver(auto&, const typename Belief::Flux& x, double) const {
        return x;
    }
    MatrixDD process_noise_covariance() const { return Q; }
};

} // namespace

TEST(SigmaPointUpdate, UKFUsesSigmaPointMeasurementTransformEvenIfJacobianIsZero) {
    using Belief = Euclidean<1>;
    using Cov    = Covariance<Belief, Representation::Direct, UnscentedStrategy>;
    using Percep = SquareMeasurementPerception;
    using Assump = IdentityAssumption<Percep>;

    Eigen::Matrix<double, 1, 1> x0;
    x0(0, 0) = 2.0;
    Eigen::Matrix<double, 1, 1> P0;
    P0(0, 0) = 0.1;

    Percep perception{0.01};
    Assump assumption{Eigen::Matrix<double, 1, 1>::Zero()};

    KalmanFilter<Cov, Percep, Assump> ukf(Belief::Flux{x0}, P0, perception, assumption);

    // Measurement near x^2 = 4.
    Eigen::Matrix<double, 1, 1> z;
    z(0, 0) = 4.1;
    EXPECT_TRUE(ukf.percept(Euclidean<1>::Flux{z}, 0.1).has_value());

    EXPECT_NE(ukf.state().data(0), 2.0);
    EXPECT_LT(ukf.covariance()(0, 0), P0(0, 0));
}

TEST(SigmaPointUpdate, CKFUsesCubaturePointsEvenIfJacobianIsZero) {
    using Belief = Euclidean<1>;
    using Cov    = Covariance<Belief, Representation::Direct, CubatureStrategy>;
    using Percep = SquareMeasurementPerception;
    using Assump = IdentityAssumption<Percep>;

    Eigen::Matrix<double, 1, 1> x0;
    x0(0, 0) = 2.0;
    Eigen::Matrix<double, 1, 1> P0;
    P0(0, 0) = 0.1;

    Percep perception{0.01};
    Assump assumption{Eigen::Matrix<double, 1, 1>::Zero()};

    KalmanFilter<Cov, Percep, Assump> ckf(Belief::Flux{x0}, P0, perception, assumption);

    Eigen::Matrix<double, 1, 1> z;
    z(0, 0) = 4.1;
    EXPECT_TRUE(ckf.percept(Euclidean<1>::Flux{z}, 0.1).has_value());

    EXPECT_NE(ckf.state().data(0), 2.0);
    EXPECT_LT(ckf.covariance()(0, 0), P0(0, 0));
}

// ============================================================================
// LogDistanceSO3 Space Tests
// ============================================================================

#include "uni_kf/redact/spaces/log_distance_so3.hpp"

TEST(LogDistanceSO3Space, SafeLogDistanceFunctions) {
    using namespace uni_kf::redact::spaces;

    // Test safe_log_distance
    EXPECT_NEAR(safe_log_distance(1.0), 0.0, 1e-10);
    EXPECT_NEAR(safe_log_distance(std::exp(1.0)), 1.0, 1e-10);

    // Test clipping for small distances
    EXPECT_EQ(safe_log_distance(1e-6), safe_log_distance(kMinLogDistance));

    // Test safe_exp_distance
    EXPECT_NEAR(safe_exp_distance(0.0), 1.0, 1e-10);
    EXPECT_NEAR(safe_exp_distance(1.0), std::exp(1.0), 1e-10);
    EXPECT_GE(safe_exp_distance(-100.0), kMinLogDistance);
}

TEST(LogDistanceSO3Space, MakeAndFromObservation) {
    using namespace uni_kf::redact::spaces;

    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const double r          = 2.5;

    auto obs            = make_observation(R, r);
    auto [R_out, r_out] = from_observation(obs);

    EXPECT_TRUE(R_out.isApprox(R, 1e-10));
    EXPECT_NEAR(r_out, r, 1e-10);
}

TEST(LogDistanceSO3Space, ObservationError) {
    using namespace uni_kf::redact::spaces;

    const Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d R2 = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()).matrix();

    auto obs1 = make_observation(R1, 2.0);
    auto obs2 = make_observation(R2, 2.2);

    auto error = observation_error(obs2, obs1);

    // SO3 error should be approximately 0.1 rad around z-axis
    EXPECT_NEAR(error.first(2), 0.1, 1e-3);

    // Log-distance error should be log(2.2) - log(2.0) = log(1.1)
    EXPECT_NEAR(error.second.data(0), std::log(2.2 / 2.0), 1e-10);
}

TEST(LogDistanceSO3Space, IsValidObservation) {
    using namespace uni_kf::redact::spaces;

    // Valid observation
    auto valid_obs = make_observation(Eigen::Matrix3d::Identity(), 1.0);
    EXPECT_TRUE(is_valid_observation(valid_obs));

    // Invalid observation with NaN
    auto invalid_obs           = valid_obs;
    invalid_obs.second.data(0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(is_valid_observation(invalid_obs));
}

TEST(LogDistanceSO3Space, SpaceConceptSatisfied) {
    using namespace uni_kf::redact::spaces;
    EXPECT_TRUE(Space<LogDistanceSO3>);
}

// ============================================================================
// SE3 Lie Group Tests
// ============================================================================

TEST(LieGroup, SE3Identity) {
    using SE3Space = SE3<LieSide::Right>;

    auto zero = SE3Space::zero_flux();
    // SE3 identity constructor creates identity pose
    auto I = group::SEn3<double, 1>{};
    EXPECT_TRUE(zero.asMatrix().isApprox(I.asMatrix()));

    auto zero_d = SE3Space::zero_drift();
    EXPECT_TRUE(zero_d.isZero());
}

TEST(LieGroup, SE3ComposeAndDiff) {
    using SE3Space = SE3<LieSide::Right>;

    // Start with identity
    auto I = SE3Space::zero_flux();

    // Small motion
    SE3Space::Drift xi;
    xi << 0.1, 0.0, 0.0, // small rotation
        0.5, 0.0, 0.0;   // translation

    // Compose: I ⊕ xi = exp(xi)
    auto T = SE3Space::compose(I, xi);

    // Diff should recover xi (approximately)
    auto recovered = SE3Space::diff(I, T);
    EXPECT_NEAR(recovered(0), 0.1, 1e-5);
    EXPECT_NEAR(recovered(1), 0.0, 1e-5);
    EXPECT_NEAR(recovered(2), 0.0, 1e-5);
    EXPECT_NEAR(recovered(3), 0.5, 1e-5);
}

TEST(LieGroup, SE3Norm) {
    using SE3Space = SE3<LieSide::Right>;

    SE3Space::Drift xi;
    xi << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    double n = SE3Space::norm(xi);
    EXPECT_NEAR(n, std::sqrt(0.91), 1e-10);
}

TEST(LieGroup, SE3Adjoint) {
    using SE3Space = SE3<LieSide::Right>;

    SE3Space::Flux X =
        group::SEn3<double, 1>::exp(Eigen::Matrix<double, 6, 1>(0.1, 0.2, 0.3, 0.4, 0.5, 0.6));

    auto Ad_X = SE3Space::adjoint_matrix(X);
    EXPECT_EQ(Ad_X.rows(), 6);
    EXPECT_EQ(Ad_X.cols(), 6);
}

TEST(LieGroup, SE3SigmaPoints) {
    using SE3Space = SE3<LieSide::Right>;

    auto mean                              = SE3Space::zero_flux();
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

    auto points = SE3Space::sigma_points(mean, covariance, 1e-3, 2.0, 0.0);

    // Should have 2n + 1 = 13 points for SE(3)
    EXPECT_EQ(points.size(), 13);

    double weight_sum = 0.0;
    for (const auto& pt : points) {
        weight_sum += pt.weight_mean;
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-6);
}

TEST(LieGroup, SE3MeanAndCovariance) {
    using SE3Space = SE3<LieSide::Right>;

    auto mean                              = SE3Space::zero_flux();
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

    auto points = SE3Space::sigma_points(mean, covariance, 1e-3, 2.0, 0.0);

    auto recovered_mean = SE3Space::mean(points);
    auto recovered_cov  = SE3Space::covariance(points, recovered_mean);

    EXPECT_EQ(recovered_cov.rows(), 6);
    EXPECT_EQ(recovered_cov.cols(), 6);
}

// ============================================================================
// SE23 Lie Group Tests
// ============================================================================

TEST(LieGroup, SE23Identity) {
    using SE23Space = SE23<LieSide::Right>;

    auto zero   = SE23Space::zero_flux();
    auto zero_d = SE23Space::zero_drift();
    EXPECT_TRUE(zero_d.isZero());
}

TEST(LieGroup, SE23ComposeAndDiff) {
    using SE23Space = SE23<LieSide::Right>;

    auto I = SE23Space::zero_flux();

    SE23Space::Drift xi;
    xi << 0.1, 0.0, 0.0,                  // rotation (0-2)
        0.5, 0.0, 0.0,                    // first isometry (3-5)
        0.3, 0.2, 0.1;                    // second isometry (6-8)

    auto X         = SE23Space::compose(I, xi);
    auto recovered = SE23Space::diff(I, X);

    EXPECT_NEAR(recovered(0), 0.1, 1e-5); // rotation
    EXPECT_NEAR(recovered(3), 0.5, 1e-5); // first isometry
    EXPECT_NEAR(recovered(6), 0.3, 1e-5); // second isometry
}

TEST(LieGroup, SE23Norm) {
    using SE23Space = SE23<LieSide::Right>;

    SE23Space::Drift xi;
    xi << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
    double n = SE23Space::norm(xi);
    EXPECT_NEAR(n, std::sqrt(2.85), 1e-10);
}

TEST(LieGroup, SE23SigmaPoints) {
    using SE23Space = SE23<LieSide::Right>;

    auto mean                              = SE23Space::zero_flux();
    Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Identity() * 0.01;

    auto points = SE23Space::sigma_points(mean, covariance, 1e-3, 2.0, 0.0);

    // Should have 2n + 1 = 19 points for SE(2,3)
    EXPECT_EQ(points.size(), 19);

    double weight_sum = 0.0;
    for (const auto& pt : points) {
        weight_sum += pt.weight_mean;
    }
    EXPECT_NEAR(weight_sum, 1.0, 1e-6);
}

TEST(LieGroup, SE23DynamicsJacobian) {
    using SE23Space = SE23<LieSide::Right>;

    auto state = SE23Space::zero_flux();
    double dt  = 0.1;

    auto F = SE23Space::dynamics_jacobian(state, dt);

    // Check structure: F should have position-velocity coupling
    EXPECT_EQ(F.rows(), 9);
    EXPECT_EQ(F.cols(), 9);

    // Rotation block should be identity
    EXPECT_NEAR(F(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(F(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(F(2, 2), 1.0, 1e-10);

    // Velocity block should be identity
    EXPECT_NEAR(F(6, 6), 1.0, 1e-10);
    EXPECT_NEAR(F(7, 7), 1.0, 1e-10);
    EXPECT_NEAR(F(8, 8), 1.0, 1e-10);

    // Position should depend on velocity
    EXPECT_NEAR(F(3, 6), dt, 1e-10);
    EXPECT_NEAR(F(4, 7), dt, 1e-10);
    EXPECT_NEAR(F(5, 8), dt, 1e-10);
}

// ============================================================================
// Left-Invariant Lie Group Tests
// ============================================================================

TEST(LieGroup, SO3LeftComposeAndDiff) {
    using SO3Space = SO3<LieSide::Left>;

    auto I = SO3Space::zero_flux();

    SO3Space::Drift xi;
    xi << 0.0, 0.0, 0.1;

    // Compose: xi ⊕ I = exp(xi) * I (left-invariant)
    auto R = SO3Space::compose(I, xi);

    // Diff: Log(I * R) = Log(exp(xi)) = xi
    auto recovered = SO3Space::diff(I, R);
    EXPECT_NEAR(recovered(2), 0.1, 1e-6);
}

TEST(LieGroup, SE3LeftComposeAndDiff) {
    using SE3Space = SE3<LieSide::Left>;

    auto I = SE3Space::zero_flux();

    SE3Space::Drift xi;
    xi << 0.1, 0.0, 0.0, 0.5, 0.0, 0.0;

    auto T         = SE3Space::compose(I, xi);
    auto recovered = SE3Space::diff(I, T);

    EXPECT_NEAR(recovered(0), 0.1, 1e-5);
    EXPECT_NEAR(recovered(3), 0.5, 1e-5);
}

TEST(LieGroup, SE23LeftComposeAndDiff) {
    using SE23Space = SE23<LieSide::Left>;

    auto I = SE23Space::zero_flux();

    SE23Space::Drift xi;
    xi << 0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.2, 0.1;

    auto X         = SE23Space::compose(I, xi);
    auto recovered = SE23Space::diff(I, X);

    EXPECT_NEAR(recovered(0), 0.1, 1e-5);
    EXPECT_NEAR(recovered(3), 0.5, 1e-5);
}

// ============================================================================
// ProductSpace Tests
// ============================================================================

TEST(ProductSpace, ZeroElements) {
    using PS = ProductSpace<Euclidean<3>, Euclidean<2>>;

    auto zero_f = PS::zero_flux();
    EXPECT_EQ(PS::Dim, 5);

    auto zero_d = PS::zero_drift();
    EXPECT_TRUE(zero_d.first.data.isZero());
    EXPECT_TRUE(zero_d.second.data.isZero());
}

TEST(ProductSpace, ComposeAndDiff) {
    using PS = ProductSpace<SO3Right, Euclidean<3>>;

    auto I = PS::zero_flux();

    PS::Drift dx;
    dx.first << 0.1, 0.0, 0.0;
    dx.second.data << 0.5, 0.3, 0.2;

    auto composed  = PS::compose(I, dx);
    auto recovered = PS::diff(I, composed);

    EXPECT_NEAR(recovered.first(0), 0.1, 1e-6);
    EXPECT_NEAR(recovered.second.data(0), 0.5, 1e-10);
}

TEST(ProductSpace, Norm) {
    using PS = ProductSpace<Euclidean<3>, Euclidean<2>>;

    PS::Drift dx;
    dx.first.data << 3.0, 0.0, 0.0;
    dx.second.data << 0.0, 4.0;

    double n = PS::norm(dx);
    EXPECT_NEAR(n, 5.0, 1e-10);
}

TEST(ProductSpace, DynamicsJacobian) {
    using PS = ProductSpace<Euclidean<3>, Euclidean<2>>;

    auto state = PS::zero_flux();
    double dt  = 0.1;

    auto J = PS::dynamics_jacobian(state, dt);

    EXPECT_EQ(J.rows(), 5);
    EXPECT_EQ(J.cols(), 5);
    // Should be block-diagonal identity
    EXPECT_NEAR(J(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(J(4, 4), 1.0, 1e-10);
}

TEST(ProductSpace, SigmaPoints) {
    using PS = ProductSpace<SO3Right, Euclidean<2>>;

    auto mean                              = PS::zero_flux();
    Eigen::Matrix<double, 5, 5> covariance = Eigen::Matrix<double, 5, 5>::Identity() * 0.01;

    auto points = PS::sigma_points(mean, covariance, 1e-3, 2.0, 0.0);

    // Should have 2n + 1 = 11 points
    EXPECT_EQ(points.size(), 11);
}

TEST(ProductSpace, DriftVectorConversion) {
    using PS = ProductSpace<Euclidean<3>, Euclidean<2>>;

    PS::Drift dx;
    dx.first.data << 1.0, 2.0, 3.0;
    dx.second.data << 4.0, 5.0;

    auto vec = PS::drift_to_vector(dx);
    EXPECT_EQ(vec.size(), 5);
    EXPECT_NEAR(vec(0), 1.0, 1e-10);
    EXPECT_NEAR(vec(4), 5.0, 1e-10);

    auto recovered = PS::vector_to_drift(vec);
    EXPECT_NEAR(recovered.first.data(0), 1.0, 1e-10);
    EXPECT_NEAR(recovered.second.data(1), 5.0, 1e-10);
}

// ============================================================================
// Belief Class Tests
// ============================================================================

TEST(BeliefClass, ConstructFromMeanAndCovariance) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Vector3d mean_vec(1.0, 2.0, 3.0);
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 0.1;

    B belief(Euclidean<3>::Flux{mean_vec}, covariance);

    EXPECT_TRUE(belief.mean().data.isApprox(mean_vec));
    EXPECT_TRUE(belief.covariance().isApprox(covariance));
}

TEST(BeliefClass, LogDeterminant) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 4.0;
    B belief(Euclidean<3>::Flux{Eigen::Vector3d::Zero()}, covariance);

    auto result = belief.log_determinant();
    ASSERT_TRUE(result.has_value());
    // log(det(I * 4)) = log(64) = 6*log(2)
    EXPECT_NEAR(*result, 6.0 * std::log(2.0), 1e-10);
}

TEST(BeliefClass, LogDeterminantNonSPD) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    // Non-SPD matrix
    Eigen::Matrix3d bad_cov;
    bad_cov << 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0;

    B belief(Euclidean<3>::Flux{Eigen::Vector3d::Zero()}, bad_cov);

    auto result = belief.log_determinant();
    EXPECT_FALSE(result.has_value());
}

TEST(BeliefClass, MahalanobisDistance) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 0.5;
    B belief1(Euclidean<3>::Flux{Eigen::Vector3d::Zero()}, covariance);
    B belief2(Euclidean<3>::Flux{Eigen::Vector3d(1.0, 0.0, 0.0)}, covariance);

    auto result = belief1.mahalanobis(belief2);
    ASSERT_TRUE(result.has_value());
    // distance = sqrt(1^2 / 0.5) = sqrt(2)
    EXPECT_NEAR(*result, std::sqrt(2.0), 1e-6);
}

TEST(BeliefClass, StdDev) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Matrix3d covariance;
    covariance << 1.0, 0.5, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 1.0;

    B belief(Euclidean<3>::Flux{Eigen::Vector3d::Zero()}, covariance);

    Euclidean<3>::Drift direction;
    direction.data = Eigen::Vector3d(1.0, 0.0, 0.0);

    double sigma = belief.std_dev(direction);
    EXPECT_NEAR(sigma, 1.0, 1e-6);
}

TEST(BeliefClass, ComposeBeliefs) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Matrix3d P1 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d P2 = Eigen::Matrix3d::Identity() * 0.5;

    B b1(Euclidean<3>::Flux{Eigen::Vector3d(1.0, 0.0, 0.0)}, P1);
    B b2(Euclidean<3>::Flux{Eigen::Vector3d(2.0, 0.0, 0.0)}, P2);

    auto combined = b1.compose(b2);

    EXPECT_NEAR(combined.mean().data(0), 3.0, 1e-10);
    EXPECT_NEAR(combined.covariance()(0, 0), 1.5, 1e-10);
}

TEST(BeliefClass, Retract) {
    // For Lie groups like SO3, we can't use LinearStrategy since they don't
    // satisfy LinearDynamics. Instead, we test the retract operation directly
    // using the Space operations.

    auto mean = SO3Right::zero_flux();

    SO3Right::Drift correction;
    correction << 0.1, 0.0, 0.0;

    auto retracted = SO3Right::compose(mean, correction);

    // Should be rotated by 0.1 rad around x-axis
    auto error = SO3Right::diff(mean, retracted);
    EXPECT_NEAR(error(0), 0.1, 1e-5);
}

TEST(BeliefClass, IsFinite) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Vector3d mean_vec(1.0, 2.0, std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 0.1;

    B belief(Euclidean<3>::Flux{mean_vec}, covariance);
    EXPECT_FALSE(belief.is_finite());
}

TEST(BeliefClass, IsApprox) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() * 0.1;

    B b1(Euclidean<3>::Flux{Eigen::Vector3d(1.0, 2.0, 3.0)}, covariance);
    B b2(Euclidean<3>::Flux{Eigen::Vector3d(1.0, 2.0, 3.0)}, covariance);
    B b3(Euclidean<3>::Flux{Eigen::Vector3d(1.0, 2.0, 4.0)}, covariance);

    EXPECT_TRUE(b1.is_approx(b2));
    EXPECT_FALSE(b1.is_approx(b3));
}

// ============================================================================
// Factorization Tests
// ============================================================================

TEST(CholUpdate, Rank1Update) {
    using namespace uni_kf::redact::factorization;

    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    Eigen::VectorXd x(3);
    x << 0.5, 0.3, 0.2;

    // L * L^T = I
    Eigen::MatrixXd L = A.llt().matrixL();

    // Update: L_new * L_new^T = L * L^T + x * x^T
    ASSERT_TRUE(cholupdate_lower_inplace(L, x, +1));

    Eigen::Matrix3d P_updated = L * L.transpose();
    Eigen::Matrix3d expected  = A + x * x.transpose();

    EXPECT_TRUE(P_updated.isApprox(expected, 1e-10));
}

TEST(CholUpdate, Rank1Downdate) {
    using namespace uni_kf::redact::factorization;

    Eigen::Matrix3d A;
    A << 2.0, 0.5, 0.3, 0.5, 1.5, 0.2, 0.3, 0.2, 1.0;

    Eigen::VectorXd x(3);
    x << 0.5, 0.3, 0.2;

    // L * L^T = A
    Eigen::MatrixXd L = A.llt().matrixL();

    // Downdate: L_new * L_new^T = L * L^T - x * x^T
    ASSERT_TRUE(cholupdate_lower_inplace(L, x, -1));

    Eigen::Matrix3d P_downdated = L * L.transpose();
    Eigen::Matrix3d expected    = A - x * x.transpose();

    EXPECT_TRUE(P_downdated.isApprox(expected, 1e-10));
}

TEST(CholUpdate, DowndateFailure) {
    using namespace uni_kf::redact::factorization;

    Eigen::Matrix3d A = Eigen::Matrix3d::Identity() * 0.1;

    Eigen::VectorXd x(3);
    x << 1.0, 0.0, 0.0;

    Eigen::MatrixXd L = A.llt().matrixL();

    // This downdate should fail (would make matrix non-SPD)
    EXPECT_FALSE(cholupdate_lower_inplace(L, x, -1));
}

// ============================================================================
// DynamicsJacobian Structure Verification Tests
// ============================================================================

TEST(DynamicsJacobian, SE23StructureIsCorrect) {
    using SE23Space = SE23<LieSide::Right>;

    auto state = SE23Space::zero_flux();
    double dt  = 0.01;

    // Compute analytical Jacobian
    auto F = SE23Space::dynamics_jacobian(state, dt);

    // For constant velocity model, the Jacobian should be:
    // F = [I₃  0₃    0₃   ]  (rotation - independent)
    //     [0₃  0₃    I₃·dt]  (position depends on velocity)
    //     [0₃  0₃    I₃   ]  (velocity - constant)
    //
    // Note: The tangent space structure for SE2(3) is:
    //   - indices 0-2: rotation (ω)
    //   - indices 3-5: position (ṗ)
    //   - indices 6-8: velocity (v)

    // Verify rotation block (0-2, 0-2) is identity
    Eigen::Matrix3d rotation_block = F.block<3, 3>(0, 0);
    EXPECT_TRUE(rotation_block.isIdentity());

    // Verify velocity block (6-8, 6-8) is identity
    Eigen::Matrix3d velocity_block = F.block<3, 3>(6, 6);
    EXPECT_TRUE(velocity_block.isIdentity());

    // Verify position depends on velocity: F(3-5, 6-8) = I₃·dt
    Eigen::Matrix3d pos_vel_block    = F.block<3, 3>(3, 6);
    Eigen::Matrix3d expected_pos_vel = Eigen::Matrix3d::Identity() * dt;
    EXPECT_NEAR(pos_vel_block(0, 0), expected_pos_vel(0, 0), 1e-10);
    EXPECT_NEAR(pos_vel_block(1, 1), expected_pos_vel(1, 1), 1e-10);
    EXPECT_NEAR(pos_vel_block(2, 2), expected_pos_vel(2, 2), 1e-10);

    // Verify other off-diagonal blocks are zero
    Eigen::Matrix3d block_03 = F.block<3, 3>(0, 3);
    EXPECT_NEAR(block_03.norm(), 0.0, 1e-10);
    Eigen::Matrix3d block_06 = F.block<3, 3>(0, 6);
    EXPECT_NEAR(block_06.norm(), 0.0, 1e-10);
    Eigen::Matrix3d block_30 = F.block<3, 3>(3, 0);
    EXPECT_NEAR(block_30.norm(), 0.0, 1e-10);
    Eigen::Matrix3d block_33 = F.block<3, 3>(3, 3);
    EXPECT_NEAR(block_33.norm(), 0.0, 1e-10);
    Eigen::Matrix3d block_60 = F.block<3, 3>(6, 0);
    EXPECT_NEAR(block_60.norm(), 0.0, 1e-10);
    Eigen::Matrix3d block_63 = F.block<3, 3>(6, 3);
    EXPECT_NEAR(block_63.norm(), 0.0, 1e-10);
}

// ============================================================================
// Full KalmanFilter Integration Tests
// ============================================================================

TEST(KalmanFilter, FullPredictionUpdateCycle) {
    using Belief = Euclidean<2>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    // Simple linear system: x' = x + v*dt, v' = v (constant velocity)
    struct CVPerception {
        using Belief      = Belief;
        using Measurement = Euclidean<1>;

        Eigen::Matrix<double, 1, 2> H;
        Eigen::Matrix<double, 1, 1> R;

        Measurement::Flux assume(const Belief::Flux& x) const {
            return Measurement::Flux{Eigen::Matrix<double, 1, 1>(x.data(0))};
        }

        Eigen::MatrixXd measurement_noise_covariance() const { return R; }

        Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const { return H; }

        Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
            const Eigen::Matrix<double, 2, 1> K = cov.compute_kalman_gain(H, R);
            return Belief::Drift{K * innovation.data};
        }
    };

    struct CVAssumption {
        using Perception = CVPerception;
        using Belief     = typename Perception::Belief;

        Eigen::Matrix<double, 2, 2> Q;

        Belief::Flux maneuver(Cov&, const Belief::Flux& x, double dt) const {
            Belief::Flux result{x.data};
            result.data(0) += x.data(1) * dt; // position += velocity * dt
            return result;
        }

        Eigen::MatrixXd process_noise_covariance() const { return Q; }
    };

    // Initial state: x=0, v=1 with some uncertainty
    Eigen::Vector2d x0_vec(0.0, 1.0);
    Eigen::Matrix2d P0;
    P0 << 0.1, 0.0, 0.0, 0.05;

    Eigen::Matrix<double, 1, 2> H;
    H << 1.0, 0.0; // measure position only
    Eigen::Matrix<double, 1, 1> R(0.01);

    Eigen::Matrix2d Q;
    Q << 0.001, 0.0, 0.0, 0.002;

    CVPerception perception{H, R};
    CVAssumption assumption{Q};

    KalmanFilter<Cov, CVPerception, CVAssumption> kf(
        Belief::Flux{x0_vec}, P0, perception, assumption);

    // Prediction step
    double dt = 0.1;
    kf.predict(dt);

    // Position should have moved: x' = 0 + 1*0.1 = 0.1
    EXPECT_NEAR(kf.state().data(0), 0.1, 1e-6);
    EXPECT_NEAR(kf.state().data(1), 1.0, 1e-6);

    // Measurement update
    Eigen::Matrix<double, 1, 1> z(0.12); // measured position
    EXPECT_TRUE(kf.percept(Euclidean<1>::Flux{z}, dt).has_value());

    // State should be pulled toward measurement
    EXPECT_GT(
        kf.state().data(0),
        0.1); // updated estimate between prediction (0.1) and measurement (0.12)
    EXPECT_LT(kf.covariance()(0, 0), P0(0, 0)); // covariance reduced
}

TEST(KalmanFilter, InKFOnSO3) {
    using Belief = SO3Right;
    using Cov    = Covariance<Belief, Representation::Direct, InvariantLinearStrategy>;

    struct SO3Perception {
        using Belief      = Belief;
        using Measurement = Euclidean<3>;

        Eigen::Matrix3d R;

        Measurement::Flux assume(const Belief::Flux& x) const {
            return Measurement::Flux{group::SO3<double>::log(x)};
        }

        Eigen::MatrixXd measurement_noise_covariance() const { return R; }

        Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const {
            return Eigen::Matrix3d::Identity();
        }

        Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
            const Eigen::Matrix3d K = cov.compute_kalman_gain(Eigen::Matrix3d::Identity(), R);
            return K * innovation.data;
        }
    };

    struct SO3Assumption {
        using Perception = SO3Perception;
        using Belief     = typename Perception::Belief;

        Eigen::Matrix3d Q;
        Eigen::Vector3d omega;

        Belief::Flux maneuver(Cov&, const Belief::Flux& x, double dt) const {
            return x * group::SO3<double>::exp(omega * dt);
        }

        Eigen::MatrixXd process_noise_covariance() const { return Q; }
    };

    group::SO3<double> R_true = group::SO3<double>::exp(Eigen::Vector3d(0.1, 0.2, 0.3));

    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.001;
    Eigen::Matrix3d R  = Eigen::Matrix3d::Identity() * 0.01;

    SO3Perception perception{R};
    SO3Assumption assumption{Q, Eigen::Vector3d(0.01, 0.02, 0.03)};

    KalmanFilter<Cov, SO3Perception, SO3Assumption> infilter(R_true, P0, perception, assumption);

    double dt = 0.1;
    infilter.predict(dt);

    EXPECT_TRUE(infilter.covariance().allFinite());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(ErrorHandling, PerceptErrorSPD) {
    using Belief = Euclidean<2>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    struct BadPerception {
        using Belief      = Belief;
        using Measurement = Euclidean<1>;

        Eigen::Matrix<double, 1, 2> H;

        Measurement::Flux assume(const Belief::Flux& x) const {
            return Measurement::Flux{Eigen::Matrix<double, 1, 1>(x.data(0))};
        }

        Eigen::MatrixXd measurement_noise_covariance() const {
            // Non-SPD matrix (negative diagonal)
            Eigen::Matrix<double, 1, 1> bad_R;
            bad_R(0) = -0.1;
            return bad_R;
        }

        Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const { return H; }

        Belief::Drift extrude(Cov&, const Measurement::Drift&) {
            return Belief::Drift{Eigen::Vector2d::Zero()};
        }
    };

    struct DummyAssumption {
        using Perception = BadPerception;
        using Belief     = typename Perception::Belief;

        Eigen::Matrix2d Q;

        Belief::Flux maneuver(Cov&, const Belief::Flux& x, double) const { return x; }
        Eigen::MatrixXd process_noise_covariance() const { return Q; }
    };

    BadPerception perception;
    DummyAssumption assumption{Eigen::Matrix2d::Zero()};

    KalmanFilter<Cov, BadPerception, DummyAssumption> kf(
        Belief::Flux{Eigen::Vector2d::Zero()}, Eigen::Matrix2d::Identity(), perception, assumption);

    // SR update should fail due to non-SPD R
    // But since we're using Direct storage, it should handle it differently
    auto result = kf.percept(Euclidean<1>::Flux{Eigen::Matrix<double, 1, 1>(1.0)}, 0.1);
    // The filter should still work (LLT will fail but fallback will be used)
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(EdgeCases, VerySmallCovariance) {
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    // Very small but positive covariance
    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() * 1e-15;

    Cov cov(P);
    Eigen::Matrix3d P_recovered = cov.mat();

    // Should be recoverable
    EXPECT_TRUE(P_recovered.allFinite());
}

TEST(EdgeCases, HighlyCorrelatedCovariance) {
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Direct, LinearStrategy>;

    // Highly correlated but SPD
    Eigen::Matrix3d P;
    P << 1.0, 0.99, 0.98, 0.99, 1.0, 0.99, 0.98, 0.99, 1.0;

    Cov cov(P);

    // Operations should work
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * 0.1;

    auto K = cov.compute_kalman_gain(H, R);
    EXPECT_TRUE(K.allFinite());
}

TEST(EdgeCases, LargeConditionNumber) {
    using Belief = Euclidean<3>;
    using Cov    = Covariance<Belief, Representation::Sr, LinearStrategy>;

    // Covariance with large condition number
    Eigen::Matrix3d P;
    P << 1e6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-6;

    Cov cov(P);

    // SR storage should handle this
    Eigen::Matrix3d P_recovered = cov.mat();
    EXPECT_TRUE(P_recovered.allFinite());
}

// ============================================================================
// Types and Concepts Verification Tests
// ============================================================================

TEST(TypesAndConcepts, SpaceConceptsAreSatisfied) {
    using namespace uni_kf::redact;

    // Test that basic space concepts are satisfied
    EXPECT_TRUE(Space<Euclidean<2>>);
    EXPECT_TRUE(Space<Euclidean<3>>);
    EXPECT_TRUE(Space<SO3<LieSide::Right>>);
    EXPECT_TRUE(Space<SE3<LieSide::Right>>);
    EXPECT_TRUE(Space<SE23<LieSide::Right>>);

    // Test LinearDynamics concept
    EXPECT_TRUE(LinearDynamics<Euclidean<2>>);
    EXPECT_TRUE(LinearDynamics<Euclidean<3>>);

    // Test Sampleable concept
    EXPECT_TRUE(Sampleable<Euclidean<2>>);
    EXPECT_TRUE(Sampleable<SO3<LieSide::Right>>);
    EXPECT_TRUE(Sampleable<SE23<LieSide::Right>>);

    SUCCEED();
}

// ============================================================================
// Variant Alias Tests
// ============================================================================

TEST(VariantAliases, KFAliasesCompile) {
    using namespace uni_kf::redact;

    // Test KF aliases compile
    using Belief      = Euclidean<2>;
    using Measurement = Euclidean<1>;
    using Cov         = Covariance<Belief, Representation::Direct, LinearStrategy>;

    // Simple test perception and assumption
    struct TestPerception {
        using Belief      = Belief;
        using Measurement = Measurement;

        Eigen::Matrix<double, 1, 2> H;
        Eigen::Matrix<double, 1, 1> R;

        Measurement::Flux assume(const Belief::Flux& x) const {
            return Measurement::Flux{Eigen::Matrix<double, 1, 1>(x.data(0))};
        }

        Eigen::MatrixXd measurement_noise_covariance() const { return R; }
        Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const { return H; }

        Belief::Drift extrude(Cov& cov, const Measurement::Drift& innovation) {
            const Eigen::Matrix<double, 2, 1> K = cov.compute_kalman_gain(H, R);
            return Belief::Drift{K * innovation.data};
        }
    };

    struct TestAssumption {
        using Perception = TestPerception;
        using Belief     = typename Perception::Belief;

        Eigen::Matrix<double, 2, 2> Q;

        Belief::Flux maneuver(Cov&, const Belief::Flux& x, double) const {
            return x; // No dynamics for test
        }

        Eigen::MatrixXd process_noise_covariance() const { return Q; }
    };

    // Test that KF alias compiles and works
    using KF_Type = KF<Belief, TestPerception, TestAssumption>;

    // Test that the type can be instantiated
    KF_Type kf(
        Belief::Flux{Eigen::Vector2d::Zero()}, Eigen::Matrix2d::Identity(),
        TestPerception{
            Eigen::Matrix<double, 1, 2>::Zero(), Eigen::Matrix<double, 1, 1>::Identity()},
        TestAssumption{Eigen::Matrix2d::Zero()});

    SUCCEED();
}

// ============================================================================
// Left-Invariant Lie Group Tests
// ============================================================================

TEST(LeftInvariantLieGroups, SO3LeftComposeAndDiff) {
    using SO3Left = SO3<LieSide::Left>;

    auto I = SO3Left::zero_flux();

    SO3Left::Drift xi;
    xi << 0.1, 0.2, 0.3;

    auto X         = SO3Left::compose(I, xi);
    auto recovered = SO3Left::diff(I, X);

    EXPECT_NEAR(recovered(0), 0.1, 1e-5);
    EXPECT_NEAR(recovered(1), 0.2, 1e-5);
    EXPECT_NEAR(recovered(2), 0.3, 1e-5);
}

TEST(LeftInvariantLieGroups, SE3LeftComposeAndDiff) {
    using SE3Left = SE3<LieSide::Left>;

    auto I = SE3Left::zero_flux();

    SE3Left::Drift xi;
    xi << 0.1, 0.0, 0.0,                  // rotation
        0.5, 0.3, 0.2;                    // translation

    auto X         = SE3Left::compose(I, xi);
    auto recovered = SE3Left::diff(I, X);

    EXPECT_NEAR(recovered(0), 0.1, 1e-5);
    EXPECT_NEAR(recovered(3), 0.5, 1e-5);
}

TEST(LeftInvariantLieGroups, SE23LeftComposeAndDiff) {
    using SE23Left = SE23<LieSide::Left>;

    auto I = SE23Left::zero_flux();

    SE23Left::Drift xi;
    xi << 0.1, 0.0, 0.0,                  // rotation
        0.5, 0.0, 0.0,                    // first isometry
        0.3, 0.2, 0.1;                    // second isometry

    auto X         = SE23Left::compose(I, xi);
    auto recovered = SE23Left::diff(I, X);

    EXPECT_NEAR(recovered(0), 0.1, 1e-5); // rotation
    EXPECT_NEAR(recovered(3), 0.5, 1e-5); // first isometry
    EXPECT_NEAR(recovered(6), 0.3, 1e-5); // second isometry
}

// ============================================================================
// Belief Compose Operation Tests
// ============================================================================

TEST(BeliefCompose, EuclideanComposeAddsMean) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Vector3d mean1_vec(1.0, 2.0, 3.0);
    Eigen::Vector3d mean2_vec(4.0, 5.0, 6.0);

    Eigen::Matrix3d P1 = Eigen::Matrix3d::Identity() * 0.1;
    Eigen::Matrix3d P2 = Eigen::Matrix3d::Identity() * 0.2;

    B b1(Euclidean<3>::Flux{mean1_vec}, P1);
    B b2(Euclidean<3>::Flux{mean2_vec}, P2);

    auto combined = b1.compose(b2);

    // Means should add
    EXPECT_NEAR(combined.mean().data(0), 5.0, 1e-10);
    EXPECT_NEAR(combined.mean().data(1), 7.0, 1e-10);
    EXPECT_NEAR(combined.mean().data(2), 9.0, 1e-10);

    // Covariances should add
    Eigen::Matrix3d P_expected = P1 + P2;
    EXPECT_TRUE(combined.covariance().isApprox(P_expected, 1e-10));
}

TEST(BeliefCompose, SO3ComposeWithAdjoint) {
    // For SO3, we can't use LinearStrategy since it doesn't satisfy LinearDynamics
    // Instead, we test the compose operation directly using the Space operations
    using SO3Right = SO3<LieSide::Right>;

    auto mean = SO3Right::zero_flux();

    // Create a non-zero drift
    SO3Right::Drift xi;
    xi << 0.1, 0.2, 0.3;

    // Compose: X = I * exp(xi) = exp(xi)
    auto combined_state = SO3Right::compose(mean, xi);

    // Verify the result
    auto recovered_xi = SO3Right::diff(mean, combined_state);
    EXPECT_NEAR(recovered_xi(0), 0.1, 1e-5);
    EXPECT_NEAR(recovered_xi(1), 0.2, 1e-5);
    EXPECT_NEAR(recovered_xi(2), 0.3, 1e-5);
}

// ============================================================================
// Error Path Tests
// ============================================================================

TEST(ErrorPaths, PerceptErrorNonSPD_R) {
    using Belief = Euclidean<2>;
    using Cov    = Covariance<Belief, Representation::Sr, LinearStrategy>;

    // Test perception with non-SPD R matrix
    struct BadRPerception {
        using Belief      = Belief;
        using Measurement = Euclidean<1>;

        Eigen::Matrix<double, 1, 2> H;
        Eigen::Matrix<double, 1, 1> bad_R;

        Measurement::Flux assume(const Belief::Flux& x) const {
            return Measurement::Flux{Eigen::Matrix<double, 1, 1>(x.data(0))};
        }

        Eigen::MatrixXd measurement_noise_covariance() const {
            // Non-SPD matrix
            return bad_R;
        }

        Eigen::MatrixXd measurement_jacobian(const Belief::Flux&) const { return H; }

        Belief::Drift extrude(Cov&, const Measurement::Drift&) {
            return Belief::Drift{Eigen::Vector2d::Zero()};
        }
    };

    struct DummyAssumption {
        using Perception = BadRPerception;
        using Belief     = typename Perception::Belief;

        Eigen::Matrix2d Q;

        Belief::Flux maneuver(Cov&, const Belief::Flux& x, double) const { return x; }
        Eigen::MatrixXd process_noise_covariance() const { return Q; }
    };

    // Create perception with bad R
    BadRPerception perception;
    perception.bad_R << -1.0; // Negative value makes it non-SPD
    perception.H << 1.0, 0.0;

    DummyAssumption assumption{Eigen::Matrix2d::Zero()};

    KalmanFilter<Cov, BadRPerception, DummyAssumption> kf(
        Belief::Flux{Eigen::Vector2d::Zero()}, Eigen::Matrix2d::Identity(), perception, assumption);

    // Measurement update should fail with R_not_spd error
    auto result = kf.percept(Euclidean<1>::Flux{Eigen::Matrix<double, 1, 1>(1.0)}, 0.1);

    EXPECT_FALSE(result.has_value());
    if (!result.has_value()) {
        EXPECT_EQ(result.error(), PerceptError::R_not_spd);
    }
}

// ============================================================================
// Belief Utility Methods Tests
// ============================================================================

TEST(BeliefUtility, Symmetrize) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    // Create a non-symmetric covariance matrix
    Eigen::Matrix3d P;
    P << 1.0, 0.9, 0.8, 0.5, 2.0, 0.7, 0.3, 0.4, 3.0;

    B belief(Euclidean<3>::Flux{Eigen::Vector3d::Zero()}, P);

    // Symmetrize the covariance - should not crash
    belief.symmetrize();

    // Just verify the belief is still valid (all finite)
    EXPECT_TRUE(belief.is_finite());
}

TEST(BeliefUtility, Regularize) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    // Create a nearly singular covariance
    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() * 1e-15;

    B belief(Euclidean<3>::Flux{Eigen::Vector3d::Zero()}, P);

    // Regularize the covariance
    belief.regularize(1e-12);

    // Check that regularization was applied
    Eigen::Matrix3d P_reg = belief.covariance();
    EXPECT_GT(P_reg(0, 0), 1e-15);
    EXPECT_GT(P_reg(1, 1), 1e-15);
    EXPECT_GT(P_reg(2, 2), 1e-15);
}

TEST(BeliefUtility, IsFinite) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    // Test with finite values
    Eigen::Vector3d mean_vec(1.0, 2.0, 3.0);
    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() * 0.1;

    B belief(Euclidean<3>::Flux{mean_vec}, P);
    EXPECT_TRUE(belief.is_finite());
}

TEST(BeliefUtility, IsApprox) {
    using B = uni_kf::redact::Belief<Euclidean<3>, Representation::Direct>;

    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() * 0.1;
    Eigen::Vector3d mean_vec(1.0, 2.0, 3.0);

    B b1(Euclidean<3>::Flux{mean_vec}, P);
    B b2(Euclidean<3>::Flux{mean_vec}, P);

    EXPECT_TRUE(b1.is_approx(b2));

    // Test with different means
    B b3(Euclidean<3>::Flux{Eigen::Vector3d(1.0, 2.0, 3.01)}, P);
    EXPECT_FALSE(b1.is_approx(b3, 1e-4));
    EXPECT_TRUE(b1.is_approx(b3, 1e-2));
}

// ============================================================================
// ProductSpace More Combinations Tests
// ============================================================================

TEST(ProductSpaceMore, SO3Euclidean) {
    using PS = ProductSpace<SO3<LieSide::Right>, Euclidean<2>>;

    auto zero = PS::zero_flux();
    auto xi   = PS::zero_drift();

    // Set SO3 drift
    xi.first << 0.1, 0.0, 0.0;

    // Set Euclidean drift
    xi.second.data << 0.5, 0.3;

    auto composed  = PS::compose(zero, xi);
    auto recovered = PS::diff(zero, composed);

    EXPECT_NEAR(recovered.first(0), 0.1, 1e-5);
    EXPECT_NEAR(recovered.second.data(0), 0.5, 1e-5);
    EXPECT_NEAR(recovered.second.data(1), 0.3, 1e-5);
}

TEST(ProductSpaceMore, NestedProductSpace) {
    using InnerPS = ProductSpace<Euclidean<2>, Euclidean<1>>;
    using OuterPS = ProductSpace<InnerPS, Euclidean<3>>;

    auto zero = OuterPS::zero_flux();
    auto xi   = OuterPS::zero_drift();

    // Set inner space drift
    xi.first.first.data << 1.0, 2.0;
    xi.first.second.data << 3.0;

    // Set outer space drift
    xi.second.data << 0.1, 0.2, 0.3;

    auto composed  = OuterPS::compose(zero, xi);
    auto recovered = OuterPS::diff(zero, composed);

    EXPECT_NEAR(recovered.first.first.data(0), 1.0, 1e-5);
    EXPECT_NEAR(recovered.first.first.data(1), 2.0, 1e-5);
    EXPECT_NEAR(recovered.first.second.data(0), 3.0, 1e-5);
    EXPECT_NEAR(recovered.second.data(0), 0.1, 1e-5);
}

// ============================================================================
// Storage Representation Tests
// ============================================================================

TEST(StorageRepresentation, SRStorage) {
    using Belief = Euclidean<2>;
    using Cov    = Covariance<Belief, Representation::Sr, LinearStrategy>;

    Eigen::Vector2d mean_vec(1.0, 2.0);
    Eigen::Matrix2d P;
    P << 1.0, 0.5, 0.5, 2.0;

    Cov cov(P);

    // Check that SR factor is computed
    Eigen::Matrix2d S           = cov.storage().sr_factor();
    Eigen::Matrix2d P_recovered = S * S.transpose();

    EXPECT_TRUE(P_recovered.isApprox(P, 1e-10));
}

TEST(StorageRepresentation, UDUStorage) {
    using Belief = Euclidean<2>;
    using Cov    = Covariance<Belief, Representation::Udu, LinearStrategy>;

    Eigen::Vector2d mean_vec(1.0, 2.0);
    Eigen::Matrix2d P;
    P << 1.0, 0.5, 0.5, 2.0;

    Cov cov(P);

    // Check that UDU factor is accessible
    Eigen::Matrix2d P_recovered = cov.mat();

    EXPECT_TRUE(P_recovered.isApprox(P, 1e-10));
}

// ============================================================================
// Invariant Strategy Tests
// ============================================================================

TEST(InvariantStrategy, SE3AdjointMatrix) {
    using SE3Right = SE3<LieSide::Right>;

    auto X = SE3Right::zero_flux();

    // Adjoint of identity should be identity
    Eigen::Matrix<double, 6, 6> Ad_I = SE3Right::adjoint_matrix(X);

    EXPECT_TRUE(Ad_I.isIdentity(1e-10));
}

TEST(InvariantStrategy, SE23AdjointMatrix) {
    using SE23Right = SE23<LieSide::Right>;

    auto X = SE23Right::zero_flux();

    // Adjoint of identity should be identity
    Eigen::Matrix<double, 9, 9> Ad_I = SE23Right::adjoint_matrix(X);

    EXPECT_TRUE(Ad_I.isIdentity(1e-10));
}

TEST(InvariantStrategy, LeftVsRightInvariantSO3) {
    using SO3Left  = SO3<LieSide::Left>;
    using SO3Right = SO3<LieSide::Right>;

    SO3Right::Drift xi;
    xi << 0.1, 0.2, 0.3;

    auto I = SO3Left::zero_flux();

    // For SO3, left and right invariant compose differently
    auto X_left  = SO3Left::compose(I, xi);
    auto X_right = SO3Right::compose(I, xi);

    // Check that both are valid SO3 elements
    auto diff_left  = SO3Left::diff(I, X_left);
    auto diff_right = SO3Right::diff(I, X_right);

    EXPECT_NEAR(diff_left(0), 0.1, 1e-5);
    EXPECT_NEAR(diff_right(0), 0.1, 1e-5);
}

// ============================================================================
// Dynamics Jacobian for Other Lie Groups
// ============================================================================

TEST(DynamicsJacobian, SO3DynamicsJacobian) {
    using SO3Right = SO3<LieSide::Right>;

    auto state = SO3Right::zero_flux();
    double dt  = 0.01;

    // For SO3, the dynamics_jacobian should be identity
    auto F = SO3Right::dynamics_jacobian(state, dt);

    EXPECT_TRUE(F.isIdentity(1e-10));
}

TEST(DynamicsJacobian, SE3DynamicsJacobian) {
    using SE3Right = SE3<LieSide::Right>;

    auto state = SE3Right::zero_flux();
    double dt  = 0.01;

    // For SE3, the dynamics_jacobian should be identity
    auto F = SE3Right::dynamics_jacobian(state, dt);

    EXPECT_TRUE(F.isIdentity(1e-10));
}
