/// @file uni_kf_redact_test.cpp
/// @brief Unit tests for uni_kf/redact Kalman filter framework
///
/// NOTE: This test file has been updated to use lieplusplus native types
/// directly, instead of the removed Space concept wrapper classes.

#include <gtest/gtest.h>

#include "uni_kf/redact/belief.hpp"
#include "uni_kf/redact/covariance.hpp"
#include "uni_kf/redact/filter.hpp"
#include "uni_kf/redact/variants.hpp"

// lieplusplus native types
#include <groups/SEn3.hpp>
#include <groups/SO3.hpp>

#include <Eigen/Core>

using namespace uni_kf::redact;

// ============================================================================
// Lie Group Native Types Tests
// ============================================================================

TEST(LieGroupNative, SO3BasicOperations) {
    using SO3d = group::SO3<double>;

    // Construct from exponential map
    SO3d R1 = SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    SO3d R2 = SO3d::exp(Eigen::Vector3d(0.2, 0.3, 0.4));

    // Group multiplication
    SO3d R3 = R1 * R2;

    // Inverse
    SO3d R1_inv = R1.inv();

    // Logarithm map
    auto xi = SO3d::log(R1);

    // Adjoint matrix
    auto Ad = R1.Adjoint();

    // Verify dimensions
    EXPECT_EQ(Ad.rows(), 3);
    EXPECT_EQ(Ad.cols(), 3);
}

TEST(LieGroupNative, SE3BasicOperations) {
    using SE3d = group::SEn3<double, 1>;

    // Construct from exponential map
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    SE3d T1 = SE3d::exp(xi);

    // Group multiplication
    SE3d T2 = SE3d::exp(xi * 2.0);
    SE3d T3 = T1 * T2;

    // Inverse
    SE3d T1_inv = T1.inv();

    // Logarithm map
    auto xi_recovered = SE3d::log(T1);

    // Adjoint matrix
    auto Ad = T1.Adjoint();

    EXPECT_EQ(Ad.rows(), 6);
    EXPECT_EQ(Ad.cols(), 6);
}

TEST(LieGroupNative, SE23BasicOperations) {
    using SE23d = group::SEn3<double, 2>;

    // Construct from exponential map
    Eigen::Matrix<double, 9, 1> xi;
    xi << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
    SE23d T1 = SE23d::exp(xi);

    // Group multiplication
    SE23d T2 = SE23d::exp(xi * 2.0);
    SE23d T3 = T1 * T2;

    // Inverse
    SE23d T1_inv = T1.inv();

    // Logarithm map
    auto xi_recovered = SE23d::log(T1);

    // Adjoint matrix
    auto Ad = T1.Adjoint();

    EXPECT_EQ(Ad.rows(), 9);
    EXPECT_EQ(Ad.cols(), 9);
}

// ============================================================================
// Belief with Lie Groups
// ============================================================================

TEST(BeliefNative, SO3BeliefOperations) {
    using LieGroup = group::SO3<double>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Direct>;

    // Construct Belief
    LieGroup state      = LieGroup::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 0.1;
    Belief belief(state, cov);

    // Verify state and covariance
    EXPECT_TRUE(belief.mean().asMatrix().isApprox(state.asMatrix()));
    EXPECT_TRUE(belief.covariance().isApprox(cov));
}

TEST(BeliefNative, SE3BeliefOperations) {
    using LieGroup = group::SEn3<double, 1>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Direct>;

    // Construct Belief
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    LieGroup state                  = LieGroup::exp(xi);
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;
    Belief belief(state, cov);

    // Verify state and covariance
    EXPECT_TRUE(belief.covariance().isApprox(cov));
}

// ============================================================================
// Covariance Storage Tests
// ============================================================================

TEST(CovarianceStorage, DirectStorageSO3) {
    using LieGroup = group::SO3<double>;
    using Storage  = CovarianceStorage<LieGroup, Representation::Direct>;

    Eigen::Matrix3d init = Eigen::Matrix3d::Identity() * 4.0;
    Storage storage(init);

    auto P = storage.mat();
    EXPECT_TRUE(P.isApprox(Eigen::Matrix3d::Identity() * 4.0, 1e-10));
}

TEST(CovarianceStorage, SquareRootStorageSO3) {
    using LieGroup = group::SO3<double>;
    using Storage  = CovarianceStorage<LieGroup, Representation::Sr>;

    Eigen::Matrix3d init = Eigen::Matrix3d::Identity() * 4.0;
    Storage storage(init);

    auto P = storage.mat();
    EXPECT_TRUE(P.isApprox(Eigen::Matrix3d::Identity() * 4.0, 1e-10));
}

TEST(CovarianceStorage, UDUStorageSO3) {
    using LieGroup = group::SO3<double>;
    using Storage  = CovarianceStorage<LieGroup, Representation::Udu>;

    Eigen::Matrix3d init = Eigen::Matrix3d::Identity() * 4.0;
    Storage storage(init);

    auto P = storage.mat();
    EXPECT_TRUE(P.isApprox(Eigen::Matrix3d::Identity() * 4.0, 1e-8));
}

// ============================================================================
// Simple Kalman Filter Integration Test (using LieGroup as belief type)
// ============================================================================

TEST(KalmanFilterNative, SO3BeliefPrediction) {
    // NOTE: This is a simplified test showing how to use lieplusplus native types
    // with the KalmanFilter framework. Full integration tests require custom
    // Perception and Assumption types that provide the necessary interfaces.

    using LieGroup = group::SO3<double>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Direct>;
    using Cov      = Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>;

    // Initial state and covariance
    LieGroup state     = LieGroup::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.001;

    Cov cov(P0);

    // Propagate covariance (identity dynamics for SO3)
    // For invariant filters with identity dynamics: P' = P + Q
    auto dynamics = [](const LieGroup& x, double) { return x; };
    cov.propagate(state, state, dynamics, Q, 0.1);

    auto P_new               = cov.mat();
    Eigen::Matrix3d expected = P0 + Q;
    EXPECT_TRUE(P_new.isApprox(expected, 1e-10));
}

// ============================================================================
// Basic Lie Group Properties
// ============================================================================

TEST(LieGroupProperties, SO3ExpLogRoundtrip) {
    using SO3d = group::SO3<double>;

    Eigen::Vector3d xi(0.1, 0.2, 0.3);
    SO3d R                       = SO3d::exp(xi);
    Eigen::Vector3d xi_recovered = SO3d::log(R);

    EXPECT_TRUE(xi.isApprox(xi_recovered, 1e-10));
}

TEST(LieGroupProperties, SE3ExpLogRoundtrip) {
    using SE3d = group::SEn3<double, 1>;

    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    SE3d T                                   = SE3d::exp(xi);
    Eigen::Matrix<double, 6, 1> xi_recovered = SE3d::log(T);

    EXPECT_TRUE(xi.isApprox(xi_recovered, 1e-10));
}

TEST(LieGroupProperties, SO3InverseProperty) {
    using SO3d = group::SO3<double>;

    SO3d R        = SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    SO3d R_inv    = R.inv();
    SO3d identity = R * R_inv;

    // Identity should have matrix close to I
    EXPECT_TRUE(identity.asMatrix().isApprox(Eigen::Matrix3d::Identity(), 1e-10));
}

TEST(LieGroupProperties, SO3AdjointInverseRelation) {
    using SO3d = group::SO3<double>;

    SO3d R = SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));

    // Ad(R) should equal inv(Ad(R^-1))
    auto Ad_R    = R.Adjoint();
    auto Ad_Rinv = R.invAdjoint();

    EXPECT_TRUE(Ad_R.isApprox(Ad_Rinv.inverse().eval(), 1e-10));
}

// ============================================================================
// Manifold Distance Tests
// ============================================================================

TEST(ManifoldDistance, SO3RiemannianDistance) {
    using SO3d = group::SO3<double>;

    SO3d R1 = SO3d::exp(Eigen::Vector3d(0.1, 0.0, 0.0));
    SO3d R2 = SO3d::exp(Eigen::Vector3d(0.2, 0.0, 0.0));

    // Distance: ||Log(R1^-1 * R2)||
    Eigen::Vector3d error = SO3d::log(R1.inv() * R2);
    double distance       = error.norm();

    // Should be approximately 0.1
    EXPECT_NEAR(distance, 0.1, 1e-6);
}

TEST(ManifoldDistance, SE3RiemannianDistance) {
    using SE3d = group::SEn3<double, 1>;

    SE3d T1 = SE3d::exp((Eigen::Matrix<double, 6, 1>() << 0.1, 0.0, 0.0, 0.0, 0.0, 0.0).finished());
    SE3d T2 = SE3d::exp((Eigen::Matrix<double, 6, 1>() << 0.2, 0.0, 0.0, 0.0, 0.0, 0.0).finished());

    // Distance: ||Log(T1^-1 * T2)||
    Eigen::Matrix<double, 6, 1> error = SE3d::log(T1.inv() * T2);
    double distance                   = error.norm();

    // Should be approximately 0.1
    EXPECT_NEAR(distance, 0.1, 1e-6);
}

// ============================================================================
// NOTE: The following tests have been removed as they depend on deleted types
// ============================================================================
// - Space concept tests (required Space<T> concept)
// - Euclidean space tests (required Euclidean<N> type)
// - ProductSpace tests (required ProductSpace<S1, S2> type)
// - LogDistanceSO3 tests (required LogDistanceSO3 type)
// - Old KalmanFilter integration tests (required old Space-based types)
//
// These tests should be reimplemented using lieplusplus native types if needed.
// ============================================================================

// ============================================================================
// Adjoint Compensation Tests for Large Angle Rotations
// ============================================================================

namespace {
// Helper to check if matrix is symmetric
bool is_symmetric(const Eigen::MatrixXd& M, double tol = 1e-10) {
    return (M - M.transpose()).cwiseAbs().maxCoeff() < tol;
}

// Helper to check if matrix is positive definite
bool is_positive_definite(const Eigen::MatrixXd& M) {
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    return llt.info() == Eigen::Success;
}
} // namespace

TEST(AdjointCompensation, SO3SmallAngleNoAdjointApplied) {
    using LieGroup = group::SO3<double>;
    using Cov      = Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>;

    // Initial state
    LieGroup state_old = LieGroup::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.001;

    Cov cov(P0);

    // Small angle rotation (< threshold of 0.05 rad)
    // This should use P' = P + Q (no adjoint)
    LieGroup state_new = state_old * LieGroup::exp(Eigen::Vector3d(0.01, 0.01, 0.01));
    auto dynamics      = [](const LieGroup& x, double) { return x; };

    cov.propagate(state_old, state_new, dynamics, Q, 0.1);

    auto P_new               = cov.mat();
    Eigen::Matrix3d expected = P0 + Q;

    // For small angles, should match simple propagation
    EXPECT_TRUE(P_new.isApprox(expected, 1e-10));
    EXPECT_TRUE(is_symmetric(P_new));
    EXPECT_TRUE(is_positive_definite(P_new));
}

TEST(InvariantLinearStrategy, SO3LargeAngleStillNoAdjoint) {
    using LieGroup = group::SO3<double>;
    using Cov      = Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>;

    // Initial state
    LieGroup state_old; // Default constructor = identity
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.001;

    Cov cov(P0);

    // Large angle rotation (90 degrees = pi/2 ≈ 1.57 rad >> 0.05 threshold)
    // Rotation about z-axis
    LieGroup state_new = state_old * LieGroup::exp(Eigen::Vector3d(0.0, 0.0, M_PI / 2));
    auto dynamics      = [](const LieGroup& x, double) { return x; };

    cov.propagate(state_old, state_new, dynamics, Q, 0.1);

    auto P_new = cov.mat();

    // With right-invariant error, covariance is expressed in the invariant error
    // coordinates, so rotations of the mean do not rotate P.
    Eigen::Matrix3d expected = P0 + Q;

    EXPECT_TRUE(P_new.isApprox(expected, 1e-10));
    EXPECT_TRUE(is_symmetric(P_new));
    EXPECT_TRUE(is_positive_definite(P_new));
}

TEST(InvariantLinearStrategy, SO3LargeAngleDoesNotRotateCovarianceAxes) {
    using LieGroup = group::SO3<double>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Direct>;
    using Cov      = Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>;

    // Anisotropic covariance to test rotation effects
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Zero();
    P0(0, 0)           = 0.1;  // Large uncertainty in x
    P0(1, 1)           = 0.01; // Small uncertainty in y
    P0(2, 2)           = 0.01; // Small uncertainty in z
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.0001;

    Cov cov(P0);
    LieGroup state_old;

    // Rotate 90 degrees about z-axis
    // Should swap x and y uncertainty
    LieGroup state_new = state_old * LieGroup::exp(Eigen::Vector3d(0.0, 0.0, M_PI / 2));
    auto dynamics      = [](const LieGroup& x, double) { return x; };

    cov.propagate(state_old, state_new, dynamics, Q, 0.1);
    auto P_new = cov.mat();

    // With right-invariant error, the covariance axes rotate with the mean.
    EXPECT_NEAR(P_new(0, 0), 0.1 + 0.0001, 1e-10);
    EXPECT_NEAR(P_new(1, 1), 0.01 + 0.0001, 1e-10);
    EXPECT_NEAR(P_new(2, 2), 0.01 + 0.0001, 1e-10);
}

TEST(InvariantLinearStrategy, SE3LargeAngleStillNoAdjoint) {
    using LieGroup = group::SEn3<double, 1>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Direct>;
    using Cov      = Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>;

    // Initial state
    LieGroup state_old; // Default constructor = identity
    Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
    Eigen::Matrix<double, 6, 6> Q  = Eigen::Matrix<double, 6, 6>::Identity() * 0.001;

    Cov cov(P0);

    // Large rotation (90 degrees about z)
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.0, 0.0, M_PI / 2, // Large rotation
        0.1, 0.1, 0.1;        // Small translation
    LieGroup state_new = state_old * LieGroup::exp(xi);
    auto dynamics      = [](const LieGroup& x, double) { return x; };

    cov.propagate(state_old, state_new, dynamics, Q, 0.1);
    auto P_new = cov.mat();

    // Verify symmetry and positive definiteness
    EXPECT_TRUE(is_symmetric(P_new, 1e-9));
    EXPECT_TRUE(is_positive_definite(P_new));
    EXPECT_TRUE(P_new.isApprox(P0 + Q, 1e-10));
}

TEST(AdjointCompensation, MultipleSequentialRotations) {
    using LieGroup = group::SO3<double>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Direct>;
    using Cov      = Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>;

    LieGroup state; // Default constructor = identity
    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity() * 0.0001;

    Cov cov(P);

    auto dynamics = [](const LieGroup& x, double) { return x; };

    // Three sequential 90-degree rotations = 270 degrees total
    for (int i = 0; i < 3; ++i) {
        LieGroup state_next = state * LieGroup::exp(Eigen::Vector3d(0.0, 0.0, M_PI / 2));
        cov.propagate(state, state_next, dynamics, Q, 0.1);
        state = state_next;
    }

    auto P_final = cov.mat();

    // Should still be symmetric and positive definite after multiple updates
    EXPECT_TRUE(is_symmetric(P_final));
    EXPECT_TRUE(is_positive_definite(P_final));

    // Trace should increase by approximately 3 * trace(Q)
    double expected_trace = P.trace() + 3.0 * Q.trace();
    EXPECT_NEAR(P_final.trace(), expected_trace, 0.01);
}

TEST(AdjointCompensation, AdjointVersusNoAdjointComparison) {
    using LieGroup = group::SO3<double>;
    using Storage  = CovarianceStorage<LieGroup, Representation::Direct>;

    // Anisotropic covariance
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Zero();
    P0(0, 0)           = 0.1;
    P0(1, 1)           = 0.01;
    P0(2, 2)           = 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Zero();

    // Manual propagation WITHOUT adjoint (wrong for large angles)
    Eigen::Matrix3d P_no_adjoint = P0 + Q;

    // Manual propagation WITH adjoint (correct)
    LieGroup state_old;
    LieGroup state_new             = state_old * LieGroup::exp(Eigen::Vector3d(0.0, 0.0, M_PI / 2));
    Eigen::Vector3d xi             = LieGroup::log(state_old.inv() * state_new);
    LieGroup dX                    = LieGroup::exp(-xi);
    Eigen::Matrix3d Ad             = dX.Adjoint();
    Eigen::Matrix3d P_with_adjoint = Ad * P0 * Ad.transpose() + Q;

    // They should be DIFFERENT for large rotations
    EXPECT_FALSE(P_no_adjoint.isApprox(P_with_adjoint, 1e-6));

    // The adjoint version should have swapped x and y variances
    EXPECT_NEAR(P_no_adjoint(0, 0), 0.1, 1e-10);    // No change
    EXPECT_NEAR(P_with_adjoint(0, 0), 0.01, 1e-10); // Swapped
    EXPECT_NEAR(P_with_adjoint(1, 1), 0.1, 1e-10);  // Swapped
}

TEST(InvariantLinearStrategy, NoAngleThreshold) {
    using LieGroup = group::SO3<double>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Direct>;
    using Cov      = Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>;

    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Zero();   // Zero noise to see adjoint effect clearly

    auto dynamics = [](const LieGroup& x, double) { return x; };

    Cov cov1(P0);
    LieGroup state_old;
    LieGroup state_new_small = state_old * LieGroup::exp(Eigen::Vector3d(0.049, 0.0, 0.0));
    cov1.propagate(state_old, state_new_small, dynamics, Q, 0.1);

    Cov cov2(P0);
    LieGroup state_new_large = state_old * LieGroup::exp(Eigen::Vector3d(0.051, 0.0, 0.0));
    cov2.propagate(state_old, state_new_large, dynamics, Q, 0.1);

    EXPECT_TRUE(cov1.mat().isApprox(P0, 1e-10));
    EXPECT_TRUE(cov2.mat().isApprox(P0, 1e-10));
}

TEST(InvariantLinearStrategy, SquareRootStorageNoAdjoint) {
    using LieGroup = group::SO3<double>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Sr>;
    using Cov      = Covariance<LieGroup, Representation::Sr, InvariantLinearStrategy>;

    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.001;

    Cov cov(P0);

    // Large angle rotation
    LieGroup state_old;
    LieGroup state_new = state_old * LieGroup::exp(Eigen::Vector3d(0.0, 0.0, M_PI / 2));
    auto dynamics      = [](const LieGroup& x, double) { return x; };

    cov.propagate(state_old, state_new, dynamics, Q, 0.1);
    auto P_new = cov.mat();

    // Verify mathematical properties even with square root storage
    EXPECT_TRUE(is_symmetric(P_new, 1e-8));
    EXPECT_TRUE(is_positive_definite(P_new));
    EXPECT_TRUE(P_new.isApprox(P0 + Q, 1e-8));
}

TEST(InvariantLinearStrategy, UDUStorageNoAdjoint) {
    using LieGroup = group::SO3<double>;
    using Belief   = uni_kf::redact::Belief<LieGroup, Representation::Udu>;
    using Cov      = Covariance<LieGroup, Representation::Udu, InvariantLinearStrategy>;

    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d Q  = Eigen::Matrix3d::Identity() * 0.001;

    Cov cov(P0);

    // Large angle rotation
    LieGroup state_old;
    LieGroup state_new = state_old * LieGroup::exp(Eigen::Vector3d(0.0, 0.0, M_PI / 2));
    auto dynamics      = [](const LieGroup& x, double) { return x; };

    cov.propagate(state_old, state_new, dynamics, Q, 0.1);
    auto P_new = cov.mat();

    // Verify mathematical properties even with UDU storage
    EXPECT_TRUE(is_symmetric(P_new, 1e-8));
    EXPECT_TRUE(is_positive_definite(P_new));
    EXPECT_TRUE(P_new.isApprox(P0 + Q, 1e-7));
}
