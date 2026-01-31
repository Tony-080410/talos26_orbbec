/// @file uni_kf_adjoint_test.cpp
/// @brief Unit tests for Adjoint compensation in covariance propagation

#include <gtest/gtest.h>

#include "uni_kf/redact/lie_ops.hpp"
#include <groups/SEn3.hpp>
#include <groups/SO3.hpp>

#include <Eigen/Core>

using namespace uni_kf::redact;
using namespace group;

// ============================================================================
// Helper Functions
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

// ============================================================================
// Adjoint Function Tests
// ============================================================================

TEST(AdjointFunction, SO3AdjointExistence) {
    SO3d R             = SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    Eigen::Matrix3d Ad = lie_ops::adjoint(R);

    // Adjoint should be 3x3
    EXPECT_EQ(Ad.rows(), 3);
    EXPECT_EQ(Ad.cols(), 3);

    // For SO(3), Adjoint equals the rotation matrix
    EXPECT_TRUE(Ad.isApprox(R.asMatrix()));
}

TEST(AdjointFunction, SE3AdjointExistence) {
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    SE3d T                         = SE3d::exp(xi);
    Eigen::Matrix<double, 6, 6> Ad = lie_ops::adjoint(T);

    // Adjoint should be 6x6
    EXPECT_EQ(Ad.rows(), 6);
    EXPECT_EQ(Ad.cols(), 6);

    // SE(3) adjoint has specific structure
    // Upper left 3x3 should be the rotation matrix
    Eigen::Matrix3d R_expected = T.R();
    Eigen::Matrix3d R_actual   = Ad.block(0, 0, 3, 3);
    EXPECT_TRUE(R_actual.isApprox(R_expected));
}

TEST(AdjointFunction, SO3AdjointProperties) {
    SO3d R1 = SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));
    SO3d R2 = SO3d::exp(Eigen::Vector3d(0.4, 0.5, 0.6));

    Eigen::Matrix3d Ad1     = lie_ops::adjoint(R1);
    Eigen::Matrix3d Ad2     = lie_ops::adjoint(R2);
    Eigen::Matrix3d Ad_prod = lie_ops::adjoint(R1 * R2);

    // Ad(R1 * R2) = Ad(R1) * Ad(R2)
    EXPECT_TRUE(Ad_prod.isApprox(Ad1 * Ad2, 1e-10));
}

TEST(AdjointFunction, SO3InvAdjointRelation) {
    SO3d R = SO3d::exp(Eigen::Vector3d(0.1, 0.2, 0.3));

    // Ad(R^(-1)) should equal Ad(R)^(-1)
    Eigen::Matrix3d Ad_R    = lie_ops::adjoint(R);
    Eigen::Matrix3d Ad_Rinv = lie_ops::adjoint(R.inv());

    EXPECT_TRUE(Ad_Rinv.isApprox(Ad_R.inverse().eval(), 1e-10));
}

// ============================================================================
// Adjoint Compensation Tests
// ============================================================================

TEST(AdjointCompensation, SO3SmallAngleApproximation) {
    // For small angles, Ad_Exp(-xi) ≈ I
    Eigen::Vector3d xi_small(0.01, 0.01, 0.01);
    SO3d dX            = SO3d::exp(-xi_small);
    Eigen::Matrix3d Ad = lie_ops::adjoint(dX);

    // Should be close to identity (use looser tolerance)
    EXPECT_TRUE(Ad.isApprox(Eigen::Matrix3d::Identity(), 0.02));
}

TEST(AdjointCompensation, SO3LargeAngleAdjointEffect) {
    // For large angles, Adjoint is significant
    Eigen::Vector3d xi_large(0.0, 0.0, M_PI / 2); // 90 degrees about z
    SO3d dX            = SO3d::exp(-xi_large);
    Eigen::Matrix3d Ad = lie_ops::adjoint(dX);

    // Should NOT be close to identity
    EXPECT_FALSE(Ad.isApprox(Eigen::Matrix3d::Identity(), 0.1));

    // Adjoint for SO(3) is the rotation matrix itself
    Eigen::Matrix3d R_expected =
        Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    EXPECT_TRUE(Ad.isApprox(R_expected, 1e-10));
}

TEST(AdjointCompensation, SO3AdjointTransformsCovariance) {
    // Anisotropic covariance
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Zero();
    P0(0, 0)           = 0.1;  // Large uncertainty in x
    P0(1, 1)           = 0.01; // Small uncertainty in y
    P0(2, 2)           = 0.01; // Small uncertainty in z

    // 90 degree rotation about z-axis
    Eigen::Vector3d xi(0.0, 0.0, M_PI / 2);
    SO3d dX            = SO3d::exp(-xi);
    Eigen::Matrix3d Ad = lie_ops::adjoint(dX);

    Eigen::Matrix3d P_new = Ad * P0 * Ad.transpose();

    // x and y uncertainties should swap
    EXPECT_NEAR(P_new(0, 0), 0.01, 1e-10); // Was y uncertainty
    EXPECT_NEAR(P_new(1, 1), 0.1, 1e-10);  // Was x uncertainty
    EXPECT_NEAR(P_new(2, 2), 0.01, 1e-10); // z unchanged
}

TEST(AdjointCompensation, SE3AdjointTransformsCovariance) {
    // SE(3) covariance with separate rotational and translational uncertainty
    Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Zero();
    P0(0, 0)                       = 0.1;  // Rotational uncertainty x
    P0(1, 1)                       = 0.01; // Rotational uncertainty y
    P0(2, 2)                       = 0.01; // Rotational uncertainty z
    P0(3, 3)                       = 0.5;  // Translational uncertainty x
    P0(4, 4)                       = 0.05; // Translational uncertainty y
    P0(5, 5)                       = 0.05; // Translational uncertainty z

    // 90 degree rotation about z-axis
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.0, 0.0, M_PI / 2, // Rotation
        0.0, 0.0, 0.0;        // No translation
    SE3d dX                        = SE3d::exp(-xi);
    Eigen::Matrix<double, 6, 6> Ad = lie_ops::adjoint(dX);

    Eigen::Matrix<double, 6, 6> P_new = Ad * P0 * Ad.transpose();

    // Check that rotational uncertainty was transformed
    EXPECT_NEAR(P_new(0, 0), 0.01, 1e-10); // Rot x uncertainty swapped with y
    EXPECT_NEAR(P_new(1, 1), 0.1, 1e-10);
    EXPECT_NEAR(P_new(2, 2), 0.01, 1e-10);

    // Check that translational uncertainty was also transformed
    EXPECT_NEAR(P_new(3, 3), 0.05, 1e-10); // Trans x swapped with y
    EXPECT_NEAR(P_new(4, 4), 0.5, 1e-10);
    EXPECT_NEAR(P_new(5, 5), 0.05, 1e-10);
}

TEST(AdjointCompensation, MultipleSequentialRotations) {
    SO3d state;                            // Identity
    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() * 0.01;

    // Three sequential 90-degree rotations
    for (int i = 0; i < 3; ++i) {
        SO3d state_next = state * SO3d::exp(Eigen::Vector3d(0.0, 0.0, M_PI / 2));

        // Compute state motion
        Eigen::Vector3d xi = SO3d::log(state.inv() * state_next);
        SO3d dX            = SO3d::exp(-xi);
        Eigen::Matrix3d Ad = lie_ops::adjoint(dX);

        // Transform covariance
        P     = Ad * P * Ad.transpose();
        state = state_next;
    }

    // After three 90-degree rotations, we're back at identity rotation
    // The covariance should be rotated three times
    EXPECT_TRUE(is_symmetric(P));
    EXPECT_TRUE(is_positive_definite(P));
}

TEST(AdjointCompensation, CovariancePreservesProperties) {
    // Test that adjoint transformation preserves key mathematical properties

    // Identity test: Ad(I) * I * Ad(I)^T = I
    SO3d R_identity;
    Eigen::Matrix3d Ad_I  = lie_ops::adjoint(R_identity);
    Eigen::Matrix3d P0    = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d P_new = Ad_I * P0 * Ad_I.transpose();
    EXPECT_TRUE(P_new.isApprox(Eigen::Matrix3d::Identity(), 1e-10));

    // Positive definiteness is preserved
    Eigen::Matrix3d P_pd = Eigen::Matrix3d::Identity() * 0.1;
    Eigen::Vector3d xi(0.5, 0.3, 0.8);
    SO3d dX                  = SO3d::exp(-xi);
    Eigen::Matrix3d Ad       = lie_ops::adjoint(dX);
    Eigen::Matrix3d P_result = Ad * P_pd * Ad.transpose();
    EXPECT_TRUE(is_positive_definite(P_result));

    // Trace is preserved (det(Ad) = 1 for SO(3))
    EXPECT_NEAR(P_result.trace(), P_pd.trace(), 1e-10);
}

TEST(AdjointCompensation, ThresholdBehavior) {
    // Test the 0.05 rad threshold used in the implementation

    // Just below threshold (0.049 rad)
    Eigen::Vector3d xi_below(0.049, 0.0, 0.0);
    SO3d dX_below            = SO3d::exp(-xi_below);
    Eigen::Matrix3d Ad_below = lie_ops::adjoint(dX_below);

    // Should be close to identity
    EXPECT_TRUE((Ad_below - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() < 0.05);

    // Just above threshold (0.051 rad)
    Eigen::Vector3d xi_above(0.051, 0.0, 0.0);
    SO3d dX_above            = SO3d::exp(-xi_above);
    Eigen::Matrix3d Ad_above = lie_ops::adjoint(dX_above);

    // Should deviate from identity
    double deviation = (Ad_above - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
    EXPECT_GT(deviation, 0.04); // Should be noticeable
}

TEST(AdjointCompensation, ExtremeRotation) {
    // Test with a very large rotation (180 degrees)
    Eigen::Vector3d xi(M_PI, 0.0, 0.0); // 180 degrees about x-axis
    SO3d dX            = SO3d::exp(-xi);
    Eigen::Matrix3d Ad = lie_ops::adjoint(dX);

    // Adjoint for 180-degree rotation
    Eigen::Matrix3d expected;
    expected << 1, 0, 0, 0, -1, 0, 0, 0, -1;

    EXPECT_TRUE(Ad.isApprox(expected, 1e-10));

    // Test covariance transformation
    Eigen::Matrix3d P0 = Eigen::Matrix3d::Identity();
    P0(1, 1)           = 0.5;
    P0(2, 2)           = 0.8;

    Eigen::Matrix3d P_new = Ad * P0 * Ad.transpose();

    // y and z uncertainties should be preserved (diagonal)
    EXPECT_NEAR(P_new(1, 1), 0.5, 1e-10);
    EXPECT_NEAR(P_new(2, 2), 0.8, 1e-10);
}
