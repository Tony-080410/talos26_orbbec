#pragma once

#include "concepts.hpp"
#include "types.hpp"
#include <Eigen/Dense>

namespace fast_tf {

// ============================================================================
// Transform Matrix Wrapper
// ============================================================================

/// 4x4 Transform matrix wrapper with utility methods
class TransformMatrix4d {
public:
    using Matrix = Eigen::Matrix4d;

    TransformMatrix4d()
        : matrix_(Matrix::Identity()) {}

    explicit TransformMatrix4d(const Matrix& m)
        : matrix_(m) {}

    // ========================================================================
    // Factory Methods
    // ========================================================================

    /// Create from RPY angles and translation
    static TransformMatrix4d from_rpy(fp_t roll, fp_t pitch, fp_t yaw, fp_t x, fp_t y, fp_t z) {
        Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q = rz * ry * rx;

        Matrix m            = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m(0, 3)             = x;
        m(1, 3)             = y;
        m(2, 3)             = z;

        return TransformMatrix4d(m);
    }

    /// Create from quaternion and translation
    static TransformMatrix4d
        from_quaternion(fp_t w, fp_t x, fp_t y, fp_t z, fp_t tx, fp_t ty, fp_t tz) {
        Eigen::Quaterniond q(w, x, y, z);
        q.normalize();

        Matrix m            = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m(0, 3)             = tx;
        m(1, 3)             = ty;
        m(2, 3)             = tz;

        return TransformMatrix4d(m);
    }

    // ========================================================================
    // Operators
    // ========================================================================

    TransformMatrix4d operator*(const TransformMatrix4d& other) const {
        return TransformMatrix4d(matrix_ * other.matrix_);
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] TransformMatrix4d inverse() const { return TransformMatrix4d(matrix_.inverse()); }

    [[nodiscard]] const Matrix& matrix() const { return matrix_; }
    [[nodiscard]] Matrix& matrix() { return matrix_; }

    [[nodiscard]] Eigen::Vector3d translation() const { return matrix_.block<3, 1>(0, 3); }

    [[nodiscard]] Eigen::Matrix3d rotation() const { return matrix_.block<3, 3>(0, 0); }

    [[nodiscard]] Eigen::Quaterniond quaternion() const { return Eigen::Quaterniond(rotation()); }

    // ========================================================================
    // Interpolation
    // ========================================================================

    static TransformMatrix4d lerp(const TransformMatrix4d& a, const TransformMatrix4d& b, fp_t t) {
        // Position linear interpolation
        Eigen::Vector3d pos = a.translation() * (1.0 - t) + b.translation() * t;

        // Quaternion spherical linear interpolation
        Eigen::Quaterniond qa(a.rotation());
        Eigen::Quaterniond qb(b.rotation());
        Eigen::Quaterniond q = qa.slerp(t, qb);

        Matrix m            = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m.block<3, 1>(0, 3) = pos;

        return TransformMatrix4d(m);
    }

private:
    Matrix matrix_;
};

// ============================================================================
// Into Specializations for TransformMatrix4d
// ============================================================================

// Forward declaration needed for concept-constrained template
struct RPYTransformData;
struct QuaternionTransformData;

} // namespace fast_tf

// ============================================================================
// std::hash specialization
// ============================================================================

namespace std {

template <>
struct hash<fast_tf::TransformMatrix4d> {
    size_t operator()(const fast_tf::TransformMatrix4d& tm) const {
        return hash<double>()(tm.matrix()(0, 0));
    }
};

} // namespace std
