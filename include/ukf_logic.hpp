#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "tracker_soa.hpp"

namespace talos {

    /**
     * @brief High-performance Kalman Filter for armor tracking
     *
     * Performance targets:
     * - predict() < 10ns
     * - update() < 250ns
     *
     * Optimizations:
     * 1. Batch matrix update (4 observations at once)
     * 2. Eigen SIMD vectorization (SSE/AVX/NEON)
     * 3. Fixed-size matrices with compile-time unrolling
     * 4. Exploit sparse H matrix structure
     * 5. LLT decomposition for numerical stability
     *
     * State vector (11D):
     * x = [px, py, pz, vx, vy, vz, yaw, v_yaw, r1, r2, _]
     */

#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_VECTORIZE
#endif

    template<int Rows, int Cols>
    using MatRM = Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>;

    using State11 = Eigen::Matrix<float, 11, 1>;
    using Cov11 = MatRM<11, 11>;
    using Vec3 = Eigen::Vector3f;
    using Vec4 = Eigen::Vector4f;
    using Mat3 = Eigen::Matrix3f;
    using Mat4 = MatRM<4, 4>;
    using Mat4x11 = MatRM<4, 11>;
    using Mat11x4 = MatRM<11, 4>;

    /**
     * @brief Kalman Filter state (stores P directly for performance)
     */
    struct alignas(64) UKFState {
        Cov11 P_; // Covariance matrix (11x11)
        State11 Q_diag; // Process noise diagonal
        Vec4 R_diag; // Measurement noise diagonal

        UKFState() {
            P_.setIdentity();
            P_ *= 0.1f;

            Q_diag << 0.01f, 0.01f, 0.01f, // pos
                    0.1f, 0.1f, 0.1f, // vel
                    0.01f, // yaw
                    0.1f, // v_yaw
                    0.001f, 0.001f, 0.0f; // radius

            R_diag << 0.05f, 0.05f, 0.1f, 0.1f;
        }

        Cov11 P() const { return P_; }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    /**
     * @brief Ultra-fast CV prediction
     */
    constexpr void predict(TrackerSoA &tracker, UKFState &ukf, float dt_ms) {
        const float dt = dt_ms * 0.001f;
        const float dt2 = dt * dt;

        // Mean prediction
        tracker.pos[0] += tracker.vel[0] * dt;
        tracker.pos[1] += tracker.vel[1] * dt;
        tracker.pos[2] += tracker.vel[2] * dt;
        tracker.yaw += tracker.v_yaw * dt;

        // Branchless yaw normalization
        constexpr float two_pi = 2.0f * static_cast<float>(M_PI);
        tracker.yaw -= two_pi * std::floor((tracker.yaw + static_cast<float>(M_PI)) / two_pi);

        // Sparse covariance prediction
        auto &P = ukf.P_;

        P.block<3, 3>(0, 0).noalias() += dt * (P.block<3, 3>(0, 3) + P.block<3, 3>(3, 0)) + dt2 * P.block<3, 3>(3, 3);
        P.block<3, 3>(0, 3).noalias() += dt * P.block<3, 3>(3, 3);
        P.block<3, 3>(3, 0) = P.block<3, 3>(0, 3).transpose();

        P(6, 6) += 2.0f * dt * P(6, 7) + dt2 * P(7, 7);
        P(6, 7) += dt * P(7, 7);
        P(7, 6) = P(6, 7);

        P.diagonal().noalias() += ukf.Q_diag * dt;
    }

    /**
     * @brief Batch update for 4 observations (pos_xyz + yaw)
     */
    constexpr void update(TrackerSoA &tracker, UKFState &ukf, const Vec3 &measured_pos, float measured_yaw) {
        auto &P = ukf.P_;

        // Innovation
        Vec4 y;
        y << measured_pos.x() - tracker.pos[0], measured_pos.y() - tracker.pos[1], measured_pos.z() - tracker.pos[2],
                measured_yaw - tracker.yaw;

        if (y(3) > static_cast<float>(M_PI))
            y(3) -= 2.0f * static_cast<float>(M_PI);
        if (y(3) < static_cast<float>(-M_PI))
            y(3) += 2.0f * static_cast<float>(M_PI);

        // P * H^T (H selects cols 0,1,2,6)
        Mat11x4 PHt;
        PHt.col(0) = P.col(0);
        PHt.col(1) = P.col(1);
        PHt.col(2) = P.col(2);
        PHt.col(3) = P.col(6);

        // S = H * P * H^T + R
        Mat4 S;
        S(0, 0) = P(0, 0);
        S(0, 1) = P(0, 1);
        S(0, 2) = P(0, 2);
        S(0, 3) = P(0, 6);
        S(1, 0) = P(1, 0);
        S(1, 1) = P(1, 1);
        S(1, 2) = P(1, 2);
        S(1, 3) = P(1, 6);
        S(2, 0) = P(2, 0);
        S(2, 1) = P(2, 1);
        S(2, 2) = P(2, 2);
        S(2, 3) = P(2, 6);
        S(3, 0) = P(6, 0);
        S(3, 1) = P(6, 1);
        S(3, 2) = P(6, 2);
        S(3, 3) = P(6, 6);
        S.diagonal() += ukf.R_diag;

        // K = PHt * S^-1 (using LLT)
        const Mat4 S_inv = S.llt().solve(Mat4::Identity());
        const Mat11x4 K = PHt * S_inv;

        // State update
        State11 dx = K * y;
        tracker.pos[0] += dx(0);
        tracker.pos[1] += dx(1);
        tracker.pos[2] += dx(2);
        tracker.vel[0] += dx(3);
        tracker.vel[1] += dx(4);
        tracker.vel[2] += dx(5);
        tracker.yaw += dx(6);
        tracker.v_yaw += dx(7);
        tracker.radius[0] += dx(8);
        tracker.radius[1] += dx(9);

        // P = (I - K*H) * P
        Mat4x11 HP;
        HP.row(0) = P.row(0);
        HP.row(1) = P.row(1);
        HP.row(2) = P.row(2);
        HP.row(3) = P.row(6);
        P.noalias() -= K * HP;

        // Symmetrize
        P = (P + P.transpose()) * 0.5f;
    }

    /**
     * @brief Scalar update for single observation
     */
    constexpr void scalar_update(TrackerSoA &tracker, UKFState &ukf, int obs_idx, float measurement, float R) {
        auto &P = ukf.P_;

        float *x_ptr;
        switch (obs_idx) {
            case 0:
                x_ptr = &tracker.pos[0];
                break;
            case 1:
                x_ptr = &tracker.pos[1];
                break;
            case 2:
                x_ptr = &tracker.pos[2];
                break;
            case 3:
                x_ptr = &tracker.vel[0];
                break;
            case 4:
                x_ptr = &tracker.vel[1];
                break;
            case 5:
                x_ptr = &tracker.vel[2];
                break;
            case 6:
                x_ptr = &tracker.yaw;
                break;
            case 7:
                x_ptr = &tracker.v_yaw;
                break;
            case 8:
                x_ptr = &tracker.radius[0];
                break;
            case 9:
                x_ptr = &tracker.radius[1];
                break;
            default:
                return;
        }

        float y = measurement - *x_ptr;
        if (obs_idx == 6) {
            if (y > static_cast<float>(M_PI))
                y -= 2.0f * static_cast<float>(M_PI);
            if (y < static_cast<float>(-M_PI))
                y += 2.0f * static_cast<float>(M_PI);
        }

        const float S = P(obs_idx, obs_idx) + R;
        const float S_inv = 1.0f / S;
        State11 K = P.col(obs_idx) * S_inv;

        tracker.pos[0] += K(0) * y;
        tracker.pos[1] += K(1) * y;
        tracker.pos[2] += K(2) * y;
        tracker.vel[0] += K(3) * y;
        tracker.vel[1] += K(4) * y;
        tracker.vel[2] += K(5) * y;
        tracker.yaw += K(6) * y;
        tracker.v_yaw += K(7) * y;
        tracker.radius[0] += K(8) * y;
        tracker.radius[1] += K(9) * y;

        P.noalias() -= K * P.row(obs_idx);
    }

    /**
     * @brief Mahalanobis distance for data association
     */
    inline float mahalanobis_distance_sq(const TrackerSoA &tracker, const UKFState &ukf, const Vec3 &measured_pos) {
        Vec3 inn;
        inn << measured_pos.x() - tracker.pos[0], measured_pos.y() - tracker.pos[1], measured_pos.z() - tracker.pos[2];

        Mat3 S = ukf.P_.block<3, 3>(0, 0);
        S(0, 0) += ukf.R_diag(0);
        S(1, 1) += ukf.R_diag(1);
        S(2, 2) += ukf.R_diag(2);

        return inn.dot(S.llt().solve(inn));
    }

    /**
     * @brief Initialize from detection
     */
    inline void init_from_detection(TrackerSoA &tracker, UKFState &ukf, const Vec3 &pos, float yaw) {
        tracker.pos[0] = pos.x();
        tracker.pos[1] = pos.y();
        tracker.pos[2] = pos.z();
        tracker.vel[0] = 0.0f;
        tracker.vel[1] = 0.0f;
        tracker.vel[2] = 0.0f;
        tracker.yaw = yaw;
        tracker.v_yaw = 0.0f;
        tracker.radius[0] = 1.0f;
        tracker.radius[1] = 1.0f;

        ukf.P_.setIdentity();
        ukf.P_ *= 0.1f;
        ukf.P_(3, 3) = 1.0f;
        ukf.P_(4, 4) = 1.0f;
        ukf.P_(5, 5) = 1.0f;
        ukf.P_(7, 7) = 1.0f;
    }

    constexpr float MAHAL_THRESHOLD_SQ = 9.0f;

    inline bool is_associated(const TrackerSoA &tracker, const UKFState &ukf, const Vec3 &measured_pos) {
        return mahalanobis_distance_sq(tracker, ukf, measured_pos) < MAHAL_THRESHOLD_SQ;
    }

} // namespace spike
