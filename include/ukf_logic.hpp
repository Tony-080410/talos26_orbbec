#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "tracker_soa.hpp"

namespace spike {

/**
 * @brief Simplified UKF for armor tracking
 *
 * Design goals:
 * - predict() < 200ns
 * - update() < 300ns
 *
 * Simplifications applied (to achieve latency targets):
 * 1. Linear constant-velocity model (no sigma points)
 * 2. Diagonal process/measurement noise (no matrix inversion)
 * 3. Fixed-size Eigen matrices (compile-time unrolling)
 * 4. Scalar Kalman gain computation (observation dimension = 1 at a time)
 *
 * State vector (11D):
 * x = [px, py, pz, vx, vy, vz, yaw, v_yaw, r1, r2, _]
 *     pos[3]     vel[3]       yaw  v_yaw  radius[2]
 */

// Type aliases for fixed-size matrices
using State11 = Eigen::Matrix<float, 11, 1>;
using Cov11 = Eigen::Matrix<float, 11, 11>;
using Vec3 = Eigen::Vector3f;

/**
 * @brief Minimal UKF state holder (separate from TrackerSoA for covariance)
 *
 * TrackerSoA holds the mean state (64 bytes, L1 cache optimized)
 * UKFState holds the covariance (484 bytes, in separate memory)
 */
struct alignas(64) UKFState {
    // Covariance matrix P (11x11 = 121 floats = 484 bytes)
    Cov11 P;

    // Process noise Q (diagonal, stored as vector)
    State11 Q_diag;

    // Measurement noise R (diagonal, 3D position + yaw = 4 elements)
    Eigen::Vector4f R_diag;

    UKFState() {
        // Initialize covariance
        P = Cov11::Identity() * 0.1f;

        // Process noise (tuned for ~500Hz update rate)
        Q_diag << 0.01f, 0.01f, 0.01f,    // position variance
                  0.1f, 0.1f, 0.1f,        // velocity variance
                  0.01f,                    // yaw variance
                  0.1f,                     // v_yaw variance
                  0.001f, 0.001f, 0.0f;    // radius variance (near constant)

        // Measurement noise (camera + PnP solver)
        R_diag << 0.05f, 0.05f, 0.1f, 0.1f; // [px, py, pz, yaw]
    }

    void reset() {
        P = Cov11::Identity() * 0.1f;
    }
};

/**
 * @brief Constant-velocity prediction model
 *
 * State transition:
 *   pos' = pos + vel * dt
 *   yaw' = yaw + v_yaw * dt
 *   vel' = vel (constant)
 *   v_yaw' = v_yaw (constant)
 *   radius' = radius (constant)
 *
 * Covariance update:
 *   P' = F * P * F^T + Q
 *
 * Since F is nearly identity (only pos/yaw depend on vel),
 * we use optimized sparse multiplication.
 *
 * @param tracker TrackerSoA state (mean) - will be modified
 * @param ukf UKFState (covariance) - will be modified
 * @param dt_ms Delta time in milliseconds
 * @return Latency in nanoseconds (for profiling)
 */
inline void predict(TrackerSoA& tracker, UKFState& ukf, float dt_ms) {
    const float dt = dt_ms * 0.001f; // Convert to seconds

    // === Mean prediction (simple CV model) ===
    tracker.pos[0] += tracker.vel[0] * dt;
    tracker.pos[1] += tracker.vel[1] * dt;
    tracker.pos[2] += tracker.vel[2] * dt;
    tracker.yaw += tracker.v_yaw * dt;

    // Normalize yaw to [-pi, pi]
    if (tracker.yaw > M_PI) tracker.yaw -= 2.0f * M_PI;
    if (tracker.yaw < -M_PI) tracker.yaw += 2.0f * M_PI;

    // === Covariance prediction (sparse F matrix) ===
    // F = I + dt * A, where A has only off-diagonal entries:
    // d(pos)/d(vel) = dt, d(yaw)/d(v_yaw) = dt
    //
    // P' = F * P * F^T + Q
    //
    // Optimized: Only update correlated blocks
    // P'[0:3, 0:3] += dt * (P[0:3, 3:6] + P[3:6, 0:3]) + dt^2 * P[3:6, 3:6]
    // P'[6, 6] += dt * (P[6, 7] + P[7, 6]) + dt^2 * P[7, 7]

    const float dt2 = dt * dt;

    // Position-velocity covariance update
    for (int i = 0; i < 3; ++i) {
        // P[i,i] += 2*dt*P[i,i+3] + dt^2*P[i+3,i+3]
        ukf.P(i, i) += 2.0f * dt * ukf.P(i, i + 3) + dt2 * ukf.P(i + 3, i + 3);

        // Cross terms
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                ukf.P(i, j) += dt * (ukf.P(i, j + 3) + ukf.P(i + 3, j));
            }
        }

        // Position-velocity correlation
        ukf.P(i, i + 3) += dt * ukf.P(i + 3, i + 3);
        ukf.P(i + 3, i) = ukf.P(i, i + 3); // Symmetry
    }

    // Yaw-v_yaw covariance update
    ukf.P(6, 6) += 2.0f * dt * ukf.P(6, 7) + dt2 * ukf.P(7, 7);
    ukf.P(6, 7) += dt * ukf.P(7, 7);
    ukf.P(7, 6) = ukf.P(6, 7);

    // Add process noise (diagonal)
    for (int i = 0; i < 11; ++i) {
        ukf.P(i, i) += ukf.Q_diag(i) * dt;
    }
}

/**
 * @brief Scalar Kalman update for single measurement dimension
 *
 * For each observed dimension independently:
 *   K = P * H^T / (H * P * H^T + R)
 *   x = x + K * (z - H * x)
 *   P = (I - K * H) * P
 *
 * Where H is a unit vector selecting the observed state.
 *
 * @param tracker TrackerSoA state - will be modified
 * @param ukf UKFState covariance - will be modified
 * @param obs_idx State index being observed (0-10)
 * @param measurement Measured value
 * @param R Measurement variance for this dimension
 */
inline void scalar_update(TrackerSoA& tracker, UKFState& ukf,
                          int obs_idx, float measurement, float R) {
    // Get current state value
    float x_prior;
    switch (obs_idx) {
        case 0: case 1: case 2:
            x_prior = tracker.pos[obs_idx];
            break;
        case 3: case 4: case 5:
            x_prior = tracker.vel[obs_idx - 3];
            break;
        case 6:
            x_prior = tracker.yaw;
            break;
        case 7:
            x_prior = tracker.v_yaw;
            break;
        case 8: case 9:
            x_prior = tracker.radius[obs_idx - 8];
            break;
        default:
            return;
    }

    // Innovation
    float y = measurement - x_prior;

    // Handle angle wrap for yaw
    if (obs_idx == 6) {
        while (y > M_PI) y -= 2.0f * M_PI;
        while (y < -M_PI) y += 2.0f * M_PI;
    }

    // Innovation variance: S = H * P * H^T + R = P[i,i] + R
    float S = ukf.P(obs_idx, obs_idx) + R;

    // Kalman gain: K = P * H^T / S = P[col_i] / S
    State11 K = ukf.P.col(obs_idx) / S;

    // State update: x = x + K * y
    // Only update non-zero entries
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

    // Covariance update: P = (I - K * H) * P
    // = P - K * H * P = P - K * P[row_i]
    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 11; ++j) {
            ukf.P(i, j) -= K(i) * ukf.P(obs_idx, j);
        }
    }

    // Ensure symmetry
    ukf.P = (ukf.P + ukf.P.transpose()) * 0.5f;
}

/**
 * @brief Full measurement update (position + yaw)
 *
 * @param tracker TrackerSoA state
 * @param ukf UKFState covariance
 * @param measured_pos Measured [x, y, z] position from PnP
 * @param measured_yaw Measured yaw angle from PnP
 */
inline void update(TrackerSoA& tracker, UKFState& ukf,
                   const Vec3& measured_pos, float measured_yaw) {
    // Sequential scalar updates (faster than matrix inversion for small obs)
    scalar_update(tracker, ukf, 0, measured_pos.x(), ukf.R_diag(0));
    scalar_update(tracker, ukf, 1, measured_pos.y(), ukf.R_diag(1));
    scalar_update(tracker, ukf, 2, measured_pos.z(), ukf.R_diag(2));
    scalar_update(tracker, ukf, 6, measured_yaw, ukf.R_diag(3));
}

/**
 * @brief Compute Mahalanobis distance for data association
 *
 * d^2 = (z - H*x)^T * S^-1 * (z - H*x)
 *
 * For position-only association:
 * d^2 = (z_pos - x_pos)^T * P[0:3,0:3]^-1 * (z_pos - x_pos)
 *
 * @param tracker Current tracker state
 * @param ukf UKF covariance
 * @param measured_pos Candidate measurement
 * @return Squared Mahalanobis distance
 */
inline float mahalanobis_distance_sq(const TrackerSoA& tracker,
                                     const UKFState& ukf,
                                     const Vec3& measured_pos) {
    Vec3 innovation;
    innovation << measured_pos.x() - tracker.pos[0],
                  measured_pos.y() - tracker.pos[1],
                  measured_pos.z() - tracker.pos[2];

    // Extract position covariance block
    Eigen::Matrix3f P_pos = ukf.P.block<3, 3>(0, 0);

    // Add measurement noise
    P_pos(0, 0) += ukf.R_diag(0);
    P_pos(1, 1) += ukf.R_diag(1);
    P_pos(2, 2) += ukf.R_diag(2);

    // Compute d^2 = y^T * S^-1 * y
    // Using ldlt for numerical stability
    return innovation.dot(P_pos.ldlt().solve(innovation));
}

/**
 * @brief Initialize tracker state from first detection
 *
 * @param tracker TrackerSoA to initialize
 * @param ukf UKFState to initialize
 * @param pos Initial position
 * @param yaw Initial yaw
 */
inline void init_from_detection(TrackerSoA& tracker, UKFState& ukf,
                                const Vec3& pos, float yaw) {
    // Set position
    tracker.pos[0] = pos.x();
    tracker.pos[1] = pos.y();
    tracker.pos[2] = pos.z();

    // Zero velocity (no prior)
    tracker.vel[0] = 0.0f;
    tracker.vel[1] = 0.0f;
    tracker.vel[2] = 0.0f;

    // Set yaw
    tracker.yaw = yaw;
    tracker.v_yaw = 0.0f;

    // Default radius
    tracker.radius[0] = 1.0f;
    tracker.radius[1] = 1.0f;

    // Reset covariance
    ukf.reset();

    // High initial uncertainty for velocity
    ukf.P(3, 3) = 1.0f;
    ukf.P(4, 4) = 1.0f;
    ukf.P(5, 5) = 1.0f;
    ukf.P(7, 7) = 1.0f; // v_yaw
}

/**
 * @brief Chi-squared threshold for data association
 *
 * For 3 DOF (position), chi-squared thresholds:
 * - 95%: 7.81
 * - 99%: 11.34
 */
constexpr float MAHAL_THRESHOLD_SQ = 9.0f; // ~97% confidence

/**
 * @brief Check if measurement is associated with tracker
 */
inline bool is_associated(const TrackerSoA& tracker, const UKFState& ukf,
                          const Vec3& measured_pos) {
    return mahalanobis_distance_sq(tracker, ukf, measured_pos) < MAHAL_THRESHOLD_SQ;
}

} // namespace spike
