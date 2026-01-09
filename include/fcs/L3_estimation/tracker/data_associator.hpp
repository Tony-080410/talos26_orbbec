#pragma once

#include "motion_model.hpp"
#include "util.hpp"

#include "fcs/core/types.hpp"

#include <Eigen/Dense>
#include <array>
#include <limits>
#include <queue>
#include <vector>

namespace fcs::L3 {

/// Template-based data associator for armor matching
/// Uses Sigma-point gating for robust observation-to-track association
template <MotionModel Model>
class DataAssociator {
public:
    static constexpr int NZ = Model::NZ;
    static constexpr int NX = Model::NX;

    using VecZ  = typename Model::VecZ;
    using VecX  = Eigen::Matrix<double, NX, 1>;
    using MatXX = Eigen::Matrix<double, NX, NX>;
    using MatZ  = Eigen::Matrix<double, NZ, NZ>;

    struct MatchResult {
        std::vector<AtLegacyArmorMeasurement> armors;
        std::vector<int> armor_ids;
    };

    /// Match observations to predicted armor hypotheses.
    ///
    /// Problem:
    ///   A single frame can contain multiple armor detections: different plates of the same target
    ///   plus clutter/false positives. We must decide:
    ///     1) which detections belong to the current track, and
    ///     2) which `armor_id` each detection corresponds to (0..ARMORS_NUM-1).
    ///
    /// Intuition:
    ///   Score each (observation, armor_id) by how plausible it is under the predicted uncertainty
    ///   (in "sigma units"). Reject implausible pairs, then pick the best one-to-one matches.
    ///
    /// Analogy:
    ///   Like matching keys to locks: only keys that fit the tolerance are considered; then choose
    ///   the best-fitting set without reusing a key or a lock.
    ///
    /// Precise (what the code does):
    ///   - Filter observations by `target_name`.
    ///   - For each armor_id, project sigma points of (x_pred, Sx_pred) through `model.h(...)` to
    ///     obtain z_pred and Pzz.
    ///   - For each observation, compute innovation nu with angle wrapping.
    ///   - Form S = Pzz + R where R is `model.R_diag(obs_z)` (heuristic, NOT from TOML).
    ///   - Compute d2 = nu^T S^{-1} nu and keep candidates with finite d2 and d2 < gate_threshold.
    ///   - Solve one-to-one assignment greedily via a min-cost priority queue (fast, not globally
    ///     optimal like Hungarian; usually fine for 3/4 armors).
    /// @param observations Batch of armor measurements
    /// @param x_pred Predicted state
    /// @param Sx_pred Predicted covariance square root
    /// @param model Motion model for measurement prediction
    /// @param target_name Target armor name to match
    /// @param gate_threshold Mahalanobis distance-squared gate (d^2).
    ///
    ///   Problem:
    ///     Without a gate, clutter and wrong armor_ids can hijack the track.
    ///
    ///   Intuition:
    ///     Larger → fewer missed matches but more wrong matches; smaller → fewer wrong matches but
    ///     more missed matches (leading to TEMP_LOST / reset in the tracker state machine).
    ///
    ///   Analogy:
    ///     A chi-square gate width in normalized units (not pixels, not Euclidean meters).
    ///
    ///   Precise:
    ///     The gate is applied to d2 computed with S = Pzz + R. Therefore, changing process noise
    ///     (which affects Pzz via Sx_pred) or changing the heuristic R model changes the effective
    ///     strictness even if gate_threshold stays the same.
    /// @param armors_num Number of armors on target
    [[nodiscard]] MatchResult match(
        const AtLegacyArmorMeasurementBatch& observations, const VecX& x_pred, const MatXX& Sx_pred,
        Model& model, ArmorName target_name, double gate_threshold, int armors_num) const noexcept {

        MatchResult result;

        if (observations.empty()) {
            return result;
        }

        const int n_obs = static_cast<int>(observations.measurements.size());

        // Cost matrix: fixed-size for armors_num <= 16, dynamic for larger
        // Using std::array to avoid heap allocation
        constexpr int MAX_ARMORS = 16;
        std::array<std::array<double, MAX_ARMORS>, MAX_ARMORS> cost{};
        cost.fill(std::array<double, MAX_ARMORS>{});
        for (auto& row : cost) {
            row.fill(std::numeric_limits<double>::infinity());
        }

        // Filter matching armors upfront to avoid repeated checks
        struct FilteredMeas {
            int obs_idx;
            VecZ z;
        };
        std::vector<FilteredMeas> filtered_obs;
        filtered_obs.reserve(n_obs);

        for (int j = 0; j < n_obs; ++j) {
            if (observations.measurements[j].name != target_name) {
                continue;
            }
            const auto& m  = observations.measurements[j];
            const auto ypd = xyz2ypd(m.transform->translation());

            VecZ z;
            z[0] = ypd[0]; // yaw
            z[1] = ypd[1]; // pitch
            z[2] = ypd[2]; // distance
            z[3] = m.transform->euler_rot().yaw;

            filtered_obs.push_back({j, z});
        }

        if (filtered_obs.empty()) {
            return result;
        }

        // Pre-compute predicted measurements and covariances for each armor_id
        struct PredMeas {
            VecZ z_pred;
            MatZ Pzz;
        };
        std::vector<PredMeas> pred_cache(armors_num);

        const double gamma = std::sqrt(static_cast<double>(NX));
        const double w     = 1.0 / (2.0 * static_cast<double>(NX));

        for (int id = 0; id < armors_num; ++id) {
            const VecZ z0 = model.h(x_pred, id);

            std::array<VecZ, 2 * NX> z_sig{};
            for (int k = 0; k < NX; ++k) {
                const VecX xp = x_pred + gamma * Sx_pred.col(k);
                const VecX xm = x_pred - gamma * Sx_pred.col(k);

                VecZ zp = model.h(xp, id);
                VecZ zm = model.h(xm, id);

                // Unwrap angles for continuity
                zp[0] = unwrap_rad(z0[0], zp[0]);
                zp[1] = unwrap_rad(z0[1], zp[1]);
                zp[3] = unwrap_rad(z0[3], zp[3]);

                zm[0] = unwrap_rad(z0[0], zm[0]);
                zm[1] = unwrap_rad(z0[1], zm[1]);
                zm[3] = unwrap_rad(z0[3], zm[3]);

                z_sig[k]      = zp;
                z_sig[NX + k] = zm;
            }

            VecZ z_bar = VecZ::Zero();
            for (const auto& zs : z_sig) {
                z_bar.noalias() += w * zs;
            }

            MatZ Pzz = MatZ::Zero();
            for (const auto& zs : z_sig) {
                VecZ dz;
                dz[0] = shortest_rad(z_bar[0], zs[0]);
                dz[1] = zs[1] - z_bar[1];
                dz[2] = zs[2] - z_bar[2];
                dz[3] = shortest_rad(z_bar[3], zs[3]);
                Pzz.noalias() += w * (dz * dz.transpose());
            }

            pred_cache[id] = {z_bar, Pzz};
        }

        // Compute costs using filtered observations only
        for (const auto& obs : filtered_obs) {
            const int j = obs.obs_idx;

            for (int id = 0; id < armors_num; ++id) {
                const auto& z_pred = pred_cache[id].z_pred;

                VecZ nu;
                nu[0] = shortest_rad(z_pred[0], obs.z[0]);
                nu[1] = obs.z[1] - z_pred[1];
                nu[2] = obs.z[2] - z_pred[2];
                nu[3] = shortest_rad(z_pred[3], obs.z[3]);

                const Eigen::Matrix<double, NZ, 1> R_diag = model.R_diag(obs.z);
                MatZ S                                    = pred_cache[id].Pzz;
                S.diagonal() += R_diag;
                S = (S + S.transpose()) * 0.5;
                S.diagonal().array() += 1e-9;

                const Eigen::LDLT<MatZ> ldlt(S);
                if (ldlt.info() != Eigen::Success) {
                    continue;
                }

                double d2 = nu.transpose() * ldlt.solve(nu);

                // Gating
                if (std::isfinite(d2) && d2 < gate_threshold) {
                    cost[j][id] = d2;
                }
            }
        }

        // Optimized greedy one-to-one assignment using priority queue
        struct CostEntry {
            double cost;
            int obs_idx;
            int armor_id;
            bool operator<(const CostEntry& other) const {
                return cost > other.cost; // Reverse for min-heap
            }
        };

        std::priority_queue<CostEntry> pq;
        for (const auto& obs : filtered_obs) {
            for (int id = 0; id < armors_num; ++id) {
                if (std::isfinite(cost[obs.obs_idx][id])) {
                    pq.push({cost[obs.obs_idx][id], obs.obs_idx, id});
                }
            }
        }

        std::vector<bool> used_obs(n_obs, false);
        std::vector<bool> used_id(armors_num, false);

        while (!pq.empty()) {
            const auto entry = pq.top();
            pq.pop();

            if (used_obs[entry.obs_idx] || used_id[entry.armor_id]) {
                continue;
            }

            used_obs[entry.obs_idx] = true;
            used_id[entry.armor_id] = true;

            result.armors.push_back(observations.measurements[entry.obs_idx]);
            result.armor_ids.push_back(entry.armor_id);
        }

        return result;
    }
};

} // namespace fcs::L3
