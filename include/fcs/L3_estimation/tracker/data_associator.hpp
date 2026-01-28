#pragma once

#include "motion_model.hpp"
#include "uni_kf_spaces.hpp"
#include "util.hpp"

#include "fcs/core/types.hpp"

#include <Eigen/Core>
#include <limits>
#include <queue>
#include <vector>

namespace fcs::L3 {

/// Template-based data associator for armor matching (pure uni_kf CKF gating).
template <typename Model>
class DataAssociator {
public:
    static constexpr int NZ = Model::NZ;
    static constexpr int NX = Model::NX;

    using StatePoint = typename Model::StatePoint;
    using MeasPoint  = typename Model::MeasPoint;
    using MatXX      = Eigen::Matrix<double, NX, NX>;
    using MatZ       = Eigen::Matrix<double, NZ, NZ>;

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
        const AtLegacyArmorMeasurementBatch& observations, const StatePoint& x_pred,
        const uni_kf::SymmetricMatrix<NX, double>& P_pred, Model& model, ArmorName target_name,
        double gate_threshold, int armors_num) const noexcept {

        MatchResult result;

        if (observations.empty()) {
            return result;
        }

        const int n_obs_total = static_cast<int>(observations.measurements.size());

        // Filter matching armors upfront to avoid repeated checks
        struct FilteredMeas {
            int obs_idx; // index into observations.measurements
            MeasPoint z;
        };
        std::vector<FilteredMeas> filtered_obs;
        filtered_obs.reserve(n_obs_total);

        for (int j = 0; j < n_obs_total; ++j) {
            if (observations.measurements[j].name != target_name) {
                continue;
            }
            const auto& m = observations.measurements[j];
            filtered_obs.push_back({j, ukf::ArmorMeas::make(m.transform->translation(), m.yaw())});
        }

        if (filtered_obs.empty()) {
            return result;
        }

        const int n_obs = static_cast<int>(filtered_obs.size());

        std::vector<std::vector<double>> cost(
            n_obs, std::vector<double>(armors_num, std::numeric_limits<double>::infinity()));

        // Pre-compute predicted measurements and covariances for each armor_id
        struct PredMeas {
            MatZ Pzz;
            MeasPoint z_pred;
        };
        std::vector<PredMeas> pred_cache(armors_num);

        for (int id = 0; id < armors_num; ++id) {
            const typename fcs::L3::UniTrackerSRCKF<Model>::ArmorMeasurementModel h{
                &model, id, MeasPoint{}};
            const auto R0            = uni_kf::SymmetricMatrix<NZ, double>::Zero();
            const auto z_ref         = model.measure(x_pred, id);
            auto [H_dummy, Pxy, Pzz] = uni_kf::CubaturePoints::measurement_covariances_about<
                typename Model::StateSpace, typename Model::MeasSpace>(
                x_pred, P_pred, h, R0, z_ref);
            (void)H_dummy;
            (void)Pxy;
            pred_cache[id] = {Pzz, z_ref};
        }

        for (int j = 0; j < n_obs; ++j) {
            const auto& obs = filtered_obs[j];
            for (int id = 0; id < armors_num; ++id) {
                const auto innovation = Model::MeasSpace::diff(pred_cache[id].z_pred, obs.z);
                const auto R_obs      = model.measurement_noise(obs.z);

                MatZ S = pred_cache[id].Pzz;
                S += R_obs.matrix();
                S = (S + S.transpose()) * 0.5;
                S.diagonal().array() += 1e-9;

                const Eigen::LDLT<MatZ> ldlt(S);
                if (ldlt.info() != Eigen::Success) {
                    continue;
                }

                const double d2 = innovation.data.transpose() * ldlt.solve(innovation.data);

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
        for (int j = 0; j < n_obs; ++j) {
            for (int id = 0; id < armors_num; ++id) {
                if (std::isfinite(cost[j][id])) {
                    pq.push({cost[j][id], j, id});
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

            result.armors.push_back(observations.measurements[filtered_obs[entry.obs_idx].obs_idx]);
            result.armor_ids.push_back(entry.armor_id);
        }

        return result;
    }
};

} // namespace fcs::L3
