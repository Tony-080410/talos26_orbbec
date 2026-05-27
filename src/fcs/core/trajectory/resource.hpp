#pragma once

#include "config.hpp"
#include "scheduler/scheduler.hpp"
#include "solver/config.hpp"
#include "solver/solver_interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fcs::core::trajectory {

/// Bullet speed filter configuration
/// Models the EMA + outlier rejection parameters as a product-type struct.
struct bullet_speed_filter_config {
    double tau_slow_s{0.4};           ///< Slow EMA time constant for stable periods
    double tau_fast_s{0.08};          ///< Fast EMA time constant for confirmed step changes
    double step_threshold_mps{0.8};   ///< |innovation| threshold to trigger step detection
    int step_confirm_count{2};        ///< Consecutive consistent deviations to confirm a step
    double reject_threshold_mps{4.0}; ///< Single-frame spike rejection (stricter than step)
    uint64_t stale_ns{200'000'000};   ///< 200ms: filter enters stale if no update for this long
};

/// Bullet speed resource with built-in EMA filtering and outlier rejection.
///
/// The raw MCU value is clamped to [min_mps, max_mps], then filtered:
///   - Extreme single-frame spikes (|innovation| > reject_threshold_mps) are dropped.
///   - Consecutive consistent deviations trigger a fast-EMA step response.
///   - During stable periods a slow EMA smooths measurement jitter.
///   - If no update arrives within stale_ns, the filter holds the last filtered value.
///
/// Consumers should read `.filtered`. `.raw` is preserved for diagnostics.
struct bullet_speed_data {
    double filtered{0.0};       ///< EMA-filtered value (consumed by L4/L5)
    double raw{0.0};            ///< Last raw MCU sample (for diagnostics)
    uint64_t last_update_ns{0}; ///< Timestamp of the last update call
    int confirm_count{0};       ///< Consecutive frames with same-direction large innovation
    bool initialized{false};    ///< Whether the filter has been seeded

    /// Apply one filter step. Returns true if `.filtered` changed.
    ///
    /// @param sample        Raw MCU bullet speed (m/s)
    /// @param timestamp_ns  Monotonic timestamp of the sample
    /// @param min_mps       Lower clamp bound (m/s)
    /// @param max_mps       Upper clamp bound (m/s)
    /// @param cfg           Filter parameters
    [[nodiscard]] auto update(
        double sample, uint64_t timestamp_ns, double min_mps, double max_mps,
        const bullet_speed_filter_config& cfg = {}) noexcept -> bool {
        raw              = std::clamp(sample, min_mps, max_mps);
        const auto delta = timestamp_ns - last_update_ns;

        // Seed on first valid sample
        if (!initialized) [[unlikely]] {
            filtered       = raw;
            last_update_ns = timestamp_ns;
            confirm_count  = 0;
            initialized    = true;
            return true;
        }

        // Stale guard: if too much time has passed since last update, hold the last filtered
        // value but still record the raw sample for diagnostics.
        if (delta > cfg.stale_ns) {
            last_update_ns = timestamp_ns;
            confirm_count  = 0;
            // Re-seed filtered to the current sample to recover from extended dropouts
            filtered = raw;
            return true;
        }

        const double dt_s       = static_cast<double>(delta) * 1e-9;
        const double innovation = raw - filtered;
        const double abs_innov  = std::abs(innovation);

        // --- Outlier rejection ---
        // Single-frame extreme spike — treat as noise and drop.
        if (abs_innov > cfg.reject_threshold_mps) {
            confirm_count  = 0;
            last_update_ns = timestamp_ns;
            return false;
        }

        // --- Step detection ---
        // A "step" is a large innovation that persists in the same direction
        // for `step_confirm_count` consecutive frames.
        const bool is_large_deviation = abs_innov > cfg.step_threshold_mps;

        if (is_large_deviation) {
            ++confirm_count;
        } else {
            confirm_count = std::max(0, confirm_count - 1);
        }

        const bool step  = confirm_count >= cfg.step_confirm_count;
        const double tau = step ? cfg.tau_fast_s : cfg.tau_slow_s;

        // --- EMA update ---
        const double alpha = 1.0 - std::exp(-dt_s / tau);
        const double prev  = filtered;
        filtered           = filtered + alpha * innovation;

        last_update_ns = timestamp_ns;

        // Return true if the filtered value meaningfully changed
        return std::abs(filtered - prev) > 1e-9;
    }
};

using trajectory_solver = talos::res<std::unique_ptr<solver::TrajectorySolver>>;
using trajectory_config = talos::res<model::ModelConfig>;

using bullet_speed     = talos::res<bullet_speed_data>;
using bullet_speed_mut = talos::res_mut<bullet_speed_data>;

constexpr void register_resource(talos::Scheduler& scheduler, TrajectoryConfig&& config) noexcept {
    scheduler.world().insert_resource(config.model.get());
    auto solver = solver::create_solver(config);
    scheduler.world().insert_resource(std::move(solver));
}

} // namespace fcs::core::trajectory
