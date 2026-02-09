#include "core/trajectory/solver/trajectory_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fcs::core::trajectory::solver {

// ============================================================================
// DirectSolver Implementation
// ============================================================================

AimSolution DirectSolver::solve(const Eigen::Vector3d& target_pos, double v0) const noexcept {
    const double target_height = target_pos.z();
    const double distance      = std::hypot(target_pos.x(), target_pos.y());

    if (distance < 1e-6) {
        return {
            .yaw            = std::atan2(target_pos.y(), target_pos.x()),
            .pitch          = 0.0,
            .time_of_flight = 0.0,
            .converged      = false,
            .iterations     = 0};
    }

    // For DirectSolver, we use iterative refinement with the model's compute_impact
    // This works for IdealModel and LinearDragModel which have fast compute_impact
    constexpr int kMaxIterations      = 20;
    constexpr double kHeightTolerance = 0.01;
    constexpr double kMaxPitch        = std::numbers::pi / 2.5;

    double iterative_height = target_height;
    double angle            = std::atan2(target_height, distance);
    double impact_height    = 0.0;
    double dh               = 0.0;
    double tof              = 0.0;
    int iterations          = 0;

    for (int i = 0; i < kMaxIterations; ++i) {
        iterations = i + 1;
        angle      = std::atan2(iterative_height, distance);

        if (std::abs(angle) > kMaxPitch) {
            return {
                .yaw            = std::atan2(target_pos.y(), target_pos.x()),
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        const auto impact = model_->compute_impact(distance, angle, v0);
        if (!impact) {
            return {
                .yaw            = std::atan2(target_pos.y(), target_pos.x()),
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        impact_height = impact->z;
        tof           = impact->tof;
        dh            = target_height - impact_height;

        if (std::abs(dh) < kHeightTolerance) {
            return {
                .yaw            = std::atan2(target_pos.y(), target_pos.x()),
                .pitch          = angle,
                .time_of_flight = tof,
                .converged      = true,
                .iterations     = iterations};
        }

        iterative_height += dh;
    }

    return {
        .yaw            = std::atan2(target_pos.y(), target_pos.x()),
        .pitch          = angle,
        .time_of_flight = tof,
        .converged      = std::abs(dh) < kHeightTolerance,
        .iterations     = iterations};
}

std::vector<std::pair<double, double>>
    DirectSolver::generate_trajectory(double pitch, double v0, double max_distance) const noexcept {
    constexpr double kTrajectoryStep = 0.03;

    if (max_distance < 0.0) {
        return {};
    }

    const size_t num_points = static_cast<size_t>(max_distance / kTrajectoryStep) + 1;
    std::vector<std::pair<double, double>> trajectory;
    trajectory.reserve(num_points);

    for (double x = 0; x < max_distance; x += kTrajectoryStep) {
        const auto impact = model_->compute_impact(x, pitch, v0);
        const double z    = impact ? impact->z : 0.0;
        trajectory.emplace_back(x, z);
    }

    return trajectory;
}

// ============================================================================
// IterativeSolver Implementation
// ============================================================================

AimSolution IterativeSolver::solve(const Eigen::Vector3d& target_pos, double v0) const noexcept {
    return detail::iterative_solve_pitch(
        *model_, target_pos, v0, static_cast<int>(config_.max_iterations), config_.height_tolerance,
        config_.max_pitch);
}

std::vector<std::pair<double, double>> IterativeSolver::generate_trajectory(
    double pitch, double v0, double max_distance) const noexcept {
    constexpr double kTrajectoryStep = 0.03;

    if (max_distance < 0.0) {
        return {};
    }

    const size_t num_points = static_cast<size_t>(max_distance / kTrajectoryStep) + 1;
    std::vector<std::pair<double, double>> trajectory;
    trajectory.reserve(num_points);

    for (double x = 0; x < max_distance; x += kTrajectoryStep) {
        const auto impact = model_->compute_impact(x, pitch, v0);
        const double z    = impact ? impact->z : 0.0;
        trajectory.emplace_back(x, z);
    }

    return trajectory;
}

// ============================================================================
// TableLookupSolver Implementation
// ============================================================================

namespace {

/// Helper to cast base_solver to concrete type for model access
const model::BallisticModel* get_model_from_solver(const TrajectorySolver* solver) noexcept {
    // Try DirectSolver
    if (const auto* direct = dynamic_cast<const DirectSolver*>(solver)) {
        return direct->model().get();
    }
    // Try IterativeSolver
    if (const auto* iterative = dynamic_cast<const IterativeSolver*>(solver)) {
        return iterative->model().get();
    }
    // Recursively check nested TableLookupSolver
    if (const auto* table = dynamic_cast<const TableLookupSolver*>(solver)) {
        return get_model_from_solver(table->base_solver().get());
    }
    return nullptr;
}

} // anonymous namespace

TableLookupSolver::TableLookupSolver(
    std::unique_ptr<TrajectorySolver> base_solver, const TableConfig& config, double v0) noexcept
    : base_solver_(std::move(base_solver))
    , config_(config) {
    rebuild_table(v0);
}

void TableLookupSolver::rebuild_table(double v0) noexcept {
    constexpr double kDefaultRangeMax  = 20.0;
    constexpr double kDefaultRangeStep = 0.01;
    constexpr double kDefaultPitchMin  = -0.6;
    constexpr double kDefaultPitchMax  = 0.9;
    constexpr double kDefaultPitchStep = 0.005;

    const double raw_range_max =
        std::isfinite(config_.range_max) ? config_.range_max : kDefaultRangeMax;
    const double raw_range_step =
        std::isfinite(config_.range_step) ? config_.range_step : kDefaultRangeStep;
    const double raw_pitch_min =
        std::isfinite(config_.pitch_min) ? config_.pitch_min : kDefaultPitchMin;
    const double raw_pitch_max =
        std::isfinite(config_.pitch_max) ? config_.pitch_max : kDefaultPitchMax;
    const double raw_pitch_step =
        std::isfinite(config_.pitch_step) ? config_.pitch_step : kDefaultPitchStep;

    const double range_step = std::max(1e-6, std::abs(raw_range_step));
    const double range_max  = std::max(range_step, std::max(0.0, raw_range_max));
    const double pitch_step = std::max(1e-6, std::abs(raw_pitch_step));
    const double pitch_min  = std::min(raw_pitch_min, raw_pitch_max);
    const double pitch_max =
        std::max(pitch_min + pitch_step, std::max(raw_pitch_min, raw_pitch_max));

    auto tbl        = std::make_unique<LookupTable>();
    tbl->range_max  = range_max;
    tbl->range_step = range_step;
    tbl->pitch_min  = pitch_min;
    tbl->pitch_max  = pitch_max;
    tbl->pitch_step = pitch_step;
    tbl->range_count =
        std::max<size_t>(2, static_cast<size_t>(std::floor(tbl->range_max / tbl->range_step)) + 1);
    tbl->pitch_count = std::max<size_t>(
        2,
        static_cast<size_t>(std::floor((tbl->pitch_max - tbl->pitch_min) / tbl->pitch_step)) + 1);

    const size_t total = tbl->pitch_count * tbl->range_count;
    tbl->z.assign(total, std::numeric_limits<double>::quiet_NaN());
    tbl->tof.assign(total, std::numeric_limits<double>::quiet_NaN());

    const auto idx = [&](size_t pi, size_t ri) { return pi * tbl->range_count + ri; };

    // Get the underlying model for table building
    const model::BallisticModel* model = get_model_from_solver(base_solver_.get());
    if (!model) {
        table_ = nullptr;
        return;
    }

    // Build table by iterating over all pitches and sampling at range intervals
    for (size_t pi = 0; pi < tbl->pitch_count; ++pi) {
        const double pitch = tbl->pitch_min + static_cast<double>(pi) * tbl->pitch_step;

        for (size_t ri = 0; ri < tbl->range_count; ++ri) {
            const double range = static_cast<double>(ri) * tbl->range_step;
            const auto impact  = model->compute_impact(range, pitch, v0);
            if (impact) {
                tbl->z[idx(pi, ri)]   = impact->z;
                tbl->tof[idx(pi, ri)] = impact->tof;
            }
        }
    }

    table_ = std::move(tbl);
}

AimSolution TableLookupSolver::solve(const Eigen::Vector3d& target_pos, double v0) const noexcept {
    const double target_height = target_pos.z();
    const double range         = std::hypot(target_pos.x(), target_pos.y());
    const double yaw           = std::atan2(target_pos.y(), target_pos.x());

    if (range < 1e-6) {
        return {
            .yaw = yaw, .pitch = 0.0, .time_of_flight = 0.0, .converged = false, .iterations = 0};
    }

    // Get the underlying model for fallback
    const model::BallisticModel* model = get_model_from_solver(base_solver_.get());
    if (!model) {
        // Fallback to base solver
        return base_solver_->solve(target_pos, v0);
    }

    if (!table_) {
        // Fallback to base solver
        return base_solver_->solve(target_pos, v0);
    }

    if (range > table_->range_max) {
        // Out of table range, use base solver
        return base_solver_->solve(target_pos, v0);
    }

    constexpr double kDhThreshold = 0.01;
    constexpr double kMaxAngle    = std::numbers::pi / 2.5;

    double pitch_est = std::atan2(target_height, range);
    double tof_est   = 0.0;

    // Initial lookup estimate
    if (lookup_pitch_tof(*table_, range, target_height, pitch_est, tof_est)) {
        if (!std::isfinite(tof_est)) {
            tof_est = 0.0;
        }
    }

    // Refine with model (1-2 iterations)
    bool have_impact = false;
    double last_dh   = std::numeric_limits<double>::infinity();

    for (int refine = 0; refine < 2; ++refine) {
        const auto impact = model->compute_impact(range, pitch_est, v0);
        if (!impact) {
            break;
        }
        have_impact     = true;
        const double dh = target_height - impact->z;
        last_dh         = dh;

        if (std::abs(dh) < kDhThreshold) {
            return {
                .yaw            = yaw,
                .pitch          = pitch_est,
                .time_of_flight = impact->tof,
                .converged      = true,
                .iterations     = 1};
        }

        // Estimate dz/dp from table for Newton-like refinement
        double dzdp = 0.0;
        if (estimate_dzdp(*table_, range, pitch_est, dzdp) && std::abs(dzdp) >= 1e-6) {
            pitch_est = std::clamp(pitch_est + dh / dzdp, table_->pitch_min, table_->pitch_max);
        } else {
            break;
        }
    }

    if (!have_impact) {
        return {
            .yaw            = yaw,
            .pitch          = pitch_est,
            .time_of_flight = tof_est,
            .converged      = false,
            .iterations     = 2};
    }

    // If still not converged after table refinement, use iterative fallback
    if (std::abs(last_dh) > kDhThreshold) {
        double iterative_height = target_height;
        double dh               = last_dh;

        for (int i = 0; i < 6; ++i) {
            pitch_est = std::atan2(iterative_height, range);
            if (std::abs(pitch_est) > kMaxAngle) {
                break;
            }

            const auto impact = model->compute_impact(range, pitch_est, v0);
            if (!impact) {
                break;
            }

            dh = target_height - impact->z;
            if (std::abs(dh) < kDhThreshold) {
                return {
                    .yaw            = yaw,
                    .pitch          = pitch_est,
                    .time_of_flight = impact->tof,
                    .converged      = true,
                    .iterations     = 2 + i + 1};
            }

            iterative_height += dh;
        }

        if (std::abs(dh) > kDhThreshold) {
            return {
                .yaw            = yaw,
                .pitch          = pitch_est,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = 8};
        }
    }

    // Get final tof
    const auto final_impact = model->compute_impact(range, pitch_est, v0);
    return {
        .yaw            = yaw,
        .pitch          = pitch_est,
        .time_of_flight = final_impact ? final_impact->tof : tof_est,
        .converged      = true,
        .iterations     = 2};
}

std::vector<std::pair<double, double>> TableLookupSolver::generate_trajectory(
    double pitch, double v0, double max_distance) const noexcept {
    if (max_distance <= 0.0) {
        return {};
    }

    constexpr double sample_step = 0.015;
    const size_t num_points      = static_cast<size_t>(max_distance / sample_step) + 1;
    std::vector<std::pair<double, double>> trajectory;
    trajectory.reserve(num_points);

    if (table_) {
        for (double x = 0; x < max_distance; x += sample_step) {
            const double z = sample_table_z(*table_, x, pitch);
            trajectory.emplace_back(x, std::isfinite(z) ? z : 0.0);
        }
        return trajectory;
    }

    // Fallback to base solver
    return base_solver_->generate_trajectory(pitch, v0, max_distance);
}

double
    TableLookupSolver::sample_table_z(const LookupTable& table, double x, double pitch) noexcept {
    if (table.range_step <= 0.0 || table.pitch_step <= 0.0 || table.range_count < 2
        || table.pitch_count < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= table.range_max) {
        x = table.range_max;
    }
    pitch = std::clamp(pitch, table.pitch_min, table.pitch_max);

    const double r_pos = x / table.range_step;
    const size_t r0    = std::min(static_cast<size_t>(r_pos), table.range_count - 2);
    const double ar    = r_pos - static_cast<double>(r0);

    const double p_pos = (pitch - table.pitch_min) / table.pitch_step;
    const size_t p0    = std::min(static_cast<size_t>(p_pos), table.pitch_count - 2);
    const double ap    = p_pos - static_cast<double>(p0);

    const auto idx = [&](size_t pi, size_t ri) { return pi * table.range_count + ri; };

    const double z00 = table.z[idx(p0, r0)];
    const double z01 = table.z[idx(p0, r0 + 1)];
    const double z10 = table.z[idx(p0 + 1, r0)];
    const double z11 = table.z[idx(p0 + 1, r0 + 1)];

    if (!std::isfinite(z00) || !std::isfinite(z01) || !std::isfinite(z10) || !std::isfinite(z11)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double z0 = z00 + ar * (z01 - z00);
    const double z1 = z10 + ar * (z11 - z10);
    return z0 + ap * (z1 - z0);
}

bool TableLookupSolver::lookup_pitch_tof(
    const LookupTable& table, double range, double target_z, double& pitch_out,
    double& tof_out) noexcept {
    if (table.range_step <= 0.0 || table.pitch_step <= 0.0 || table.range_count < 2
        || table.pitch_count < 2) {
        return false;
    }

    if (range <= 0.0 || range > table.range_max) {
        return false;
    }

    const double r_pos = range / table.range_step;
    const size_t r0    = std::min(static_cast<size_t>(r_pos), table.range_count - 2);
    const double ar    = r_pos - static_cast<double>(r0);

    const auto idx = [&](size_t pi, size_t ri) { return pi * table.range_count + ri; };

    auto z_at = [&](size_t pi) -> double {
        const double z0 = table.z[idx(pi, r0)];
        const double z1 = table.z[idx(pi, r0 + 1)];
        if (!std::isfinite(z0) || !std::isfinite(z1)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return z0 + ar * (z1 - z0);
    };

    auto t_at = [&](size_t pi) -> double {
        const double t0 = table.tof[idx(pi, r0)];
        const double t1 = table.tof[idx(pi, r0 + 1)];
        if (!std::isfinite(t0) || !std::isfinite(t1)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return t0 + ar * (t1 - t0);
    };

    double z_prev = z_at(0);
    if (!std::isfinite(z_prev)) {
        return false;
    }
    if (target_z < z_prev) {
        return false;
    }

    for (size_t pi = 1; pi < table.pitch_count; ++pi) {
        const double z_cur = z_at(pi);
        if (!std::isfinite(z_cur)) {
            continue;
        }
        if (target_z <= z_cur) {
            const double p0    = table.pitch_min + static_cast<double>(pi - 1) * table.pitch_step;
            const double p1    = table.pitch_min + static_cast<double>(pi) * table.pitch_step;
            const double denom = (z_cur - z_prev);
            const double ap =
                denom > 1e-9 ? std::clamp((target_z - z_prev) / denom, 0.0, 1.0) : 0.0;

            pitch_out = p0 + ap * (p1 - p0);

            const double t0 = t_at(pi - 1);
            const double t1 = t_at(pi);
            if (std::isfinite(t0) && std::isfinite(t1)) {
                tof_out = t0 + ap * (t1 - t0);
            } else {
                tof_out = std::numeric_limits<double>::quiet_NaN();
            }
            return true;
        }
        z_prev = z_cur;
    }

    return false;
}

bool TableLookupSolver::estimate_dzdp(
    const LookupTable& table, double range, double pitch, double& dzdp_out) noexcept {
    if (table.range_step <= 0.0 || table.pitch_step <= 0.0 || table.range_count < 2
        || table.pitch_count < 2) {
        return false;
    }

    if (table.pitch_count == 2) {
        const double p_lo  = table.pitch_min;
        const double p_hi  = table.pitch_min + table.pitch_step;
        const double z_lo  = sample_table_z(table, range, p_lo);
        const double z_hi  = sample_table_z(table, range, p_hi);
        const double denom = (p_hi - p_lo);
        if (!std::isfinite(z_lo) || !std::isfinite(z_hi) || std::abs(denom) < 1e-12) {
            return false;
        }
        dzdp_out = (z_hi - z_lo) / denom;
        return true;
    }

    pitch              = std::clamp(pitch, table.pitch_min, table.pitch_max);
    const double p_pos = (pitch - table.pitch_min) / table.pitch_step;
    size_t pi          = static_cast<size_t>(std::round(p_pos));
    if (pi < 1) {
        pi = 1;
    }
    if (pi + 1 >= table.pitch_count) {
        pi = table.pitch_count - 2;
    }

    const double p_lo  = table.pitch_min + static_cast<double>(pi - 1) * table.pitch_step;
    const double p_hi  = table.pitch_min + static_cast<double>(pi + 1) * table.pitch_step;
    const double z_lo  = sample_table_z(table, range, p_lo);
    const double z_hi  = sample_table_z(table, range, p_hi);
    const double denom = (p_hi - p_lo);
    if (!std::isfinite(z_lo) || !std::isfinite(z_hi) || std::abs(denom) < 1e-12) {
        return false;
    }
    dzdp_out = (z_hi - z_lo) / denom;
    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

namespace detail {

AimSolution iterative_solve_pitch(
    const model::BallisticModel& model, const Eigen::Vector3d& target_pos, double v0,
    int max_iterations, double height_tolerance, double max_pitch) noexcept {

    const double target_height = target_pos.z();
    const double distance      = std::hypot(target_pos.x(), target_pos.y());
    const double yaw           = std::atan2(target_pos.y(), target_pos.x());

    if (distance < 1e-6) {
        return {
            .yaw = yaw, .pitch = 0.0, .time_of_flight = 0.0, .converged = false, .iterations = 0};
    }

    double iterative_height = target_height;
    double angle            = std::atan2(target_height, distance);
    double impact_height    = 0.0;
    double dh               = 0.0;
    double tof              = 0.0;
    int iterations          = 0;

    for (int i = 0; i < max_iterations; ++i) {
        iterations = i + 1;
        angle      = std::atan2(iterative_height, distance);

        if (std::abs(angle) > max_pitch) {
            return {
                .yaw            = yaw,
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        const auto impact = model.compute_impact(distance, angle, v0);
        if (!impact) {
            return {
                .yaw            = yaw,
                .pitch          = angle,
                .time_of_flight = 0.0,
                .converged      = false,
                .iterations     = iterations};
        }

        impact_height = impact->z;
        tof           = impact->tof;
        dh            = target_height - impact_height;

        if (std::abs(dh) < height_tolerance) {
            return {
                .yaw            = yaw,
                .pitch          = angle,
                .time_of_flight = tof,
                .converged      = true,
                .iterations     = iterations};
        }

        iterative_height += dh;
    }

    return {
        .yaw            = yaw,
        .pitch          = angle,
        .time_of_flight = tof,
        .converged      = std::abs(dh) < height_tolerance,
        .iterations     = iterations};
}

} // namespace detail

} // namespace fcs::core::trajectory::solver
