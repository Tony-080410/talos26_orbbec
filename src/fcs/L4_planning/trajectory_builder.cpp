#include "L4_planning/trajectory_builder.hpp"
#include "core/math/normalize.hpp"

#include <algorithm>
#include <vector>

namespace fcs::L4 {
namespace {

using fcs::core::math::normalize_angle;

using core::trajectory::ReferenceTrajectory;
using StateMatrix = ReferenceTrajectory::StateMatrix;

struct ReferenceSample {
    double yaw{0.0};
    double pitch{0.0};
    double distance{0.0};
    double diff_yaw{0.0};
    double time_of_flight{0.0};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    int selected_armor_id{0};
    int rough_selected_armor_id{0};
};

[[nodiscard]] double
    finite_difference(const std::vector<double>& samples, int index, double dt) noexcept {
    const int size = static_cast<int>(samples.size());
    if (size <= 1) {
        return 0.0;
    }
    if (index <= 0) {
        return (samples[1] - samples[0]) / dt;
    }
    if (index >= size - 1) {
        return (samples[size - 1] - samples[size - 2]) / dt;
    }
    return (samples[index + 1] - samples[index - 1]) / (2.0 * dt);
}

[[nodiscard]] double
    finite_difference_angle(const std::vector<double>& samples, int index, double dt) noexcept {
    const int size = static_cast<int>(samples.size());
    if (size <= 1) {
        return 0.0;
    }
    if (index <= 0) {
        return normalize_angle(samples[1] - samples[0]) / dt;
    }
    if (index >= size - 1) {
        return normalize_angle(samples[size - 1] - samples[size - 2]) / dt;
    }
    return normalize_angle(samples[index + 1] - samples[index - 1]) / (2.0 * dt);
}

template <typename TargetState>
[[nodiscard]] std::expected<ReferenceSample, std::string> build_reference_sample(
    const Aimer& aimer, const TargetState& state, ArmorAimContext context,
    uint64_t state_timestamp_ns, uint64_t current_ns, double sample_offset_s,
    const Aimer::GimbalTransform& gimbal, const Aimer::MuzzleTransform& muzzle,
    const core::trajectory::solver::TrajectorySolver& solver, double bullet_speed) noexcept {
    const auto prediction = aimer.aim(
        state, context, gimbal, muzzle, state_timestamp_ns, current_ns, sample_offset_s,
        bullet_speed, solver);
    if (!prediction) {
        return std::unexpected(prediction.error());
    }

    return ReferenceSample{
        .yaw                     = prediction->aim_yaw,
        .pitch                   = prediction->aim_pitch,
        .distance                = prediction->distance,
        .diff_yaw                = prediction->diff_yaw,
        .time_of_flight          = prediction->flying_time,
        .position                = prediction->predicted_position,
        .selected_armor_id       = prediction->selected_armor_id,
        .rough_selected_armor_id = prediction->rough_selected_armor_id,
    };
}

} // namespace

[[nodiscard]] std::expected<ReferenceTrajectory, std::string> build_reference_trajectory(
    const PlannerSeed& seed, const Aimer& aimer, int horizon_ahead, int horizon_back, double dt,
    double pitch_min, double pitch_max, uint64_t current_ns, const Aimer::GimbalTransform& gimbal,
    const Aimer::MuzzleTransform& muzzle,
    const core::trajectory::solver::TrajectorySolver& trajectory_solver,
    double bullet_speed) noexcept {

    const int horizon = horizon_ahead + horizon_back + 1;

    ReferenceTrajectory trajectory{
        .state                    = StateMatrix(4, horizon),
        .distances                = std::vector<double>(horizon, 0.0),
        .diff_yaws                = std::vector<double>(horizon, 0.0),
        .time_of_flights          = std::vector<double>(horizon, 0.0),
        .positions                = std::vector<Eigen::Vector3d>(horizon, Eigen::Vector3d::Zero()),
        .selected_armor_ids       = std::vector<int>(horizon, 0),
        .rough_selected_armor_ids = std::vector<int>(horizon, 0),
    };

    std::vector<double> yaw_samples(horizon, 0.0);
    std::vector<double> pitch_samples(horizon, 0.0);

    ArmorAimContext context;
    context.target_jumped      = seed.target_jumped;
    context.phase              = seed.aim_phase;
    context.preferred_armor_id = seed.selected_armor_id;

    for (int k = 0; k < horizon; ++k) {
        const double sample_offset = (static_cast<double>(k) - horizon_back) * dt;

        std::expected<ReferenceSample, std::string> sample = std::unexpected("empty planner seed");
        if (const auto* robot = seed.robot_state()) {
            sample = build_reference_sample(
                aimer, *robot, context, seed.state_timestamp_ns, current_ns, sample_offset, gimbal,
                muzzle, trajectory_solver, bullet_speed);
        } else if (const auto* outpost = seed.outpost_state()) {
            sample = build_reference_sample(
                aimer, *outpost, context, seed.state_timestamp_ns, current_ns, sample_offset,
                gimbal, muzzle, trajectory_solver, bullet_speed);
        }

        if (!sample) {
            return std::unexpected(sample.error());
        }

        context.preferred_armor_id = sample->selected_armor_id;

        yaw_samples[k]                         = sample->yaw;
        pitch_samples[k]                       = std::clamp(sample->pitch, pitch_min, pitch_max);
        trajectory.distances[k]                = sample->distance;
        trajectory.diff_yaws[k]                = sample->diff_yaw;
        trajectory.time_of_flights[k]          = sample->time_of_flight;
        trajectory.positions[k]                = sample->position;
        trajectory.selected_armor_ids[k]       = sample->selected_armor_id;
        trajectory.rough_selected_armor_ids[k] = sample->rough_selected_armor_id;
    }

    trajectory.yaw_origin = yaw_samples[horizon_back];

    for (int k = 0; k < horizon; ++k) {
        trajectory.state(0, k) = normalize_angle(yaw_samples[k] - trajectory.yaw_origin);
        trajectory.state(1, k) = finite_difference_angle(yaw_samples, k, dt);
        trajectory.state(2, k) = pitch_samples[k];
        trajectory.state(3, k) = finite_difference(pitch_samples, k, dt);
    }

    return trajectory;
}

} // namespace fcs::L4
