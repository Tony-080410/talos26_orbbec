#pragma once

#include "L3_estimation/energy_meter_solver/motion_model.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace energy_meter {

struct BigFitResult {
    double a{0.0};
    double omega{0.0};
    double phase{0.0};
    double c{0.0};
    double t{0.0};
    double b{0.0};
    bool valid{false};
};

template <typename ModelT>
struct Tracker {
    using Model = ModelT;

    struct Params {
        int lost_thres{10};
        int tracking_thres{3};
        double matcher_gate{25.0};
        double angle_match_gate_rad{0.45};
        int blade_unlock_frames{10};
        typename Model::Params model_params{};

        int direction_lock_frames{10};
        int min_fit_samples{24};
        int max_fit_samples{160};
        int fit_iterations{3};
        double fit_window_sec{3.0};
        double fit_inlier_threshold_rad{0.16};
        double fit_penalty_a{2.0};
        double fit_penalty_w{2.0};
        double fit_temporal_weight{1.0};
        double fit_filter_alpha{0.35};
    } params;

    enum State : uint8_t {
        IDLE,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } state{IDLE};

    Tracker() = default;
    explicit Tracker(const Params& p) { set_params(p); }

    void set_params(const Params& p) { params = p; }

    [[nodiscard]] bool has_filter() const noexcept { return initialized_; }
    [[nodiscard]] bool has_ekf() const noexcept { return has_filter(); }

    void reset_filter() noexcept;
    void reset_ekf() noexcept { reset_filter(); }

    [[nodiscard]] auto getRadius() const noexcept -> double { return radius_; }
    void setRadius(double radius) noexcept { radius_ = radius; }

    [[nodiscard]] auto getCurrentBladeId() const noexcept -> int { return blade_id_; }
    [[nodiscard]] auto getDirection() const noexcept -> int { return dir_; }
    [[nodiscard]] auto current_roll() const noexcept -> double {
        return reference_roll_ + continuous_angle_;
    }
    [[nodiscard]] auto current_yaw() const noexcept -> double { return yaw_; }
    [[nodiscard]] auto current_pitch() const noexcept -> double { return pitch_; }
    [[nodiscard]] auto current_center() const noexcept -> const Eigen::Vector3d& { return center_; }
    [[nodiscard]] auto current_time() const noexcept -> double { return current_time_sec_; }
    [[nodiscard]] auto model_valid() const noexcept -> bool { return model_valid_; }
    [[nodiscard]] auto big_fit() const noexcept -> const BigFitResult& { return fit_; }

    bool first_meet_u(
        const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
        const Eigen::Quaterniond& target_quat);

    void predict(double dt) noexcept;

    bool update(
        const Eigen::Vector3d& rcenter_position,
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

    void step(
        double dt, const Eigen::Vector3d& rcenter_position,
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats);

private:
public:
    struct AngleSample {
        double t{0.0};
        double angle{0.0};
    };

private:
    void reset_runtime_state() noexcept;
    void state_machine(bool found) noexcept;
    void initialize_reference(
        const Eigen::Vector3d& r_center, const Eigen::Vector3d& target_pos,
        const Eigen::Quaterniond& target_quat) noexcept;
    void update_direction(double new_angle) noexcept;
    void update_fit();

    [[nodiscard]] auto select_target(
        const std::vector<Eigen::Vector3d>& target_positions,
        const std::vector<Eigen::Quaterniond>& target_quats) const -> std::optional<std::size_t>;

    [[nodiscard]] auto instantaneous_speed(double t) const noexcept -> double;

    double radius_{0.7};
    Eigen::Vector3d center_{Eigen::Vector3d::Zero()};
    double yaw_{0.0};
    double pitch_{0.0};
    double dt_{0.0};

    bool initialized_{false};
    uint64_t reference_timestamp_ns_{0};
    Eigen::Matrix3d reference_rotation_{Eigen::Matrix3d::Identity()};

    double current_time_sec_{0.0};
    double raw_angle_{0.0};
    double reference_roll_{0.0};
    double continuous_angle_{0.0};
    int cumulative_switches_{0};
    int blade_id_{0};

    int dir_{0};
    int dir_votes_pos_{0};
    int dir_votes_neg_{0};
    int dir_detect_count_{0};

    int detecting_count_{0};
    int lost_count_{0};

    std::vector<AngleSample> samples_{};
    BigFitResult fit_{};
    bool model_valid_{false};
};

using SmallTracker = Tracker<SmallRuneModel>;
using BigTracker   = Tracker<BigRuneModel>;
using AnyTracker   = std::variant<std::monostate, SmallTracker, BigTracker>;

} // namespace energy_meter
