#pragma once

#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <string>

namespace energy_meter {

enum EnergyState : uint8_t {
    XC,
    YC,
    ZC,
    ROLL,
    V_ROLL,
    STATE_MAX,
};

enum Measure : uint8_t {
    ARMOR_X,
    ARMOR_Y,
    ARMOR_Z,
    ARMOR_YAW,
    MEASURE_MAX,
};

enum class Direction { UNKNOWN, STABLE, ANTI_CLOCKWISE, CLOCKWISE };
enum class Convexity { UNKNOWN, CONCAVE, CONVEX };

inline double normalize_rad(double a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a <= 0.0) {
        a += 2.0 * M_PI;
    }
    return a - M_PI;
}

inline double shortest_rad(double from, double to) { return normalize_rad(to - from); }

inline double unwrap_rad(double prev, double raw) {
    const double d = shortest_rad(prev, raw);
    return prev + d;
}

inline auto xyz2ypd(const Eigen::Vector3d& xyz) -> Eigen::Vector3d {
    const double x = xyz.x();
    const double y = xyz.y();
    const double z = xyz.z();

    const double distance = xyz.norm();
    const double yaw      = std::atan2(y, x);
    const double pitch    = std::atan2(z, std::sqrt(x * x + y * y));

    return Eigen::Vector3d(yaw, pitch, distance);
}

inline auto orientation2roll(const Eigen::Quaterniond& q) -> double {
    const Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    return std::atan2(R(2, 1), R(2, 2));
}

struct ObservedFrame {
    double timestamp{};
    Eigen::Vector3d R_center_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d target_center_world{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond target_orientation{Eigen::Quaterniond::Identity()};
    Eigen::Matrix3d target_rotation_matrix{Eigen::Matrix3d::Identity()};
    bool is_valid{true};
};

/// Angle model fitted from observations.
///
/// Big rune:  θ(t) = -a·cos(ω·(t+t0)) + b·t + c
///   params = [a, ω, t0, b, c]
///
/// Small rune: constant rotation_speed (rad/s)
struct AngleModel {
    bool is_big_rune{false};

    // ── big rune 5-parameter model ──
    double a{0.470};     // amplitude
    double omega{1.942}; // angular frequency
    double t0{0.0};      // phase offset (internal)
    double b{1.178};     // linear drift
    double c{0.0};       // constant term

    // ── small rune ──
    double rotation_speed{1.04719}; // rad/s (~60°/s)

    // ── fit quality ──
    double fit_error{0.0};
    bool is_valid{false};
    int data_points_used{0};
    double last_fit_time{0.0};

    // ── tracking state ──
    double last_continuous_angle{0.0};
    double last_observation_time{0.0};
};

struct PredictionResult {
    double predicted_angle{0.0};
    Eigen::Vector3d predicted_position{Eigen::Vector3d::Zero()};
    bool is_valid{false};

    double prediction_time_offset{0.0};
};

// Legacy config struct for backward compatibility (deprecated)
// Use fcs::energy_meter::EnergyMeterL3Config for new code
struct [[deprecated("Use fcs::energy_meter::EnergyMeterL3Config instead")]] EnergyMeterConfig {
    std::string rune_type;
    double predict_time{};
    std::string gimbal_frame;
    std::string world_frame;

    double smooth_alpha{};
    int min_data_points{};

    double control_delay{};
};

} // namespace energy_meter
