#pragma once
#include "aimer.hpp"
#include "detector.hpp"
#include "output_interface.hpp"
#include "primitives/buffer_pool.hpp"
#include "primitives/latency_stats.hpp"
#include "scheduler/strategy.hpp"
#include "shooter.hpp"
#include "target.hpp"
#include "tracker.hpp"
#include "types.hpp"

class TalosStrategy {
public:
    // SchedulerStrategy concept methods
    void detect(const scheduler::Captured &feat, talos::ArmorSlots &armor) noexcept;
    void solve(const talos::ArmorSlots &armor, auto_aim::Poses3D &poses3d) const noexcept;
    void track(talos::TrackerSoA &tracker_soa, const talos::ArmorSlots &armor,
               const auto_aim::Poses3D &poses3d) noexcept;
    void output() noexcept;

    // Additional methods (not part of concept)
    void preprocess(scheduler::Captured &sf);

    TalosStrategy(const bool debug, const SpVisionConfig &config, const size_t buffer_count, const int rows,
                  const int cols) noexcept :
        detector(config, debug), solver(config.solver), tracker(config.tracker),
        aimer_(to_aimer_config(config.aimer)), shooter_(to_shooter_config(config.shooter)),
        frame_pool(buffer_count, rows, cols, CV_8UC3), gray_pool(buffer_count, rows, cols, CV_8UC1),
        small_pool(buffer_count, rows, cols, CV_8UC1),
        image_width_(static_cast<uint32_t>(cols)), image_height_(static_cast<uint32_t>(rows)) {
        // 存储相机标定参数
        for (int i = 0; i < 9; ++i) {
            camera_matrix_[i] = config.solver.camera_matrix.data()[i];
        }
        for (int i = 0; i < 5; ++i) {
            distortion_.push_back(config.solver.distort_coefficient.data()[i]);
        }
    }

    /**
     * @brief 设置输出接口 (IPC 或 USB)
     */
    void set_output_interface(auto_aim::OutputInterface output) noexcept {
        output_interface_.emplace(std::move(output));
    }

    /**
     * @brief 设置当前弹速 (从 IMU 获取)
     */
    void set_bullet_speed(const double speed) noexcept { bullet_speed_ = speed; }

    /**
     * @brief 设置当前云台角度 (从 IMU 获取)
     */
    void set_gimbal_yaw(const double yaw) noexcept { gimbal_yaw_ = yaw; }

    /**
     * @brief 设置完整的云台姿态 (从 IMU 获取，用于 TF 发布)
     */
    void set_gimbal_pose(double roll, double pitch, double yaw) noexcept {
        gimbal_roll_ = roll;
        gimbal_pitch_ = pitch;
        gimbal_yaw_ = yaw;
    }

    /**
     * @brief 获取云台姿态 (用于 TF 发布)
     */
    [[nodiscard]] double get_gimbal_roll() const noexcept { return gimbal_roll_; }
    [[nodiscard]] double get_gimbal_pitch() const noexcept { return gimbal_pitch_; }
    [[nodiscard]] double get_gimbal_yaw() const noexcept { return gimbal_yaw_; }

    /**
     * @brief 获取检测器工作缓冲区 (用于调试/可视化)
     */
    [[nodiscard]] const auto_aim::DetectorWorkBuffers& get_detector_buffers() const noexcept {
        return detector_buffers_;
    }

    /**
     * @brief 获取跟踪目标 (用于调试/可视化)
     * 注意: 返回非 const 引用，因为 converged() 可能修改内部状态
     */
    [[nodiscard]] auto_aim::Target& get_target() noexcept { return target_; }

    /**
     * @brief 获取最新的云台控制命令 (用于可视化)
     */
    [[nodiscard]] const auto_aim::GimbalCommand& get_last_gimbal_cmd() const noexcept {
        return last_gimbal_cmd_;
    }

    /**
     * @brief 获取相机标定参数 (用于可视化)
     */
    [[nodiscard]] uint32_t get_image_width() const noexcept { return image_width_; }
    [[nodiscard]] uint32_t get_image_height() const noexcept { return image_height_; }
    [[nodiscard]] const std::array<double, 9>& get_camera_matrix() const noexcept { return camera_matrix_; }
    [[nodiscard]] const std::vector<double>& get_distortion() const noexcept { return distortion_; }

    // Initialize pipeline components (stateless)
    auto_aim::TraditionalDetector detector;
    auto_aim::Solver solver;
    auto_aim::Tracker tracker;

    // Buffer pool (复用 buffer)
    scheduler::BufferPool frame_pool;
    scheduler::BufferPool gray_pool;
    scheduler::BufferPool small_pool;

    // 统计
    scheduler::LatencyStats capture_latency;
    scheduler::LatencyStats preprocess_latency;
    scheduler::LatencyStats detect_latency;
    scheduler::LatencyStats track_latency;
    scheduler::LatencyStats e2e_latency;

    std::atomic<uint64_t> frame_count{0};
    std::atomic<uint64_t> processed_count{0};
    std::atomic<uint64_t> output_count{0};

private:
    // 配置转换辅助函数
    static auto_aim::AimerConfig to_aimer_config(const SpVisionAimerConfig &cfg) noexcept {
        return {
            .yaw_offset = cfg.yaw_offset,
            .pitch_offset = cfg.pitch_offset,
            .comming_angle = cfg.comming_angle,
            .leaving_angle = cfg.leaving_angle,
            .high_speed_delay_time = cfg.high_speed_delay_time,
            .low_speed_delay_time = cfg.low_speed_delay_time,
            .decision_speed = cfg.decision_speed,
            .left_yaw_offset = cfg.left_yaw_offset,
            .right_yaw_offset = cfg.right_yaw_offset,
        };
    }

    static auto_aim::ShooterConfig to_shooter_config(const SpVisionShooterConfig &cfg) noexcept {
        return {
            .judge_distance = cfg.judge_distance,
            .first_tolerance = cfg.first_tolerance,
            .second_tolerance = cfg.second_tolerance,
            .auto_fire = cfg.auto_fire,
        };
    }

    // Internal state owned by strategy
    auto_aim::Target target_{};                          // EKF 跟踪目标
    auto_aim::DetectorWorkBuffers detector_buffers_{};   // 检测器工作缓冲
    auto_aim::GimbalCommand last_gimbal_cmd_{};          // 最近一次云台命令 (用于可视化)

    // Aimer + Shooter
    auto_aim::Aimer aimer_;
    auto_aim::Shooter shooter_;

    // 输出接口
    std::optional<auto_aim::OutputInterface> output_interface_{};

    // 运行时参数 (从外部设置)
    double bullet_speed_{23.0};  // m/s (默认 23)
    double gimbal_roll_{0.0};    // rad (当前云台 roll)
    double gimbal_pitch_{0.0};   // rad (当前云台 pitch)
    double gimbal_yaw_{0.0};     // rad (当前云台 yaw)

    // 相机标定参数 (用于可视化)
    uint32_t image_width_{0};
    uint32_t image_height_{0};
    std::array<double, 9> camera_matrix_{};
    std::vector<double> distortion_;
};
static_assert(scheduler::SchedulerStrategy<TalosStrategy>);
