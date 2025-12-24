#include "strategy.hpp"
#include "strategy_wrapper.hpp"

#include <opencv2/imgproc.hpp>

auto TalosStrategy::preprocess(scheduler::Captured& sf) -> void {
    const auto t0 = std::chrono::steady_clock::now();

    const auto small = small_pool.acquire();
    sf.preprocessed  = small;
    detector.preprocess(sf.image->mat(), sf.preprocessed->mat());

    const auto t1 = std::chrono::steady_clock::now();
    preprocess_latency.record(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

auto TalosStrategy::detect(const scheduler::Captured& feat, talos::ArmorSlots& armor) noexcept
    -> void {
    const auto t0 = std::chrono::steady_clock::now();

    if (!feat.preprocessed || feat.preprocessed->rows() <= 0) [[unlikely]] {
        return;
    }

    armor = detector.detect(
        detector_buffers_, feat.image->mat(), feat.preprocessed->mat(), feat.frame_seq, 0);

    processed_count.fetch_add(1, std::memory_order_relaxed);

    const auto t1 = std::chrono::steady_clock::now();
    detect_latency.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    small_pool.release(feat.preprocessed);
}

auto TalosStrategy::solve(
    const talos::ArmorSlots& armor, std::array<auto_aim::ArmorPose3D, 10>& poses3d,
    uint64_t timestamp_ns) const noexcept -> void {
    solver.solve_all(armor, poses3d, timestamp_ns);
}

auto TalosStrategy::track(
    talos::TrackerSoA& tracker_soa, const talos::ArmorSlots& armor,
    const std::array<auto_aim::ArmorPose3D, 10>& poses3d,
    [[maybe_unused]] uint64_t timestamp_ns) noexcept -> void {
    const auto t0 = std::chrono::steady_clock::now();

    const auto timestamp = std::chrono::steady_clock::now();
    tracker.track(tracker_soa, target_, armor, poses3d, timestamp);

    spdlog::info("state={}", talos::tracker_state(tracker_soa.state.load()));
    const auto t1 = std::chrono::steady_clock::now();
    track_latency.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

auto TalosStrategy::output() noexcept -> void {
    output_count.fetch_add(1, std::memory_order_relaxed);

    // 检查输出接口是否已初始化
    if (!output_interface_) {
        last_gimbal_cmd_ = {false, false, 0.0, 0.0, 0.0};
        return;
    }

    // 检查目标是否有效 (依赖 track() 的结果)
    if (!target_.converged()) {
        // 发送无效命令
        last_gimbal_cmd_ = {false, false, 0.0, 0.0, 0.0};
        output_interface_->send(last_gimbal_cmd_);
        return;
    }

    // 1. Aimer: 计算瞄准命令
    const auto timestamp = std::chrono::steady_clock::now();
    auto cmd             = aimer_.aim(target_, timestamp, bullet_speed_, true);

    if (!cmd.valid) {
        last_gimbal_cmd_ = cmd;
        output_interface_->send(cmd);
        return;
    }

    // 2. Shooter: 判断是否发射
    const bool should_fire = shooter_.should_shoot(cmd, aimer_, target_, gimbal_yaw_);
    cmd.fire_advice        = should_fire;
    if (should_fire) {
        spdlog::info("fire!!!");
    }

    // 3. 保存并发送命令
    last_gimbal_cmd_ = cmd;
    output_interface_->send(cmd);
}
