#include "strategy.hpp"
#include "strategy_wrapper.hpp"

#include <opencv2/imgproc.hpp>

auto TalosStrategy::preprocess(scheduler::SharedFeature& sf) -> void {
    const auto t0 = std::chrono::steady_clock::now();

    const auto small = small_pool.acquire();
    sf.preprocessed  = small;
    detector.preprocess(sf.image->mat(), sf.preprocessed->mat());

    const auto t1 = std::chrono::steady_clock::now();
    preprocess_latency.record(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

auto TalosStrategy::detect(const scheduler::SharedFeature& feat, talos::ArmorSlots& armor) noexcept
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
    const talos::ArmorSlots& armor, std::array<auto_aim::ArmorPose3D, 10>& poses3d) const noexcept
    -> void {
    solver.solve_all(armor, poses3d);
}

auto TalosStrategy::track(
    talos::TrackerSoA& tracker_soa, const talos::ArmorSlots& armor,
    const std::array<auto_aim::ArmorPose3D, 10>& poses3d) noexcept -> void {
    const auto t0 = std::chrono::steady_clock::now();

    const auto timestamp = std::chrono::steady_clock::now();
    tracker.track(tracker_soa, target_, armor, poses3d, timestamp);

    spdlog::info("state={}", talos::tracker_state(tracker_soa.state.load()));
    const auto t1 = std::chrono::steady_clock::now();
    track_latency.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

auto TalosStrategy::output() noexcept -> void {
    output_count.fetch_add(1, std::memory_order_relaxed);
}
