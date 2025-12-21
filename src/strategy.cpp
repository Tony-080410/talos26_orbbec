#include "strategy.hpp"

#include <opencv2/imgproc.hpp>

auto TalosStrategy::preprocess(scheduler::SharedFeature &sf) -> void {
    const auto t0 = std::chrono::steady_clock::now();

    const auto small = small_pool.acquire();
    sf.preprocessed = small;
    detector.preprocess(sf.image->mat(), sf.preprocessed->mat());

    const auto t1 = std::chrono::steady_clock::now();
    preprocess_latency.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

auto TalosStrategy::detect(const scheduler::SharedFeature &sf) -> scheduler::Detection {
    const auto t0 = std::chrono::steady_clock::now();

    scheduler::Detection det;
    det.frame_seq = sf.frame_seq;
    det.timestamp_ns = sf.timestamp_ns;


    if (!sf.preprocessed || sf.preprocessed->rows() <= 0) [[unlikely]] {
        return det;
    }

    slots = detector.detect(sf.image->mat(), sf.preprocessed->mat(), sf.frame_seq, 0);

    processed_count.fetch_add(1, std::memory_order_relaxed);

    const auto t1 = std::chrono::steady_clock::now();
    detect_latency.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    small_pool.release(sf.preprocessed);
    return det;
}

auto TalosStrategy::track(const scheduler::Detection &det, [[maybe_unused]] const scheduler::SharedFeature &sf)
        -> scheduler::TrackResult {
    const auto t0 = std::chrono::steady_clock::now();

    scheduler::TrackResult tr;
    tr.frame_seq = det.frame_seq;
    tr.timestamp_ns = det.timestamp_ns;

    const auto timestamp = std::chrono::steady_clock::now();
    tracker.track(slots, poses3d, timestamp);


    const auto t1 = std::chrono::steady_clock::now();
    track_latency.record(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    return tr;
}

auto TalosStrategy::solve(const scheduler::TrackResult &tr) -> scheduler::SolverOutput {
    scheduler::SolverOutput out;
    out.frame_seq = tr.frame_seq;
    out.timestamp_ns = tr.timestamp_ns;

    solver.solve_all(slots, poses3d);
    return out;
}

auto TalosStrategy::output(const scheduler::SolverOutput &out) -> void {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    e2e_latency.record(now - out.timestamp_ns);
    spdlog::info("armor count: {}", slots.count_valid());
    spdlog::info("state: {}", tracker_state(tracker.tracker_soa().state));
    output_count.fetch_add(1, std::memory_order_relaxed);
}
