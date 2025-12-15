#include "scheduler/scheduler.hpp"
#include <spdlog/spdlog.h>
#include <chrono>

namespace scheduler {

Scheduler::Scheduler() = default;

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::init(const RuntimeConfig& config) {
    std::atomic_store(&config_, std::make_shared<RuntimeConfig>(config));
    spdlog::info("[Scheduler] initialized with {} detector threads", config.detector_threads);
}

void Scheduler::start() {
    if (running_.exchange(true)) {
        return;  // 已经在运行
    }

    spdlog::info("[Scheduler] starting...");

    detector_thread_ = std::thread(&Scheduler::detector_loop, this);
    tracker_thread_ = std::thread(&Scheduler::tracker_loop, this);
    solver_thread_ = std::thread(&Scheduler::solver_loop, this);

    spdlog::info("[Scheduler] started");
}

void Scheduler::stop() {
    if (!running_.exchange(false)) {
        return;  // 已经停止
    }

    spdlog::info("[Scheduler] stopping...");

    // 唤醒可能阻塞的线程
    frame_event_.notify_all();
    tracker_event_.notify_all();

    if (detector_thread_.joinable()) detector_thread_.join();
    if (tracker_thread_.joinable()) tracker_thread_.join();
    if (solver_thread_.joinable()) solver_thread_.join();

    spdlog::info("[Scheduler] stopped");
}

bool Scheduler::push_frame(Frame frame) {
    stats_.frame_in.fetch_add(1, std::memory_order_relaxed);

    // 检查队列是否已满
    if (frame_queue_size_.load(std::memory_order_relaxed) >= kMaxFrameQueueSize) {
        stats_.frame_drop.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    frame_queue_.push(std::move(frame));
    frame_queue_size_.fetch_add(1, std::memory_order_relaxed);
    frame_event_.notify();
    return true;
}

void Scheduler::update_config(const RuntimeConfig& config) {
    std::atomic_store(&config_, std::make_shared<RuntimeConfig>(config));
    spdlog::debug("[Scheduler] config updated");
}

RuntimeConfig Scheduler::get_config() const {
    return *std::atomic_load(&config_);
}

void Scheduler::detector_loop() {
    spdlog::debug("[Detector] loop started");

    Frame frame;
    while (running_.load(std::memory_order_relaxed)) {
        // 等待新帧通知
        if (!frame_event_.wait_for(std::chrono::milliseconds(100))) {
            continue;  // 超时，检查 running 状态
        }

        // 处理队列中所有帧
        while (frame_queue_.try_pop(frame)) {
            frame_queue_size_.fetch_sub(1, std::memory_order_relaxed);

            const auto t0 = std::chrono::steady_clock::now();
            [[maybe_unused]] auto cfg = std::atomic_load(&config_);

            // 写入共享缓冲
            auto& shared = shared_buffer_.borrow_mut();
            shared.frame_seq = frame.seq;
            shared.timestamp_ns = frame.timestamp_ns;
            shared.image = std::move(frame.image);

            // 预处理 (用户可用 TBB parallel_for 并行化)
            if (preprocess_fn_) {
                preprocess_fn_(shared);
            }

            shared_buffer_.publish();

            // 执行检测（用户回调）
            if (detect_fn_) {
                Detection det = detect_fn_(shared);
                detection_queue_.push(std::move(det));
                tracker_event_.notify();
            }

            const auto t1 = std::chrono::steady_clock::now();
            const auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            stats_.detect_latency.store(static_cast<uint64_t>(latency), std::memory_order_relaxed);
            stats_.detect_hist.record(static_cast<uint64_t>(latency));
            stats_.detect_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    spdlog::debug("[Detector] loop ended");
}

void Scheduler::tracker_loop() {
    spdlog::debug("[Tracker] loop started");

    while (running_.load(std::memory_order_relaxed)) {
        // 等待通知
        if (!tracker_event_.wait_for(std::chrono::milliseconds(100))) {
            continue;  // 超时，检查 running 状态
        }

        Detection det;
        while (detection_queue_.try_pop(det)) {
            const auto t0 = std::chrono::steady_clock::now();

            if (track_fn_) {
                if (auto shared_opt = shared_buffer_.borrow(); shared_opt) {
                    TrackResult result = track_fn_(det, **shared_opt);
                    track_result_.write(std::move(result));
                }
            }

            const auto t1 = std::chrono::steady_clock::now();
            const auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            stats_.track_latency.store(static_cast<uint64_t>(latency), std::memory_order_relaxed);
            stats_.track_hist.record(static_cast<uint64_t>(latency));
            stats_.track_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    spdlog::debug("[Tracker] loop ended");
}

void Scheduler::solver_loop() {
    spdlog::debug("[Solver] loop started");

    uint64_t last_version = 0;
    TrackResult tr;

    while (running_.load(std::memory_order_relaxed)) {
        // 轮询最新跟踪结果
        if (!track_result_.try_read(tr, last_version)) {
            // 无新数据，短暂让出 CPU
            std::this_thread::yield();
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();

        if (solve_fn_) {
            SolverOutput out = solve_fn_(tr);
            if (output_fn_) {
                output_fn_(out);
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        const auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        stats_.solve_latency.store(static_cast<uint64_t>(latency), std::memory_order_relaxed);
        stats_.solve_hist.record(static_cast<uint64_t>(latency));
        stats_.solve_count.fetch_add(1, std::memory_order_relaxed);
    }

    spdlog::debug("[Solver] loop ended");
}

} // namespace scheduler
