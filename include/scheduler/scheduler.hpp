#pragma once

#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/global_control.h>

#include <thread>
#include <atomic>
#include <functional>
#include <memory>

#include "scheduler/types.hpp"
#include "scheduler/primitives/triple_buffer.hpp"
#include "scheduler/primitives/latest_value.hpp"
#include "scheduler/primitives/event_flag.hpp"

namespace scheduler {

/**
 * @brief 高性能视觉流水线调度器
 *
 * 数据流:
 *   Frame(160Hz) → Detector(TBB parallel) → Tracker(event-driven) → Solver(polling)
 *
 * 性能目标:
 * - 调度开销 < 1μs
 * - 无锁或细粒度锁
 * - Cache 友好
 */
class Scheduler {
public:
    // 回调类型
    using DetectorFunc = std::function<Detection(const SharedFeature&)>;
    using TrackerFunc = std::function<TrackResult(const Detection&, const SharedFeature&)>;
    using SolverFunc = std::function<SolverOutput(const TrackResult&)>;
    using OutputFunc = std::function<void(const SolverOutput&)>;

    Scheduler();
    ~Scheduler();

    // 禁止拷贝
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // ============ 生命周期 ============

    /**
     * @brief 初始化调度器
     */
    void init(const RuntimeConfig& config);

    /**
     * @brief 启动所有工作线程
     */
    void start();

    /**
     * @brief 停止所有工作线程
     */
    void stop();

    /**
     * @brief 检查是否在运行
     */
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    // ============ 数据输入 ============

    /**
     * @brief 推送新帧 (非阻塞)
     * @return true 如果成功入队, false 如果队列满
     */
    bool push_frame(Frame frame);

    // ============ 配置热更新 ============

    /**
     * @brief 热更新配置 (RCU 模式，无锁)
     */
    void update_config(const RuntimeConfig& config);

    /**
     * @brief 获取当前配置
     */
    [[nodiscard]] RuntimeConfig get_config() const;

    // ============ 注册回调 ============

    void set_detector(DetectorFunc fn) { detect_fn_ = std::move(fn); }
    void set_tracker(TrackerFunc fn) { track_fn_ = std::move(fn); }
    void set_solver(SolverFunc fn) { solve_fn_ = std::move(fn); }
    void set_output(OutputFunc fn) { output_fn_ = std::move(fn); }

    // ============ 统计 ============

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_.reset(); }

private:
    void detector_loop();
    void tracker_loop();
    void solver_loop();

    // 数据通道
    tbb::concurrent_queue<Frame> frame_queue_;
    std::atomic<size_t> frame_queue_size_{0};
    static constexpr size_t kMaxFrameQueueSize = 8;
    EventFlag frame_event_;  // 通知 detector 有新帧
    TripleBuffer<SharedFeature> shared_buffer_;
    tbb::concurrent_queue<Detection> detection_queue_;
    EventFlag tracker_event_;
    LatestValue<TrackResult> track_result_;

    // 配置 (使用 atomic 自由函数)
    std::shared_ptr<RuntimeConfig> config_;

    // 回调
    DetectorFunc detect_fn_;
    TrackerFunc track_fn_;
    SolverFunc solve_fn_;
    OutputFunc output_fn_;

    // 线程
    std::thread detector_thread_;
    std::thread tracker_thread_;
    std::thread solver_thread_;
    std::atomic<bool> running_{false};

    // 统计
    Stats stats_;
};

} // namespace scheduler
