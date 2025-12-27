#pragma once

#include <oneapi/tbb/concurrent_queue.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <spdlog/spdlog.h>
#include <thread>
#include <utility>

#include "auto_aim/pipeline_types.hpp"
#include "auto_aim/types.hpp"
#include "detector_state.hpp"
#include "primitives/native_event.hpp"
#include "primitives/thread_affinity.hpp"
#include "primitives/triple_buffer.hpp"
#include "strategy.hpp"
#include "tracker_soa.hpp"
#include "types.hpp"

// ============ 编译器提示宏 ============
#if defined(__GNUC__) || defined(__clang__)
# define LIKELY(x)   __builtin_expect(!!(x), 1)
# define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
# define LIKELY(x)   (x)
# define UNLIKELY(x) (x)
#endif

namespace scheduler {

/**
 * @brief 高性能零成本抽象视觉流水线调度器
 *
 * 使用CRTP + C++20 concepts实现静态多态，消除std::function虚调开销。
 * 使用平台原生事件原语（futex/dispatch_semaphore）替代condition_variable。
 * 缓存行对齐避免false sharing。
 *
 * 数据流:
 *   Frame(160Hz) → Detect → Solve → Track → Output
 *
 * 数据所有权:
 * - Scheduler 持有: TrackerSoA, poses3d
 * - Strategy 持有: Target, DetectorWorkBuffers (内部状态)
 *
 * 性能目标:
 * - 调度开销 < 500ns (from 1-2µs baseline)
 * - 零动态分发开销
 * - Cache友好（无false sharing）
 */
template <SchedulerStrategy Strategy>
class Scheduler {
public:
    Scheduler() noexcept
        : config_(nullptr)
        , strategy_(nullptr) {}

    ~Scheduler() noexcept { stop(); }

    // 禁止拷贝
    Scheduler(const Scheduler&)            = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // 禁止移动
    Scheduler(Scheduler&&)            = delete;
    Scheduler& operator=(Scheduler&&) = delete;

    // ============ 生命周期 ============

    /**
     * @brief 初始化调度器
     *
     * @param config 运行时配置
     * @param strategy 用户实现的Strategy对象（指针，调用者保证生命周期）
     */
    void init(const RuntimeConfig& config, Strategy* strategy) noexcept {
        strategy_ = strategy;
        config_   = std::make_shared<RuntimeConfig>(config);
        spdlog::info("[scheduler] initialized.");
    }

    /**
     * @brief 启动所有工作线程
     */
    void start() noexcept {
        if (running_.exchange(true, std::memory_order_release)) {
            return; // 已经在运行
        }

        spdlog::info("[scheduler] starting...");

        detector_thread_ = std::thread([this] { detector_loop(); });
        tracker_thread_  = std::thread([this] { tracker_loop(); });
        output_thread_   = std::thread([this] { output_loop(); });

        spdlog::info("[scheduler] started");
    }

    /**
     * @brief 停止所有工作线程
     */
    void stop() noexcept {
        if (!running_.exchange(false, std::memory_order_release)) {
            return; // 已经停止
        }

        spdlog::info("[scheduler] stopping...");

        // 唤醒可能阻塞的线程
        frame_event_.notify_all();
        tracker_event_.notify_all();
        output_event_.notify_all();

        if (detector_thread_.joinable()) {
            detector_thread_.join();
        }
        if (tracker_thread_.joinable()) {
            tracker_thread_.join();
        }
        if (output_thread_.joinable()) {
            output_thread_.join();
        }

        spdlog::info("[scheduler] stopped");
    }

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
    bool push_frame(Frame frame) noexcept {
        stats_.frame_in.fetch_add(1, std::memory_order_relaxed);

        // 检查队列是否已满
        if (UNLIKELY(frame_queue_size_.load(std::memory_order_relaxed) >= kMaxFrameQueueSize)) {
            stats_.frame_drop.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        auto& mut = frame_queue_.borrow_mut();
        mut       = std::move(frame);
        frame_queue_.publish();
        frame_queue_size_.fetch_add(1, std::memory_order_relaxed);
        frame_event_.notify_one();
        return true;
    }

    // ============ 配置热更新 ============

    /**
     * @brief 热更新配置 (RCU 模式，无锁)
     */
    void update_config(const RuntimeConfig& config) noexcept {
        config_ = std::make_shared<RuntimeConfig>(config);
        spdlog::info("[scheduler] config updated");
    }

    /**
     * @brief 获取当前配置
     */
    [[nodiscard]] RuntimeConfig get_config() const noexcept { return *config_; }

    // ============ 统计 ============

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_.reset(); }

    // Event驱动
    alignas(CACHE_LINE_SIZE) NativeEvent tracker_event_; // 通知tracker有新detection
    alignas(CACHE_LINE_SIZE) NativeEvent output_event_;  // 通知output线程

private:
    // ============ 数据通道 ============

    // Frame队列 + Event驱动detector
    alignas(CACHE_LINE_SIZE) TripleBuffer<Frame> frame_queue_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> frame_queue_size_{0};
    static constexpr size_t kMaxFrameQueueSize = 8;
    alignas(CACHE_LINE_SIZE) NativeEvent frame_event_; // 通知detector有新帧

    // Triple Buffer: 共享特征缓冲（zero-copy）
    alignas(CACHE_LINE_SIZE) TripleBuffer<Captured> shared_buffer_;

    // ArmorSlots + poses3d: 检测和解算结果缓冲
    // 注: 命名为 DetectSolveBundle 避免与 auto_aim::DetectionResult variant 冲突
    // Phase 2 将迁移到 variant 类型
    struct DetectSolveBundle {
        uint64_t timestamp_ns{0}; // 帧时间戳 (用于tf查询)
        talos::ArmorSlots armor_slots{};
        auto_aim::Poses3D poses3d = {};
    };
    alignas(CACHE_LINE_SIZE) TripleBuffer<DetectSolveBundle> detection_buffer_;

    // Scheduler 持有的状态
    alignas(CACHE_LINE_SIZE) talos::TrackerSoA tracker_soa_{};

    // ============ 配置 + 策略 ============

    std::shared_ptr<RuntimeConfig> config_;
    Strategy* strategy_{nullptr};

    // ============ 线程管理 ============

    std::thread detector_thread_;
    std::thread tracker_thread_;
    std::thread output_thread_;
    alignas(CACHE_LINE_SIZE) std::atomic<bool> running_{false};

    // ============ 统计 ============

    Stats stats_;

    // ============ 热循环 ============

    static constexpr auto realtime_policy(const std::string& prefix, const uint32_t core) {
        if (const auto result = ThreadAffinity::pin_to_core(core); !result) {
            spdlog::warn("[{}] failed to pin: {}", prefix, error_message(result.error()));
        } else if (const auto realtime_result = ThreadAffinity::set_realtime_priority();
                   !realtime_result) {
            spdlog::warn(
                "[{}] failed to set realtime policy: {}", prefix,
                error_message(realtime_result.error()));
        } else {
            spdlog::debug("[{}] pinned to core {}", prefix, core);
        }
    }

    /**
     * @brief Detector线程主循环
     *
     * 执行: detect → solve → 通知 tracker
     */
    void detector_loop() noexcept {
        realtime_policy(__FUNCTION__, 0);
        while (LIKELY(running_.load(std::memory_order_relaxed))) {
            // 等待新帧通知
            if (UNLIKELY(!frame_event_.wait_for(100))) {
                continue;
            }

            // 处理队列中所有帧
            const auto borrow_ = frame_queue_.borrow();
            if (!borrow_) {
                continue;
            }
            frame_queue_size_.fetch_sub(1, std::memory_order_relaxed);

            const auto borrow = *borrow_;

            // 写入共享缓冲
            auto& shared        = shared_buffer_.borrow_mut();
            shared.frame_seq    = borrow->seq;
            shared.timestamp_ns = borrow->timestamp_ns;
            shared.image        = borrow->image;

            // 预处理
            strategy_->preprocess(shared);
            shared_buffer_.publish();

            // 检测
            auto& result = detection_buffer_.borrow_mut();

            result.timestamp_ns             = borrow->timestamp_ns; // 保存帧时间戳
            result.armor_slots.timestamp_ns = borrow->timestamp_ns;

            const auto t_detect_start = std::chrono::steady_clock::now();
            result.armor_slots.clear();
            strategy_->detect(shared, result.armor_slots);
            const auto t_detect_end = std::chrono::steady_clock::now();

            // 解算 (传递帧时间戳用于tf查询)
            strategy_->solve(result.armor_slots, result.poses3d, result.timestamp_ns);
            const auto t_solve_end = std::chrono::steady_clock::now();

            detection_buffer_.publish();

            // 通知 tracker
            tracker_event_.notify_one();

            // 记录延迟
            const auto detect_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t_detect_end - t_detect_start)
                    .count();
            const auto solve_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t_solve_end - t_detect_end)
                    .count();
            stats_.detect_latency.store(detect_ns, std::memory_order_relaxed);
            stats_.detect_hist.record(detect_ns);
            stats_.solve_latency.store(solve_ns, std::memory_order_relaxed);
            stats_.solve_hist.record(solve_ns);
        }
    }

    /**
     * @brief Tracker线程主循环
     *
     * 执行: track → 通知 output
     */
    void tracker_loop() noexcept {
        realtime_policy(__FUNCTION__, 1);
        spdlog::debug("[tracker] loop started");

        while (LIKELY(running_.load(std::memory_order_relaxed))) {
            // 等待新detection通知
            if (UNLIKELY(!tracker_event_.wait_for(100))) {
                continue;
            }

            const auto t0 = std::chrono::steady_clock::now();

            auto detection_opt = detection_buffer_.borrow();
            if (UNLIKELY(!detection_opt)) {
                continue;
            }

            const auto result = *detection_opt;

            // 调用 track，传入 scheduler 持有的 TrackerSoA 和帧时间戳
            strategy_->track(
                tracker_soa_, result->armor_slots, result->poses3d, result->timestamp_ns);

            // 通知 output
            output_event_.notify_one();

            const auto t1 = std::chrono::steady_clock::now();
            const auto latency_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            stats_.track_latency.store(latency_ns, std::memory_order_relaxed);
            stats_.track_hist.record(latency_ns);
        }
    }

    /**
     * @brief Output线程主循环
     */
    void output_loop() noexcept {
        realtime_policy(__FUNCTION__, 2);
        spdlog::debug("[output] loop started");

        while (LIKELY(running_.load(std::memory_order_relaxed))) {
            if (UNLIKELY(!output_event_.wait_for(100))) {
                continue;
            }

            const auto t0 = std::chrono::steady_clock::now();

            strategy_->output();

            const auto t1 = std::chrono::steady_clock::now();
            const auto latency_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            stats_.output_latency.store(latency_ns, std::memory_order_relaxed);
            stats_.output_hist.record(latency_ns);
        }
    }
};
} // namespace scheduler

// ============ 显式模板实例化 (见scheduler_template.cpp) ============
