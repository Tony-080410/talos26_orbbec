#pragma once

#include <oneapi/tbb/concurrent_queue.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <spdlog/spdlog.h>
#include <thread>

#include "scheduler/primitives/latest_value.hpp"
#include "scheduler/primitives/native_event.hpp"
#include "scheduler/primitives/thread_affinity.hpp" // Phase C: CPU pinning
#include "scheduler/primitives/triple_buffer.hpp"
#include "scheduler/strategy.hpp"
#include "scheduler/types.hpp"

// ============ 编译器提示宏 ============
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
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
     *   Frame(160Hz) → Detector(TBB parallel) → Tracker(event-driven) → Solver(polling)
     *
     * 性能目标:
     * - 调度开销 < 500ns (from 1-2µs baseline)
     * - 零动态分发开销
     * - Cache友好（无false sharing）
     */
    template<SchedulerStrategy Strategy>
    class Scheduler {
    public:
        Scheduler() noexcept : config_(nullptr), strategy_(nullptr) {}

        ~Scheduler() noexcept { stop(); }

        // 禁止拷贝
        Scheduler(const Scheduler &) = delete;
        Scheduler &operator=(const Scheduler &) = delete;

        // 允许移动
        Scheduler(Scheduler &&) = default;
        Scheduler &operator=(Scheduler &&) = default;

        // ============ 生命周期 ============

        /**
         * @brief 初始化调度器
         *
         * @param config 运行时配置
         * @param strategy 用户实现的Strategy对象（指针，调用者保证生命周期）
         */
        void init(const RuntimeConfig &config, Strategy *strategy) noexcept {
            strategy_ = strategy;
            config_ = std::make_shared<RuntimeConfig>(config);
            spdlog::info("[Scheduler] initialized with {} detector threads", config.detector_threads);
        }

        /**
         * @brief 启动所有工作线程
         */
        void start() noexcept {
            if (running_.exchange(true, std::memory_order_release)) {
                return; // 已经在运行
            }

            spdlog::info("[Scheduler] starting...");

            detector_thread_ = std::thread([this] { detector_loop(); });
            tracker_thread_ = std::thread([this] { tracker_loop(); });
            solver_thread_ = std::thread([this] { solver_loop(); });

            spdlog::info("[Scheduler] started");
        }

        /**
         * @brief 停止所有工作线程
         */
        void stop() noexcept {
            if (!running_.exchange(false, std::memory_order_release)) {
                return; // 已经停止
            }

            spdlog::info("[Scheduler] stopping...");

            // 唤醒可能阻塞的线程
            frame_event_.notify_all();
            tracker_event_.notify_all();

            if (detector_thread_.joinable())
                detector_thread_.join();
            if (tracker_thread_.joinable())
                tracker_thread_.join();
            if (solver_thread_.joinable())
                solver_thread_.join();

            spdlog::info("[Scheduler] stopped");
        }

        /**
         * @brief 检查是否在运行
         */
        [[nodiscard]] bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }

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

            frame_queue_.push(std::move(frame));
            frame_queue_size_.fetch_add(1, std::memory_order_relaxed);
            frame_event_.notify_one();
            return true;
        }

        // ============ 配置热更新 ============

        /**
         * @brief 热更新配置 (RCU 模式，无锁)
         */
        void update_config(const RuntimeConfig &config) noexcept {
            config_ = std::make_shared<RuntimeConfig>(config);
            spdlog::debug("[Scheduler] config updated");
        }

        /**
         * @brief 获取当前配置
         */
        [[nodiscard]] RuntimeConfig get_config() const noexcept { return *config_; }

        // ============ 统计 ============

        [[nodiscard]] const Stats &stats() const noexcept { return stats_; }
        void reset_stats() noexcept { stats_.reset(); }

    private:
        // ============ 数据通道 ============

        // Frame队列 + Event驱动detector
        tbb::concurrent_queue<Frame> frame_queue_;
        alignas(CACHE_LINE_SIZE) std::atomic<size_t> frame_queue_size_{0};
        static constexpr size_t kMaxFrameQueueSize = 8;
        NativeEvent frame_event_; // 通知detector有新帧

        // Triple Buffer: 共享特征缓冲（zero-copy）
        TripleBuffer<SharedFeature> shared_buffer_;

        // Detection队列 + Event驱动tracker
        tbb::concurrent_queue<Detection> detection_queue_;
        NativeEvent tracker_event_; // 通知tracker有新detection

        // Track Result: 双缓冲（solver轮询读取）
        LatestValue<TrackResult> track_result_;

        // ============ 配置 + 策略 ============

        std::shared_ptr<RuntimeConfig> config_;
        Strategy *strategy_{nullptr};

        // ============ 线程管理 ============

        std::thread detector_thread_;
        std::thread tracker_thread_;
        std::thread solver_thread_;
        alignas(CACHE_LINE_SIZE) std::atomic<bool> running_{false};

        // ============ 统计 ============

        Stats stats_;

        // ============ 热循环 ============

        /**
         * @brief Detector线程主循环
         *
         * 关键热路径优化：
         * 1. CRTP直接调用strategy_->detect() - 编译器可内联
         * 2. NativeEvent避免condition_variable开销
         * 3. 缓存行对齐避免false sharing
         * 4. Phase C: CPU pinning减少缓存失效
         */
        void detector_loop() noexcept {
            // Phase C: 绑定到核心0 (detector是最频繁的线程)
            ThreadAffinity::pin_to_core(0);
            spdlog::debug("[Detector] loop started, pinned to core 0");

            Frame frame;
            while (LIKELY(running_.load(std::memory_order_relaxed))) {
                // 等待新帧通知（原生事件，5-10ns hot path）
                if (UNLIKELY(!frame_event_.wait_for(100))) {
                    continue; // 超时，检查 running 状态
                }

                // 处理队列中所有帧
                while (frame_queue_.try_pop(frame)) {
                    frame_queue_size_.fetch_sub(1, std::memory_order_relaxed);

                    const auto t0 = std::chrono::steady_clock::now();

                    // 写入共享缓冲
                    auto &shared = shared_buffer_.borrow_mut();
                    shared.frame_seq = frame.seq;
                    shared.timestamp_ns = frame.timestamp_ns;
                    shared.image = std::move(frame.image);

                    // 预处理（可选，使用if constexpr检测）
                    if constexpr (requires { std::declval<Strategy>().preprocess(std::declval<SharedFeature &>()); }) {
                        strategy_->preprocess(shared);
                    }

                    shared_buffer_.publish();

                    // ⭐ 关键热点：CRTP直接调用，编译器会内联
                    // 性能：0ns (vs std::function的10-20ns)
                    Detection det = strategy_->detect(shared);

                    detection_queue_.push(std::move(det));
                    tracker_event_.notify_one();

                    const auto t1 = std::chrono::steady_clock::now();
                    const auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    stats_.detect_latency.store(latency_ns, std::memory_order_relaxed);
                    stats_.detect_hist.record(latency_ns);
                }
            }
        }

        /**
         * @brief Tracker线程主循环
         * Phase C: CPU pinning to core 1
         */
        void tracker_loop() noexcept {
            // Phase C: 绑定到核心1
            ThreadAffinity::pin_to_core(1);
            spdlog::debug("[Tracker] loop started, pinned to core 1");

            Detection det;
            while (LIKELY(running_.load(std::memory_order_relaxed))) {
                // 等待新detection通知
                if (UNLIKELY(!tracker_event_.wait_for(100))) {
                    continue;
                }

                while (detection_queue_.try_pop(det)) {
                    const auto t0 = std::chrono::steady_clock::now();

                    auto shared_opt = shared_buffer_.borrow();
                    if (UNLIKELY(!shared_opt)) {
                        continue; // 共享缓冲无效
                    }

                    // ⭐ 关键热点：CRTP直接调用
                    TrackResult tr = strategy_->track(det, **shared_opt);
                    track_result_.write(tr);

                    const auto t1 = std::chrono::steady_clock::now();
                    const auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    stats_.track_latency.store(latency_ns, std::memory_order_relaxed);
                    stats_.track_hist.record(latency_ns);
                }
            }
        }

        /**
         * @brief Solver线程主循环
         *
         * 轮询模式：读取最新的track result，无竞争
         * Phase C: CPU pinning to core 2
         */
        void solver_loop() noexcept {
            // Phase C: 绑定到核心2
            ThreadAffinity::pin_to_core(2);
            spdlog::debug("[Solver] loop started, pinned to core 2");

            TrackResult tr;
            uint64_t last_version = 0;

            while (LIKELY(running_.load(std::memory_order_relaxed))) {
                if (LIKELY(track_result_.try_read(tr, last_version))) {
                    const auto t0 = std::chrono::steady_clock::now();

                    // ⭐ 关键热点：CRTP直接调用
                    SolverOutput out = strategy_->solve(tr);

                    // 可选：输出回调（如果strategy实现了）
                    if constexpr (requires { std::declval<Strategy>().output(std::declval<SolverOutput>()); }) {
                        strategy_->output(out);
                    }

                    const auto t1 = std::chrono::steady_clock::now();
                    const auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    stats_.solve_latency.store(latency_ns, std::memory_order_relaxed);
                    stats_.solve_hist.record(latency_ns);
                } else {
                    // 数据不可用，让出CPU
                    std::this_thread::yield();
                }
            }
        }
    };

} // namespace scheduler

// ============ 显式模板实例化 (见scheduler_template.cpp) ============
