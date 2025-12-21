#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace scheduler {

    /**
     * @brief 无锁延迟直方图
     *
     * 用于收集延迟分布，支持百分位数计算。
     * 使用对数分桶，覆盖 100ns ~ 100ms 范围。
     *
     * 桶分布 (对数):
     * - [0]: 0-100ns
     * - [1]: 100-200ns
     * - [2]: 200-500ns
     * - [3]: 500ns-1us
     * - [4]: 1-2us
     * - [5]: 2-5us
     * - [6]: 5-10us
     * - [7]: 10-20us
     * - [8]: 20-50us
     * - [9]: 50-100us
     * - [10]: 100-200us
     * - [11]: 200-500us
     * - [12]: 500us-1ms
     * - [13]: 1-2ms
     * - [14]: 2-5ms
     * - [15]: 5-10ms
     * - [16]: 10-20ms
     * - [17]: 20-50ms
     * - [18]: 50-100ms
     * - [19]: >100ms
     */
    class LatencyHistogram {
    public:
        static constexpr size_t kNumBuckets = 20;

        LatencyHistogram() = default;

        /**
         * @brief 记录一个延迟样本 (纳秒)
         */
        void record(const uint64_t latency_ns) noexcept {
            const size_t bucket = latency_to_bucket(latency_ns);
            buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
            count_.fetch_add(1, std::memory_order_relaxed);
            sum_.fetch_add(latency_ns, std::memory_order_relaxed);

            // 更新 min/max (无锁)
            uint64_t old_min = min_.load(std::memory_order_relaxed);
            while (latency_ns < old_min &&
                   !min_.compare_exchange_weak(old_min, latency_ns, std::memory_order_relaxed)) {
            }

            uint64_t old_max = max_.load(std::memory_order_relaxed);
            while (latency_ns > old_max &&
                   !max_.compare_exchange_weak(old_max, latency_ns, std::memory_order_relaxed)) {
            }
        }

        /**
         * @brief 重置所有统计
         */
        void reset() noexcept {
            for (auto &b: buckets_) {
                b.store(0, std::memory_order_relaxed);
            }
            count_.store(0, std::memory_order_relaxed);
            sum_.store(0, std::memory_order_relaxed);
            min_.store(UINT64_MAX, std::memory_order_relaxed);
            max_.store(0, std::memory_order_relaxed);
        }

        /**
         * @brief 获取样本数
         */
        [[nodiscard]] uint64_t count() const noexcept { return count_.load(std::memory_order_relaxed); }

        /**
         * @brief 获取平均延迟 (纳秒)
         */
        [[nodiscard]] double mean() const noexcept {
            const auto c = count_.load(std::memory_order_relaxed);
            if (c == 0)
                return 0.0;
            return static_cast<double>(sum_.load(std::memory_order_relaxed)) / c;
        }

        /**
         * @brief 获取最小延迟 (纳秒)
         */
        [[nodiscard]] uint64_t min() const noexcept {
            const auto m = min_.load(std::memory_order_relaxed);
            return m == UINT64_MAX ? 0 : m;
        }

        /**
         * @brief 获取最大延迟 (纳秒)
         */
        [[nodiscard]] uint64_t max() const noexcept { return max_.load(std::memory_order_relaxed); }

        /**
         * @brief 获取百分位数 (纳秒)
         * @param p 百分位 (0.0-1.0)
         */
        [[nodiscard]] uint64_t percentile(const double p) const noexcept {
            const auto total = count_.load(std::memory_order_relaxed);
            if (total == 0)
                return 0;

            const auto target = static_cast<uint64_t>(p * static_cast<double>(total));
            uint64_t cumulative = 0;

            for (size_t i = 0; i < kNumBuckets; ++i) {
                cumulative += buckets_[i].load(std::memory_order_relaxed);
                if (cumulative >= target) {
                    return bucket_to_latency(i);
                }
            }
            return bucket_to_latency(kNumBuckets - 1);
        }

        /**
         * @brief 快捷方法: p50
         */
        [[nodiscard]] uint64_t p50() const noexcept { return percentile(0.50); }

        /**
         * @brief 快捷方法: p90
         */
        [[nodiscard]] uint64_t p90() const noexcept { return percentile(0.90); }

        /**
         * @brief 快捷方法: p99
         */
        [[nodiscard]] uint64_t p99() const noexcept { return percentile(0.99); }

        /**
         * @brief 获取桶计数 (用于调试/可视化)
         */
        [[nodiscard]] std::array<uint64_t, kNumBuckets> snapshot() const noexcept {
            std::array<uint64_t, kNumBuckets> result{};
            for (size_t i = 0; i < kNumBuckets; ++i) {
                result[i] = buckets_[i].load(std::memory_order_relaxed);
            }
            return result;
        }

    private:
        // 桶边界 (纳秒)
        static constexpr std::array<uint64_t, kNumBuckets> kBucketBounds = {
                100, // 0: 0-100ns
                200, // 1: 100-200ns
                500, // 2: 200-500ns
                1000, // 3: 500ns-1us
                2000, // 4: 1-2us
                5000, // 5: 2-5us
                10000, // 6: 5-10us
                20000, // 7: 10-20us
                50000, // 8: 20-50us
                100000, // 9: 50-100us
                200000, // 10: 100-200us
                500000, // 11: 200-500us
                1000000, // 12: 500us-1ms
                2000000, // 13: 1-2ms
                5000000, // 14: 2-5ms
                10000000, // 15: 5-10ms
                20000000, // 16: 10-20ms
                50000000, // 17: 20-50ms
                100000000, // 18: 50-100ms
                UINT64_MAX // 19: >100ms
        };

        static constexpr size_t latency_to_bucket(const uint64_t latency_ns) noexcept {
            for (size_t i = 0; i < kNumBuckets; ++i) {
                if (latency_ns < kBucketBounds[i]) {
                    return i;
                }
            }
            return kNumBuckets - 1;
        }

        static constexpr uint64_t bucket_to_latency(const size_t bucket) noexcept {
            if (bucket == 0) {
                return 50; // 中点
            }
            if (bucket >= kNumBuckets) {
                return kBucketBounds[kNumBuckets - 2];
            }
            // 返回桶的中点
            return (kBucketBounds[bucket - 1] + kBucketBounds[bucket]) / 2;
        }

        alignas(64) std::array<std::atomic<uint64_t>, kNumBuckets> buckets_{};
        alignas(64) std::atomic<uint64_t> count_{0};
        alignas(64) std::atomic<uint64_t> sum_{0};
        alignas(64) std::atomic<uint64_t> min_{UINT64_MAX};
        alignas(64) std::atomic<uint64_t> max_{0};
    };

} // namespace scheduler
