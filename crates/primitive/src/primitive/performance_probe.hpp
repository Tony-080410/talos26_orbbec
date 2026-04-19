#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace talos::primitive {

/**
 * @brief Low-overhead latency probe
 *
 * steady_clock: monotonic, no TZ adjustment, ~20ns overhead per call.
 */
class LatencyProbe {
public:
    LatencyProbe() noexcept
        : start_ns_(now_ns()) {}

    ~LatencyProbe() = default;

    [[nodiscard]] uint64_t elapsed_ns() const noexcept { return now_ns() - start_ns_; }

    uint64_t snapshot_and_reset() noexcept {
        const auto now     = now_ns();
        const auto elapsed = now - start_ns_;
        start_ns_          = now;
        return elapsed;
    }

private:
    uint64_t start_ns_;

    static uint64_t now_ns() noexcept {
        return static_cast<uint64_t>(
            std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now())
                .time_since_epoch()
                .count());
    }
};

/**
 * @brief Lock-free SPSC latency histogram
 *
 * record(): one relaxed load + one store + one release store (~5ns).
 * Keeps the most recent kCapacity samples in a power-of-2 ring buffer.
 * Heap-allocated — safe to use as member without blowing stack.
 */
class LatencyHistogram {
public:
    static constexpr uint32_t kCapacity = 4096;
    static constexpr uint32_t kMask     = kCapacity - 1;
    static_assert((kCapacity & kMask) == 0, "kCapacity must be power of 2");

    LatencyHistogram()
        : write_pos_(0)
        , samples_(kCapacity, 0) {}

    LatencyHistogram(LatencyHistogram&& other) noexcept
        : write_pos_(other.write_pos_.load(std::memory_order_acquire))
        , samples_(std::move(other.samples_)) {}

    LatencyHistogram& operator=(LatencyHistogram&& other) noexcept {
        if (this != &other) {
            samples_ = std::move(other.samples_);
            write_pos_.store(
                other.write_pos_.load(std::memory_order_acquire), std::memory_order_release);
        }
        return *this;
    }

    LatencyHistogram(const LatencyHistogram&)            = delete;
    LatencyHistogram& operator=(const LatencyHistogram&) = delete;

    void record(const uint64_t latency_ns) noexcept {
        const auto pos        = write_pos_.load(std::memory_order_relaxed);
        samples_[pos & kMask] = latency_ns;
        write_pos_.store(pos + 1, std::memory_order_release);
    }

    struct Stats {
        uint64_t min_ns;
        uint64_t max_ns;
        uint64_t mean_ns;
        uint64_t p50_ns;
        uint64_t p95_ns;
        uint64_t p99_ns;
        uint64_t p999_ns;
        double stddev_ns;
        size_t count;
    };

    [[nodiscard]] Stats compute() const {
        const uint64_t end = write_pos_.load(std::memory_order_acquire);
        const size_t total = static_cast<size_t>(end);
        const size_t count = std::min(total, static_cast<size_t>(kCapacity));

        if (count == 0) {
            return {0, 0, 0, 0, 0, 0, 0, 0.0, 0};
        }

        std::vector<uint64_t> snap(count);
        if (total <= kCapacity) {
            std::copy_n(samples_.data(), count, snap.data());
        } else {
            const size_t start = static_cast<size_t>(end) & kMask;
            const size_t tail  = kCapacity - start;
            std::copy_n(samples_.data() + start, tail, snap.data());
            std::copy_n(samples_.data(), start, snap.data() + tail);
        }

        std::ranges::sort(snap);

        const auto min_val  = snap.front();
        const auto max_val  = snap.back();
        const auto sum      = std::accumulate(snap.begin(), snap.end(), 0ULL);
        const auto mean_val = sum / count;

        const auto p50_idx  = std::min(count * 50 / 100, count - 1);
        const auto p95_idx  = std::min(count * 95 / 100, count - 1);
        const auto p99_idx  = std::min(count * 99 / 100, count - 1);
        const auto p999_idx = std::min(count * 999 / 1000, count - 1);

        double sum_sq       = 0.0;
        const double mean_d = static_cast<double>(mean_val);
        for (auto v : snap) {
            const double diff = static_cast<double>(v) - mean_d;
            sum_sq += diff * diff;
        }

        return {min_val,        max_val,
                mean_val,       snap[p50_idx],
                snap[p95_idx],  snap[p99_idx],
                snap[p999_idx], std::sqrt(sum_sq / static_cast<double>(count)),
                count};
    }

    void print_stats() const {
        const auto s = compute();
        std::cout << "Latency Statistics (ns):\n"
                  << "  Count:  " << s.count << "\n"
                  << "  Min:    " << s.min_ns << "\n"
                  << "  P50:    " << s.p50_ns << "\n"
                  << "  P95:    " << s.p95_ns << "\n"
                  << "  P99:    " << s.p99_ns << "\n"
                  << "  P99.9:  " << s.p999_ns << "\n"
                  << "  Mean:   " << s.mean_ns << "\n"
                  << "  Max:    " << s.max_ns << "\n"
                  << "  StdDev: " << s.stddev_ns << "\n";
    }

private:
    alignas(64) std::atomic<uint64_t> write_pos_{0};
    std::vector<uint64_t> samples_;
};

} // namespace talos::primitive
