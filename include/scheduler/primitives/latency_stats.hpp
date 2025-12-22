#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>
#include <mutex>
#include <vector>

namespace scheduler {
    using namespace std::chrono_literals;

    // ============ 统计工具 ============

    struct LatencyStats {
        std::vector<uint64_t> samples;
        std::mutex mutex;

        void record(const uint64_t ns) {
            std::scoped_lock const lock(mutex);
            samples.push_back(ns);
        }

        void print(const std::string &name) {
            if (samples.empty()) {
                std::cout << "[" << name << "] No samples\n";
                return;
            }

            std::ranges::sort(samples);
            const auto sum = std::accumulate(samples.begin(), samples.end(), 0ULL);
            const auto avg = sum / samples.size();
            const auto p50 = samples[samples.size() / 2];
            const auto p90 = samples[samples.size() * 90 / 100];
            const auto p99 = samples[samples.size() * 99 / 100];
            const auto max_val = samples.back();

            std::cout << "[" << name << "] "
                      << "n=" << samples.size() << ", avg=" << static_cast<double>(avg) / 1000.0 << "us"
                      << ", p50=" << static_cast<double>(p50) / 1000.0 << "us"
                      << ", p90=" << static_cast<double>(p90) / 1000.0 << "us"
                      << ", p99=" << static_cast<double>(p99) / 1000.0 << "us"
                      << ", max=" << static_cast<double>(max_val) / 1000.0 << "us\n";
        }

        void reset() {
            std::scoped_lock lock(mutex);
            samples.clear();
        }
    };
} // namespace scheduler
