#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

namespace primitive {

// ============ 统计工具 ============

struct LatencyStats {
    std::vector<uint64_t> samples;
    std::mutex mutex;

    void record(const uint64_t ns) {
        std::scoped_lock const lock(mutex);
        samples.push_back(ns);
    }

    void print(const std::string& name) {
        std::vector<uint64_t> local;
        {
            std::scoped_lock const lock(mutex);
            local = samples;
        }

        if (local.empty()) {
            std::cout << "[" << name << "] No samples\n";
            return;
        }

        std::ranges::sort(local);
        const auto sum     = std::accumulate(local.begin(), local.end(), 0ULL);
        const auto avg     = sum / local.size();
        const auto p50     = local[local.size() / 2];
        const auto p90     = local[local.size() * 90 / 100];
        const auto p99     = local[local.size() * 99 / 100];
        const auto max_val = local.back();

        std::cout << "[" << name << "] "
                  << "n=" << local.size() << ", avg=" << static_cast<double>(avg) / 1000.0 << "us"
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
} // namespace primitive
