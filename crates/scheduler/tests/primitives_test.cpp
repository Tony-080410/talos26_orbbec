#include <algorithm>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "primitive/spmc_triple_buffer.hpp"

using namespace talos;
using namespace std::chrono_literals;

namespace {

class CoutRedirect {
public:
    explicit CoutRedirect(const std::ostream& to)
        : old_(std::cout.rdbuf(to.rdbuf())) {}

    ~CoutRedirect() { std::cout.rdbuf(old_); }

    CoutRedirect(const CoutRedirect&)            = delete;
    CoutRedirect& operator=(const CoutRedirect&) = delete;
    CoutRedirect(CoutRedirect&&)                 = delete;
    CoutRedirect& operator=(CoutRedirect&&)      = delete;

private:
    std::streambuf* old_;
};

} // namespace

struct GenData {
    std::uint64_t gen{0};
    std::array<std::uint64_t, 32> pad{};

    explicit GenData(const std::uint64_t g)
        : gen(g) {
        pad.fill(g);
    }

    [[nodiscard]] bool consistent() const noexcept {
        return std::ranges::all_of(pad, [gen = this->gen](const auto v) { return v == gen; });
    }
};

TEST(SpmcTripleBufferTest, ConcurrentReadersSeeConsistentSnapshots) {
    auto [w, r] = primitive::SpmcTripleBuffer<GenData>::create();

    std::atomic stop{false};
    constexpr int kReaders = 4;

    std::thread writer([&] {
        std::uint64_t g = 1;
        while (!stop.load(std::memory_order_acquire)) {
            w.write(GenData(g++));
        }
    });

    std::atomic violated{false};
    std::vector<std::thread> readers;
    readers.reserve(kReaders);
    for (int i = 0; i < kReaders; ++i) {
        auto reader = r.clone();
        readers.emplace_back([reader = std::move(reader), &stop, &violated]() mutable {
            std::uint64_t last = 0;
            while (!stop.load(std::memory_order_acquire)) {
                if (const auto v = reader.read()) {
                    if (!v->consistent()) {
                        violated.store(true, std::memory_order_relaxed);
                    }
                    if (v->gen < last) {
                        violated.store(true, std::memory_order_relaxed);
                    }
                    last = v->gen;
                }
            }
        });
    }

    std::this_thread::sleep_for(100ms);
    stop.store(true, std::memory_order_release);

    writer.join();
    for (auto& th : readers) {
        th.join();
    }

    EXPECT_FALSE(violated.load(std::memory_order_acquire));
}
