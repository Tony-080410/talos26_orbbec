#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <latch>
#include <mutex>
#include <ostream>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

#include "scheduler/primitives/buffer_pool.hpp"
#include "scheduler/primitives/frame.hpp"
#include "scheduler/primitives/latency_histogram.hpp"
#include "scheduler/primitives/latency_stats.hpp"
#include "scheduler/primitives/native_event.hpp"
#include "scheduler/primitives/spmc_triple_buffer.hpp"

using namespace primitive;
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

TEST(AutoResetEventTest, SignalWakesOne) {
    AutoResetEvent ev;
    std::atomic woke{false};

    std::thread waiter([&] {
        const auto st = ev.wait();
        EXPECT_EQ(st, WaitStatus::Signaled);
        woke.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(10ms);
    ev.signal();

    waiter.join();
    EXPECT_TRUE(woke.load(std::memory_order_acquire));
}

TEST(AutoResetEventTest, SignalBeforeWaitIsConsumedOnce) {
    AutoResetEvent ev;
    ev.signal();

    EXPECT_EQ(ev.wait_for(10ms), WaitStatus::Signaled);
    EXPECT_EQ(ev.wait_for(10ms), WaitStatus::Timeout);
}

TEST(AutoResetEventTest, SignalWakesExactlyOneWaiter) {
    AutoResetEvent ev;

    std::atomic woke{0};
    std::latch ready(2);

    std::thread a([&] {
        ready.count_down();
        EXPECT_EQ(ev.wait(), WaitStatus::Signaled);
        woke.fetch_add(1, std::memory_order_relaxed);
    });
    std::thread b([&] {
        ready.count_down();
        EXPECT_EQ(ev.wait(), WaitStatus::Signaled);
        woke.fetch_add(1, std::memory_order_relaxed);
    });

    ready.wait();
    std::this_thread::sleep_for(10ms);

    ev.signal();
    const auto deadline = std::chrono::steady_clock::now() + 200ms;
    while (std::chrono::steady_clock::now() < deadline
           && woke.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
    }
    EXPECT_EQ(woke.load(std::memory_order_acquire), 1);

    ev.signal();
    a.join();
    b.join();
    EXPECT_EQ(woke.load(std::memory_order_acquire), 2);
}

TEST(AutoResetEventTest, WaitForTimeout) {
    AutoResetEvent ev;
    const auto st = ev.wait_for(5ms);
    EXPECT_EQ(st, WaitStatus::Timeout);
}

TEST(AutoResetEventTest, CloseUnblocks) {
    AutoResetEvent ev;
    std::atomic st{WaitStatus::Timeout};

    std::thread waiter([&] { st.store(ev.wait(), std::memory_order_release); });
    std::this_thread::sleep_for(10ms);
    ev.close();
    waiter.join();

    EXPECT_EQ(st.load(std::memory_order_acquire), WaitStatus::Closed);
}

TEST(ManualResetEventTest, SetStaysSignaledUntilReset) {
    ManualResetEvent ev;
    ev.set();
    EXPECT_EQ(ev.wait_for(1ms), WaitStatus::Signaled);
    EXPECT_EQ(ev.wait_for(1ms), WaitStatus::Signaled);
    ev.reset();
    EXPECT_EQ(ev.wait_for(1ms), WaitStatus::Timeout);
}

TEST(ManualResetEventTest, SetWakesAllWaiters) {
    ManualResetEvent ev;

    constexpr int kWaiters = 4;
    std::atomic woke{0};
    std::latch ready(kWaiters);

    std::vector<std::thread> threads;
    threads.reserve(kWaiters);
    for (int i = 0; i < kWaiters; ++i) {
        threads.emplace_back([&] {
            ready.count_down();
            EXPECT_EQ(ev.wait(), WaitStatus::Signaled);
            woke.fetch_add(1, std::memory_order_relaxed);
        });
    }

    ready.wait();
    std::this_thread::sleep_for(10ms);
    ev.set();

    for (auto& t : threads) {
        t.join();
    }
    EXPECT_EQ(woke.load(std::memory_order_acquire), kWaiters);

    EXPECT_EQ(ev.wait_for(1ms), WaitStatus::Signaled);
    ev.reset();
    EXPECT_EQ(ev.wait_for(5ms), WaitStatus::Timeout);
}

TEST(NativeEventTest, BackwardCompatibleApi) {
    NativeEvent ev;

    EXPECT_FALSE(ev.wait_for(5));

    ev.notify_one();
    EXPECT_TRUE(ev.wait_for(100));

    EXPECT_EQ(ev.wait_for_status(1), WaitStatus::Timeout);
    ev.close();
    EXPECT_EQ(ev.wait_for_status(1), WaitStatus::Closed);
}

TEST(NativeEventTest, NotifyAllActsAsSingleSignal) {
    NativeEvent ev;

    std::atomic woke{0};
    std::latch ready(2);

    std::thread a([&] {
        ready.count_down();
        ev.wait();
        woke.fetch_add(1, std::memory_order_relaxed);
    });
    std::thread b([&] {
        ready.count_down();
        ev.wait();
        woke.fetch_add(1, std::memory_order_relaxed);
    });

    ready.wait();
    std::this_thread::sleep_for(10ms);

    ev.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + 200ms;
    while (std::chrono::steady_clock::now() < deadline
           && woke.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
    }
    EXPECT_EQ(woke.load(std::memory_order_acquire), 1);

    ev.notify_one();
    a.join();
    b.join();
    EXPECT_EQ(woke.load(std::memory_order_acquire), 2);
}

TEST(BufferPoolTest, ReturnsToPoolOnLastRef) {
    BufferPool pool(1, 8, 8, CV_8UC1, BufferPool::ExhaustPolicy::Fail);

    auto a = pool.acquire();
    ASSERT_TRUE(static_cast<bool>(a));
    const auto* addr_a = a.get();

    EXPECT_EQ(pool.available_approx(), 0U);
    EXPECT_FALSE(static_cast<bool>(pool.try_acquire()));

    auto copy = a;
    a.reset();
    EXPECT_EQ(pool.available_approx(), 0U); // still held by copy

    copy.reset();
    EXPECT_EQ(pool.available_approx(), 1U);

    const auto b = pool.acquire();
    ASSERT_TRUE(static_cast<bool>(b));
    EXPECT_EQ(b.get(), addr_a);             // same pooled object reused
}

TEST(BufferPoolTest, FailPolicyReturnsNullWhenExhausted) {
    BufferPool pool(2, 8, 8, CV_8UC1, BufferPool::ExhaustPolicy::Fail);

    auto a       = pool.acquire();
    const auto b = pool.acquire();
    ASSERT_TRUE(static_cast<bool>(a));
    ASSERT_TRUE(static_cast<bool>(b));
    EXPECT_EQ(pool.available_approx(), 0U);

    EXPECT_FALSE(static_cast<bool>(pool.try_acquire()));

    a.reset();
    EXPECT_EQ(pool.available_approx(), 1U);
    const auto c = pool.try_acquire();
    ASSERT_TRUE(static_cast<bool>(c));
}

TEST(BufferPoolTest, AllocatePolicyOverflowDoesNotReturnToPool) {
    BufferPool pool(1, 8, 8, CV_8UC1, BufferPool::ExhaustPolicy::Allocate);

    auto pooled = pool.acquire();
    ASSERT_TRUE(static_cast<bool>(pooled));
    EXPECT_EQ(pool.available_approx(), 0U);

    auto overflow = pool.acquire();
    ASSERT_TRUE(static_cast<bool>(overflow));
    EXPECT_NE(pooled.get(), overflow.get());
    const auto allocs_after_acquire = pool.overflow_allocations();
    EXPECT_GE(allocs_after_acquire, 1U);

    overflow.reset();
    EXPECT_EQ(pool.available_approx(), 0U);

    const auto allocs_before_try = pool.overflow_allocations();
    auto overflow2               = pool.try_acquire();
    ASSERT_TRUE(static_cast<bool>(overflow2));
    EXPECT_NE(overflow2.get(), pooled.get());
    EXPECT_GE(pool.overflow_allocations(), allocs_before_try + 1);
    EXPECT_EQ(pool.available_approx(), 0U);

    pooled.reset();
    EXPECT_EQ(pool.available_approx(), 1U);
}

TEST(BufferPoolTest, AllocateOnExhaustion) {
    BufferPool pool(0, 8, 8, CV_8UC1, BufferPool::ExhaustPolicy::Allocate);
    const auto a = pool.acquire();
    ASSERT_TRUE(static_cast<bool>(a));
    EXPECT_GE(pool.overflow_allocations(), 1U);
}

TEST(BufferPoolTest, BlockPolicyBlocksUntilReturn) {
    BufferPool pool(1, 8, 8, CV_8UC1, BufferPool::ExhaustPolicy::Block);

    auto a = pool.acquire();
    ASSERT_TRUE(static_cast<bool>(a));

    std::atomic attempting{false};
    std::atomic acquired{false};
    std::chrono::nanoseconds waited{0};

    std::thread t([&] {
        attempting.store(true, std::memory_order_release);
        const auto start = std::chrono::steady_clock::now();
        const auto b     = pool.acquire();
        waited           = std::chrono::steady_clock::now() - start;
        ASSERT_TRUE(static_cast<bool>(b));
        acquired.store(true, std::memory_order_release);
    });

    while (!attempting.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    std::this_thread::sleep_for(30ms);
    EXPECT_FALSE(acquired.load(std::memory_order_acquire));

    a.reset();
    t.join();

    EXPECT_TRUE(acquired.load(std::memory_order_acquire));
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(waited).count(), 20);
}

TEST(BufferPoolTest, NoDuplicateCheckoutUnderContention) {
    constexpr std::size_t kCapacity    = 4;
    constexpr int kThreads             = 8;
    constexpr int kIterationsPerThread = 200;

    BufferPool pool(kCapacity, 8, 8, CV_8UC1, BufferPool::ExhaustPolicy::Block);

    std::mutex mu;
    std::unordered_set<const void*> in_flight;
    std::atomic violated{false};

    std::latch start(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            start.count_down();
            start.wait();

            for (int i = 0; i < kIterationsPerThread; ++i) {
                auto img = pool.acquire();
                ASSERT_TRUE(static_cast<bool>(img));

                {
                    std::lock_guard lock(mu);
                    if (!in_flight.insert(img.get()).second) {
                        violated.store(true, std::memory_order_relaxed);
                    }
                }

                std::this_thread::sleep_for(200us);

                {
                    std::lock_guard lock(mu);
                    in_flight.erase(img.get());
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_FALSE(violated.load(std::memory_order_acquire));
}

TEST(LatencyHistogramTest, PercentileClampsAndNoOverflow) {
    LatencyHistogram hist;

    hist.record(10'000); // 10us
    hist.record(10'000);
    hist.record(10'000);

    EXPECT_EQ(hist.percentile(0.0), hist.min());
    EXPECT_EQ(hist.percentile(1.0), hist.max());

    // 10us falls into bucket (10us-20us) => midpoint 15us
    EXPECT_EQ(hist.p50(), 15'000U);

    hist.record(200'000'000);     // 200ms (open-ended last bucket)
    const auto p99 = hist.p99();
    EXPECT_GT(p99, 0U);
    EXPECT_LE(p99, 100'000'000U); // last bucket returns lower bound (100ms)
}

TEST(LatencyHistogramTest, ConcurrentRecordCounts) {
    LatencyHistogram hist;

    constexpr int kThreads    = 4;
    constexpr int kIterations = 50'000;

    std::latch start(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            start.count_down();
            start.wait();
            for (int i = 0; i < kIterations; ++i) {
                hist.record(100 + static_cast<std::uint64_t>(t));
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(hist.count(), static_cast<std::uint64_t>(kThreads) * kIterations);
    const auto snap   = hist.snapshot();
    std::uint64_t sum = 0;
    for (const auto c : snap) {
        sum += c;
    }
    EXPECT_EQ(sum, hist.count());
}

TEST(LatencyStatsTest, ConcurrentRecordAndPrintIsSafe) {
    LatencyStats stats;

    constexpr int kThreads    = 4;
    constexpr int kIterations = 20'000;

    std::ostringstream sink;
    CoutRedirect redirect(sink);

    std::atomic stop{false};
    std::thread printer([&] {
        while (!stop.load(std::memory_order_acquire)) {
            stats.print("test");
        }
    });

    std::vector<std::thread> writers;
    writers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&] {
            for (int i = 0; i < kIterations; ++i) {
                stats.record(1000);
            }
        });
    }

    for (auto& th : writers) {
        th.join();
    }
    stop.store(true, std::memory_order_release);
    printer.join();

    EXPECT_EQ(stats.samples.size(), static_cast<std::size_t>(kThreads) * kIterations);
}

TEST(ImageBufferTest, WrapOwnedKeepsOwnerAlive) {
    std::atomic freed{0};
    auto owner = std::shared_ptr<void>(new int(123), [&](void* p) {
        freed.fetch_add(1, std::memory_order_relaxed);
        delete static_cast<int*>(p);
    });

    std::uint8_t data[64] = {};
    auto img              = ImageBuffer::wrap_owned(data, 8, 8, CV_8UC1, /*step=*/8, owner);

    owner.reset();
    EXPECT_EQ(freed.load(std::memory_order_acquire), 0);

    img.reset();
    EXPECT_EQ(freed.load(std::memory_order_acquire), 1);
}

struct GenData {
    std::uint64_t gen{0};
    std::array<std::uint64_t, 32> pad{};

    explicit GenData(const std::uint64_t g)
        : gen(g) {
        pad.fill(g);
    }

    [[nodiscard]] bool consistent() const noexcept {
        for (const auto v : pad) {
            if (v != gen) {
                return false;
            }
        }
        return true;
    }
};

TEST(SpmcTripleBufferTest, ConcurrentReadersSeeConsistentSnapshots) {
    auto [w, r] = SpmcTripleBuffer<GenData>::create();

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
