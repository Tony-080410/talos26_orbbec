#pragma once

#include <cstdint>
#include <cstring>
#include <expected>
#include <fmt/format.h>
#include <string>
#include <thread>

#ifdef __APPLE__
# include <mach/mach.h>
# include <mach/thread_policy.h>
# include <pthread.h>
#elif __linux__
# include <pthread.h>
# include <sched.h>
#endif

namespace talos::primitive {

using CoreIdResult   = std::expected<std::uint32_t, std::string>;
using AffinityResult = std::expected<void, std::string>;

/**
 * @brief 线程亲和性工具 - CPU绑定
 *
 * 性能优化原理:
 * - 减少缓存失效 (线程固定在同一核心)
 * - 减少上下文切换开销
 * - 更好的L1/L2缓存利用率
 *
 * 平台支持:
 * - Linux: pthread_setaffinity_np (精确绑核)
 * - macOS: thread_policy_set (QoS提示，非严格绑定)
 */
class ThreadAffinity {
public:
    ThreadAffinity() = delete; // 纯静态工具类，禁止实例化

    /**
     * @brief 将当前线程绑定到指定CPU核心
     *
     * @param core_id CPU核心ID (0-based)
     * @return AffinityResult 成功返回void，失败返回具体错误
     */
    [[nodiscard]] static auto pin_to_core(const std::uint32_t core_id) noexcept -> AffinityResult {
        if (core_id >= get_num_cores()) {
            return std::unexpected(
                fmt::format("pin_to_core: core_id {} >= num_cores {}", core_id, get_num_cores()));
        }

#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(core_id), &cpuset);

        pthread_t thread = pthread_self();
        const int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

        if (result != 0) {
            return std::unexpected(
                fmt::format(
                    "pin_to_core: pthread_setaffinity_np(core {}) failed: {}", core_id,
                    std::strerror(result)));
        }
        return {};
#elif defined(__APPLE__) && defined(__arm64__)
        // Apple Silicon: THREAD_AFFINITY_POLICY not supported (kern_return=46)
        // The OS handles P-core/E-core scheduling; affinity tags are Intel-only.
        return {};
#elif defined(__APPLE__)
        // Intel Mac: affinity tag is a scheduling hint (not strict binding)
        thread_affinity_policy_data_t policy = {static_cast<integer_t>(core_id + 1)};
        const thread_port_t mach_thread      = pthread_mach_thread_np(pthread_self());

        const kern_return_t kr = thread_policy_set(
            mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&policy),
            THREAD_AFFINITY_POLICY_COUNT);
        if (kr != KERN_SUCCESS) {
            return std::unexpected(
                fmt::format(
                    "pin_to_core: thread_policy_set(affinity, tag={}) failed: {} (kern_return={})",
                    core_id + 1, mach_error_string(kr), kr));
        }
        return {};
#else
        return std::unexpected("pin_to_core: unsupported platform");
#endif
    }

    [[nodiscard]] static auto
        pin_thread_to_core(std::thread& thread, const std::uint32_t core_id) noexcept
        -> AffinityResult {
        if (core_id >= get_num_cores()) {
            return std::unexpected(
                fmt::format(
                    "pin_thread_to_core: core_id {} >= num_cores {}", core_id, get_num_cores()));
        }

#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(core_id), &cpuset);

        const int result =
            pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);

        if (result != 0) {
            return std::unexpected(
                fmt::format(
                    "pin_thread_to_core: pthread_setaffinity_np(core {}) failed: {}", core_id,
                    std::strerror(result)));
        }
        return {};

#elif defined(__APPLE__)
        static_cast<void>(thread);
        static_cast<void>(core_id);
        return std::unexpected(
            "pin_thread_to_core: macOS does not support setting affinity from external thread");
#else
        static_cast<void>(thread);
        static_cast<void>(core_id);
        return std::unexpected("pin_thread_to_core: unsupported platform");
#endif
    }

    /**
     * @brief 实时优先级配置
     */
    struct RealtimeConfig {
        std::uint8_t priority        = 50;     // Linux: 1-99, 99最高
        std::uint32_t computation_ns = 50000;  // macOS: 计算时间（纳秒）
        std::uint32_t constraint_ns  = 100000; // macOS: 约束时间（纳秒）
        bool preemptible             = false;  // macOS: 是否可抢占
    };

    /**
     * @brief 设置线程为实时优先级 (需要root权限)
     *
     * @param config 实时优先级配置
     * @return AffinityResult 成功返回void，失败返回具体错误
     */
    [[nodiscard]] static auto set_realtime_priority(
        const RealtimeConfig config = RealtimeConfig{50, 50000, 100000, false}) noexcept
        -> AffinityResult {
#ifdef __linux__
        struct sched_param param{};
        param.sched_priority = static_cast<int>(config.priority);

        const int result = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        if (result != 0) {
            return std::unexpected(
                fmt::format(
                    "set_realtime_priority: pthread_setschedparam(SCHED_FIFO, prio={}) failed: {}",
                    config.priority, std::strerror(result)));
        }
        return {};

#elif defined(__APPLE__)
        thread_time_constraint_policy_data_t policy{};
        policy.period      = 0;
        policy.computation = static_cast<std::uint32_t>(config.computation_ns);
        policy.constraint  = static_cast<std::uint32_t>(config.constraint_ns);
        policy.preemptible = config.preemptible ? TRUE : FALSE;

        const thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
        const kern_return_t kr          = thread_policy_set(
            mach_thread, THREAD_TIME_CONSTRAINT_POLICY, reinterpret_cast<thread_policy_t>(&policy),
            THREAD_TIME_CONSTRAINT_POLICY_COUNT);

        if (kr != KERN_SUCCESS) {
            return std::unexpected(
                fmt::format(
                    "set_realtime_priority: thread_policy_set(time_constraint, comp={}ns, "
                    "constr={}ns) failed: {} (kern_return={})",
                    config.computation_ns, config.constraint_ns, mach_error_string(kr), kr));
        }
        return {};
#else
        static_cast<void>(config);
        return std::unexpected("set_realtime_priority: unsupported platform");
#endif
    }

    /**
     * @brief 获取可用CPU核心数
     */
    [[nodiscard]] static auto get_num_cores() noexcept -> std::uint32_t {
        return std::thread::hardware_concurrency();
    }

    /**
     * @brief 获取当前线程运行的CPU核心ID
     *
     * @return CoreIdResult 成功返回核心ID，失败返回错误
     */
    [[nodiscard]] static auto get_current_core() noexcept -> CoreIdResult {
#ifdef __linux__
        const int cpu = sched_getcpu();
        if (cpu < 0) {
            return std::unexpected("get_current_core: sched_getcpu() failed");
        }
        return static_cast<std::uint32_t>(cpu);
#elif defined(__APPLE__)
        return std::unexpected("get_current_core: not available on macOS");
#else
        return std::unexpected("get_current_core: unsupported platform");
#endif
    }
};

} // namespace talos::primitive
