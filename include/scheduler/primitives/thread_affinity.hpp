#pragma once

#include <expected>
#include <string_view>
#include <thread>

#ifdef __APPLE__
# include <mach/mach.h>
# include <mach/thread_policy.h>
# include <pthread.h>
#elif __linux__
# include <pthread.h>
# include <sched.h>
#endif

namespace primitive {

/**
 * @brief 线程亲和性操作的错误类型
 */
enum class AffinityError : std::uint8_t {
    PlatformNotSupported,       // 当前平台不支持此操作
    InvalidCoreId,              // 无效的核心ID
    SetAffinityFailed,          // 设置亲和性失败
    SetPriorityFailed,          // 设置优先级失败
    InsufficientPermission,     // 权限不足（如需要root）
    ExternalThreadNotSupported, // 不支持外部线程设置（macOS限制）
};

/**
 * @brief 获取错误描述
 */
[[nodiscard]] constexpr auto error_message(const AffinityError err) noexcept -> std::string_view {
    switch (err) {
    case AffinityError::PlatformNotSupported: return "platform does not support this operation";
    case AffinityError::InvalidCoreId: return "invalid CPU core ID specified";
    case AffinityError::SetAffinityFailed: return "failed to set thread affinity";
    case AffinityError::SetPriorityFailed: return "failed to set thread priority";
    case AffinityError::InsufficientPermission: return "insufficient permission (may require root)";
    case AffinityError::ExternalThreadNotSupported:
        return "cannot set affinity for external thread on this platform";
    }
    // 穷尽所有枚举值，此处不可达
    return "unknown error";
}

/**
 * @brief 获取当前核心ID的结果类型
 *
 * 使用 std::expected 替代裸 int + 魔数(-1)
 */
using CoreIdResult = std::expected<std::uint32_t, AffinityError>;

/**
 * @brief 无返回值操作的结果类型
 */
using AffinityResult = std::expected<void, AffinityError>;

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
            return std::unexpected(AffinityError::InvalidCoreId);
        }

#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(core_id), &cpuset);

        pthread_t thread = pthread_self();
        const int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

        if (result != 0) {
            return std::unexpected(AffinityError::SetAffinityFailed);
        }
        return {};
#elif defined(__APPLE__)
        // macOS不支持严格的CPU绑定，使用affinity tag作为提示
        // 相同tag的线程会倾向于调度到同一核心组
        thread_affinity_policy_data_t policy = {static_cast<integer_t>(core_id + 1)}; // tag从1开始
        const thread_port_t mach_thread      = pthread_mach_thread_np(pthread_self());

        const kern_return_t result = thread_policy_set(
            mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&policy),
            THREAD_AFFINITY_POLICY_COUNT);
        if (result != KERN_SUCCESS) {
            return std::unexpected(AffinityError::SetAffinityFailed);
        }
        return {};
#else
        static_cast<void>(core_id);
        return std::unexpected(AffinityError::PlatformNotSupported);
#endif
    }

    /**
     * @brief 将指定std::thread绑定到CPU核心
     *
     * @note macOS不支持从外部设置线程亲和性，必须在线程内部调用pin_to_core
     */
    [[nodiscard]] static auto
        pin_thread_to_core(std::thread& thread, const std::uint32_t core_id) noexcept
        -> AffinityResult {
        if (core_id >= get_num_cores()) {
            return std::unexpected(AffinityError::InvalidCoreId);
        }

#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<int>(core_id), &cpuset);

        const int result =
            pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);

        if (result != 0) {
            return std::unexpected(AffinityError::SetAffinityFailed);
        }
        return {};

#elifdef __APPLE__
        // macOS: 需要在线程内部设置
        static_cast<void>(thread);
        static_cast<void>(core_id);
        return std::unexpected(AffinityError::ExternalThreadNotSupported);
#else
        static_cast<void>(thread);
        static_cast<void>(core_id);
        return std::unexpected(AffinityError::PlatformNotSupported);
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
            // EPERM表示权限不足
            if (result == EPERM) {
                return std::unexpected(AffinityError::InsufficientPermission);
            }
            return std::unexpected(AffinityError::SetPriorityFailed);
        }
        return {};

#elifdef __APPLE__
        thread_time_constraint_policy_data_t policy{};
        policy.period      = 0;
        policy.computation = static_cast<std::uint32_t>(config.computation_ns);
        policy.constraint  = static_cast<std::uint32_t>(config.constraint_ns);
        policy.preemptible = config.preemptible ? TRUE : FALSE;

        const thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
        const kern_return_t result      = thread_policy_set(
            mach_thread, THREAD_TIME_CONSTRAINT_POLICY, reinterpret_cast<thread_policy_t>(&policy),
            THREAD_TIME_CONSTRAINT_POLICY_COUNT);

        if (result != KERN_SUCCESS) {
            return std::unexpected(AffinityError::SetPriorityFailed);
        }
        return {};
#else
        static_cast<void>(config);
        return std::unexpected(AffinityError::PlatformNotSupported);
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
            return std::unexpected(AffinityError::PlatformNotSupported);
        }
        return static_cast<std::uint32_t>(cpu);
#elifdef __APPLE__
        // macOS没有直接的API
        return std::unexpected(AffinityError::PlatformNotSupported);
#else
        return std::unexpected(AffinityError::PlatformNotSupported);
#endif
    }
};

} // namespace primitive
