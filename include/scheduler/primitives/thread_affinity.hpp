#pragma once

#include <thread>
#include <cstdint>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#elif __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace scheduler {

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
    /**
     * @brief 将当前线程绑定到指定CPU核心
     *
     * @param core_id CPU核心ID (0-based)
     * @return true 如果成功，false 如果失败
     */
    static bool pin_to_core(int core_id) noexcept {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);

        pthread_t thread = pthread_self();
        int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        return result == 0;

#elif __APPLE__
        // macOS不支持严格的CPU绑定，使用affinity tag作为提示
        // 相同tag的线程会倾向于调度到同一核心组
        thread_affinity_policy_data_t policy = {core_id + 1};  // tag从1开始
        thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());

        kern_return_t result = thread_policy_set(
            mach_thread,
            THREAD_AFFINITY_POLICY,
            reinterpret_cast<thread_policy_t>(&policy),
            THREAD_AFFINITY_POLICY_COUNT
        );
        return result == KERN_SUCCESS;
#else
        (void)core_id;
        return false;
#endif
    }

    /**
     * @brief 将指定std::thread绑定到CPU核心
     */
    static bool pin_thread_to_core(std::thread& thread, int core_id) noexcept {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);

        int result = pthread_setaffinity_np(
            thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        return result == 0;

#elif __APPLE__
        // macOS: 需要在线程内部设置，这里返回false
        // 应该在线程启动时调用pin_to_core
        (void)thread;
        (void)core_id;
        return false;
#else
        (void)thread;
        (void)core_id;
        return false;
#endif
    }

    /**
     * @brief 设置线程为实时优先级 (需要root权限)
     *
     * @param priority 优先级 (1-99, 99最高)
     * @return true 如果成功
     */
    static bool set_realtime_priority(int priority = 50) noexcept {
#ifdef __linux__
        struct sched_param param;
        param.sched_priority = priority;
        return pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0;

#elif __APPLE__
        // macOS: 使用thread_policy_set设置时间约束
        thread_time_constraint_policy_data_t policy;
        policy.period = 0;
        policy.computation = 50000;  // 50us计算时间
        policy.constraint = 100000;  // 100us约束
        policy.preemptible = false;

        thread_port_t mach_thread = pthread_mach_thread_np(pthread_self());
        kern_return_t result = thread_policy_set(
            mach_thread,
            THREAD_TIME_CONSTRAINT_POLICY,
            reinterpret_cast<thread_policy_t>(&policy),
            THREAD_TIME_CONSTRAINT_POLICY_COUNT
        );
        return result == KERN_SUCCESS;
#else
        (void)priority;
        return false;
#endif
    }

    /**
     * @brief 获取可用CPU核心数
     */
    static unsigned int get_num_cores() noexcept {
        return std::thread::hardware_concurrency();
    }

    /**
     * @brief 获取当前线程运行的CPU核心ID
     */
    static int get_current_core() noexcept {
#ifdef __linux__
        return sched_getcpu();
#elif __APPLE__
        // macOS没有直接的API，返回-1
        return -1;
#else
        return -1;
#endif
    }
};

}  // namespace scheduler
