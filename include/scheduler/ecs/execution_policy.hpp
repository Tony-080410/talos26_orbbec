#pragma once

#include <cstdint>
#include <type_traits>

namespace talos {

// ============================================================================
// Execution Policy Tags
// ============================================================================

/// Exclusive with notify: dedicated thread, notifies scheduler after each tick
/// Use for: data sources that should trigger compute (e.g., Camera)
///
/// @tparam FreqHz      Execution frequency in Hz, 0 = run as fast as possible
/// @tparam Affinity    CPU core affinity, -1 = no binding
/// @tparam Priority    Thread priority, 0 = default
template <std::uint32_t FreqHz = 0, std::int32_t Affinity = -1, std::int32_t Priority = 0>
struct exclusive {
    static constexpr std::uint32_t frequency_hz   = FreqHz;
    static constexpr std::int32_t cpu_affinity    = Affinity;
    static constexpr std::int32_t thread_priority = Priority;
    static constexpr bool notifies                = true;
};

/// Exclusive silent: dedicated thread, does NOT notify scheduler
/// Use for: high-frequency data sources (e.g., IMU) that update silently
///
/// @tparam FreqHz      Execution frequency in Hz, 0 = run as fast as possible
/// @tparam Affinity    CPU core affinity, -1 = no binding
/// @tparam Priority    Thread priority, 0 = default
template <std::uint32_t FreqHz = 0, std::int32_t Affinity = -1, std::int32_t Priority = 0>
struct exclusive_silent {
    static constexpr std::uint32_t frequency_hz   = FreqHz;
    static constexpr std::int32_t cpu_affinity    = Affinity;
    static constexpr std::int32_t thread_priority = Priority;
    static constexpr bool notifies                = false;
};

/// Pool compute: TBB thread pool, triggered by exclusive notify
struct pool_compute {};

/// Default policy
using default_policy = pool_compute;

// ============================================================================
// Policy Traits
// ============================================================================

namespace detail {

template <typename T>
struct is_exclusive_policy : std::false_type {};

template <std::uint32_t F, std::int32_t A, std::int32_t P>
struct is_exclusive_policy<exclusive<F, A, P>> : std::true_type {};

template <std::uint32_t F, std::int32_t A, std::int32_t P>
struct is_exclusive_policy<exclusive_silent<F, A, P>> : std::true_type {};

template <typename T>
struct is_pool_policy : std::is_same<T, pool_compute> {};

template <typename T>
struct is_notifying_policy : std::false_type {};

template <std::uint32_t F, std::int32_t A, std::int32_t P>
struct is_notifying_policy<exclusive<F, A, P>> : std::true_type {};

template <typename T>
struct is_silent_policy : std::false_type {};

template <std::uint32_t F, std::int32_t A, std::int32_t P>
struct is_silent_policy<exclusive_silent<F, A, P>> : std::true_type {};

} // namespace detail

template <typename T>
inline constexpr bool is_exclusive_policy_v = detail::is_exclusive_policy<T>::value;

template <typename T>
inline constexpr bool is_pool_policy_v = detail::is_pool_policy<T>::value;

template <typename T>
inline constexpr bool is_notifying_policy_v = detail::is_notifying_policy<T>::value;

template <typename T>
inline constexpr bool is_silent_policy_v = detail::is_silent_policy<T>::value;

// ============================================================================
// Runtime Policy Info
// ============================================================================

enum class PolicyKind : std::uint8_t {
    Exclusive,
    PoolCompute,
};

struct PolicyInfo {
    PolicyKind kind              = PolicyKind::PoolCompute;
    std::uint32_t frequency_hz   = 0;
    std::int32_t cpu_affinity    = -1;
    std::int32_t thread_priority = 0;
    bool notifies                = false;

    [[nodiscard]] constexpr bool is_exclusive() const noexcept {
        return kind == PolicyKind::Exclusive;
    }
    [[nodiscard]] constexpr bool is_pool() const noexcept {
        return kind == PolicyKind::PoolCompute;
    }
};

template <typename Policy>
[[nodiscard]] constexpr auto make_policy_info() noexcept -> PolicyInfo {
    if constexpr (is_exclusive_policy_v<Policy>) {
        return PolicyInfo{
            .kind            = PolicyKind::Exclusive,
            .frequency_hz    = Policy::frequency_hz,
            .cpu_affinity    = Policy::cpu_affinity,
            .thread_priority = Policy::thread_priority,
            .notifies        = Policy::notifies,
        };
    } else {
        return PolicyInfo{.kind = PolicyKind::PoolCompute};
    }
}

} // namespace talos
