#pragma once

/**
 * @file
 * @brief System metadata and type utilities
 *
 * This module provides:
 * - Type traits for extracting channel and resource types from system parameters
 * - Metadata structures for describing systems at runtime
 * - Channel key identification for dependency tracking
 */

#include "channel_types.hpp"
#include "execution_policy.hpp"

#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

namespace talos {

// ============================================================================
// Channel Key: (Type, Topic) for unique identification
// ============================================================================

/**
 * @brief Uniquely identifies a channel by (data type, topic) pair
 *
 * Used for tracking channel usage across systems and detecting conflicts.
 */
struct ChannelKey {
    std::type_index type;
    std::type_index topic;

    constexpr bool operator==(const ChannelKey&) const noexcept = default;
};

/**
 * @brief Hash function for ChannelKey
 */
struct ChannelKeyHash {
    constexpr std::size_t operator()(const ChannelKey& k) const noexcept {
        return std::hash<std::type_index>{}(k.type) ^ std::hash<std::type_index>{}(k.topic) << 1;
    }
};

// ============================================================================
// Function Traits
// ============================================================================

namespace detail {

template <typename F>
struct function_traits;

/**
 * @brief Extract function traits from a function pointer
 */
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
    using return_type                  = R;
    using args_tuple                   = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
};

/**
 * @brief Extract function traits from a member function
 */
template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...)> : function_traits<R (*)(Args...)> {};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const> : function_traits<R (*)(Args...)> {};

/**
 * @brief Extract function traits from a lambda/functor
 */
template <typename F>
struct function_traits : function_traits<decltype(&F::operator())> {};

// ============================================================================
// Inner Type Extraction
// ============================================================================

/**
 * @brief Default: inner type is the type itself, topic is void
 */
template <typename T>
struct inner_type {
    using type  = T;
    using topic = void;
};

template <typename T>
concept topic_type = requires(T) {
    { sizeof(T) == 0 };
};

/**
 * @brief Extract inner type from spsc<T, Topic>
 */
template <typename T, typename Topic>
struct inner_type<spsc<T, Topic>> {
    using type  = T;
    using topic = Topic;
};

/**
 * @brief Extract inner type from spsc_mut<T, Topic>
 */
template <typename T, typename Topic>
struct inner_type<spsc_mut<T, Topic>> {
    using type  = T;
    using topic = Topic;
};

/**
 * @brief Extract inner type from spmc<T, Topic>
 */
template <typename T, typename Topic>
struct inner_type<spmc<T, Topic>> {
    using type  = T;
    using topic = Topic;
};

/**
 * @brief Extract inner type from spmc_mut<T, Topic>
 */
template <typename T, typename Topic>
struct inner_type<spmc_mut<T, Topic>> {
    using type  = T;
    using topic = Topic;
};

/**
 * @brief Extract inner type from atomic_ref<T, Topic>
 */
template <typename T, typename Topic>
struct inner_type<atomic_ref<T, Topic>> {
    using type  = T;
    using topic = Topic;
};

/**
 * @brief Extract inner type from res<T>
 */
template <typename T>
struct inner_type<res<T>> {
    using type  = T;
    using topic = void;
};

/**
 * @brief Extract inner type from res_mut<T>
 */
template <typename T>
struct inner_type<res_mut<T>> {
    using type  = T;
    using topic = void;
};

/**
 * @brief Extract inner type from local<T>
 */
template <typename T>
struct inner_type<local<T>> {
    using type  = T;
    using topic = void;
};

template <typename T>
using inner_type_t = inner_type<T>::type;

template <typename T>
using inner_topic_t = inner_type<T>::topic;

// ============================================================================
// Type Predicates
// ============================================================================

/**
 * @brief Check if type is an SPSC reader
 */
template <typename T>
struct is_spsc_reader : std::bool_constant<T::is_spsc && T::is_reader> {};

/**
 * @brief Check if type is an SPSC writer
 */
template <typename T>
struct is_spsc_writer : std::bool_constant<T::is_spsc && !T::is_reader> {};

/**
 * @brief Check if type is an SPMC reader
 */
template <typename T>
struct is_spmc_reader : std::bool_constant<T::is_spmc && T::is_reader> {};

/**
 * @brief Check if type is an SPMC writer
 */
template <typename T>
struct is_spmc_writer : std::bool_constant<T::is_spmc && !T::is_reader> {};

/**
 * @brief Check if type is an atomic reference
 */
template <typename T>
struct is_atomic_type : std::bool_constant<T::is_atomic> {};

/**
 * @brief Check if type is a resource (immutable read)
 */
template <typename T>
struct is_res_type : std::bool_constant<T::is_res> {};

/**
 * @brief Check if type is a mutable resource
 */
template <typename T>
struct is_res_mut_type : std::bool_constant<T::is_res_mut> {};

/**
 * @brief Check if type is a local variable (defaults to false if is_local not defined)
 */
template <typename T, typename = void>
struct is_local_type : std::false_type {};

template <typename T>
struct is_local_type<T, std::void_t<decltype(T::is_local)>> : std::bool_constant<T::is_local> {};

} // namespace detail

// ============================================================================
// System Metadata
// ============================================================================

/**
 * @brief Metadata for a single channel used by a system
 */
struct ChannelMeta {
    std::type_index type;
    std::type_index topic;
    bool is_reader;
    bool is_spsc;
    bool is_spmc;
};

/**
 * @brief Runtime metadata for a system
 *
 * Contains information about the system's policy, channels, and resources.
 */
struct SystemMeta {
    std::string name;
    PolicyInfo policy;
    std::vector<ChannelMeta> spsc_channels;
    std::vector<ChannelMeta> spmc_channels;
    std::vector<std::type_index> atomics;
    std::vector<std::type_index> reads;
    std::vector<std::type_index> writes;
};

namespace detail {

/**
 * @brief Extract channel/resource metadata from a single parameter
 */
template <typename T>
constexpr void extract_one_param(SystemMeta& meta) noexcept {
    using Inner = inner_type_t<T>;
    using Topic = inner_topic_t<T>;

    if constexpr (is_spsc_reader<T>::value || is_spsc_writer<T>::value) {
        meta.spsc_channels.emplace_back(
            ChannelMeta{typeid(Inner), typeid(Topic), T::is_reader, true, false});
    } else if constexpr (is_spmc_reader<T>::value || is_spmc_writer<T>::value) {
        meta.spmc_channels.emplace_back(
            ChannelMeta{typeid(Inner), typeid(Topic), T::is_reader, false, true});
    } else if constexpr (is_atomic_type<T>::value) {
        meta.atomics.emplace_back(typeid(Inner));
    } else if constexpr (is_res_type<T>::value) {
        meta.reads.emplace_back(typeid(Inner));
    } else if constexpr (is_res_mut_type<T>::value) {
        meta.writes.emplace_back(typeid(Inner));
    } else if constexpr (is_local_type<T>::value) {
        // Local variables don't participate in dependency analysis
    }
}

} // namespace detail

/**
 * @brief Extract system metadata from a function type
 *
 * Analyzes the function's parameters to determine:
 * - What channels and resources the system uses
 * - Which are read vs write
 * - The execution policy
 *
 * ## Template parameters
 *
 * - `F`: function type (or lambda/functor)
 * - `Policy`: execution policy (defaults to default_policy)
 *
 * ## Parameters
 *
 * - `name`: system name for identification
 *
 * ## Returns
 *
 * SystemMeta describing the system
 */
template <typename F, typename Policy = default_policy>
constexpr SystemMeta extract_system_meta(std::string name) noexcept {
    SystemMeta meta;
    meta.name   = std::move(name);
    meta.policy = make_policy_info<Policy>();

    using traits = detail::function_traits<F>;
    using args   = traits::args_tuple;

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (detail::extract_one_param<std::tuple_element_t<Is, args>>(meta), ...);
    }(std::make_index_sequence<traits::arity>{});

    return meta;
}

} // namespace talos
