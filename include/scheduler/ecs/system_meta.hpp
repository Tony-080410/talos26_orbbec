#pragma once

#include "channel_types.hpp"
#include "execution_policy.hpp"

#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

namespace talos {

// ============================================================================
// Channel Key: (Type, Tag) for unique identification
// ============================================================================

struct ChannelKey {
    std::type_index type;
    std::type_index tag;

    bool operator==(const ChannelKey&) const = default;
};

struct ChannelKeyHash {
    std::size_t operator()(const ChannelKey& k) const noexcept {
        return std::hash<std::type_index>{}(k.type) ^ std::hash<std::type_index>{}(k.tag) << 1;
    }
};

// ============================================================================
// Function Traits
// ============================================================================

namespace detail {

template <typename F>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
    using return_type                  = R;
    using args_tuple                   = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...)> : function_traits<R (*)(Args...)> {};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const> : function_traits<R (*)(Args...)> {};

template <typename F>
struct function_traits : function_traits<decltype(&F::operator())> {};

// ============================================================================
// Inner Type Extraction
// ============================================================================

template <typename T>
struct inner_type {
    using type = T;
    using tag  = void;
};

template <typename T, typename Tag>
struct inner_type<spsc<T, Tag>> {
    using type = T;
    using tag  = Tag;
};
template <typename T, typename Tag>
struct inner_type<spsc_mut<T, Tag>> {
    using type = T;
    using tag  = Tag;
};
template <typename T, typename Tag>
struct inner_type<spmc<T, Tag>> {
    using type = T;
    using tag  = Tag;
};
template <typename T, typename Tag>
struct inner_type<spmc_mut<T, Tag>> {
    using type = T;
    using tag  = Tag;
};
template <typename T, typename Tag>
struct inner_type<atomic_ref<T, Tag>> {
    using type = T;
    using tag  = Tag;
};
template <typename T>
struct inner_type<res<T>> {
    using type = T;
    using tag  = void;
};
template <typename T>
struct inner_type<res_mut<T>> {
    using type = T;
    using tag  = void;
};

template <typename T>
using inner_type_t = inner_type<T>::type;

template <typename T>
using inner_tag_t = inner_type<T>::tag;

// ============================================================================
// Type Predicates
// ============================================================================

template <typename T>
struct is_spsc_reader : std::bool_constant<T::is_spsc && T::is_reader> {};

template <typename T>
struct is_spsc_writer : std::bool_constant<T::is_spsc && !T::is_reader> {};

template <typename T>
struct is_spmc_reader : std::bool_constant<T::is_spmc && T::is_reader> {};

template <typename T>
struct is_spmc_writer : std::bool_constant<T::is_spmc && !T::is_reader> {};

template <typename T>
struct is_atomic_type : std::bool_constant<T::is_atomic> {};

template <typename T>
struct is_res_type : std::bool_constant<T::is_res> {};

template <typename T>
struct is_res_mut_type : std::bool_constant<T::is_res_mut> {};

} // namespace detail

// ============================================================================
// System Metadata
// ============================================================================

struct ChannelMeta {
    std::type_index type;
    std::type_index tag;
    bool is_reader;
    bool is_spsc;
    bool is_spmc;
};

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

template <typename T>
void extract_one_param(SystemMeta& meta) {
    using Inner = inner_type_t<T>;
    using Tag   = inner_tag_t<T>;

    if constexpr (is_spsc_reader<T>::value || is_spsc_writer<T>::value) {
        meta.spsc_channels.emplace_back(
            ChannelMeta{typeid(Inner), typeid(Tag), T::is_reader, true, false});
    } else if constexpr (is_spmc_reader<T>::value || is_spmc_writer<T>::value) {
        meta.spmc_channels.emplace_back(
            ChannelMeta{typeid(Inner), typeid(Tag), T::is_reader, false, true});
    } else if constexpr (is_atomic_type<T>::value) {
        meta.atomics.emplace_back(typeid(Inner));
    } else if constexpr (is_res_type<T>::value) {
        meta.reads.emplace_back(typeid(Inner));
    } else if constexpr (is_res_mut_type<T>::value) {
        meta.writes.emplace_back(typeid(Inner));
    }
}

} // namespace detail

template <typename F, typename Policy = default_policy>
SystemMeta extract_system_meta(std::string name) {
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
