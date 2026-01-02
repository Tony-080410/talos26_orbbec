#pragma once

#include "system_meta.hpp"
#include "world.hpp"

#include <functional>
#include <memory>
#include <utility>

namespace talos {

// ============================================================================
// System Base Interface
// ============================================================================

class SystemBase {
public:
    virtual ~SystemBase()                  = default;
    virtual bool run(World& world)         = 0; // Returns true if any channel was written
    virtual const SystemMeta& meta() const = 0;
};

// ============================================================================
// Argument Construction
// ============================================================================

namespace detail {

template <typename T>
auto make_arg(World& world) {
    using Inner = inner_type_t<T>;
    using Tag   = inner_tag_t<T>;

    if constexpr (is_res_type<T>::value) {
        return world.get_res<Inner>();
    } else if constexpr (is_res_mut_type<T>::value) {
        return world.get_res_mut<Inner>();
    } else if constexpr (is_spsc_reader<T>::value) {
        return world.get_spsc_reader<Inner, Tag>();
    } else if constexpr (is_spsc_writer<T>::value) {
        return world.get_spsc_writer<Inner, Tag>();
    } else if constexpr (is_spmc_reader<T>::value) {
        return world.get_spmc_reader<Inner, Tag>();
    } else if constexpr (is_spmc_writer<T>::value) {
        return world.get_spmc_writer<Inner, Tag>();
    } else if constexpr (is_atomic_type<T>::value) {
        return world.get_atomic<Inner, Tag>();
    } else {
        static_assert(sizeof(T) == 0, "Unknown parameter type");
    }
}

} // namespace detail

// ============================================================================
// Concrete System Implementation
// ============================================================================

template <typename F, typename Policy = default_policy>
class System : public SystemBase {
    using traits     = detail::function_traits<F>;
    using args_tuple = traits::args_tuple;

    F func_;
    SystemMeta meta_;
    std::optional<args_tuple> cached_args_;
    bool written_ = false;

    // Recursively set written_flag_ for all writer arguments
    template <std::size_t I = 0>
    void bind_written_flags() {
        if constexpr (I < std::tuple_size_v<args_tuple>) {
            using ArgType = std::tuple_element_t<I, args_tuple>;
            if constexpr (
                detail::is_spsc_writer<ArgType>::value || detail::is_spmc_writer<ArgType>::value) {
                std::get<I>(*cached_args_).written_flag_ = &written_;
            }
            bind_written_flags<I + 1>();
        }
    }

public:
    System(std::string name, F func)
        : func_(std::move(func))
        , meta_(extract_system_meta<F, Policy>(std::move(name))) {}

    bool run(World& world) override {
        // Cache arguments on first run (readers/writers created once, reused)
        if (!cached_args_) {
            cached_args_ = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return args_tuple{detail::make_arg<std::tuple_element_t<Is, args_tuple>>(world)...};
            }(std::make_index_sequence<traits::arity>{});
            bind_written_flags();
        }

        written_ = false;
        std::apply(func_, *cached_args_);
        return written_;
    }

    const SystemMeta& meta() const override { return meta_; }
};

// ============================================================================
// Type-erased System Runner (for scheduler)
// ============================================================================

using SystemRunner = std::function<void(World&)>;

template <typename F, typename Policy = default_policy>
[[nodiscard]] std::unique_ptr<SystemBase> make_system(std::string name, F&& func) {
    return std::make_unique<System<std::decay_t<F>, Policy>>(
        std::move(name), std::forward<F>(func));
}

} // namespace talos
