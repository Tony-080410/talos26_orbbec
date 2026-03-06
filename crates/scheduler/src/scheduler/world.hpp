#pragma once

#include "demangle.hpp"
#include "system/components.hpp"
#include "system/system_meta.hpp"

#include <any>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>
#include <typeindex>
#include <unordered_map>

namespace talos::scheduler {

// ============================================================================
// Resource Wrapper
// ============================================================================

template <typename T>
struct Resource {
    T value;

    explicit Resource(T val) noexcept
        : value(std::move(val)) {}
};

// ============================================================================
// Design Note: Type Erasure for Heterogeneous Storage
// ============================================================================
// World needs to store arbitrary types (resources, channels, atomics) in maps.
// std::any is used here as a type-erasure pattern - all accesses go through
// templates that know the exact type, providing compile-time type safety at
// the call site while allowing heterogeneous storage internally.
//
// Alternative approaches considered:
// - Template-based storage: Would require users to specify all types upfront
// - std::variant: Not scalable for open-ended type sets
// - std::shared_ptr<void>: Same type-safety issues as std::any
//
// This is an internal implementation detail; the public API is fully type-safe.
// ============================================================================

// ============================================================================
// SPSC Storage: holds split writer and reader (wrapped in shared_ptr for any)
// ============================================================================

template <typename T>
struct SpscStorage {
    using Channel    = primitive::SpscChannel<T>;
    using Writer     = Channel::Writer;
    using Reader     = Channel::Reader;
    using value_type = T;

    std::shared_ptr<Writer> writer;
    std::shared_ptr<Reader> reader;

    static std::shared_ptr<SpscStorage> create() {
        auto ch         = primitive::make_spsc_channel<T>();
        auto [w, r]     = ch.split();
        auto storage    = std::make_shared<SpscStorage>();
        storage->writer = std::make_shared<Writer>(std::move(w));
        storage->reader = std::make_shared<Reader>(std::move(r));
        return storage;
    }
};

// ============================================================================
// SPMC Storage: holds channel and readers
// ============================================================================

template <typename T>
struct SpmcStorage {
    using Channel    = primitive::SpmcChannel<T>;
    using Reader     = Channel::Reader;
    using value_type = T;

    std::shared_ptr<Channel> channel;
    std::vector<std::shared_ptr<Reader>> readers;

    static std::shared_ptr<SpmcStorage> create() {
        auto storage     = std::make_shared<SpmcStorage>();
        storage->channel = std::make_shared<Channel>(primitive::make_spmc_channel<T>());
        return storage;
    }

    std::shared_ptr<Reader> add_reader() {
        auto reader = std::make_shared<Reader>(channel->clone_reader());
        readers.push_back(reader);
        return reader;
    }
};

// ============================================================================
// World: Resource + Channel Container
// ============================================================================
using namespace system;
using namespace talos::scheduler::detail;

class World {
public:
    // ==================== Resource API ====================

    template <typename T>
    void insert_resource(T resource) {
        resources_[typeid(T)] = std::make_shared<Resource<T>>(std::move(resource));
    }

    template <typename T>
    [[nodiscard]] bool has_resource() const noexcept {
        return resources_.contains(typeid(T));
    }

    /// Get read-only resource. Panics if resource not found (bind-phase error).
    template <typename T>
    [[nodiscard]] res<T> get_res() const noexcept {
        return get_resource_helper<T, res<T>>(
            [](auto* storage) { return res<T>{&storage->value}; });
    }

    /// Get mutable resource. Panics if resource not found (bind-phase error).
    template <typename T>
    [[nodiscard]] res_mut<T> get_res_mut() noexcept {
        return get_resource_helper<T, res_mut<T>>(
            [](auto* storage) { return res_mut<T>{&storage->value}; });
    }

    // ==================== Type-Dispatched Access ====================
    //
    // Unified get() that uses tag dispatch (zero if-constexpr, zero pasta)
    //
    // Usage: world.get<res<Config>>(); world.get<spsc<Image, Tag>>();

    template <typename ComponentT>
    [[nodiscard]] auto get() noexcept {
        return get_impl(ComponentT{});
    }

private:
private:
    // Unified channel accessor: one template for all member-pointer-based access
    template <typename Ret, typename Topic, typename Storage, auto MemberPtr>
    [[nodiscard]] auto get_channel_member() noexcept {
        auto* s = get_channel_storage<Storage, Topic>();
        return Ret{(s->*MemberPtr).get()};
    }

    // Channel implementations (3 cases use unified get_channel_member, 1 special case)
    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_impl(spsc<T, Topic>) noexcept {
        return get_channel_member<spsc<T, Topic>, Topic, SpscStorage<T>, &SpscStorage<T>::reader>();
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_impl(spsc_mut<T, Topic>) noexcept {
        return get_channel_member<
            spsc_mut<T, Topic>, Topic, SpscStorage<T>, &SpscStorage<T>::writer>();
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_impl(spmc<T, Topic>) noexcept {
        // Special case: spmc reader calls add_reader()
        auto* s = get_channel_storage<SpmcStorage<T>, Topic>();
        return spmc<T, Topic>{s->add_reader().get()};
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_impl(spmc_mut<T, Topic>) noexcept {
        return get_channel_member<
            spmc_mut<T, Topic>, Topic, SpmcStorage<T>, &SpmcStorage<T>::channel>();
    }

    template <typename T>
    [[nodiscard]] auto get_impl(res<T>) noexcept {
        return get_res<T>();
    }

    template <typename T>
    [[nodiscard]] auto get_impl(res_mut<T>) noexcept {
        return get_res_mut<T>();
    }

    // Original storage accessor (kept for get_channel_storage compatibility)
    template <typename Storage, typename Topic = DefaultTopic>
    [[nodiscard]] auto* get_channel_storage() noexcept {
        using T = typename Storage::value_type;
        const ChannelKey key{typeid(T), typeid(Topic)};
        auto& storage_map = select_storage_map<Storage>();
        if (!storage_map.contains(key)) {
            storage_map[key] = Storage::create();
        }
        return std::any_cast<std::shared_ptr<Storage>>(storage_map.at(key)).get();
    }

private:
    std::unordered_map<std::type_index, std::any> resources_;
    std::unordered_map<ChannelKey, std::any, ChannelKeyHash> spsc_storage_;
    std::unordered_map<ChannelKey, std::any, ChannelKeyHash> spmc_storage_;

    /// Helper: Get resource storage with unified error handling
    template <typename T, typename Return, typename MapFn>
    [[nodiscard]] Return get_resource_helper(MapFn&& map_fn) const noexcept {
        const auto it = resources_.find(typeid(T));
        if (it == resources_.end()) [[unlikely]] {
            SPDLOG_CRITICAL("Resource not found: {}", demangle(typeid(T).name()));
            std::abort();
        }
        auto storage = std::any_cast<std::shared_ptr<Resource<T>>>(it->second);
        return map_fn(storage.get());
    }

    /// Helper: Select the right storage map based on storage type
    template <typename Storage>
    auto& select_storage_map() noexcept {
        if constexpr (std::is_same_v<Storage, SpscStorage<typename Storage::value_type>>) {
            return spsc_storage_;
        } else {
            return spmc_storage_;
        }
    }
};

template <typename T, typename Topic = DefaultTopic>
using publish = spmc_mut<T, Topic>;

template <typename T, typename Topic = DefaultTopic>
using subscribe = spmc<T, Topic>;

} // namespace talos::scheduler

// ============================================================================
// Convenience aliases for talos::
// ============================================================================

namespace talos {

template <typename T, typename Topic = scheduler::DefaultTopic>
using subscribe = scheduler::spmc<T, Topic>;

template <typename T, typename Topic = scheduler::DefaultTopic>
using publish = scheduler::spmc_mut<T, Topic>;

} // namespace talos
