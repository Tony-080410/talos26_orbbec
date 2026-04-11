#pragma once

#include "demangle.hpp"
#include "error.hpp"
#include "system/components.hpp"
#include "system/system_meta.hpp"

#include <any>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace talos::scheduler {

namespace rclcompat {
template <typename T, typename Topic>
struct PubSlot;
} // namespace rclcompat

// ============================================================================
// Resource Wrapper
// ============================================================================

template <typename T>
struct Resource {
    T value;

    Resource() noexcept(std::is_nothrow_default_constructible_v<T>)
        requires std::default_initializable<T>
        = default;

    template <typename... Args>
    explicit Resource(std::in_place_t, Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args&&...>)
        : value(std::forward<Args>(args)...) {}

    template <typename U>
    explicit Resource(U&& val) noexcept(std::is_nothrow_constructible_v<T, U&&>)
        : value(std::forward<U>(val)) {}
};

struct WorldLifetimeToken {};

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

    std::unique_ptr<Writer> writer;
    std::unique_ptr<Reader> reader;
    bool writer_claimed = false;
    bool reader_claimed = false;

    static std::shared_ptr<SpscStorage> create() {
        auto ch         = primitive::make_spsc_channel<T>();
        auto [w, r]     = ch.split();
        auto storage    = std::make_shared<SpscStorage>();
        storage->writer = std::make_unique<Writer>(std::move(w));
        storage->reader = std::make_unique<Reader>(std::move(r));
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
    void insert_resource(T&& resource) {
        using U = std::remove_cvref_t<T>;
        ensure_resource_structure_mutable<U>();
        resources_[typeid(U)] = std::make_shared<Resource<U>>(std::forward<T>(resource));
    }

    template <typename T, typename... Args>
    [[nodiscard]] T& emplace_resource(Args&&... args) {
        ensure_resource_structure_mutable<T>();
        auto storage = std::make_shared<Resource<T>>(std::in_place, std::forward<Args>(args)...);
        auto& value  = storage->value;
        resources_[typeid(T)] = std::move(storage);
        return value;
    }

    template <typename T>
    void unsafe_insert_resource(T&& resource) {
        using U               = std::remove_cvref_t<T>;
        resources_[typeid(U)] = std::make_shared<Resource<U>>(std::forward<T>(resource));
    }

    template <typename T, typename... Args>
    [[nodiscard]] T& unsafe_emplace_resource(Args&&... args) {
        auto storage = std::make_shared<Resource<T>>(std::in_place, std::forward<Args>(args)...);
        auto& value  = storage->value;
        resources_[typeid(T)] = std::move(storage);
        return value;
    }

    void freeze_resource_structure() noexcept {
        resource_structure_frozen_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool resource_structure_frozen() const noexcept {
        return resource_structure_frozen_.load(std::memory_order_acquire);
    }

    void freeze_resource_identity() noexcept { freeze_resource_structure(); }

    [[nodiscard]] bool resource_identity_frozen() const noexcept {
        return resource_structure_frozen();
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
    // Channel components are only valid during scheduler-controlled bind phases.
    //
    // Channel endpoints are bind-phase only. Calling channel getters outside scheduler bind()
    // aborts because it would mutate or claim graph-visible topology behind the scheduler's back.

    template <typename ComponentT>
    [[nodiscard]] auto get() noexcept {
        return get_impl(ComponentT{});
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_spsc_reader() noexcept {
        return get<spsc<T, Topic>>();
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_spsc_writer() noexcept {
        return get<spsc_mut<T, Topic>>();
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_spmc_reader() noexcept {
        return get<spmc<T, Topic>>();
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_spmc_writer() noexcept {
        return get<spmc_mut<T, Topic>>();
    }

private:
private:
    friend class Scheduler;
    template <typename T, typename Topic>
    friend struct rclcompat::PubSlot;

    void open_channel_binding() noexcept {
        channel_binding_depth_.fetch_add(1, std::memory_order_acq_rel);
    }

    void close_channel_binding() noexcept {
        auto depth = channel_binding_depth_.load(std::memory_order_acquire);
        while (true) {
            if (depth == 0) [[unlikely]] {
                panic("Channel binding scope underflow");
            }
            if (channel_binding_depth_.compare_exchange_weak(
                    depth, depth - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
        }
    }

    [[nodiscard]] bool channel_binding_open() const noexcept {
        return channel_binding_depth_.load(std::memory_order_acquire) != 0;
    }

    // Unified channel accessor: one template for all member-pointer-based access
    template <typename Ret, typename Topic, typename Storage, auto MemberPtr>
    [[nodiscard]] auto get_channel_member() noexcept {
        auto* s = get_channel_storage<Storage, Topic>();
        return Ret{(s->*MemberPtr).get()};
    }

    // Channel implementations (3 cases use unified get_channel_member, 1 special case)
    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_impl(spsc<T, Topic>) noexcept {
        auto* s = get_channel_storage<SpscStorage<T>, Topic>();
        if (s->reader_claimed) [[unlikely]] {
            SPDLOG_CRITICAL(
                "SPSC reader already bound: {}@{}", demangle(typeid(T).name()),
                demangle(typeid(Topic).name()));
            std::abort();
        }
        s->reader_claimed = true;
        return spsc<T, Topic>{s->reader.get()};
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] auto get_impl(spsc_mut<T, Topic>) noexcept {
        auto* s = get_channel_storage<SpscStorage<T>, Topic>();
        if (s->writer_claimed) [[unlikely]] {
            SPDLOG_CRITICAL(
                "SPSC writer already bound: {}@{}", demangle(typeid(T).name()),
                demangle(typeid(Topic).name()));
            std::abort();
        }
        s->writer_claimed = true;
        return spsc_mut<T, Topic>{s->writer.get()};
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
        ensure_channel_binding_open();
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
    std::atomic<bool> resource_structure_frozen_{false};
    std::atomic<std::uint32_t> channel_binding_depth_{0};
    std::shared_ptr<WorldLifetimeToken> lifetime_token_ = std::make_shared<WorldLifetimeToken>();

    template <typename T>
    void ensure_resource_structure_mutable() const noexcept {
        if (resource_structure_frozen_.load(std::memory_order_acquire)) [[unlikely]] {
            panic(
                "Mutating resource structure for '{}' after build() breaks scheduler invariants; "
                "use unsafe_insert_resource()/unsafe_emplace_resource() only if you need the "
                "explicit escape hatch",
                demangle(typeid(T).name()));
        }
    }

    void ensure_channel_binding_open() const noexcept {
        if (channel_binding_depth_.load(std::memory_order_acquire) == 0) [[unlikely]] {
            panic("Channel access through World is only allowed during scheduler bind()");
        }
    }

public:
    [[nodiscard]] std::weak_ptr<WorldLifetimeToken> lifetime_token() const noexcept {
        return lifetime_token_;
    }

private:
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
