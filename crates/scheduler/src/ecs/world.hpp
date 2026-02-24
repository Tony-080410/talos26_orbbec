#pragma once

#include "channel_types.hpp"
#include "rclcompat/demangle.hpp"
#include "system_meta.hpp"

#include <any>
#include <atomic>
#include <cxxabi.h>
#include <expected>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>
#include <typeindex>
#include <unordered_map>

namespace talos {

// ============================================================================
// Versioned Resource Wrapper
// ============================================================================

template <typename T>
struct VersionedResource {
    T value;
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> version{0};

    explicit VersionedResource(T val) noexcept
        : value(std::move(val)) {}

    uint64_t update(T new_value) noexcept {
        value = std::move(new_value);
        return version.fetch_add(1, std::memory_order_release) + 1;
    }
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
    using Channel = primitive::SpscChannel<T>;
    using Writer  = Channel::Writer;
    using Reader  = Channel::Reader;

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
    using Channel = primitive::SpmcChannel<T>;
    using Reader  = Channel::Reader;

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

class World {
public:
    std::unordered_map<std::type_index, std::any> resources_;
    std::unordered_map<ChannelKey, std::any, ChannelKeyHash> spsc_storage_;
    std::unordered_map<ChannelKey, std::any, ChannelKeyHash> spmc_storage_;
    std::unordered_map<std::type_index, std::any> atomics_;

    // ==================== Resource API ====================

    template <typename T>
    void insert_resource(T resource) {
        resources_[typeid(T)] = std::make_shared<VersionedResource<T>>(std::move(resource));
    }

    template <typename T>
    [[nodiscard]] bool has_resource() const noexcept {
        return resources_.contains(typeid(T));
    }

    /// Get read-only resource. Panics if resource not found (bind-phase error).
    template <typename T>
    [[nodiscard]] res<T> get_res() const noexcept {
        const auto it = resources_.find(typeid(T));
        if (it == resources_.end()) [[unlikely]] {
            SPDLOG_CRITICAL(
                "Resource not found: {}", rclcompat::detail::demangle(typeid(T).name()));
            std::abort();
        }
        auto storage = std::any_cast<std::shared_ptr<VersionedResource<T>>>(it->second);
        return res<T>{&storage->value, &storage->version, 0};
    }

    /// Get mutable resource. Panics if resource not found (bind-phase error).
    template <typename T>
    [[nodiscard]] res_mut<T> get_res_mut() noexcept {
        const auto it = resources_.find(typeid(T));
        if (it == resources_.end()) [[unlikely]] {
            SPDLOG_CRITICAL(
                "Resource not found: {}", rclcompat::detail::demangle(typeid(T).name()));
            std::abort();
        }
        auto storage = std::any_cast<std::shared_ptr<VersionedResource<T>>>(it->second);
        return res_mut<T>{&storage->value, &storage->version, 0};
    }

    template <typename T>
    [[nodiscard]] std::expected<uint64_t, std::string> update_resource(T new_value) {
        auto it = resources_.find(typeid(T));
        if (it == resources_.end()) {
            return std::unexpected("Resource not found: " + std::string(typeid(T).name()));
        }

        auto storage = std::any_cast<std::shared_ptr<VersionedResource<T>>>(it->second);
        return storage->update(std::move(new_value));
    }

    // ==================== SPSC Channel API ====================

    template <typename T, typename Topic = DefaultTopic>
    void ensure_spsc() {
        if (const ChannelKey key{typeid(T), typeid(Topic)}; !spsc_storage_.contains(key)) {
            spsc_storage_[key] = SpscStorage<T>::create();
        }
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] spsc<T, Topic> get_spsc_reader() noexcept {
        const ChannelKey key{typeid(T), typeid(Topic)};
        ensure_spsc<T, Topic>();
        auto storage_ptr = std::any_cast<std::shared_ptr<SpscStorage<T>>>(spsc_storage_.at(key));
        return spsc<T, Topic>{storage_ptr->reader.get()};
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] spsc_mut<T, Topic> get_spsc_writer() noexcept {
        const ChannelKey key{typeid(T), typeid(Topic)};
        ensure_spsc<T, Topic>();
        auto storage_ptr = std::any_cast<std::shared_ptr<SpscStorage<T>>>(spsc_storage_.at(key));
        return spsc_mut<T, Topic>{storage_ptr->writer.get()};
    }

    // ==================== SPMC Channel API ====================

    template <typename T, typename Topic = DefaultTopic>
    void ensure_spmc() {
        if (const ChannelKey key{typeid(T), typeid(Topic)}; !spmc_storage_.contains(key)) {
            spmc_storage_[key] = SpmcStorage<T>::create();
        }
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] spmc<T, Topic> get_spmc_reader() noexcept {
        const ChannelKey key{typeid(T), typeid(Topic)};
        ensure_spmc<T, Topic>();
        auto storage_ptr = std::any_cast<std::shared_ptr<SpmcStorage<T>>>(spmc_storage_.at(key));
        auto reader      = storage_ptr->add_reader();
        return spmc<T, Topic>{reader.get()};
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] spmc_mut<T, Topic> get_spmc_writer() noexcept {
        const ChannelKey key{typeid(T), typeid(Topic)};
        ensure_spmc<T, Topic>();
        auto storage_ptr = std::any_cast<std::shared_ptr<SpmcStorage<T>>>(spmc_storage_.at(key));
        return spmc_mut<T, Topic>{storage_ptr->channel.get()};
    }

    // ==================== Atomic API ====================

    template <typename T>
    void insert_atomic(T initial_value) {
        atomics_[typeid(T)] = std::make_shared<std::atomic<T>>(initial_value);
    }

    template <typename T, typename Topic = DefaultTopic>
    [[nodiscard]] atomic_ref<T, Topic> get_atomic() noexcept {
        const auto it = atomics_.find(typeid(T));
        if (it == atomics_.end()) [[unlikely]] {
            SPDLOG_CRITICAL("Atomic not found: {}", rclcompat::detail::demangle(typeid(T).name()));
            std::abort();
        }
        auto& ptr = std::any_cast<std::shared_ptr<std::atomic<T>>&>(it->second);
        return atomic_ref<T, Topic>{ptr.get()};
    }
};

template <typename T, typename Topic = DefaultTopic>
using publish = spmc_mut<T, Topic>;

template <typename T, typename Topic = DefaultTopic>
using subscribe = spmc<T, Topic>;

} // namespace talos
