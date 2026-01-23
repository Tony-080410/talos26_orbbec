#pragma once

#include "ecs/system_meta.hpp"

#include "demangle.hpp"
#include <cstdlib>
#include <cxxabi.h>
#include <mutex>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace talos::rclcompat {

// ============================================================================
// OwnershipRegistry: Track channel ownership
// ============================================================================

/**
 * @brief Registry for tracking channel ownership
 *
 * Ensures single-publisher-per-channel constraint at runtime.
 * Thread-safe for registration operations.
 */
class OwnershipRegistry {
public:
    /**
     * @brief Register a channel owner
     *
     * ## Parameters
     *
     * - `key`: channel identifier
     * - `owner`: name of the owning system
     *
     * ## Thread safety
     *
     * Thread-safe (mutex protected)
     */
    void register_owner(ChannelKey key, std::string owner) noexcept {
        std::lock_guard lock(lock_);
        if (owners_.contains(key)) {
            SPDLOG_CRITICAL(
                "Channel {}@{} is already owned by '{}', but '{}' attempted to register",
                detail::demangle(key.type.name()), detail::demangle(key.topic.name()), owners_[key],
                owner);
            std::abort();
        }
        owners_[key] = std::move(owner);
    }

    /**
     * @brief Try to claim a publisher handle
     *
     * A channel can only be claimed once (enforces move-only semantics at runtime).
     *
     * ## Parameters
     *
     * - `key`: channel identifier
     *
     * ## Returns
     *
     * `true` if claim succeeded, `false` if already claimed
     *
     * ## Thread safety
     *
     * Thread-safe (mutex protected)
     */
    [[nodiscard]] bool try_claim(ChannelKey key) noexcept {
        std::lock_guard lock(lock_);
        if (claimed_.contains(key)) {
            return false;
        }
        claimed_.insert(key);
        return true;
    }

    /**
     * @brief Release a claimed publisher handle
     *
     * Called when Publisher is moved or destroyed.
     *
     * ## Parameters
     *
     * - `key`: channel identifier
     *
     * ## Thread safety
     *
     * Thread-safe (mutex protected)
     */
    void release_claim(ChannelKey key) noexcept {
        std::lock_guard lock(lock_);
        claimed_.erase(key);
    }

    /**
     * @brief Assert that a channel is owned by a specific system
     *
     * ## Parameters
     *
     * - `key`: channel identifier
     * - `caller`: name of the caller for error reporting
     *
     * ## Panics
     *
     * Aborts if the channel is not owned by the caller
     */
    void assert_owner(ChannelKey key, std::string_view caller) const noexcept {
        std::lock_guard lock(lock_);
        auto it = owners_.find(key);
        if (it == owners_.end()) {
            SPDLOG_CRITICAL(
                "Channel {}@{} is not registered, but '{}' attempted to access", key.type.name(),
                key.topic.name(), caller);
            std::abort();
        }
    }

private:
    mutable std::mutex lock_;
    std::unordered_map<ChannelKey, std::string, ChannelKeyHash> owners_;
    std::unordered_set<ChannelKey, ChannelKeyHash> claimed_;
};

// ============================================================================
// Callback Context: Track current executing callback
// ============================================================================

namespace detail {} // namespace detail

} // namespace talos::rclcompat
