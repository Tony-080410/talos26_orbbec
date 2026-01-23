#pragma once

/**
 * @file
 * @brief ECS Application builder
 *
 * This module provides a simple builder pattern for constructing
 * ECS applications with resources and systems.
 *
 * ## Usage
 *
 * ```cpp
 * auto app = talos::app()
 *     .add_resource(Config{...})
 *     .add_system("camera", [](auto camera_in, auto frame_out) {
 *         // Process camera frame
 *     })
 *     .add_system("tracker", [](auto frame_in, auto targets_out) {
 *         // Track targets
 *     });
 *
 * app.build();
 *
 * while (true) {
 *     app.run_once();
 * }
 * ```
 *
 * ## Differences from Scheduler
 *
 * - **Single-threaded**: `run_once()` executes all systems sequentially
 * - **Simpler**: No dependency graph or thread management
 * - **Debug-friendly**: Useful for testing and offline processing
 *
 * ## Thread safety
 *
 * The App is not thread-safe. Configure and run from a single thread.
 */

#include "system.hpp"
#include "world.hpp"

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

namespace talos {

// ============================================================================
// App: ECS Application Builder
// ============================================================================

/**
 * @brief ECS Application builder for single-threaded execution
 *
 * Provides a simple builder pattern for constructing and running
 * ECS applications without the complexity of threading and scheduling.
 */
class App {
    World world_;
    std::vector<std::unique_ptr<SystemBase>> systems_;
    bool built_ = false;

public:
    App()                      = default;
    App(const App&)            = delete;
    App& operator=(const App&) = delete;
    App(App&&) noexcept        = default;
    App& operator=(App&&)      = default;

    // ==================== Resource Registration ====================

    /**
     * @brief Add a resource to the world
     *
     * Resources are immutable state accessible by all systems.
     * Only one instance of each type can exist.
     *
     * ## Template parameters
     *
     * - `T`: resource type
     *
     * ## Parameters
     *
     * - `resource`: the resource to add (moved)
     *
     * ## Returns
     *
     * Reference to this App for chaining
     */
    template <typename T>
    App& add_resource(T resource) {
        world_.insert_resource(std::move(resource));
        return *this;
    }

    /**
     * @brief Add an atomic value
     *
     * Atomics provide thread-safe shared state between systems.
     *
     * ## Template parameters
     *
     * - `T`: atomic type (must be trivially copyable)
     *
     * ## Parameters
     *
     * - `initial_value`: initial value of the atomic
     *
     * ## Returns
     *
     * Reference to this App for chaining
     */
    template <typename T>
    App& add_atomic(T initial_value) {
        world_.insert_atomic(std::move(initial_value));
        return *this;
    }

    // ==================== System Registration ====================

    /**
     * @brief Add a system with default policy (pool_compute)
     *
     * ## Template parameters
     *
     * - `F`: callable type for the system function
     *
     * ## Parameters
     *
     * - `name`: human-readable system name
     * - `func`: the system function to execute
     *
     * ## Returns
     *
     * Reference to this App for chaining
     */
    template <typename F>
    App& add_system(std::string name, F&& func) {
        systems_.push_back(make_system<F, default_policy>(std::move(name), std::forward<F>(func)));
        return *this;
    }

    /**
     * @brief Add a system with explicit policy
     *
     * ## Template parameters
     *
     * - `Policy`: execution policy (fixed_rate, fixed_rate_silent, pool_compute)
     * - `F`: callable type for the system function
     *
     * ## Parameters
     *
     * - `name`: human-readable system name
     * - `func`: the system function to execute
     *
     * ## Returns
     *
     * Reference to this App for chaining
     */
    template <typename Policy, typename F>
    App& add_system(std::string name, F&& func) {
        systems_.push_back(make_system<F, Policy>(std::move(name), std::forward<F>(func)));
        return *this;
    }

    // ==================== Build & Run ====================

    /**
     * @brief Build the application
     *
     * Binds all systems to the world, preparing them for execution.
     * Safe to call multiple times (idempotent).
     */
    void build() {
        if (built_)
            return;
        for (const auto& sys : systems_) {
            sys->bind(world_);
        }
        built_ = true;
    }

    /**
     * @brief Run all systems once
     *
     * Executes each system in registration order.
     * Automatically builds if not already built.
     */
    void run_once() {
        if (!built_)
            build();
        for (const auto& sys : systems_) {
            sys->run(world_);
        }
    }

    // ==================== Inspection ====================

    /**
     * @brief Print registered systems info
     *
     * Outputs to stdout in a formatted table.
     */
    void print_systems() const {
        for (const auto& sys : systems_) {
            const auto& m   = sys->meta();
            auto policy_str = "pool";
            if (m.policy.is_fixed_rate()) {
                policy_str = m.policy.notifies ? "fixed_rate" : "silent";
            }
            printf(
                "  [%s] policy:%s spsc:%zu spmc:%zu res:%zu res_mut:%zu\n", m.name.c_str(),
                policy_str, m.spsc_channels.size(), m.spmc_channels.size(), m.reads.size(),
                m.writes.size());
        }
    }

    /**
     * @brief Get number of registered systems
     *
     * ## Returns
     *
     * Count of systems
     */
    [[nodiscard]] std::size_t system_count() const noexcept { return systems_.size(); }

    /**
     * @brief Get system by index
     *
     * ## Parameters
     *
     * - `idx`: system index (0 to system_count()-1)
     *
     * ## Returns
     *
     * Const reference to the system
     *
     * ## Panics
     *
     * Throws if idx is out of bounds.
     */
    [[nodiscard]] const SystemBase& system_at(const std::size_t idx) const {
        return *systems_.at(idx);
    }

    // ==================== World Access ====================

    /**
     * @brief Get mutable reference to the world
     *
     * ## Returns
     *
     * Mutable world reference
     */
    [[nodiscard]] World& world() noexcept { return world_; }

    /**
     * @brief Get const reference to the world
     *
     * ## Returns
     *
     * Const world reference
     */
    [[nodiscard]] const World& world() const noexcept { return world_; }

    // ==================== System Metadata Access (for scheduler) ====================

    /**
     * @brief Get metadata for all systems
     *
     * ## Returns
     *
     * Vector of pointers to system metadata
     */
    [[nodiscard]] std::vector<const SystemMeta*> all_meta() const {
        std::vector<const SystemMeta*> result;
        result.reserve(systems_.size());
        for (const auto& sys : systems_) {
            result.push_back(&sys->meta());
        }
        return result;
    }

    /**
     * @brief Get systems filtered by policy kind
     *
     * ## Parameters
     *
     * - `kind`: policy kind to filter by
     *
     * ## Returns
     *
     * Vector of (metadata, system) pairs
     */
    [[nodiscard]] std::vector<std::pair<const SystemMeta*, SystemBase*>>
        systems_by_policy(const PolicyKind kind) const {
        std::vector<std::pair<const SystemMeta*, SystemBase*>> result;
        for (auto& sys : systems_) {
            if (sys->meta().policy.kind == kind) {
                result.emplace_back(&sys->meta(), sys.get());
            }
        }
        return result;
    }
};

/**
 * @brief Create a new App
 *
 * Factory function for creating an App instance.
 *
 * ## Returns
 *
 * New empty App
 */
[[nodiscard]] inline App app() { return App{}; }

} // namespace talos
