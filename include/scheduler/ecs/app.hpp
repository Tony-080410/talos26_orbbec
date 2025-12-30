#pragma once

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

    template <typename T>
    App& add_resource(T resource) {
        world_.insert_resource(std::move(resource));
        return *this;
    }

    template <typename T>
    App& add_atomic(T initial_value) {
        world_.insert_atomic(std::move(initial_value));
        return *this;
    }

    // ==================== System Registration ====================

    /// Add system with default policy (pool_compute)
    template <typename F>
    App& add_system(std::string name, F&& func) {
        systems_.push_back(make_system<F, default_policy>(std::move(name), std::forward<F>(func)));
        return *this;
    }

    /// Add system with explicit policy
    template <typename Policy, typename F>
    App& add_system(std::string name, F&& func) {
        systems_.push_back(make_system<F, Policy>(std::move(name), std::forward<F>(func)));
        return *this;
    }

    // ==================== Build & Run ====================

    void build() {
        if (built_)
            return;
        // Pre-create channels based on system metadata could go here
        built_ = true;
    }

    void run_once() {
        if (!built_)
            build();
        for (const auto& sys : systems_) {
            sys->run(world_);
        }
    }

    // ==================== Inspection ====================

    void print_systems() const {
        for (const auto& sys : systems_) {
            const auto& m   = sys->meta();
            auto policy_str = "pool";
            if (m.policy.is_exclusive()) {
                policy_str = m.policy.notifies ? "exclusive" : "silent";
            }
            printf(
                "  [%s] policy:%s spsc:%zu spmc:%zu res:%zu res_mut:%zu\n", m.name.c_str(),
                policy_str, m.spsc_channels.size(), m.spmc_channels.size(), m.reads.size(),
                m.writes.size());
        }
    }

    [[nodiscard]] std::size_t system_count() const noexcept { return systems_.size(); }

    [[nodiscard]] const SystemBase& system_at(const std::size_t idx) const {
        return *systems_.at(idx);
    }

    // ==================== World Access ====================

    [[nodiscard]] World& world() noexcept { return world_; }
    [[nodiscard]] const World& world() const noexcept { return world_; }

    // ==================== System Metadata Access (for scheduler) ====================

    [[nodiscard]] std::vector<const SystemMeta*> all_meta() const {
        std::vector<const SystemMeta*> result;
        result.reserve(systems_.size());
        for (const auto& sys : systems_) {
            result.push_back(&sys->meta());
        }
        return result;
    }

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

[[nodiscard]] inline App app() { return App{}; }

} // namespace talos
