#include "scheduler/world.hpp"

namespace talos::scheduler {

void World::freeze_resource_structure() noexcept {
    resource_structure_frozen_.store(true, std::memory_order_release);
}

bool World::resource_structure_frozen() const noexcept {
    return resource_structure_frozen_.load(std::memory_order_acquire);
}

void World::freeze_resource_identity() noexcept { freeze_resource_structure(); }

bool World::resource_identity_frozen() const noexcept { return resource_structure_frozen(); }

void World::open_channel_binding() noexcept {
    channel_binding_depth_.fetch_add(1, std::memory_order_acq_rel);
}

void World::close_channel_binding() noexcept {
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

bool World::channel_binding_open() const noexcept {
    return channel_binding_depth_.load(std::memory_order_acquire) != 0;
}

void World::ensure_channel_binding_open() const noexcept {
    if (channel_binding_depth_.load(std::memory_order_acquire) == 0) [[unlikely]] {
        panic("Channel access through World is only allowed during scheduler bind()");
    }
}

std::weak_ptr<WorldLifetimeToken> World::lifetime_token() const noexcept { return lifetime_token_; }

} // namespace talos::scheduler
