#pragma once

#include <atomic>
#include <memory>

namespace fcs::L2 {

/// Config storage with thread-safe access
/// Eliminates atomic_load/store boilerplate duplication
template <typename Config>
class ConfigHolder {
public:
    using ConfigPtr = std::shared_ptr<const Config>;

    ConfigPtr config_;

    [[nodiscard]] ConfigPtr get_config() const noexcept { return std::atomic_load(&config_); }

    void set_config(ConfigPtr cfg) noexcept { std::atomic_store(&config_, std::move(cfg)); }
};

} // namespace fcs::L2
