#pragma once

#include "ecs/execution_policy.hpp"
#include "ecs/system_meta.hpp"
#include <string>

namespace talos::rclcompat {

// ============================================================================
// SystemMetaBuilder: Runtime construction of SystemMeta
// ============================================================================

/**
 * @brief Builder for runtime construction of SystemMeta
 *
 * Used by rclcompat systems that need to construct metadata at runtime
 * rather than extracting it from function signatures.
 *
 * ## Example
 *
 * ```cpp
 * auto meta = SystemMetaBuilder("camera_pub")
 *     .policy(make_policy_info<pool_compute>())
 *     .add_spmc_writer(typeid(ImageFrame), typeid(CameraTag))
 *     .build();
 * ```
 */
class SystemMetaBuilder {
public:
    /**
     * @brief Construct a builder with the system name
     *
     * ## Parameters
     *
     * - `name`: system name for identification
     */
    explicit SystemMetaBuilder(std::string name) noexcept { meta_.name = std::move(name); }

    /**
     * @brief Set the execution policy
     *
     * ## Parameters
     *
     * - `p`: policy info
     *
     * ## Returns
     *
     * Reference to this builder for chaining
     */
    SystemMetaBuilder& policy(PolicyInfo p) noexcept {
        meta_.policy = p;
        return *this;
    }

    /**
     * @brief Add an SPMC writer channel
     *
     * ## Parameters
     *
     * - `type`: data type index
     * - `topic`: topic type index
     *
     * ## Returns
     *
     * Reference to this builder for chaining
     */
    SystemMetaBuilder& add_spmc_writer(std::type_index type, std::type_index topic) noexcept {
        meta_.spmc_channels.push_back(
            ChannelMeta{
                .type      = type,
                .topic     = topic,
                .is_reader = false,
                .is_spsc   = false,
                .is_spmc   = true,
            });
        return *this;
    }

    /**
     * @brief Add an SPMC reader channel
     *
     * ## Parameters
     *
     * - `type`: data type index
     * - `topic`: topic type index
     *
     * ## Returns
     *
     * Reference to this builder for chaining
     */
    SystemMetaBuilder& add_spmc_reader(std::type_index type, std::type_index topic) noexcept {
        meta_.spmc_channels.push_back(
            ChannelMeta{
                .type      = type,
                .topic     = topic,
                .is_reader = true,
                .is_spsc   = false,
                .is_spmc   = true,
            });
        return *this;
    }

    /**
     * @brief Add an SPSC writer channel
     *
     * ## Parameters
     *
     * - `type`: data type index
     * - `topic`: topic type index
     *
     * ## Returns
     *
     * Reference to this builder for chaining
     */
    SystemMetaBuilder& add_spsc_writer(std::type_index type, std::type_index topic) noexcept {
        meta_.spsc_channels.push_back(
            ChannelMeta{
                .type      = type,
                .topic     = topic,
                .is_reader = false,
                .is_spsc   = true,
                .is_spmc   = false,
            });
        return *this;
    }

    /**
     * @brief Add an SPSC reader channel
     *
     * ## Parameters
     *
     * - `type`: data type index
     * - `topic`: topic type index
     *
     * ## Returns
     *
     * Reference to this builder for chaining
     */
    SystemMetaBuilder& add_spsc_reader(std::type_index type, std::type_index topic) noexcept {
        meta_.spsc_channels.push_back(
            ChannelMeta{
                .type      = type,
                .topic     = topic,
                .is_reader = true,
                .is_spsc   = true,
                .is_spmc   = false,
            });
        return *this;
    }

    /**
     * @brief Build the SystemMeta (rvalue version)
     *
     * ## Returns
     *
     * The constructed SystemMeta
     */
    [[nodiscard]] SystemMeta build() && noexcept { return std::move(meta_); }

    /**
     * @brief Build the SystemMeta (lvalue version)
     *
     * ## Returns
     *
     * Copy of the constructed SystemMeta
     */
    [[nodiscard]] SystemMeta build() const& noexcept { return meta_; }

private:
    SystemMeta meta_;
};

}; // namespace talos::rclcompat
