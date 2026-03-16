#pragma once

#include "toml/detail/common.hpp"
#include "toml/detail/core_deserialize.hpp"

namespace toml_helper {

template <typename T>
[[nodiscard]] std::expected<T, std::string> read(const toml::table& table, std::string_view key) {
    return Deserialize<T>::read(table, key);
}

template <typename T>
[[nodiscard]] std::expected<std::optional<T>, std::string>
    read_optional(const toml::table& table, std::string_view key) {
    return Deserialize<T>::read_optional(table, key);
}

[[nodiscard]] inline std::expected<void, std::string>
    ensure_all_read(const toml::table& table, std::string_view context = {}) noexcept {
    return detail::error_if_unread(table, context);
}

template <typename T>
[[nodiscard]] std::expected<T, std::string> take(const toml::table& table, std::string_view key) {
    return Deserialize<T>::take(table, key);
}

template <typename T>
[[nodiscard]] std::expected<std::optional<T>, std::string>
    take_optional(const toml::table& table, std::string_view key) {
    return Deserialize<T>::take_optional(table, key);
}

template <ReadFrom T>
[[nodiscard]] std::expected<void, std::string>
    read_flatten_into(const toml::table& table, T& out, std::string_view context) noexcept {
    auto result = out.read_from(table);
    if (result) {
        return {};
    }
    return std::unexpected(detail::prefixed_error(context, result.error()));
}

#define READ_OPT(tbl, val)                                               \
    do {                                                                 \
        auto res = toml_helper::take_optional<decltype(val)>(tbl, #val); \
        if (!res) {                                                      \
            return std::unexpected(res.error());                         \
        }                                                                \
        if (*res) {                                                      \
            (val) = std::move(**res);                                    \
        }                                                                \
    } while (0)

#define READ(tbl, val)                                          \
    do {                                                        \
        auto res = toml_helper::take<decltype(val)>(tbl, #val); \
        if (!res) {                                             \
            return std::unexpected(res.error());                \
        }                                                       \
        (val) = std::move(*res);                                \
    } while (0)

#define READ_FLATTEN(tbl, val)                                         \
    do {                                                               \
        auto res = toml_helper::read_flatten_into((tbl), (val), #val); \
        if (!res) {                                                    \
            return std::unexpected(res.error());                       \
        }                                                              \
    } while (0)

[[nodiscard]] inline std::expected<toml::table, std::string>
    merge_configs(const toml::table& base, const toml::table& override_table) noexcept {
    toml::table result = base;

    for (const auto& [key, override_node] : override_table) {
        auto* base_node               = result.get(key);
        const auto* base_subtable     = base_node ? base_node->as_table() : nullptr;
        const auto* override_subtable = override_node.as_table();

        if (base_subtable && override_subtable) {
            auto merged = merge_configs(*base_subtable, *override_subtable);
            if (!merged) {
                return std::unexpected(merged.error());
            }
            result.insert_or_assign(key, std::move(*merged));
            continue;
        }

        result.insert_or_assign(key, override_node);
    }

    return result;
}

} // namespace toml_helper
