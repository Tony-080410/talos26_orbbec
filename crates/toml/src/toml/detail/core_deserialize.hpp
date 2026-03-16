#pragma once

#include "toml/detail/common.hpp"

namespace toml_helper {

template <typename T, typename Enable = void>
struct Deserialize {
    [[nodiscard]] static std::expected<T, std::string> read(const toml::table&, std::string_view) {
        static_assert(
            detail::always_false_v<T>,
            "No toml_helper::Deserialize<T> specialization available for this type");
        return std::unexpected("No deserializer available");
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        read_optional(const toml::table& table, std::string_view key) {
        const toml::node* node = table.get(key);
        if (!node) {
            return std::optional<T>{};
        }
        return detail::as_optional(read(table, key));
    }

    [[nodiscard]] static std::expected<T, std::string>
        take(const toml::table& table, std::string_view key) {
        return detail::erase_on_success(read(table, key), table, key);
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        take_optional(const toml::table& table, std::string_view key) {
        return detail::erase_on_success_optional(read_optional(table, key), table, key);
    }
};

template <TomlScalarValue T>
struct Deserialize<T, void> {
    [[nodiscard]] static std::expected<T, std::string>
        read(const toml::table& table, std::string_view key) {
        return detail::read_required<T>(table, key, [&](const toml::node& node) {
            return detail::parse_scalar_node<T>(node, key);
        });
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        read_optional(const toml::table& table, std::string_view key) {
        return detail::read_optional_value<T>(table, key, [&](const toml::node& node) {
            return detail::parse_scalar_node<T>(node, key);
        });
    }

    [[nodiscard]] static std::expected<T, std::string>
        take(const toml::table& table, std::string_view key) {
        return detail::erase_on_success(read(table, key), table, key);
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        take_optional(const toml::table& table, std::string_view key) {
        return detail::erase_on_success_optional(read_optional(table, key), table, key);
    }
};

template <typename T>
requires(std::is_enum_v<T>) struct Deserialize<T, void> {
    [[nodiscard]] static std::expected<T, std::string>
        read(const toml::table& table, std::string_view key) {
        return detail::read_required<T>(table, key, [&](const toml::node& node) {
            return detail::parse_enum_node<T>(node, key);
        });
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        read_optional(const toml::table& table, std::string_view key) {
        return detail::read_optional_value<T>(table, key, [&](const toml::node& node) {
            return detail::parse_enum_node<T>(node, key);
        });
    }

    [[nodiscard]] static std::expected<T, std::string>
        take(const toml::table& table, std::string_view key) {
        return detail::erase_on_success(read(table, key), table, key);
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        take_optional(const toml::table& table, std::string_view key) {
        return detail::erase_on_success_optional(read_optional(table, key), table, key);
    }
};

template <ReadFrom T>
struct Deserialize<T, void> {
    [[nodiscard]] static std::expected<T, std::string>
        read(const toml::table& table, std::string_view key) {
        auto parsed = detail::parse_read_from_key<T>(table, key);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        return std::move(parsed->value);
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        read_optional(const toml::table& table, std::string_view key) {
        const toml::node* node = table.get(key);
        if (!node) {
            return std::optional<T>{};
        }

        auto parsed = detail::parse_read_from_node<T>(*node, key);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        return std::optional<T>{std::move(parsed->value)};
    }

    [[nodiscard]] static std::expected<T, std::string>
        take(const toml::table& table, std::string_view key) {
        auto parsed = detail::parse_read_from_key<T>(table, key);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }

        if (auto unread = detail::error_if_unread(*parsed->parsed_table, key); !unread) {
            return std::unexpected(unread.error());
        }

        (void)const_cast<toml::table&>(table).erase(key);
        return std::move(parsed->value);
    }

    [[nodiscard]] static std::expected<std::optional<T>, std::string>
        take_optional(const toml::table& table, std::string_view key) {
        const toml::node* node = table.get(key);
        if (!node) {
            return std::optional<T>{};
        }

        auto parsed = detail::parse_read_from_node<T>(*node, key);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }

        if (auto unread = detail::error_if_unread(*parsed->parsed_table, key); !unread) {
            return std::unexpected(unread.error());
        }

        (void)const_cast<toml::table&>(table).erase(key);
        return std::optional<T>{std::move(parsed->value)};
    }
};

} // namespace toml_helper
