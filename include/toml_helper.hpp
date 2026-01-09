#pragma once

#include <Eigen/Dense>
#include <expected>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#define MAGIC_ENUM_RANGE_MIN 0
#define MAGIC_ENUM_RANGE_MAX 16
#include <magic_enum.hpp>
#define TOML_HEADER_ONLY 0
#define TOML_EXCEPTIONS  0
#include <toml++/toml.hpp>

namespace toml_helper {
namespace detail {
[[nodiscard]] inline const char* node_type_name(toml::node_type t) noexcept {
    switch (t) {
    case toml::node_type::none: return "none";
    case toml::node_type::table: return "table";
    case toml::node_type::array: return "array";
    case toml::node_type::string: return "string";
    case toml::node_type::integer: return "integer";
    case toml::node_type::floating_point: return "float";
    case toml::node_type::boolean: return "bool";
    case toml::node_type::date: return "date";
    case toml::node_type::time: return "time";
    case toml::node_type::date_time: return "datetime";
    }
    return "unknown";
}

[[nodiscard]] inline std::string join_unread_keys(const toml::table& table) {
    std::string out;
    for (const auto& [k, v] : table) {
        if (!out.empty()) {
            out += ", ";
        }
        out += fmt::format("'{}'({})", std::string_view{k}, node_type_name(v.type()));
    }
    return out;
}

[[nodiscard]] inline std::expected<void, std::string>
    error_if_unread(const toml::table& table, std::string_view context) noexcept {
    if (table.empty()) {
        return {};
    }
    if (context.empty()) {
        return std::unexpected(fmt::format("Unread keys: {}", join_unread_keys(table)));
    }
    return std::unexpected(fmt::format("{}: unread keys: {}", context, join_unread_keys(table)));
}

template <typename T>
[[nodiscard]] constexpr const char* expected_type_name() noexcept {
    if constexpr (std::is_same_v<T, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string";
    } else if constexpr (std::is_floating_point_v<T>) {
        return "float";
    } else if constexpr (std::is_integral_v<T>) {
        return "integer";
    } else {
        return "value";
    }
}

template <typename T>
[[nodiscard]] std::optional<T> value_with_numeric_cast(const toml::node& node) noexcept {
    if (auto v = node.value<T>()) {
        return *v;
    }

    // toml++ does not always implicitly convert integer <-> float across all T,
    // so we do a small amount of safe numeric bridging where it is unambiguous.
    if constexpr (std::is_floating_point_v<T>) {
        if (auto i = node.value<int64_t>()) {
            return static_cast<T>(*i);
        }
        if (auto u = node.value<uint64_t>()) {
            return static_cast<T>(*u);
        }
    }

    return std::nullopt;
}
} // namespace detail

template <typename T>
concept ReadFrom = requires(T& t, const toml::table& table) {
    { t.read_from(table) } -> std::same_as<std::expected<void, std::string>>;
};

template <typename T>
[[nodiscard]] std::expected<T, std::string> read(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::unexpected(fmt::format("Missing key '{}'", key));
    }

    if (auto val = detail::value_with_numeric_cast<T>(*node)) {
        return *val;
    }

    return std::unexpected(
        fmt::format(
            "Invalid value for key '{}': expected {}, got {}", key, detail::expected_type_name<T>(),
            detail::node_type_name(node->type())));
}

template <typename T>
[[nodiscard]] std::expected<std::optional<T>, std::string>
    read_optional(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::optional<T>{};
    }

    if (auto val = detail::value_with_numeric_cast<T>(*node)) {
        return std::optional<T>{*val};
    }

    return std::unexpected(
        fmt::format(
            "Invalid value for key '{}': expected {}, got {}", key, detail::expected_type_name<T>(),
            detail::node_type_name(node->type())));
}

template <ReadFrom T>
[[nodiscard]] std::expected<T, std::string> read(const toml::table& table, std::string_view key) {
    if (const auto _t = table[key].as_table()) {
        T t;
        auto result = t.read_from(*_t);
        if (!result) {
            return std::unexpected(fmt::format("{}: {}", key, result.error()));
        }
        return t;
    }
    if (!table.get(key)) {
        return std::unexpected(fmt::format("Missing table '{}'", key));
    }
    return std::unexpected(fmt::format("Invalid value for key '{}': expected table", key));
}

template <ReadFrom T>
[[nodiscard]] std::expected<T, std::string> take(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::unexpected(fmt::format("Missing table '{}'", key));
    }

    const auto sub = node->as_table();
    if (!sub) {
        return std::unexpected(fmt::format("Invalid value for key '{}': expected table", key));
    }

    T out;
    auto result = out.read_from(*sub);
    if (!result) {
        return std::unexpected(fmt::format("{}: {}", key, result.error()));
    }

    if (auto unread = detail::error_if_unread(*sub, key); !unread) {
        return std::unexpected(unread.error());
    }

    (void)const_cast<toml::table&>(table).erase(key);
    return out;
}

template <ReadFrom T>
[[nodiscard]] std::expected<std::optional<T>, std::string>
    read_optional(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::optional<T>{};
    }
    if (const auto t = node->as_table()) {
        T out;
        auto result = out.read_from(*t);
        if (!result) {
            return std::unexpected(fmt::format("{}: {}", key, result.error()));
        }
        return std::optional<T>{std::move(out)};
    }
    return std::unexpected(fmt::format("Invalid value for key '{}': expected table", key));
}

template <ReadFrom T>
[[nodiscard]] std::expected<std::optional<T>, std::string>
    take_optional(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::optional<T>{};
    }
    const auto sub = node->as_table();
    if (!sub) {
        return std::unexpected(fmt::format("Invalid value for key '{}': expected table", key));
    }

    T out;
    auto result = out.read_from(*sub);
    if (!result) {
        return std::unexpected(fmt::format("{}: {}", key, result.error()));
    }

    if (auto unread = detail::error_if_unread(*sub, key); !unread) {
        return std::unexpected(unread.error());
    }

    (void)const_cast<toml::table&>(table).erase(key);
    return std::optional<T>{std::move(out)};
}

template <typename T>
requires(std::is_enum_v<T>) [[nodiscard]] std::string enum_options() {
    auto all = magic_enum::enum_names<T>();
    std::string options;
    for (const auto& name : all) {
        if (!options.empty()) {
            options += ", ";
        }
        options += fmt::format("'{}'", name);
    }
    return options;
}

template <typename T>
requires(std::is_enum_v<T>)
[[nodiscard]] std::expected<T, std::string> read(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::unexpected(fmt::format("Missing key '{}'", key));
    }

    if (auto s = node->value<std::string>()) {
        if (auto enum_val = magic_enum::enum_cast<T>(*s, magic_enum::case_insensitive)) {
            return *enum_val;
        }
        return std::unexpected(
            fmt::format(
                "Invalid enum value for key '{}': '{}', must be one of {}", key, *s,
                enum_options<T>()));
    }

    // Allow numeric enum values for convenience (e.g. backend_type = 0).
    if (auto i = node->value<int64_t>()) {
        using U = std::underlying_type_t<T>;
        if (auto enum_val = magic_enum::enum_cast<T>(static_cast<U>(*i))) {
            return *enum_val;
        }
        return std::unexpected(
            fmt::format(
                "Invalid enum value for key '{}': {}, must be one of {}", key, *i,
                enum_options<T>()));
    }

    return std::unexpected(
        fmt::format(
            "Invalid value for key '{}': expected enum as string/integer, got {}; options: {}", key,
            detail::node_type_name(node->type()), enum_options<T>()));
}

template <typename T>
requires(std::is_enum_v<T>) [[nodiscard]] std::expected<std::optional<T>, std::string>
    read_optional(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::optional<T>{};
    }

    if (auto s = node->value<std::string>()) {
        if (auto enum_val = magic_enum::enum_cast<T>(*s, magic_enum::case_insensitive)) {
            return std::optional<T>{*enum_val};
        }
        return std::unexpected(
            fmt::format(
                "Invalid enum value for key '{}': '{}', must be one of {}", key, *s,
                enum_options<T>()));
    }

    if (auto i = node->value<int64_t>()) {
        using U = std::underlying_type_t<T>;
        if (auto enum_val = magic_enum::enum_cast<T>(static_cast<U>(*i))) {
            return std::optional<T>{*enum_val};
        }
        return std::unexpected(
            fmt::format(
                "Invalid enum value for key '{}': {}, must be one of {}", key, *i,
                enum_options<T>()));
    }

    return std::unexpected(
        fmt::format(
            "Invalid value for key '{}': expected enum as string/integer, got {}; options: {}", key,
            detail::node_type_name(node->type()), enum_options<T>()));
}

template <typename Mat>
requires(Mat::RowsAtCompileTime != Eigen::Dynamic && Mat::ColsAtCompileTime != Eigen::Dynamic)
std::expected<Mat, std::string> read(const toml::table& table, std::string_view key) {
    constexpr int expected_size = Mat::RowsAtCompileTime * Mat::ColsAtCompileTime;
    using Scalar                = typename Mat::Scalar;

    std::vector<Scalar> scalar_array;
    scalar_array.reserve(expected_size);

    const toml::node* node = table.get(key);
    if (!node) {
        return std::unexpected(fmt::format("Missing key '{}'", key));
    }

    if (const auto arr = node->as_array()) {
        for (const auto& v : *arr) {
            if (const auto val = v.value<Scalar>()) {
                scalar_array.push_back(*val);
            } else if constexpr (std::is_floating_point_v<Scalar>) {
                if (const auto i = v.value<int64_t>()) {
                    scalar_array.push_back(static_cast<Scalar>(*i));
                } else if (const auto u = v.value<uint64_t>()) {
                    scalar_array.push_back(static_cast<Scalar>(*u));
                } else {
                    return std::unexpected(fmt::format("'{}' must be an array of numbers", key));
                }
            } else {
                return std::unexpected(fmt::format("'{}' must be an array of numbers", key));
            }
        }
        if (scalar_array.size() != expected_size) {
            return std::unexpected(
                fmt::format("'{}' must have exactly '{}' elements", key, expected_size));
        }
        // Eigen reads raw pointer in the matrix's native storage order; make it explicit.
        return Eigen::Map<const Mat>(scalar_array.data());
    }

    return std::unexpected(
        fmt::format(
            "Invalid value for key '{}': expected array, got {}", key,
            detail::node_type_name(node->type())));
}

template <typename Mat>
requires(Mat::RowsAtCompileTime != Eigen::Dynamic && Mat::ColsAtCompileTime != Eigen::Dynamic)
[[nodiscard]] std::expected<std::optional<Mat>, std::string>
    read_optional(const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::optional<Mat>{};
    }

    if (auto m = read<Mat>(table, key)) {
        return std::optional<Mat>{std::move(*m)};
    } else {
        return std::unexpected(m.error());
    }
}

[[nodiscard]] inline std::expected<void, std::string>
    ensure_all_read(const toml::table& table, std::string_view context = {}) noexcept {
    return detail::error_if_unread(table, context);
}

template <typename T>
[[nodiscard]] std::expected<T, std::string> take(const toml::table& table, std::string_view key) {
    auto v = read<T>(table, key);
    if (!v) {
        return std::unexpected(v.error());
    }
    (void)const_cast<toml::table&>(table).erase(key);
    return std::move(*v);
}

template <typename T>
[[nodiscard]] std::expected<std::optional<T>, std::string>
    take_optional(const toml::table& table, std::string_view key) {
    auto v = read_optional<T>(table, key);
    if (!v) {
        return std::unexpected(v.error());
    }
    if (*v) {
        (void)const_cast<toml::table&>(table).erase(key);
    }
    return std::move(*v);
}

// Read a nested `ReadFrom` object from the *current* table instead of a subtable.
// This allows config layouts like:
//   width = 1920
//   height = 1080
// where `width/height` are fields of a nested struct, without requiring `[nested]`.
template <ReadFrom T>
[[nodiscard]] std::expected<void, std::string>
    read_flatten_into(const toml::table& table, T& out, std::string_view context) noexcept {
    auto result = out.read_from(table);
    if (result) {
        return {};
    }
    if (!context.empty()) {
        return std::unexpected(fmt::format("{}: {}", context, result.error()));
    }
    return std::unexpected(result.error());
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

// ============================================================================
// Configuration Merge/Override Utilities
// ============================================================================

/// Recursively merge two TOML tables. Values in `override_table` take precedence.
/// @param base The base configuration table (inherited)
/// @param override_table The override configuration table (takes precedence)
/// @return Merged table with override values winning on conflicts
[[nodiscard]] inline std::expected<toml::table, std::string>
    merge_configs(const toml::table& base, const toml::table& override_table) noexcept {
    toml::table result = base; // 先拷贝一份 base

    for (const auto& [k, v] : override_table) {
        // 检查 base 中是否也存在这个键，且两者都是 table
        auto* base_node = result.get(k);

        if (base_node && base_node->as_table() && v.as_table()) {
            // 递归合并子表
            auto merged = merge_configs(*base_node->as_table(), *v.as_table());
            if (!merged)
                return std::unexpected(merged.error());

            // 替换为合并后的子表
            result.insert_or_assign(k, std::move(*merged));
        } else {
            // 如果不是 table，或者 base 中不存在，则直接覆盖/插入
            // 使用 insert_or_assign 确保 override 胜出
            result.insert_or_assign(k, v);
        }
    }

    return result;
}
} // namespace toml_helper
