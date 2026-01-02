#pragma once

#include <Eigen/Dense>
#include <expected>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <type_traits>
#define MAGIC_ENUM_RANGE_MIN 0
#define MAGIC_ENUM_RANGE_MAX 16
#include <magic_enum/magic_enum.hpp>
#include <optional>
#define TOML_HEADER_ONLY 0
#define TOML_EXCEPTIONS  0
#include <toml++/toml.hpp>

namespace toml_helper {
template <typename T>
concept ReadFrom = requires(T& t, const toml::table& table) {
    { t.read_from(table) } -> std::same_as<std::expected<void, std::string>>;
};

template <typename T>
constexpr std::expected<T, std::string> read(const toml::table& table, std::string_view key) {
    if (auto val = table[key].value<T>(); val) {
        return *val;
    } else {
        return std::unexpected(fmt::format("'{}' does not exists, or have a invalid value", key));
    }
}

template <ReadFrom T>
constexpr std::expected<T, std::string> read(const toml::table& table, std::string_view key) {
    if (const auto _t = table[key].as_table()) {
        T t;
        auto result = t.read_from(*_t);
        if (!result) {
            return std::unexpected(result.error());
        }
        return t;
    }
    return std::unexpected(fmt::format("'{}' is not a table", key));
}

template <typename T>
requires(std::is_enum_v<T>)
constexpr std::expected<T, std::string> read(const toml::table& table, std::string_view key) {
    auto val = read<std::string>(table, key); // Read as string
    if (!val) {
        return std::unexpected(val.error());
    }
    auto enum_val = magic_enum::enum_cast<T>(*val, magic_enum::case_insensitive);
    if (enum_val) {
        return *enum_val;
    }
    auto all = magic_enum::enum_names<T>();
    std::string options;
    for (const auto& name : all) {
        if (!options.empty()) {
            options += ", ";
        }
        options += fmt::format("'{}'", name);
    }
    return std::unexpected(
        fmt::format("'{}' has invalid enum value '{}', must be one of {}", key, *val, options));
}

template <typename Mat>
requires(Mat::RowsAtCompileTime != Eigen::Dynamic && Mat::ColsAtCompileTime != Eigen::Dynamic)
std::expected<Mat, std::string> read(const toml::table& table, std::string_view key) {
    constexpr int expected_size = Mat::RowsAtCompileTime * Mat::ColsAtCompileTime;
    std::vector<double> double_array;
    double_array.reserve(expected_size);
    if (const auto arr = table[key].as_array()) {
        for (const auto& v : *arr) {
            if (const auto val = v.value<double>()) {
                double_array.push_back(*val);
            } else {
                return std::unexpected(fmt::format("'{}' must be an array of doubles", key));
            }
        }
        if (double_array.size() != expected_size) {
            return std::unexpected(
                fmt::format("'{}' must have exactly '{}' elements", key, expected_size));
        }
        return Mat(double_array.data());
    }
    return std::unexpected(fmt::format("'{}' is not an array", key));
}
} // namespace toml_helper

#define READ_OPT(tbl, val)                                      \
    do {                                                        \
        auto res = toml_helper::read<decltype(val)>(tbl, #val); \
        if (res) {                                              \
            (val) = std::move(*res);                            \
        }                                                       \
    } while (0)

#define READ(tbl, val)                                          \
    do {                                                        \
        auto res = toml_helper::read<decltype(val)>(tbl, #val); \
        if (!res) {                                             \
            return std::unexpected(res.error());                \
        }                                                       \
        (val) = std::move(*res);                                \
    } while (0)
