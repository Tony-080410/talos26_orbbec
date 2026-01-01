#pragma once

#include "toml++/toml.hpp"
#include <Eigen/Dense>
#include <expected>
#include <optional>

template <typename Mat>
requires(Mat::RowsAtCompileTime != Eigen::Dynamic && Mat::ColsAtCompileTime != Eigen::Dynamic)
std::expected<void, std::string>
    parse_double_linear_impl(const toml::table& t, const std::string& key, Mat& mat) {
    constexpr int expected_size = Mat::RowsAtCompileTime * Mat::ColsAtCompileTime;
    std::vector<double> double_array;
    double_array.reserve(expected_size);
    if (const auto arr = t[key].as_array()) {
        for (const auto& v : *arr) {
            if (const auto val = v.value<double>()) {
                double_array.push_back(*val);
            } else {
                return std::unexpected(key + " contains non-double value");
            }
        }
        if (double_array.size() != expected_size) {
            return std::unexpected(
                key + " must have " + std::to_string(expected_size) + " elements");
        }
        mat = Mat(double_array.data());
        return {};
    }
    return std::unexpected(key + " not found");
}

#define READ_SUB_OPT(val, tbl) (val).read_from(tbl);

#define READ_SUB(val, tbl)                         \
    if (auto read = (val).read_from(tbl); !read) { \
        return read;                               \
    }

#define READ_MATRIX(var, t, container)                              \
    auto(var) = parse_double_linear_impl(t, #container, container); \
    if (!(var)) {                                                   \
        return std::unexpected((var).error());                      \
    }

#define READ(tbl, key, field)                                  \
    if (auto val = (tbl)[key].value<decltype(field)>(); val) { \
        (field) = *val;                                        \
    } else {                                                   \
        return std::unexpected(#key " not set or invalid");    \
    }

#define READ_KEY(tbl, field)                                      \
    if (auto val = (tbl)[#field].value<decltype(field)>(); val) { \
        (field) = *val;                                           \
    } else {                                                      \
        return std::unexpected(#field " not set or invalid");     \
    }

#define READ_KEY_OPT(tbl, field)                                  \
    if (auto val = (tbl)[#field].value<decltype(field)>(); val) { \
        (field) = *val;                                           \
    }

#define READ_ENUM(tbl, field)                                                             \
    if (auto val = (tbl)[#field].value<std::underlying_type_t<decltype(field)>>(); val) { \
        (field) = static_cast<decltype(field)>(*val);                                     \
    } else {                                                                              \
        return std::unexpected(#field " not set or invalid");                             \
    }
