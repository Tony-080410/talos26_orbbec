#pragma once

#include "toml++/toml.hpp"

template<typename Container>
std::expected<void, std::string> parse_double_array_impl(const toml::table &t, const std::string &key,
                                                         Container &container, const size_t expected_size) {
    std::vector<double> double_array;
    double_array.reserve(expected_size);
    if (const auto arr = t[key].as_array()) {
        for (const auto &v: *arr) {
            if (const auto val = v.value<double>()) {
                double_array.push_back(*val);
            } else {
                return std::unexpected(key + " contains non-double value");
            }
        }
        if (double_array.size() != expected_size) {
            return std::unexpected(key + " must have " + std::to_string(expected_size) + " elements");
        }
        return {};
    }
    container = decltype(container)(double_array);
    return std::unexpected(key + " not found");
}

#define PARSE_DOUBLE_ARRAY(var, t, container, size)                                                               \
    auto(var) = parse_double_array_impl(t, #container, container, size);                                                      \
    if (!(var)) {                                                                                                      \
        return std::unexpected((var).error());                                                                         \
    }

#define READ(tbl, key, field)                                                                                          \
    if (auto val = (tbl)[key].value<decltype(field)>(); val) {                                                         \
        (field) = *val;                                                                                                \
    } else {                                                                                                           \
        return std::unexpected(#key " not set or invalid");                                                            \
    }

#define READ_KEY(tbl, field)                                                                                           \
    if (auto val = (tbl)[#field].value<decltype(field)>(); val) {                                                      \
        (field) = *val;                                                                                                \
    } else {                                                                                                           \
        return std::unexpected(#field " not set or invalid");                                                          \
    }
