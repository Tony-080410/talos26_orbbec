#pragma once

#include "toml_helper.hpp"

#include <Eigen/Core>

namespace toml_helper {

template <typename Mat>
requires(Mat::RowsAtCompileTime != Eigen::Dynamic && Mat::ColsAtCompileTime != Eigen::Dynamic)
std::expected<Mat, std::string>
    read_impl(type_tag<Mat>, const toml::table& table, std::string_view key) {
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
    read_optional_impl(type_tag<Mat>, const toml::table& table, std::string_view key) {
    const toml::node* node = table.get(key);
    if (!node) {
        return std::optional<Mat>{};
    }

    if (auto m = read_impl(type_tag<Mat>{}, table, key)) {
        return std::optional<Mat>{std::move(*m)};
    } else {
        return std::unexpected(m.error());
    }
}

} // namespace toml_helper
