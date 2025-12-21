#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <algorithm>

namespace fast_tf {

// 编译期字符串用于坐标系标识符
template<std::size_t N>
struct CompileTimeString {
    std::array<char, N> data{};
    std::size_t size = 0;

    constexpr CompileTimeString() = default;

    // 从字符串字面量构造
    constexpr CompileTimeString(const char (&str)[N + 1]) {
        std::copy(str, str + N, data.begin());
        size = N;
    }

    // 获取字符串视图
    [[nodiscard]] constexpr std::string_view view() const {
        return std::string_view(data.data(), size);
    }

    // 获取C字符串
    [[nodiscard]] constexpr const char* c_str() const {
        return data.data();
    }

    // 比较操作
    constexpr bool operator==(const CompileTimeString& other) const {
        if (size != other.size) return false;
        return std::equal(data.begin(), data.begin() + size,
                         other.data.begin());
    }

    constexpr bool operator!=(const CompileTimeString& other) const {
        return !(*this == other);
    }

    constexpr bool operator<(const CompileTimeString& other) const {
        if (size != other.size) return size < other.size;
        return std::lexicographical_compare(
            data.begin(), data.begin() + size,
            other.data.begin(), other.data.begin() + other.size);
    }

    // 哈希函数（用于编译期和运行期）
    [[nodiscard]] constexpr uint64_t hash() const {
        // FNV-1a 哈希算法
        uint64_t hash_value = 14695981039346656037ULL; // FNV offset basis
        for (std::size_t i = 0; i < size; ++i) {
            hash_value ^= static_cast<uint64_t>(data[i]);
            hash_value *= 1099511628211ULL; // FNV prime
        }
        return hash_value;
    }
};

// 允许从字符串字面量创建 CompileTimeString
template<std::size_t N>
CompileTimeString(const char (&)[N]) -> CompileTimeString<N - 1>;

// 推导指南已经通过构造函数隐式定义

// 快速哈希函数，接受 CompileTimeString
template<std::size_t N>
[[nodiscard]] constexpr uint64_t hash_compile_time_string(
    const CompileTimeString<N>& str) {
    return str.hash();
}

// 支持std::string_view的编译期哈希
constexpr uint64_t hash_string_view(std::string_view str) {
    uint64_t hash_value = 14695981039346656037ULL;
    for (char c : str) {
        hash_value ^= static_cast<uint64_t>(c);
        hash_value *= 1099511628211ULL;
    }
    return hash_value;
}

} // namespace fast_tf
