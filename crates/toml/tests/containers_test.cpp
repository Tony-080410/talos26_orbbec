#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <vector>

#include "toml/core.hpp"
#include "toml/ext/containers.hpp"

namespace {

using namespace toml_helper;

struct Point {
    int x = 0;
    int y = 0;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, x);
        READ(tbl, y);
        return {};
    }
};

TEST(TomlContainers, OptionalReadRequiresKey) {
    const auto parsed = toml::parse("");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::optional<int>>::read(parsed.table(), "value");
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), "Missing key 'value'");
}

TEST(TomlContainers, OptionalReadParsesValue) {
    const auto parsed = toml::parse("value = 42");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::optional<int>>::read(parsed.table(), "value");
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->has_value());
    EXPECT_EQ(**result, 42);
}

TEST(TomlContainers, ReadOptLeavesOptionalDefaultOnMissingKey) {
    struct Config {
        std::optional<int> exposure = 42;

        std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
            READ_OPT(tbl, exposure);
            return {};
        }
    };

    const auto parsed = toml::parse("");
    ASSERT_TRUE(parsed);

    Config config;
    auto result = config.read_from(parsed.table());
    ASSERT_TRUE(result);
    ASSERT_TRUE(config.exposure.has_value());
    EXPECT_EQ(*config.exposure, 42);
}

TEST(TomlContainers, ReadOptUpdatesOptionalFieldWhenPresent) {
    struct Config {
        std::optional<int> exposure = 42;

        std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
            READ_OPT(tbl, exposure);
            return {};
        }
    };

    const auto parsed = toml::parse("exposure = 7");
    ASSERT_TRUE(parsed);

    Config config;
    auto result = config.read_from(parsed.table());
    ASSERT_TRUE(result);
    ASSERT_TRUE(config.exposure.has_value());
    EXPECT_EQ(*config.exposure, 7);
}

TEST(TomlContainers, VectorReadParsesScalars) {
    const auto parsed = toml::parse("values = [1, 2, 3, 4, 5]");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::vector<int>>::read(parsed.table(), "values");
    ASSERT_TRUE(result);
    EXPECT_EQ(result->size(), 5U);
    EXPECT_EQ((*result)[0], 1);
    EXPECT_EQ((*result)[4], 5);
}

TEST(TomlContainers, VectorReadParsesNestedArrays) {
    const auto parsed = toml::parse(R"(
        waypoints = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0]
        ]
    )");
    ASSERT_TRUE(parsed);

    auto result =
        Deserialize<std::vector<std::array<double, 3>>>::read(parsed.table(), "waypoints");
    ASSERT_TRUE(result);
    EXPECT_EQ(result->size(), 3U);
    EXPECT_DOUBLE_EQ((*result)[1][0], 1.0);
    EXPECT_DOUBLE_EQ((*result)[2][1], 0.0);
}

TEST(TomlContainers, VectorTakeRejectsUnreadKeysInNestedTables) {
    const auto parsed = toml::parse(R"(
        points = [
            { x = 1, y = 2, extra = 3 }
        ]
    )");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::vector<Point>>::take(parsed.table(), "points");
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("points[0]: unread keys"), std::string::npos);
    EXPECT_NE(result.error().find("'extra'(integer)"), std::string::npos);
}

TEST(TomlContainers, ArrayReadValidatesSize) {
    const auto parsed = toml::parse("values = [1, 2]");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::array<int, 3>>::read(parsed.table(), "values");
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("expected 3 elements, got 2"), std::string::npos);
}

TEST(TomlContainers, ArrayReadParsesNestedTables) {
    const auto parsed = toml::parse(R"(
        triangle = [
            { x = 0, y = 0 },
            { x = 1, y = 0 },
            { x = 0, y = 1 }
        ]
    )");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::array<Point, 3>>::read(parsed.table(), "triangle");
    ASSERT_TRUE(result);
    EXPECT_EQ((*result)[0].x, 0);
    EXPECT_EQ((*result)[1].x, 1);
    EXPECT_EQ((*result)[2].y, 1);
}

TEST(TomlContainers, ReadOptionalForOptionalReturnsOuterPresence) {
    const auto parsed = toml::parse("value = 9");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::optional<int>>::read_optional(parsed.table(), "value");
    ASSERT_TRUE(result);
    ASSERT_TRUE(result->has_value());
    ASSERT_TRUE((*result)->has_value());
    EXPECT_EQ(**result, 9);
}

TEST(TomlContainers, ReadOptionalVectorMissingKeyReturnsOuterNullopt) {
    const auto parsed = toml::parse("");
    ASSERT_TRUE(parsed);

    auto result = Deserialize<std::vector<int>>::read_optional(parsed.table(), "values");
    ASSERT_TRUE(result);
    EXPECT_FALSE(result->has_value());
}

} // namespace
