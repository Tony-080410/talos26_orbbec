#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "toml/core.hpp"
#include "toml/ext/containers.hpp"

namespace {

using namespace toml_helper;

struct SerialConfig {
    int baud = 115200;
    std::optional<int> port{};
    std::optional<int> timeout_ms = 200;
};

struct CameraIntrinsics {
    double fx = 1000.0;
    double fy = 1000.0;
    std::optional<double> cx{};
    std::optional<double> cy = 540.0;
};

struct CameraConfig {
    std::string model = "HIK";
    SerialConfig serial{};
    flatten<CameraIntrinsics> intrinsics{};
    std::optional<int> fps = 60;
};

struct Waypoint {
    double x = 0.0;
    double y = 0.0;
};

struct PathConfig {
    std::vector<Waypoint> points;
};

TEST(TomlReflection, FromTableReadsDefaultsAndFlattenFields) {
    const auto parsed = toml::parse(R"(
        model = "MV-CA013"
        fx = 900.0
        fy = 901.0
        cx = 320.0

        [serial]
        port = 2
    )");
    ASSERT_TRUE(parsed);

    auto result = from_table<CameraConfig>(parsed.table());
    ASSERT_TRUE(result);

    EXPECT_EQ(result->model, "MV-CA013");
    EXPECT_EQ(result->serial.baud, 115200);
    ASSERT_TRUE(result->serial.port.has_value());
    EXPECT_EQ(*result->serial.port, 2);
    ASSERT_TRUE(result->serial.timeout_ms.has_value());
    EXPECT_EQ(*result->serial.timeout_ms, 200);
    EXPECT_DOUBLE_EQ(result->intrinsics->fx, 900.0);
    EXPECT_DOUBLE_EQ(result->intrinsics->fy, 901.0);
    ASSERT_TRUE(result->intrinsics->cx.has_value());
    EXPECT_DOUBLE_EQ(*result->intrinsics->cx, 320.0);
    ASSERT_TRUE(result->intrinsics->cy.has_value());
    EXPECT_DOUBLE_EQ(*result->intrinsics->cy, 540.0);
    ASSERT_TRUE(result->fps.has_value());
    EXPECT_EQ(*result->fps, 60);

    auto check = ensure_all_read(parsed.table(), "camera");
    ASSERT_TRUE(check);
}

TEST(TomlReflection, ReadIntoPreservesExistingDefaultsForMissingKeys) {
    const auto parsed = toml::parse(R"(
        fx = 700.0
        fy = 710.0
        cx = 300.0

        [serial]
        port = 4
    )");
    ASSERT_TRUE(parsed);

    CameraConfig config;
    config.model = "existing";
    config.fps   = 144;

    auto result = read_into(parsed.table(), config);
    ASSERT_TRUE(result);

    EXPECT_EQ(config.model, "existing");
    ASSERT_TRUE(config.fps.has_value());
    EXPECT_EQ(*config.fps, 144);
    EXPECT_EQ(config.serial.baud, 115200);
    ASSERT_TRUE(config.serial.port.has_value());
    EXPECT_EQ(*config.serial.port, 4);
    EXPECT_DOUBLE_EQ(config.intrinsics->fx, 700.0);
    EXPECT_DOUBLE_EQ(config.intrinsics->fy, 710.0);
    ASSERT_TRUE(config.intrinsics->cx.has_value());
    EXPECT_DOUBLE_EQ(*config.intrinsics->cx, 300.0);

    auto check = ensure_all_read(parsed.table(), "camera");
    ASSERT_TRUE(check);
}

TEST(TomlReflection, MissingRequiredOptionalInsideFlattenIsPrefixed) {
    const auto parsed = toml::parse(R"(
        model = "MV-CA013"
        fx = 900.0
        fy = 901.0

        [serial]
        port = 2
    )");
    ASSERT_TRUE(parsed);

    auto result = from_table<CameraConfig>(parsed.table());
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("intrinsics: Missing key 'cx'"), std::string::npos);
}

TEST(TomlReflection, TakeReflectiveObjectRejectsUnreadKeysInNestedTable) {
    const auto parsed = toml::parse(R"(
        serial = { port = 7, extra = 1 }
    )");
    ASSERT_TRUE(parsed);

    auto result = take<SerialConfig>(parsed.table(), "serial");
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("serial: unread keys"), std::string::npos);
    EXPECT_NE(result.error().find("'extra'(integer)"), std::string::npos);
}

TEST(TomlReflection, NulloptOptionalFieldIsRequired) {
    struct RequiredConfig {
        std::optional<int> port{};
    };

    const auto parsed = toml::parse("");
    ASSERT_TRUE(parsed);

    auto result = from_table<RequiredConfig>(parsed.table());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), "Missing key 'port'");
}

TEST(TomlReflection, ContainerParsersSupportReflectiveElementTypes) {
    const auto parsed = toml::parse(R"(
        points = [
            { x = 1.0, y = 2.0 },
            { x = 3.0, y = 4.0 }
        ]
    )");
    ASSERT_TRUE(parsed);

    auto result = from_table<PathConfig>(parsed.table());
    ASSERT_TRUE(result);
    ASSERT_EQ(result->points.size(), 2U);
    EXPECT_DOUBLE_EQ(result->points[0].x, 1.0);
    EXPECT_DOUBLE_EQ(result->points[1].y, 4.0);

    auto check = ensure_all_read(parsed.table(), "path");
    ASSERT_TRUE(check);
}

} // namespace
