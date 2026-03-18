#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "toml/core.hpp"
#include "toml/ext/containers.hpp"

namespace {

using namespace toml_helper;

struct NoDefault {
    NoDefault() = delete;
    explicit NoDefault(int value)
        : value(value) {}

    int value;
};

static_assert(!std::default_initializable<NoDefault>);
static_assert(std::default_initializable<required<NoDefault>>);

struct SerialConfig {
    required<int> baud{};
    std::optional<int> port{};
    int timeout_ms{200};
};

struct CameraIntrinsics {
    required<double> fx{};
    required<double> fy{};
    std::optional<double> cx{};
    double cy{540.0};
};

struct CameraConfig {
    std::string model{"HIK"};
    SerialConfig serial{};
    flatten<CameraIntrinsics> intrinsics{};
    int fps{60};
};

struct Waypoint {
    double x = 0.0;
    double y = 0.0;
};

struct PathConfig {
    std::vector<Waypoint> points;
};

TEST(TomlReflection, FromTableReadsRequiredOptionalAndPlainDefaultFields) {
    const auto parsed = toml::parse(R"(
        model = "MV-CA013"
        fps = 60
        fx = 900.0
        fy = 901.0
        cx = 320.0

        [serial]
        baud = 115200
        port = 2
    )");
    ASSERT_TRUE(parsed);

    auto result = from_table<CameraConfig>(parsed.table());
    ASSERT_TRUE(result);

    EXPECT_EQ(result->model, "MV-CA013");
    EXPECT_EQ(result->fps, 60);
    EXPECT_EQ(result->serial.baud.get(), 115200);
    ASSERT_TRUE(result->serial.port.has_value());
    EXPECT_EQ(*result->serial.port, 2);
    EXPECT_EQ(result->serial.timeout_ms, 200);
    EXPECT_DOUBLE_EQ(result->intrinsics->fx.get(), 900.0);
    EXPECT_DOUBLE_EQ(result->intrinsics->fy.get(), 901.0);
    ASSERT_TRUE(result->intrinsics->cx.has_value());
    EXPECT_DOUBLE_EQ(*result->intrinsics->cx, 320.0);
    EXPECT_DOUBLE_EQ(result->intrinsics->cy, 540.0);

    auto check = ensure_all_read(parsed.table(), "camera");
    ASSERT_TRUE(check);
}

TEST(TomlReflection, ReadIntoReplacesExistingObjectAndResetsByFieldPolicy) {
    const auto parsed = toml::parse(R"(
        fx = 700.0
        fy = 710.0

        [serial]
        baud = 115200
    )");
    ASSERT_TRUE(parsed);

    CameraConfig config;
    config.model             = "existing";
    config.serial.port       = 4;
    config.serial.timeout_ms = 200;
    config.intrinsics->cx    = 300.0;
    config.intrinsics->cy    = 540.0;
    config.fps               = 144;

    auto result = read_into(parsed.table(), config);
    ASSERT_TRUE(result);

    EXPECT_EQ(config.model, "HIK");
    EXPECT_EQ(config.serial.baud.get(), 115200);
    EXPECT_FALSE(config.serial.port.has_value());
    EXPECT_EQ(config.serial.timeout_ms, 200);
    EXPECT_DOUBLE_EQ(config.intrinsics->fx.get(), 700.0);
    EXPECT_DOUBLE_EQ(config.intrinsics->fy.get(), 710.0);
    EXPECT_FALSE(config.intrinsics->cx.has_value());
    EXPECT_DOUBLE_EQ(config.intrinsics->cy, 540.0);
    EXPECT_EQ(config.fps, 60);

    auto check = ensure_all_read(parsed.table(), "camera");
    ASSERT_TRUE(check);
}

TEST(TomlReflection, ReadIntoFailureDoesNotMutateDestination) {
    const auto parsed = toml::parse(R"(
        fx = 700.0

        [serial]
        baud = 115200
    )");
    ASSERT_TRUE(parsed);

    CameraConfig config;
    config.model             = "existing";
    config.serial.baud       = 9600;
    config.serial.port       = 4;
    config.serial.timeout_ms = 250;
    config.intrinsics->fx    = 1.0;
    config.intrinsics->fy    = 2.0;
    config.intrinsics->cx    = 300.0;
    config.fps               = 144;

    auto result = read_into(parsed.table(), config);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), "intrinsics: Missing key 'fy'");

    EXPECT_EQ(config.model, "existing");
    EXPECT_EQ(config.serial.baud.get(), 9600);
    ASSERT_TRUE(config.serial.port.has_value());
    EXPECT_EQ(*config.serial.port, 4);
    EXPECT_EQ(config.serial.timeout_ms, 250);
    EXPECT_DOUBLE_EQ(config.intrinsics->fx.get(), 1.0);
    EXPECT_DOUBLE_EQ(config.intrinsics->fy.get(), 2.0);
    ASSERT_TRUE(config.intrinsics->cx.has_value());
    EXPECT_DOUBLE_EQ(*config.intrinsics->cx, 300.0);
    EXPECT_EQ(config.fps, 144);
}

TEST(TomlReflection, MissingRequiredFieldInsideFlattenIsPrefixed) {
    const auto parsed = toml::parse(R"(
        model = "MV-CA013"
        fy = 901.0

        [serial]
        baud = 115200
        port = 2
    )");
    ASSERT_TRUE(parsed);

    auto result = from_table<CameraConfig>(parsed.table());
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("intrinsics: Missing key 'fx'"), std::string::npos);
}

TEST(TomlReflection, TakeReflectiveObjectRejectsUnreadKeysInNestedTable) {
    const auto parsed = toml::parse(R"(
        serial = { baud = 115200, port = 7, extra = 1 }
    )");
    ASSERT_TRUE(parsed);

    auto result = take<SerialConfig>(parsed.table(), "serial");
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("serial: unread keys"), std::string::npos);
    EXPECT_NE(result.error().find("'extra'(integer)"), std::string::npos);
}

TEST(TomlReflection, FromTableRejectsUnreadTopLevelKeys) {
    const auto parsed = toml::parse(R"(
        model = "MV-CA013"
        extra = 1
        fx = 900.0
        fy = 901.0

        [serial]
        baud = 115200
    )");
    ASSERT_TRUE(parsed);

    auto result = from_table<CameraConfig>(parsed.table());
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("Unread keys"), std::string::npos);
    EXPECT_NE(result.error().find("'extra'(integer)"), std::string::npos);
}

TEST(TomlReflection, PlainFieldMayBeMissingAndKeepsInitializer) {
    struct PlainDefaultConfig {
        int port = 7;
    };

    const auto parsed = toml::parse("");
    ASSERT_TRUE(parsed);

    auto result = from_table<PlainDefaultConfig>(parsed.table());
    ASSERT_TRUE(result);
    EXPECT_EQ(result->port, 7);
}

TEST(TomlReflection, RequiredFieldMustBePresent) {
    struct RequiredConfig {
        required<int> port{};
    };

    const auto parsed = toml::parse("");
    ASSERT_TRUE(parsed);

    auto result = from_table<RequiredConfig>(parsed.table());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), "Missing key 'port'");
}

TEST(TomlReflection, RequiredWrapperThrowsOnEmptyAccess) {
    required<int> port{};
    const required<int>& const_port = port;

    EXPECT_FALSE(port.has_value());
    EXPECT_THROW((void)port.get(), std::bad_optional_access);
    EXPECT_THROW((void)const_port.get(), std::bad_optional_access);
    EXPECT_THROW((void)port.operator->(), std::bad_optional_access);
    EXPECT_THROW((void)const_port.operator->(), std::bad_optional_access);
    EXPECT_THROW((void)*port, std::bad_optional_access);
}

TEST(TomlReflection, OptionalFieldMayBeMissingEvenWithEngagedInitializer) {
    struct OptionalConfig {
        std::optional<int> port = 7;
    };

    const auto parsed = toml::parse("");
    ASSERT_TRUE(parsed);

    auto result = from_table<OptionalConfig>(parsed.table());
    ASSERT_TRUE(result);
    EXPECT_FALSE(result->port.has_value());
}

TEST(TomlReflection, MergeConfigsKeepsFileLevelOverrideSemantics) {
    const auto base = toml::parse(R"(
        model = "MV-CA013"
        fps = 120
        fx = 900.0
        fy = 901.0
        cy = 540.0

        [serial]
        baud = 115200
        timeout_ms = 200
    )");
    ASSERT_TRUE(base);

    const auto override_cfg = toml::parse(R"(
        [serial]
        port = 2
    )");
    ASSERT_TRUE(override_cfg);

    auto merged = merge_configs(base.table(), override_cfg.table());
    ASSERT_TRUE(merged);

    auto result = from_table<CameraConfig>(*merged);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->model, "MV-CA013");
    EXPECT_EQ(result->fps, 120);
    ASSERT_TRUE(result->serial.port.has_value());
    EXPECT_EQ(*result->serial.port, 2);
    EXPECT_EQ(result->serial.timeout_ms, 200);
    EXPECT_DOUBLE_EQ(result->intrinsics->cy, 540.0);
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
