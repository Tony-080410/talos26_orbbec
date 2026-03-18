#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#define TALOS_DISABLE_PROGRAM_MAIN
#include "../../main.cpp"
#undef TALOS_DISABLE_PROGRAM_MAIN

namespace {

std::filesystem::path source_root() { return std::filesystem::path(TALOS_SOURCE_DIR); }

class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(const std::filesystem::path& path)
        : old_path_(std::filesystem::current_path()) {
        std::filesystem::current_path(path);
    }

    ~ScopedCurrentPath() { std::filesystem::current_path(old_path_); }

private:
    std::filesystem::path old_path_;
};

void write_file(const std::filesystem::path& path, std::string_view contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << contents;
}

void copy_file_from_repo(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::filesystem::create_directories(to.parent_path());
    std::filesystem::copy_file(
        source_root() / from, to, std::filesystem::copy_options::overwrite_existing);
}

std::filesystem::path make_temp_dir(std::string_view name) {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path   = std::filesystem::temp_directory_path()
                      / ("talos-read-config-" + std::string(name) + "-" + std::to_string(unique));
    std::filesystem::create_directories(path);
    return path;
}

TEST(ReadConfig, RepositoryLaunchConfigLoadsAndMergesOverrideSemantics) {
    const ScopedCurrentPath cwd(source_root());

    auto result = read_config();
    ASSERT_TRUE(result.has_value()) << to_string(result.error());

    auto [robot, daedalus, hardware, foxglove, vision] = std::move(*result);
    EXPECT_EQ(robot, fcs::RobotType::Infantry);
    EXPECT_TRUE(daedalus);
    ASSERT_TRUE(hardware.has_value());
    EXPECT_EQ(hardware->type.get(), fcs::RobotType::Infantry);
    EXPECT_DOUBLE_EQ(hardware->mcu->bullet_speed_default, 22.0);

    EXPECT_TRUE(foxglove.enabled);
    EXPECT_EQ(foxglove.host, "0.0.0.0");
    EXPECT_EQ(foxglove.port, 8765);

    EXPECT_EQ(vision.trajectory.model->type, fcs::core::trajectory::model::ModelType::Ideal);
    EXPECT_DOUBLE_EQ(vision.trajectory.model->gravity, 9.81);
    EXPECT_TRUE(vision.at_legacy->traditional.advanced_binary);
    EXPECT_EQ(vision.at_legacy->default_detect_color, fcs::ArmorColor::Blue);
}

TEST(ReadConfig, MissingRequiredLaunchFieldReturnsEntryConfigError) {
    const auto temp_dir = make_temp_dir("missing-launch-field");
    {
        const ScopedCurrentPath cwd(temp_dir);
        std::ofstream(temp_dir / "at_vision.toml") << "daedalus = false\nvision = 'std'\n";

        auto result = read_config();
        ASSERT_FALSE(result.has_value());
        ASSERT_TRUE(std::holds_alternative<std::string>(result.error()));
        EXPECT_EQ(std::get<std::string>(result.error()), "entry config: Missing key 'robot'");
    }
    std::filesystem::remove_all(temp_dir);
}

TEST(ReadConfig, NonDaedalusLaunchReturnsHardwareAndMergedBaseVision) {
    const auto temp_dir = make_temp_dir("non-daedalus-success");
    copy_file_from_repo("config/vision_base.toml", temp_dir / "config/vision_base.toml");
    copy_file_from_repo("config/robot/hero.toml", temp_dir / "config/robot/hero.toml");
    write_file(temp_dir / "config/vision/std.toml", "");
    write_file(temp_dir / "at_vision.toml", "daedalus = false\nrobot = 'hero'\nvision = 'std'\n");

    {
        const ScopedCurrentPath cwd(temp_dir);
        auto result = read_config();
        ASSERT_TRUE(result.has_value()) << to_string(result.error());

        auto [robot, daedalus, hardware, foxglove, vision] = std::move(*result);
        EXPECT_EQ(robot, fcs::RobotType::Hero);
        EXPECT_FALSE(daedalus);
        ASSERT_TRUE(hardware.has_value());
        EXPECT_EQ(hardware->type.get(), fcs::RobotType::Hero);
        EXPECT_TRUE(foxglove.enabled);
        EXPECT_EQ(
            vision.trajectory.model->type, fcs::core::trajectory::model::ModelType::LinearDrag);
    }

    std::filesystem::remove_all(temp_dir);
}

TEST(ReadConfig, InvalidMergedVisionConfigIsReported) {
    const auto temp_dir = make_temp_dir("invalid-vision-config");
    write_file(temp_dir / "config/vision_base.toml", "backend_type = [1]\n");
    write_file(temp_dir / "config/vision/std.toml", "");
    write_file(temp_dir / "at_vision.toml", "daedalus = true\nrobot = 'hero'\nvision = 'std'\n");

    {
        const ScopedCurrentPath cwd(temp_dir);
        auto result = read_config();
        ASSERT_FALSE(result.has_value());
        ASSERT_TRUE(std::holds_alternative<std::string>(result.error()));
        const auto& error = std::get<std::string>(result.error());
        EXPECT_NE(error.find("vision config:"), std::string::npos);
        EXPECT_NE(error.find("backend_type"), std::string::npos);
    }

    std::filesystem::remove_all(temp_dir);
}

TEST(ReadConfig, InvalidHardwareConfigIsReported) {
    const auto temp_dir = make_temp_dir("invalid-hardware-config");
    copy_file_from_repo("config/vision_base.toml", temp_dir / "config/vision_base.toml");
    write_file(temp_dir / "config/vision/std.toml", "");
    write_file(temp_dir / "config/robot/hero.toml", "");
    write_file(temp_dir / "at_vision.toml", "daedalus = false\nrobot = 'hero'\nvision = 'std'\n");

    {
        const ScopedCurrentPath cwd(temp_dir);
        auto result = read_config();
        ASSERT_FALSE(result.has_value());
        ASSERT_TRUE(std::holds_alternative<std::string>(result.error()));
        EXPECT_EQ(std::get<std::string>(result.error()), "hardware config: Missing key 'type'");
    }

    std::filesystem::remove_all(temp_dir);
}

} // namespace
