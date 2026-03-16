// Example usage of reflection-based parsing in toml_helper.
//
// This example demonstrates how to use:
// - std::optional<T> for required/defaulted fields
// - std::vector<T> and std::array<T, N>
// - toml_helper::flatten<T> for serde-like flattened structs

#include <array>
#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <vector>

#include "toml/core.hpp"
#include "toml/ext/containers.hpp"

struct CameraIntrinsics {
    double fx = 1000.0;
    double fy = 1000.0;
    std::optional<double> cx{};
    std::optional<double> cy = 540.0;
};

struct RobotConfig {
    // Required because the default is nullopt.
    std::optional<std::string> team_color{};
    // Defaulted because the optional starts engaged.
    std::optional<double> max_speed = 8.0;

    std::vector<int> camera_ids;
    std::vector<std::string> allies;

    std::array<double, 3> position{0.0, 0.0, 0.0};
    std::array<double, 4> orientation{1.0, 0.0, 0.0, 0.0};

    toml_helper::flatten<CameraIntrinsics> camera{};
};

int main() {
    constexpr std::string_view config_toml = R"(
        team_color = "blue"

        camera_ids = [0, 1, 2]
        allies = ["robot1", "robot2"]

        position = [1.0, 2.0, 3.0]
        orientation = [1.0, 0.0, 0.0, 0.0]

        fx = 920.0
        fy = 918.0
        cx = 640.0
    )";

    auto result = toml::parse(config_toml);
    if (!result) {
        std::cerr << "Failed to parse TOML\n";
        return 1;
    }

    auto config = toml_helper::from_table<RobotConfig>(result.table());
    if (!config) {
        std::cerr << fmt::format("Failed to read config: {}\n", config.error());
        return 1;
    }

    auto unread = toml_helper::ensure_all_read(result.table(), "robot");
    if (!unread) {
        std::cerr << fmt::format("Unread keys: {}\n", unread.error());
        return 1;
    }

    std::cout << "=== Robot Configuration ===\n";
    std::cout << fmt::format("Team Color: {}\n", *config->team_color);
    std::cout << fmt::format("Max Speed: {} m/s\n", *config->max_speed);
    std::cout << fmt::format("Camera IDs: [{}]\n", fmt::join(config->camera_ids, ", "));
    std::cout << fmt::format("Allies: [{}]\n", fmt::join(config->allies, ", "));
    std::cout << fmt::format(
        "Position: [{}, {}, {}]\n", config->position[0], config->position[1], config->position[2]);
    std::cout << fmt::format(
        "Orientation: [{}, {}, {}, {}]\n", config->orientation[0], config->orientation[1],
        config->orientation[2], config->orientation[3]);
    std::cout << fmt::format(
        "Camera Intrinsics: fx={}, fy={}, cx={}, cy={}\n", config->camera->fx, config->camera->fy,
        *config->camera->cx, *config->camera->cy);

    return 0;
}
