#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "L2_perception/rune/types.hpp"
#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/selected_target_snapshot.hpp"
#include "L4_planning/target_selection_trace.hpp"
#include "L5_weapon/fire_control.hpp"
#include "config.hpp"
#include "core/channel_topics.hpp"
#include "core/runtime.hpp"
#include "core/time.hpp"
#include "core/trajectory/resource.hpp"
#include "core/types.hpp"
#include "runtime/capturer.hpp"
#include "scheduler/scheduler.hpp"

namespace {

namespace fs = std::filesystem;
using json   = nlohmann::json;
using namespace std::chrono_literals;

enum class ProducerMode {
    Continuous,
    StaleDetector,
    BurstThenIdle,
    ControlBurstThenIdle,
};

auto make_temp_dir(std::string_view name) -> fs::path {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path   = fs::temp_directory_path()
                    / ("talos-capturer-test-" + std::string(name) + "-" + std::to_string(unique));
    fs::create_directories(path);
    return path;
}

auto read_jsonl(const fs::path& path) -> std::vector<json> {
    std::ifstream in(path);
    std::vector<json> rows;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) {
            rows.push_back(json::parse(line));
        }
    }
    return rows;
}

auto read_json_file(const fs::path& path) -> json {
    std::ifstream in(path);
    return json::parse(in);
}

auto read_text_file(const fs::path& path) -> std::string {
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

auto find_run_dir(const fs::path& output_dir) -> fs::path {
    for (const auto& entry : fs::directory_iterator(output_dir)) {
        if (entry.is_directory()) {
            return entry.path();
        }
    }
    return {};
}

auto source_root() -> fs::path { return fs::path(TALOS_SOURCE_DIR); }

void register_test_producer(talos::Scheduler& scheduler, ProducerMode mode) {
    scheduler.add_system<talos::fixed_rate<20>>(
        "capturer_test_producer",
        [mode, start_ns = fcs::clock::now_ns()](
            talos::spmc_mut<fcs::AtLegacyImageFrame, fcs::AtLegacyImageChannelTopic> image_out,
            talos::spmc_mut<fcs::AtLegacyArmorDetectionBatch, fcs::AtLegacyDetectionChannelTopic>
                detection_out,
            talos::spmc_mut<
                fcs::AtLegacyArmorMeasurementBatch, fcs::AtLegacyMeasurementChannelTopic>
                measurement_out,
            talos::spmc_mut<fcs::L3::TrackerOutputs, fcs::TrackerOutputChannelTopic> tracker_out,
            talos::spmc_mut<fcs::rune::RuneObservation, fcs::RuneObservationChannelTopic>
                rune_observation_out,
            talos::spmc_mut<fcs::rune::RuneDebugFrame, fcs::RuneDebugFrameChannelTopic>
                rune_debug_out,
            talos::spmc_mut<fcs::energy_meter::EnergyMeterState, fcs::EnergyMeterStateChannelTopic>
                energy_meter_out,
            talos::spmc_mut<
                fcs::L4::SelectedTargetSnapshot, fcs::SelectedTargetSnapshotChannelTopic>
                selected_target_out,
            talos::spmc_mut<fcs::L4::TargetSelectionTrace, fcs::TargetSelectionTraceChannelTopic>
                target_selection_trace_out,
            talos::spmc_mut<fcs::L4::GimbalPlan, fcs::GimbalPlanChannelTopic> gimbal_plan_out,
            talos::spmc_mut<fcs::L5::WeaponCommand, fcs::WeaponCommandChannelTopic> weapon_out,
            talos::spmc_mut<
                fcs::core::ControlResourceSnapshot, fcs::RuntimeControlStateChannelTopic>
                control_state_out) mutable {
            const uint64_t now_ns       = fcs::clock::now_ns();
            const bool stale_phase      = (now_ns - start_ns) >= 900'000'000ULL;
            const bool burst_idle_phase = (now_ns - start_ns) >= 1200'000'000ULL;
            const uint64_t camera_frame = stale_phase ? 2 : 1;
            const cv::Scalar color = stale_phase ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
            cv::Mat image(4, 4, CV_8UC3, color);

            image_out.write(fcs::AtLegacyImageFrame{image.clone(), now_ns, camera_frame});

            const bool publish_detector = [&]() {
                switch (mode) {
                case ProducerMode::Continuous:
                case ProducerMode::ControlBurstThenIdle: return true;
                case ProducerMode::StaleDetector: return !stale_phase;
                case ProducerMode::BurstThenIdle: return !burst_idle_phase;
                }
                return false;
            }();

            std::optional<fcs::L3::TrackerOutput> selected_tracker;
            if (publish_detector) {
                detection_out.write(
                    fcs::AtLegacyArmorDetectionBatch{{}, image.clone(), now_ns, camera_frame});
                measurement_out.write(fcs::AtLegacyArmorMeasurementBatch{{}, now_ns, camera_frame});

                fcs::L3::TrackerOutputs trackers;
                fcs::L3::TrackerOutput tracker;
                tracker.timestamp_ns = now_ns;
                tracker.status       = stale_phase ? fcs::L3::TrackerStatus::Tracking
                                                   : fcs::L3::TrackerStatus::Detecting;
                tracker.target_name  = stale_phase ? fcs::ArmorName::Two : fcs::ArmorName::One;
                selected_tracker     = tracker;
                trackers.push_back(tracker);
                tracker_out.write(trackers);
            }

            fcs::rune::RuneObservation rune_observation;
            rune_observation.timestamp_ns = now_ns;
            rune_observation.frame_id     = camera_frame;
            rune_observation_out.write(rune_observation);

            fcs::rune::RuneDebugFrame rune_debug;
            rune_debug.timestamp_ns = now_ns;
            rune_debug.frame_id     = camera_frame;
            rune_debug.enabled      = true;
            rune_debug_out.write(rune_debug);

            fcs::energy_meter::EnergyMeterState energy_meter;
            energy_meter.timestamp_ns = now_ns;
            energy_meter_out.write(energy_meter);

            fcs::L4::SelectedTargetSnapshot selected_target;
            selected_target.timestamp_ns = now_ns;
            if (selected_tracker) {
                selected_target.distance            = 1.0;
                selected_target.predicted_future_ns = now_ns + 10'000'000ULL;
                selected_target.source              = fcs::L4::GimbalPlanSource::Armor;
                selected_target.tracker             = *selected_tracker;
            }
            selected_target_out.write(selected_target);

            fcs::L4::TargetSelectionTrace target_selection_trace;
            target_selection_trace.timestamp_ns = now_ns;
            target_selection_trace_out.write(target_selection_trace);

            fcs::L4::GimbalPlan gimbal_plan;
            gimbal_plan.timestamp_ns = now_ns;
            gimbal_plan_out.write(gimbal_plan);

            fcs::L5::WeaponCommand weapon_command;
            weapon_command.timestamp_ns = now_ns;
            weapon_out.write(weapon_command);

            fcs::core::ImuState imu_state;
            imu_state.timestamp_ns           = now_ns;
            imu_state.yaw                    = stale_phase ? 0.2 : 0.1;
            imu_state.pitch                  = stale_phase ? 0.05 : -0.05;
            imu_state.roll                   = 0.0;
            imu_state.yaw_vel                = 0.01;
            imu_state.pitch_vel              = -0.02;
            imu_state.roll_vel               = 0.0;
            imu_state.attitude_valid         = true;
            imu_state.angular_velocity_valid = true;

            const bool publish_control_state = [&]() {
                switch (mode) {
                case ProducerMode::Continuous:
                case ProducerMode::StaleDetector:
                case ProducerMode::BurstThenIdle: return true;
                case ProducerMode::ControlBurstThenIdle: return !burst_idle_phase;
                }
                return false;
            }();

            if (publish_control_state) {
                control_state_out.write(
                    fcs::core::ControlResourceSnapshot{
                        .sample_timestamp_ns = now_ns,
                        .imu                 = imu_state,
                        .detecting_color     = fcs::ArmorColor::Blue,
                        .bullet_speed        = 20.0,
                    });
            }
        });
}

void run_scheduler_for(talos::Scheduler& scheduler, std::chrono::milliseconds duration) {
    const auto build_result = scheduler.build();
    ASSERT_TRUE(build_result.has_value()) << "scheduler build failed";

    std::optional<talos::SchedulerError> run_error;
    std::thread scheduler_thread([&]() {
        if (const auto result = scheduler.run(); !result) {
            run_error = result.error();
        }
    });

    std::this_thread::sleep_for(duration);
    scheduler.stop();
    scheduler_thread.join();
    ASSERT_FALSE(run_error.has_value());
}

void seed_runtime_resources(talos::World& world) {
    (void)world.emplace_resource<fast_tf::CoordinateSystem>();
    world.insert_resource(fcs::core::ImuState{});
    world.insert_resource(fcs::ArmorColor::Blue);
    world.insert_resource(fcs::core::trajectory::bullet_speed_data{.bullet_speed = 20.0});
}

TEST(Capturer, LowDiskStillWritesSnapshotRows) {
    const fs::path output_dir = make_temp_dir("low-disk");

    talos::World world;
    talos::Scheduler scheduler(world);
    seed_runtime_resources(world);

    register_test_producer(scheduler, ProducerMode::Continuous);

    fcs::CapturerConfig config;
    config.enabled             = true;
    config.output_dir          = output_dir.string();
    config.reserved_free_bytes = std::numeric_limits<uint64_t>::max();

    fcs::runtime::register_runtime_capturer_system(
        scheduler, config, {.daedalus = false, .robot = "hero", .vision = "std"},
        fcs::RobotType::Hero, fcs::CameraConfig{});

    run_scheduler_for(scheduler, 2200ms);

    const fs::path run_dir = find_run_dir(output_dir);
    ASSERT_FALSE(run_dir.empty());

    const auto rows = read_jsonl(run_dir / "snapshots.jsonl");
    ASSERT_GE(rows.size(), 2u);

    for (const auto& row : rows) {
        EXPECT_EQ(row.at("capture_status"), "low_disk");
        ASSERT_TRUE(row.at("streams").contains("selected_target"));
    }

    fs::remove_all(output_dir);
}

TEST(Capturer, RunMetadataStoresConfigSnapshotCopies) {
    const fs::path output_dir = make_temp_dir("config-snapshot");

    talos::World world;
    talos::Scheduler scheduler(world);
    seed_runtime_resources(world);

    fcs::CapturerConfig config;
    config.enabled             = true;
    config.output_dir          = output_dir.string();
    config.reserved_free_bytes = 0;

    fcs::runtime::register_runtime_capturer_system(
        scheduler, config, {.daedalus = false, .robot = "hero", .vision = "std"},
        fcs::RobotType::Hero, fcs::CameraConfig{});

    const fs::path run_dir = find_run_dir(output_dir);
    ASSERT_FALSE(run_dir.empty());

    const auto run_json = read_json_file(run_dir / "run.json");
    ASSERT_TRUE(run_json.contains("config_snapshot"));

    const auto& config_snapshot = run_json.at("config_snapshot");
    EXPECT_EQ(config_snapshot.at("dir").get<std::string>(), "config");

    const auto& files = config_snapshot.at("files");
    const std::array<std::pair<std::string_view, fs::path>, 4> expected_files{
        {
         {"entry", "at_vision.toml"},
         {"vision_base", "config/vision_base.toml"},
         {"vision_override", "config/vision/std.toml"},
         {"hardware", "config/robot/hero.toml"},
         }
    };

    for (const auto& [key, source_path] : expected_files) {
        ASSERT_TRUE(files.contains(std::string(key)));
        const auto& file = files.at(std::string(key));

        EXPECT_EQ(file.at("source_path").get<std::string>(), source_path.generic_string());
        ASSERT_TRUE(file.at("snapshot_path").is_string());

        const fs::path snapshot_path = run_dir / file.at("snapshot_path").get<std::string>();
        ASSERT_TRUE(fs::exists(snapshot_path));
        EXPECT_EQ(read_text_file(snapshot_path), read_text_file(source_root() / source_path));
    }

    fs::remove_all(output_dir);
}

TEST(Capturer, StaleDetectorKeepsItsOwnImagePath) {
    const fs::path output_dir = make_temp_dir("stale-detector");

    talos::World world;
    talos::Scheduler scheduler(world);
    seed_runtime_resources(world);

    register_test_producer(scheduler, ProducerMode::StaleDetector);

    fcs::CapturerConfig config;
    config.enabled             = true;
    config.output_dir          = output_dir.string();
    config.reserved_free_bytes = 0;

    fcs::runtime::register_runtime_capturer_system(
        scheduler, config, {.daedalus = false, .robot = "hero", .vision = "std"},
        fcs::RobotType::Hero, fcs::CameraConfig{});

    run_scheduler_for(scheduler, 3500ms);

    const fs::path run_dir = find_run_dir(output_dir);
    ASSERT_FALSE(run_dir.empty());

    const auto rows = read_jsonl(run_dir / "snapshots.jsonl");
    ASSERT_FALSE(rows.empty());

    bool found_stale_row = false;
    for (const auto& row : rows) {
        const auto& camera   = row.at("streams").at("camera");
        const auto& detector = row.at("streams").at("detector");

        if (!camera.value("present", false) || !detector.value("present", false)) {
            continue;
        }
        if (camera.at("frame_id") != 2 || detector.at("frame_id") != 1) {
            continue;
        }
        if (!detector.at("stale").get<bool>()) {
            continue;
        }

        found_stale_row = true;
        EXPECT_FALSE(detector.at("updated_this_tick").get<bool>());
        EXPECT_TRUE(detector.at("stale").get<bool>());
        ASSERT_TRUE(camera.at("image_path").is_string());
        ASSERT_TRUE(detector.at("image_path").is_string());
        EXPECT_NE(
            camera.at("image_path").get<std::string>(),
            detector.at("image_path").get<std::string>());
        break;
    }

    EXPECT_TRUE(found_stale_row);
    fs::remove_all(output_dir);
}

TEST(Capturer, SnapshotReferencesRecordedControlWindow) {
    const fs::path output_dir = make_temp_dir("control-window");

    talos::World world;
    talos::Scheduler scheduler(world);
    seed_runtime_resources(world);

    register_test_producer(scheduler, ProducerMode::Continuous);

    fcs::CapturerConfig config;
    config.enabled             = true;
    config.output_dir          = output_dir.string();
    config.reserved_free_bytes = 0;

    fcs::runtime::register_runtime_capturer_system(
        scheduler, config, {.daedalus = false, .robot = "hero", .vision = "std"},
        fcs::RobotType::Hero, fcs::CameraConfig{});

    run_scheduler_for(scheduler, 2200ms);

    const fs::path run_dir = find_run_dir(output_dir);
    ASSERT_FALSE(run_dir.empty());

    const auto rows = read_jsonl(run_dir / "snapshots.jsonl");
    ASSERT_FALSE(rows.empty());
    const auto run_json = read_json_file(run_dir / "run.json");

    bool found_written_window = false;
    for (const auto& row : rows) {
        const auto& control_window = row.at("control_window");
        if (!control_window.at("written").get<bool>()) {
            continue;
        }

        found_written_window = true;
        EXPECT_EQ(control_window.at("frequency_hz"), 250);
        EXPECT_GT(control_window.at("sample_count").get<size_t>(), 0u);
        ASSERT_TRUE(control_window.at("path").is_string());
        ASSERT_TRUE(control_window.at("chunk_header_size").is_number_unsigned());
        ASSERT_TRUE(control_window.at("sample_size").is_number_unsigned());

        const fs::path control_path = run_dir / control_window.at("path").get<std::string>();
        ASSERT_TRUE(fs::exists(control_path));
        EXPECT_GT(fs::file_size(control_path), 0u);
        EXPECT_EQ(control_window.at("encoding"), "little_endian_explicit");
        EXPECT_EQ(run_json.at("control").at("encoding"), "little_endian_explicit");
        EXPECT_EQ(
            fs::file_size(control_path),
            control_window.at("chunk_header_size").get<uint64_t>()
                + (control_window.at("sample_count").get<uint64_t>()
                   * control_window.at("sample_size").get<uint64_t>()));
        break;
    }

    EXPECT_TRUE(found_written_window);
    fs::remove_all(output_dir);
}

TEST(Capturer, ControlResourcesGoStaleWhenPublisherIdles) {
    const fs::path output_dir = make_temp_dir("control-stale");

    talos::World world;
    talos::Scheduler scheduler(world);
    seed_runtime_resources(world);

    register_test_producer(scheduler, ProducerMode::ControlBurstThenIdle);

    fcs::CapturerConfig config;
    config.enabled             = true;
    config.output_dir          = output_dir.string();
    config.reserved_free_bytes = 0;

    fcs::runtime::register_runtime_capturer_system(
        scheduler, config, {.daedalus = false, .robot = "hero", .vision = "std"},
        fcs::RobotType::Hero, fcs::CameraConfig{});

    run_scheduler_for(scheduler, 3500ms);

    const fs::path run_dir = find_run_dir(output_dir);
    ASSERT_FALSE(run_dir.empty());

    const auto rows = read_jsonl(run_dir / "snapshots.jsonl");
    ASSERT_FALSE(rows.empty());

    std::optional<uint64_t> last_fresh_sample_timestamp_ns;
    bool found_stale_row = false;
    for (const auto& row : rows) {
        const auto& resources = row.at("resources");
        if (!resources.value("present", false)) {
            continue;
        }

        const uint64_t sample_timestamp_ns = resources.at("sample_timestamp_ns").get<uint64_t>();
        if (resources.at("updated_this_tick").get<bool>()) {
            last_fresh_sample_timestamp_ns = sample_timestamp_ns;
            continue;
        }
        if (!resources.at("stale").get<bool>() || !last_fresh_sample_timestamp_ns.has_value()) {
            continue;
        }

        found_stale_row = true;
        EXPECT_EQ(sample_timestamp_ns, *last_fresh_sample_timestamp_ns);
        EXPECT_LT(sample_timestamp_ns, row.at("tick_timestamp_ns").get<uint64_t>());
        break;
    }

    EXPECT_TRUE(found_stale_row);
    fs::remove_all(output_dir);
}

TEST(Capturer, ControlResourcesExposeImuPresenceBitWhenPresent) {
    const fs::path output_dir = make_temp_dir("control-imu-present");

    talos::World world;
    talos::Scheduler scheduler(world);
    seed_runtime_resources(world);

    register_test_producer(scheduler, ProducerMode::Continuous);

    fcs::CapturerConfig config;
    config.enabled             = true;
    config.output_dir          = output_dir.string();
    config.reserved_free_bytes = 0;

    fcs::runtime::register_runtime_capturer_system(
        scheduler, config, {.daedalus = false, .robot = "hero", .vision = "std"},
        fcs::RobotType::Hero, fcs::CameraConfig{});

    run_scheduler_for(scheduler, 2200ms);

    const fs::path run_dir = find_run_dir(output_dir);
    ASSERT_FALSE(run_dir.empty());

    const auto rows = read_jsonl(run_dir / "snapshots.jsonl");
    ASSERT_FALSE(rows.empty());

    bool found_resources_row = false;
    for (const auto& row : rows) {
        const auto& resources = row.at("resources");
        if (!resources.value("present", false)) {
            continue;
        }

        found_resources_row = true;
        ASSERT_TRUE(resources.contains("imu_state"));
        const auto& imu_state = resources.at("imu_state");
        ASSERT_TRUE(imu_state.contains("present"));
        EXPECT_TRUE(imu_state.at("present").get<bool>());
        EXPECT_TRUE(imu_state.at("timestamp_ns").is_number_unsigned());
        break;
    }

    EXPECT_TRUE(found_resources_row);
    fs::remove_all(output_dir);
}

TEST(Capturer, ChannelGenerationSamplingTracksUpdatedAndStale) {
    struct TestTopic {};

    auto channel = talos::primitive::make_spmc_channel<int>();
    auto split   = channel.split();
    auto& writer = split.writer;
    auto& reader = split.reader;

    talos::spmc<int, TestTopic> wrapped_reader{.ptr_ = &reader};

    const auto sample = [&]() {
        const auto previous_generation = wrapped_reader.last_generation();
        auto value                     = wrapped_reader.read_current();
        const bool updated_this_tick =
            value.has_value() && wrapped_reader.last_generation() > previous_generation;
        return std::pair{value, updated_this_tick};
    };

    auto [initial_value, initial_updated] = sample();
    EXPECT_FALSE(initial_value.has_value());
    EXPECT_FALSE(initial_updated);

    writer.write(42);
    auto [fresh_value, fresh_updated] = sample();
    ASSERT_TRUE(fresh_value.has_value());
    EXPECT_EQ(*fresh_value, 42);
    EXPECT_TRUE(fresh_updated);

    auto [stale_value, stale_updated] = sample();
    ASSERT_TRUE(stale_value.has_value());
    EXPECT_EQ(*stale_value, 42);
    EXPECT_FALSE(stale_updated);

    writer.write(43);
    auto [next_value, next_updated] = sample();
    ASSERT_TRUE(next_value.has_value());
    EXPECT_EQ(*next_value, 43);
    EXPECT_TRUE(next_updated);
}

} // namespace
