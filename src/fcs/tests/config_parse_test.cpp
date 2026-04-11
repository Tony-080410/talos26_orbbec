#include <gtest/gtest.h>

#include <array>
#include <expected>
#include <filesystem>
#include <fmt/format.h>
#include <string>

#include "config.hpp"
#include "core/trajectory/resource.hpp"
#include "runtime/l1_l2_setup.hpp"
#include "scheduler/scheduler.hpp"
#include "toml_helper.hpp"

namespace {

std::filesystem::path source_root() { return std::filesystem::path(TALOS_SOURCE_DIR); }

template <typename T>
std::expected<T, std::string> parse_config_file(const std::filesystem::path& path) {
    auto parsed = toml::parse_file(path.string());
    if (!parsed) {
        return std::unexpected(path.string() + ": " + std::string(parsed.error().description()));
    }

    auto config = toml_helper::from_table<T>(parsed.table());
    if (!config) {
        return std::unexpected(path.string() + ": " + config.error());
    }
    return *config;
}

template <typename Derived, std::size_t N>
void expect_matrix_values(
    const Eigen::MatrixBase<Derived>& actual, const std::array<double, N>& expected) {
    ASSERT_EQ(static_cast<std::size_t>(actual.size()), expected.size());
    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual(i), expected[static_cast<std::size_t>(i)]) << "index=" << i;
    }
}

TEST(ConfigParse, HeroRobotTomlMatchesExpectedSemanticsAndValues) {
    auto result = parse_config_file<fcs::HardwareConfig>(source_root() / "config/robot/hero.toml");
    ASSERT_TRUE(result.has_value()) << result.error();
    const auto& config = *result;

    EXPECT_EQ(config.type.get(), fcs::RobotType::Hero);

    expect_matrix_values(
        config.camera->camera_matrix,
        std::array<double, 9>{2333.06, 0.0, 657.19, 0.0, 2323.58, 592.108, 0.0, 0.0, 1.0});
    expect_matrix_values(
        config.camera->distort_coefficient,
        std::array<double, 5>{-0.103011, 0.125266, 0.00137156, -0.00445701, 0.347987});
    EXPECT_EQ(config.camera->width, 1440u);
    EXPECT_EQ(config.camera->height, 1080u);

    EXPECT_FALSE(config.camera->profile.trigger_mode);
    EXPECT_FALSE(config.camera->profile.invert_image);
    EXPECT_EQ(config.camera->profile.exposure_time_us, 5000u);
    EXPECT_DOUBLE_EQ(config.camera->profile.gain, 16.7);
    EXPECT_EQ(config.camera->profile.rotate_angle, fcs::RotateType::None);

    expect_matrix_values(
        config.extrinsic->gimbal_yaw.gimbal_pitch.translation,
        std::array<double, 3>{-0.06492, 0.0, 0.164});
    expect_matrix_values(
        config.extrinsic->gimbal_yaw.gimbal_pitch.camera_link.translation,
        std::array<double, 3>{0.25391, 0.0735, 0.00639});
    EXPECT_DOUBLE_EQ(config.extrinsic->gimbal_yaw.gimbal_pitch.camera_link.roll, 0.0);
    EXPECT_DOUBLE_EQ(config.extrinsic->gimbal_yaw.gimbal_pitch.camera_link.pitch, -1.0);
    EXPECT_DOUBLE_EQ(config.extrinsic->gimbal_yaw.gimbal_pitch.camera_link.yaw, -1.0);
    expect_matrix_values(
        config.extrinsic->gimbal_yaw.gimbal_pitch.muzzle_link.translation,
        std::array<double, 3>{0.17605, 0.0, -0.03332});

    EXPECT_EQ(config.mcu->mcu_vendor_id, 0x0483);
    EXPECT_FALSE(config.mcu->mcu_product_id.has_value());
    EXPECT_TRUE(config.mcu->untrust_roll);
    EXPECT_TRUE(config.mcu->dynamic_self_color);
    EXPECT_TRUE(config.mcu->dynamic_bullet_speed);
    EXPECT_DOUBLE_EQ(config.mcu->bullet_speed_default, 11.0);
    EXPECT_DOUBLE_EQ(config.mcu->bullet_speed_min, 10.0);
    EXPECT_DOUBLE_EQ(config.mcu->bullet_speed_max, 20.0);
}

TEST(ConfigParse, VisionBaseTomlMatchesExpectedSemanticsAndValues) {
    auto result = parse_config_file<fcs::VisionConfig>(source_root() / "config/vision_base.toml");
    ASSERT_TRUE(result.has_value()) << result.error();
    const auto& config = *result;

    EXPECT_EQ(config.at_legacy->backend_type, fcs::L2::BackendTypeConfig::NeuralNetworkTup);
    EXPECT_EQ(config.at_legacy->default_detect_color, fcs::ArmorColor::Blue);

    EXPECT_FALSE(config.at_legacy->traditional.advanced_binary);
    EXPECT_FLOAT_EQ(config.at_legacy->traditional.dark_percentage, 0.5F);
    EXPECT_EQ(config.at_legacy->traditional.binary_threshold, 85);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.light_min_ratio, 0.0001);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.light_max_ratio, 1.0);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.light_max_angle, 40.0);
    EXPECT_EQ(config.at_legacy->traditional.light_color_diff_thresh, 30);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.armor_min_light_ratio, 0.8);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.armor_min_small_center_distance, 0.8);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.armor_max_small_center_distance, 3.5);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.armor_min_large_center_distance, 3.5);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.armor_max_large_center_distance, 8.0);
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.armor_max_angle, 40.0);
    EXPECT_EQ(config.at_legacy->traditional.classifier_model_path, "models/mlp.onnx");
    EXPECT_DOUBLE_EQ(config.at_legacy->traditional.classifier_confidence_threshold, 0.5);
    EXPECT_TRUE(config.at_legacy->traditional.classifier_enable_type_filtering);
    EXPECT_TRUE(config.at_legacy->traditional.classifier_use_softmax);

    EXPECT_EQ(config.at_legacy->nn.model_path, "models/best.onnx");
    EXPECT_EQ(config.at_legacy->nn.device, "CPU");
    EXPECT_EQ(config.at_legacy->nn.device_priorities, "");
    EXPECT_FALSE(config.at_legacy->nn.enable_profiling);
    EXPECT_FALSE(config.at_legacy->nn.enable_multi_thread);
    EXPECT_DOUBLE_EQ(config.at_legacy->nn.confidence_threshold, 0.75);
    EXPECT_DOUBLE_EQ(config.at_legacy->nn.nms_threshold, 0.30);
    EXPECT_EQ(config.at_legacy->nn.top_k, 128);

    EXPECT_EQ(config.at_legacy->nn_tup.model_path, "models/opt-1208-001.onnx");
    EXPECT_EQ(config.at_legacy->nn_tup.device, "CPU");
    EXPECT_EQ(config.at_legacy->nn_tup.device_priorities, "");
    EXPECT_FALSE(config.at_legacy->nn_tup.enable_profiling);
    EXPECT_FALSE(config.at_legacy->nn_tup.enable_multi_thread);
    EXPECT_DOUBLE_EQ(config.at_legacy->nn_tup.confidence_threshold, 0.75);
    EXPECT_DOUBLE_EQ(config.at_legacy->nn_tup.nms_threshold, 0.30);
    EXPECT_EQ(config.at_legacy->nn_tup.top_k, 128);

    EXPECT_EQ(config.at_legacy->trt.model_path, "models/best_fp16.onnx");
    EXPECT_EQ(config.at_legacy->trt.engine_path, "models/best_fp16_sm87_fp16.engine");
    EXPECT_EQ(config.at_legacy->trt.engine_cache_dir, "/tmp/tensorrt_cache");
    EXPECT_EQ(config.at_legacy->trt.device_id, 0);
    EXPECT_EQ(config.at_legacy->trt.num_streams, 2);
    EXPECT_EQ(config.at_legacy->trt.compute_capability, "87");
    EXPECT_TRUE(config.at_legacy->trt.enable_fp16);
    EXPECT_TRUE(config.at_legacy->trt.enable_dla);
    EXPECT_EQ(config.at_legacy->trt.dla_core, 0);
    EXPECT_DOUBLE_EQ(config.at_legacy->trt.confidence_threshold, 0.20);
    EXPECT_DOUBLE_EQ(config.at_legacy->trt.nms_threshold, 0.30);
    EXPECT_EQ(config.at_legacy->trt.top_k, 128);

    EXPECT_EQ(config.at_legacy->axera.model_path, "models/compiled.axmodel");
    EXPECT_EQ(config.at_legacy->axera.input_width, 768);
    EXPECT_EQ(config.at_legacy->axera.input_height, 576);
    EXPECT_EQ(config.at_legacy->axera.num_colors, 4);
    EXPECT_EQ(config.at_legacy->axera.num_kpts, 4);
    EXPECT_EQ(config.at_legacy->axera.num_pairs, 12);
    EXPECT_EQ(config.at_legacy->axera.strides.size(), 3u);
    EXPECT_EQ(config.at_legacy->axera.strides[0], 8);
    EXPECT_EQ(config.at_legacy->axera.strides[1], 16);
    EXPECT_EQ(config.at_legacy->axera.strides[2], 32);
    EXPECT_DOUBLE_EQ(config.at_legacy->axera.confidence_threshold, 0.15);
    EXPECT_DOUBLE_EQ(config.at_legacy->axera.nms_threshold, 0.70);
    EXPECT_EQ(config.at_legacy->axera.top_k, 128);
    EXPECT_TRUE(config.at_legacy->readback_roi.enabled);
    EXPECT_DOUBLE_EQ(config.at_legacy->readback_roi.stale_timeout_s, 0.20);
    EXPECT_DOUBLE_EQ(config.at_legacy->readback_roi.margin_ratio_x, 0.20);
    EXPECT_DOUBLE_EQ(config.at_legacy->readback_roi.margin_ratio_y, 0.20);
    EXPECT_DOUBLE_EQ(config.at_legacy->readback_roi.box_size_m[0], 1.0);
    EXPECT_DOUBLE_EQ(config.at_legacy->readback_roi.box_size_m[1], 1.0);
    EXPECT_DOUBLE_EQ(config.at_legacy->readback_roi.box_size_m[2], 0.6);

    EXPECT_DOUBLE_EQ(config.tracker().robot.matcher_gate, 10.0);
    EXPECT_DOUBLE_EQ(config.tracker().robot.lost_threshold, 1.0);
    EXPECT_EQ(config.tracker().robot.tracking_threshold, 5u);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.sigma_a_xy, 10.0);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.sigma_a_z, 2.0);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.sigma_a_yaw, 20.0);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.sigma_r0, 1.0);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.sigma_r1, 1.0);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.sigma_h, 0.8);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.meas_yaw_var_floor, 4e-3);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.meas_pitch_var_floor, 4e-3);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.meas_dist_var_floor, 0.3);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.meas_dist_delta_angle_k, 1.0);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.meas_armor_yaw_var_floor, 9e-2);
    EXPECT_DOUBLE_EQ(config.tracker().robot.model.meas_armor_yaw_range_k, 0.005);

    EXPECT_DOUBLE_EQ(config.tracker().robot_inekf.radius0, 0.23);
    EXPECT_DOUBLE_EQ(config.tracker().robot_inekf.radius1, 0.23);
    EXPECT_DOUBLE_EQ(config.tracker().robot_inekf.height, 0.05);

    EXPECT_DOUBLE_EQ(config.tracker().outpost.matcher_gate, 4.0);
    EXPECT_DOUBLE_EQ(config.tracker().outpost.lost_threshold, 2.0);
    EXPECT_EQ(config.tracker().outpost.tracking_threshold, 4u);
    EXPECT_DOUBLE_EQ(config.tracker().outpost.model.sigma_q_xy, 10.0);
    EXPECT_DOUBLE_EQ(config.tracker().outpost.model.sigma_q_z, 2.0);
    EXPECT_DOUBLE_EQ(config.tracker().outpost.model.sigma_a_yaw, 8.0);
    EXPECT_DOUBLE_EQ(config.tracker().outpost.model.meas_dist_k, 0.43);
    EXPECT_DOUBLE_EQ(config.tracker().outpost.model.yaw_log_k, 0.005);

    EXPECT_DOUBLE_EQ(config.l4->fire_decision.fire_thresh, 0.01744444);
    EXPECT_DOUBLE_EQ(config.l4->fire_decision.delay_gate_error_rad, 3.1744444);
    EXPECT_DOUBLE_EQ(config.l4->fire_decision.shooting_range_h, 0.135);
    EXPECT_DOUBLE_EQ(config.l4->fire_decision.shooting_range_w, 0.135);
    EXPECT_DOUBLE_EQ(config.l4->aimer.delay, 0.04);
    EXPECT_DOUBLE_EQ(config.l4->aimer.single_whole_armor_v_yaw_range[0], 1.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.single_whole_armor_v_yaw_range[1], 2.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.whole_armor_center_v_yaw_range[0], 6.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.whole_armor_center_v_yaw_range[1], 7.5);
    EXPECT_EQ(config.l4->aimer.single_to_whole_armor_frames, 20);
    EXPECT_EQ(config.l4->aimer.phase_transition_frames, 50);
    EXPECT_DOUBLE_EQ(config.l4->aimer.front_window_deg, 90.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.coming_angle_deg, 60.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.leaving_angle_deg, 20.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.yaw_limit_deg, 40.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.min_enable_yaw_deg, 0.25);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.temp_lost_state_score, 0.05);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.optical_stale_timeout_s, 0.25);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.switch_margin, 0.08);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.weights.image_center, 1.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.weights.track_state, 3.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.weights.tof, 1.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.weights.distance, 1.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.weights.gimbal_effort, 1.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.weights.armor_name, 6.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.refs.image_center_ref_px, 300.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.refs.tof_ref_s, 0.35);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.refs.distance_ref_m, 10.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.refs.yaw_effort_ref_deg, 20.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.refs.pitch_effort_ref_deg, 10.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.sentry, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.one, 1.0);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.two, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.three, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.four, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.five, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.outpost, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.base, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.base_large, 0.5);
    EXPECT_DOUBLE_EQ(config.l4->aimer.target_selection.armor_name_score.invalid, 0.0);

    EXPECT_EQ(config.trajectory.model->type, fcs::core::trajectory::model::ModelType::LinearDrag);
    EXPECT_EQ(config.trajectory.model->iteration_times, 10u);
    EXPECT_DOUBLE_EQ(config.trajectory.model->gravity, 9.7997);
    EXPECT_DOUBLE_EQ(config.trajectory.model->resistance, 0.0005);
    EXPECT_DOUBLE_EQ(config.trajectory.model->diameter, 0.017);
    EXPECT_DOUBLE_EQ(config.trajectory.model->mass, 0.00312);
    EXPECT_DOUBLE_EQ(config.trajectory.model->rho, 1.225);

    EXPECT_TRUE(config.weapon().enabled);
    EXPECT_TRUE(config.weapon().enable_debug);
    EXPECT_DOUBLE_EQ(config.weapon().pitch_min, -0.69813169);
    EXPECT_DOUBLE_EQ(config.weapon().pitch_max, 0.69813169);
    EXPECT_EQ(config.weapon().mpc.horizon_ahead, 50);
    EXPECT_EQ(config.weapon().mpc.horizon_back, 50);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.dt, 0.01);
    EXPECT_EQ(config.weapon().mpc.max_iterations, 10);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.abs_tol, 0.001);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.rho, 2.0);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.yaw.q_pos, 9e5);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.yaw.q_vel, 0.0);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.yaw.r, 10.0);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.yaw.max_acc, 50.0);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.pitch.q_pos, 9e5);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.pitch.q_vel, 0.0);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.pitch.r, 100.0);
    EXPECT_DOUBLE_EQ(config.weapon().mpc.pitch.max_acc, 100.0);

    EXPECT_FALSE(config.rune_detector.enabled);
    EXPECT_EQ(config.rune_detector.detect_color, fcs::ArmorColor::Red);
    EXPECT_EQ(config.rune_detector.arrow_threshold, 170);
    EXPECT_EQ(config.rune_detector.target_threshold, 130);
    EXPECT_EQ(config.rune_detector.rcenter_threshold, 120);

    EXPECT_TRUE(config.energy_meter.is_big_rune);
    EXPECT_DOUBLE_EQ(config.energy_meter.control_delay_s, 0.5);
    EXPECT_DOUBLE_EQ(config.energy_meter.angle_fitter.smooth_alpha, 0.8);
    EXPECT_EQ(config.energy_meter.angle_fitter.min_data_points, 15);
    EXPECT_EQ(config.energy_meter.tracker.lost_thres, 10);
    EXPECT_EQ(config.energy_meter.tracker.tracking_thres, 3);
    EXPECT_DOUBLE_EQ(config.energy_meter.tracker.matcher_gate, 25.0);
    EXPECT_DOUBLE_EQ(config.energy_meter.tracker.energy_params.sigma_xy, 10.0);
    EXPECT_DOUBLE_EQ(config.energy_meter.tracker.energy_params.sigma_z, 1.0);
    EXPECT_DOUBLE_EQ(config.energy_meter.tracker.energy_params.sigma_roll, 1.0);
    EXPECT_DOUBLE_EQ(config.energy_meter.tracker.energy_params.radius, 0.7);
    EXPECT_DOUBLE_EQ(config.energy_meter.tracker.energy_params.meas_dist_k, 0.43);
}

TEST(ConfigParse, InlineHelpersAndFormattersBehaveAsExpected) {
    auto hardware_result =
        parse_config_file<fcs::HardwareConfig>(source_root() / "config/robot/hero.toml");
    ASSERT_TRUE(hardware_result.has_value()) << hardware_result.error();
    const auto& hardware = *hardware_result;

    auto vision_result =
        parse_config_file<fcs::VisionConfig>(source_root() / "config/vision_base.toml");
    ASSERT_TRUE(vision_result.has_value()) << vision_result.error();
    const auto& vision = *vision_result;

    EXPECT_EQ(fmt::format("{}", hardware.type.get()), "Hero");
    EXPECT_EQ(fmt::format("{}", hardware.camera->profile.rotate_angle), "None");

    const auto rotation = hardware.extrinsic->gimbal_yaw.gimbal_pitch.camera_link.rotation();
    const auto [roll, pitch, yaw] = rotation.rpy();
    EXPECT_DOUBLE_EQ(roll, 0.0);
    EXPECT_DOUBLE_EQ(pitch, -1.0);
    EXPECT_DOUBLE_EQ(yaw, -1.0);

    const auto tracker_ptr = vision.l3->tracker_ptr();
    ASSERT_TRUE(tracker_ptr);
    EXPECT_DOUBLE_EQ(tracker_ptr->robot.matcher_gate, vision.tracker().robot.matcher_gate);
    EXPECT_DOUBLE_EQ(tracker_ptr->outpost.model.yaw_log_k, 0.005);

    EXPECT_TRUE(fcs::L4::is_on_target(vision.l4->fire_decision, 0.0, 0.0, 0.001, 0.001, 100.0));
    EXPECT_FALSE(fcs::L4::is_on_target(vision.l4->fire_decision, 0.0, 0.0, 0.2, 0.2, 100.0));
}

TEST(ConfigParse, MutableAccessorsAndTrajectoryHelpersUseWrappedConfigFields) {
    auto vision_result =
        parse_config_file<fcs::VisionConfig>(source_root() / "config/vision_base.toml");
    ASSERT_TRUE(vision_result.has_value()) << vision_result.error();
    auto vision = *vision_result;

    vision.tracker().robot.matcher_gate = 12.0;
    EXPECT_DOUBLE_EQ(vision.l3->tracker.robot.matcher_gate, 12.0);

    vision.weapon().enable_debug = false;
    EXPECT_FALSE(vision.l5->mpc_weapon.enable_debug);

    fcs::core::trajectory::TrajectoryConfig ideal{};
    ideal.model->type    = fcs::core::trajectory::model::ModelType::Ideal;
    ideal.model->gravity = 9.81;
    auto ideal_solver    = fcs::core::trajectory::solver::create_solver(ideal);
    ASSERT_NE(ideal_solver, nullptr);

    fcs::core::trajectory::TrajectoryConfig linear{};
    linear.model->type       = fcs::core::trajectory::model::ModelType::LinearDrag;
    linear.model->gravity    = 9.79;
    linear.model->resistance = 0.001;
    auto linear_solver       = fcs::core::trajectory::solver::create_solver(linear);
    ASSERT_NE(linear_solver, nullptr);

    talos::World world;
    talos::Scheduler scheduler(world);
    fcs::core::trajectory::register_resource(scheduler, std::move(linear));
}

TEST(ConfigParse, SetupL1CoversUsbMcuWrapperBranchesBeforeStubCameraFails) {
    auto hardware_result =
        parse_config_file<fcs::HardwareConfig>(source_root() / "config/robot/hero.toml");
    ASSERT_TRUE(hardware_result.has_value()) << hardware_result.error();
    auto hardware = *hardware_result;

    {
        talos::World world;
        talos::Scheduler scheduler(world);
        auto result = fcs::runtime::setup_l1(world, scheduler, false, &hardware);
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("init hik input"), std::string::npos);
    }

    hardware.mcu->mcu_product_id = 0x1234;
    {
        talos::World world;
        talos::Scheduler scheduler(world);
        auto result = fcs::runtime::setup_l1(world, scheduler, false, &hardware);
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("init hik input"), std::string::npos);
    }
}

} // namespace
