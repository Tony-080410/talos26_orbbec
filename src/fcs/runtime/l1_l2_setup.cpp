#include "runtime/l1_l2_setup.hpp"

#include "L1_sensor/camera_interface.hpp"
#include "L1_sensor/output_interface.hpp"
#include "L5_weapon/fire_control.hpp"
#include "config.hpp"
#include "core/channel_topics.hpp"
#include "core/runtime.hpp"
#include "core/time.hpp"
#include "core/trajectory/resource.hpp"
#include "core/types.hpp"
#include "euler.hpp"
#include "frame.hpp"
#include "scheduler/scheduler.hpp"
#include "talos_gimbal/mcu_device.hpp"
#include "talos_gimbal/packet.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace fcs::runtime {
namespace {

template <fast_tf::frame From, fast_tf::frame To>
[[nodiscard]] auto capture_control_transform_snapshot(
    const fast_tf::CoordinateSystem& tf_buffer, uint64_t timestamp_ns)
    -> core::ControlTransformSnapshot {
    core::ControlTransformSnapshot snapshot;
    auto tf = fast_tf::lookup_clamped<From, To>(tf_buffer, timestamp_ns);
    if (!tf) {
        return snapshot;
    }

    snapshot.present       = true;
    const auto translation = tf->translation();
    snapshot.translation   = {translation.x(), translation.y(), translation.z()};

    const auto quaternion = tf->quaternion();
    snapshot.quaternion   = {quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()};
    return snapshot;
}

void publish_control_snapshot(
    talos::spmc_mut<core::ControlResourceSnapshot, RuntimeControlStateChannelTopic> control_out,
    const fast_tf::CoordinateSystem& tf_buffer, const core::ImuState& imu_state,
    ArmorColor detecting_color, double bullet_speed_raw, double bullet_speed,
    uint64_t sample_timestamp_ns) {
    control_out.write(
        core::ControlResourceSnapshot{
            .sample_timestamp_ns = sample_timestamp_ns,
            .imu                 = imu_state,
            .detecting_color     = detecting_color,
            .bullet_speed_raw    = bullet_speed_raw,
            .bullet_speed        = bullet_speed,
            .odom_to_gimbal_pitch =
                capture_control_transform_snapshot<fast_tf::odom, fast_tf::gimbal_pitch_fuxk_frame>(
                    tf_buffer, sample_timestamp_ns),
            .gimbal_to_camera_link =
                capture_control_transform_snapshot<fast_tf::gimbal, fast_tf::camera>(
                    tf_buffer, sample_timestamp_ns),
            .odom_to_camera_optical =
                capture_control_transform_snapshot<fast_tf::odom, fast_tf::camera_optical>(
                    tf_buffer, sample_timestamp_ns),
            .odom_to_muzzle =
                capture_control_transform_snapshot<fast_tf::odom, fast_tf::muzzle_link_fuxk_frame>(
                    tf_buffer, sample_timestamp_ns),
        });
}

struct DaedalusImu {
    std::shared_ptr<ipc::ShmClient> ipc_device;

    void system(
        talos::res_mut<fast_tf::CoordinateSystem> tf_buffer,
        [[maybe_unused]] core::trajectory::bullet_speed_mut bullet_speed,
        [[maybe_unused]] core::detecting_color_mut detecting_color,
        [[maybe_unused]] core::imu_state_mut imu_state,
        talos::spmc_mut<core::ControlResourceSnapshot, RuntimeControlStateChannelTopic>
            control_out) noexcept {
        bool control_updated           = false;
        uint64_t control_timestamp_ns  = 0;
        const auto note_control_update = [&](uint64_t timestamp_ns) {
            control_updated      = true;
            control_timestamp_ns = std::max(control_timestamp_ns, timestamp_ns);
        };

        if (const auto pose = ipc_device->recv_pose(ipc::POSE_GIMBAL)) {
            const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
            const auto euler              = math_fuxk::rpy(q);
            const auto [roll, pitch, yaw] = euler.rpy();
            // const Eigen::Quaterniond q(1.0, 0, 0, 0);
            fast_tf::update(
                *tf_buffer,
                fast_tf::EdgeTransform<fast_tf::gimbal>::from_quaternion_xyz(
                    q, pose->x, pose->y, pose->z),
                pose->timestamp_ns);
            fast_tf::update(
                *tf_buffer,
                fast_tf::EdgeTransform<fast_tf::gimbal_pitch_fuxk_frame>::from_rpy(0, 0, 0),
                pose->timestamp_ns);
            imu_state->timestamp_ns           = pose->timestamp_ns;
            imu_state->yaw                    = yaw;
            imu_state->pitch                  = pitch;
            imu_state->roll                   = roll;
            imu_state->yaw_vel                = 0.0;
            imu_state->pitch_vel              = 0.0;
            imu_state->roll_vel               = 0.0;
            imu_state->attitude_valid         = true;
            imu_state->angular_velocity_valid = false;
            note_control_update(pose->timestamp_ns);
        }
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_ODOM)) {
            fast_tf::update(
                *tf_buffer,
                fast_tf::EdgeTransform<fast_tf::odom>::from_translation(pose->x, pose->y, pose->z),
                pose->timestamp_ns);
            note_control_update(pose->timestamp_ns);
        }
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_CAMERA)) {
            const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
            fast_tf::update(
                *tf_buffer,
                fast_tf::EdgeTransform<fast_tf::camera_link_fuxk_frame>::from_quaternion_xyz(
                    q, pose->x, pose->y, pose->z),
                pose->timestamp_ns);
            note_control_update(pose->timestamp_ns);
        }
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_MUZZLE)) {
            const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
            fast_tf::update(
                *tf_buffer,
                fast_tf::EdgeTransform<fast_tf::muzzle_link_fuxk_frame>::from_quaternion_xyz(
                    q, pose->x, pose->y, pose->z),
                pose->timestamp_ns);
            note_control_update(pose->timestamp_ns);
        }
        if (const auto obs = ipc_device->recv_chassis_observation()) {
            // TODO
        }
        if (control_updated) {
            publish_control_snapshot(
                control_out, *tf_buffer, *imu_state, *detecting_color, bullet_speed->bullet_speed,
                bullet_speed->bullet_speed, control_timestamp_ns);
        }
    }
};

using McuHandle = talos_gimbal::McuDeviceHandle;
static talos_gimbal::ReceiveImuData g_imu_data{};

struct McuImu {
    McuConfig config;
    std::shared_ptr<McuHandle> device;
    talos_gimbal::ReceiveImuData* imu_data = nullptr;
    std::optional<uint32_t> last_mcu_timestamp_ms{};

    void system(
        talos::res_mut<fast_tf::CoordinateSystem> tf_buffer,
        core::trajectory::bullet_speed_mut bullet_speed, core::detecting_color_mut detecting_color,
        core::imu_state_mut imu_state,
        talos::spmc_mut<core::ControlResourceSnapshot, RuntimeControlStateChannelTopic>
            control_out) noexcept {
        device->handle_events();
        if (imu_data->header.sof != talos_gimbal::HeaderFrame::SoF()
            || imu_data->eof != talos_gimbal::HeaderFrame::EoF()) [[unlikely]] {
            return;
        }
        if (last_mcu_timestamp_ms.has_value() && *last_mcu_timestamp_ms == imu_data->time_stamp)
            [[unlikely]] {
            return;
        }
        last_mcu_timestamp_ms       = imu_data->time_stamp;
        const uint64_t timestamp_ns = fcs::clock::now_ns();
        fast_tf::update_rotate_only<fast_tf::gimbal_yaw_fuxk_frame>(
            *tf_buffer, math_fuxk::rpy<double>(imu_data->data.roll, 0.0, imu_data->data.yaw),
            timestamp_ns);
        fast_tf::update_rotate_only<fast_tf::gimbal_pitch_fuxk_frame>(
            *tf_buffer, math_fuxk::rpy<double>(0.0, imu_data->data.pitch, 0.0), timestamp_ns);
        if (config.mcu_authoritative_bullet_speed && imu_data->data.bullet_speed > 0) {
            double new_speed = std::clamp(
                static_cast<double>(imu_data->data.bullet_speed), config.bullet_speed_min,
                config.bullet_speed_max);
            if (new_speed != bullet_speed->bullet_speed) {
                SPDLOG_INFO(
                    "bullet speed changed: {} -> {}", bullet_speed->bullet_speed, new_speed);
                bullet_speed->bullet_speed = new_speed;
            }
        }
        if (config.mcu_authoritative_self_color) {
            auto new_detecting_color = imu_data->data.self_color == talos_gimbal::Color::Blue
                                         ? ArmorColor::Red
                                         : ArmorColor::Blue;
            if (new_detecting_color != *detecting_color) [[unlikely]] {
                SPDLOG_INFO(
                    "detecting color changed: {} -> {}", *detecting_color, new_detecting_color);
                *detecting_color = new_detecting_color;
            }
        }
        imu_state->timestamp_ns           = timestamp_ns;
        imu_state->yaw                    = imu_data->data.yaw;
        imu_state->pitch                  = imu_data->data.pitch;
        imu_state->roll                   = imu_data->data.roll;
        imu_state->yaw_vel                = imu_data->data.yaw_vel;
        imu_state->pitch_vel              = imu_data->data.pitch_vel;
        imu_state->roll_vel               = imu_data->data.roll_vel;
        imu_state->attitude_valid         = true;
        imu_state->angular_velocity_valid = true;
        publish_control_snapshot(
            control_out, *tf_buffer, *imu_state, *detecting_color, imu_data->data.bullet_speed,
            bullet_speed->bullet_speed, timestamp_ns);
    }
};

struct Imu {
    std::variant<DaedalusImu, McuImu> impl;

    void system(
        talos::res_mut<fast_tf::CoordinateSystem> tf_buffer,
        core::trajectory::bullet_speed_mut bullet_speed, core::detecting_color_mut detecting_color,
        core::imu_state_mut imu_state,
        talos::spmc_mut<core::ControlResourceSnapshot, RuntimeControlStateChannelTopic>
            control_out) noexcept {
        std::visit(
            [&](auto& inner) {
                inner.system(tf_buffer, bullet_speed, detecting_color, imu_state, control_out);
            },
            impl);
    }
};

} // namespace

std::expected<L1L2SetupResult, std::string> setup_l1(
    talos::World& world, talos::Scheduler& scheduler, bool daedalus,
    const HardwareConfig* hardware) {
    std::shared_ptr<fcs::L1::CameraInterface> camera;
    std::shared_ptr<fcs::L1::OutputInterface> output;
    auto imu       = std::make_shared<Imu>();
    bool imu_ready = false;
    world.insert_resource(core::ImuState{});
    if (daedalus) {
        SPDLOG_INFO("connecting to daedalus ipc...");
        auto ipc_ = ipc::ShmClient::connect();
        if (!ipc_) {
            return std::unexpected(fmt::format("connect ipc: {}", ipc_.error()));
        }
        auto ipc_device = std::make_shared<ipc::ShmClient>(std::move(*ipc_));

        SPDLOG_INFO("initializing daedalus input interface...");
        auto input_result = fcs::L1::CameraInterface::create_ipc(ipc_device);
        if (!input_result) {
            return std::unexpected(fmt::format("init daedalus input: {}", input_result.error()));
        }

        SPDLOG_INFO("initializing daedalus output interface...");
        auto out_result = fcs::L1::OutputInterface::create_ipc(ipc_device);
        if (!out_result) {
            return std::unexpected(fmt::format("init daedalus output: {}", out_result.error()));
        }

        camera    = std::make_shared<fcs::L1::CameraInterface>(std::move(*input_result));
        output    = std::make_shared<fcs::L1::OutputInterface>(std::move(*out_result));
        imu->impl = DaedalusImu{ipc_device};
        imu_ready = true;
    } else {
        if (hardware == nullptr) {
            return std::unexpected("hardware config is required when daedalus=false");
        }

        talos_gimbal::Stm32Parser::latest_imu = &g_imu_data;

        const auto& config = *hardware;
        auto mcu_config    = config.mcu;
        if (!hardware->mcu->pretend_mcu) {
            std::shared_ptr<McuHandle> mcu_device;
            if (hardware->mcu->mcu_authoritative_self_color) {
                SPDLOG_INFO("MCU authoritative self color enabled");
            } else {
                SPDLOG_INFO("MCU authoritative self color disabled");
            }
            if (hardware->mcu->mcu_authoritative_bullet_speed) {
                SPDLOG_INFO("MCU authoritative bullet speed enabled");
            } else {
                SPDLOG_INFO("MCU authoritative bullet speed disabled");
            }
            if (mcu_config->mcu_backend == fcs::McuBackend::Serial) {
                SPDLOG_INFO(
                    "connecting to mcu via serial with {} @ {} baud", mcu_config->serial_device,
                    mcu_config->serial_baud_rate);
                mcu_device = McuHandle::create_serial(
                    mcu_config->serial_device, mcu_config->serial_baud_rate);
            } else {
                if (mcu_config->mcu_product_id) {
                    SPDLOG_INFO(
                        "connecting to mcu via USB with {:#x}:{:#x}", mcu_config->mcu_vendor_id,
                        mcu_config->mcu_product_id);
                } else {
                    SPDLOG_INFO(
                        "connecting to mcu via USB with {:#x} (product_id unspecified)",
                        mcu_config->mcu_vendor_id);
                }
                mcu_device =
                    McuHandle::create_usb(mcu_config->mcu_vendor_id, mcu_config->mcu_product_id);
            }
            if (!mcu_device) {
                return std::unexpected(
                    fmt::format("failed connecting to mcu via {}", mcu_config->mcu_backend));
            }
            SPDLOG_INFO("connected to mcu via {}", mcu_config->mcu_backend);
            output    = std::make_shared<fcs::L1::OutputInterface>(fcs::L1::McuOutput(mcu_device));
            imu->impl = McuImu{mcu_config, mcu_device, &g_imu_data};
            imu_ready = true;
        }
        SPDLOG_INFO("initializing hikrobot camera...");
        auto input_result = fcs::L1::CameraInterface::create_hik(config.camera);
        if (!input_result) {
            return std::unexpected(fmt::format("init hikrobot camera: {}", input_result.error()));
        }

        camera = std::make_shared<fcs::L1::CameraInterface>(std::move(*input_result));
    }

    const auto& cam_info = camera->camera_info();
    world.insert_resource(cam_info);

    if (imu_ready) {
        // IMU: 1000Hz, silent update (no notify)
        scheduler.add_system<talos::fixed_rate_silent<1000, 1>>(
            "imu_reader",
            [imu = std::move(imu)](
                talos::res_mut<fast_tf::CoordinateSystem> tf_buffer,
                core::trajectory::bullet_speed_mut bullet_speed,
                core::detecting_color_mut detecting_color, core::imu_state_mut imu_state,
                talos::spmc_mut<core::ControlResourceSnapshot, RuntimeControlStateChannelTopic>
                    control_out) {
                imu->system(tf_buffer, bullet_speed, detecting_color, imu_state, control_out);
            });
    } else {
        SPDLOG_WARN("pretending mcu is connected, no mcu data will be available");
    }

    // Camera: 160Hz, notify (triggers compute)
    auto last_seq = std::make_shared<std::atomic<uint64_t>>(0);
    scheduler.add_system<talos::fixed_rate<250, 1>>(
        "camera_reader", [camera = std::move(camera),
                          last_seq](talos::spmc_mut<ImageFrame, ImageChannelTopic> cam_out) {
            const auto frame = camera->try_recv();
            if (!frame) [[unlikely]] {
                return;
            }

            const auto seq  = frame->seq;
            const auto prev = last_seq->load(std::memory_order_relaxed);
            if (seq != prev + 1 && prev > 0) {
                SPDLOG_DEBUG("skip {} frame", seq - prev - 1);
            }
            last_seq->store(seq, std::memory_order_relaxed);

            auto img = frame->image;
            cam_out.write(fcs::ImageFrame{std::move(img), frame->timestamp_ns, seq});
        });

    if (imu_ready) {
        // L1 Output System: Send WeaponCommand to device
        scheduler.add_system<talos::fixed_rate<250, 1>>(
            "weapon_output",
            [output = std::move(output)](
                talos::spmc<fcs::L5::WeaponCommand, fcs::WeaponCommandChannelTopic> cmd_in) {
                auto cmd = cmd_in.read();
                if (!cmd) [[unlikely]] {
                    return;
                }
                output->send(*cmd);
            });
    }

    return L1L2SetupResult{.camera_config = cam_info};
}

} // namespace fcs::runtime
