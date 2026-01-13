#include "fcs/runtime/l1_l2_setup.hpp"

#include "auto_aim/config.hpp"
#include "fcs/L1_sensor/camera_interface.hpp"
#include "fcs/L1_sensor/output_interface.hpp"
#include "fcs/L2_perception/at_legacy_systems.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/time.hpp"
#include "fcs/core/types.hpp"
#include "math/euler.hpp"
#include "math/fast_tf/frame.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"
#include "talos_gimbal/stm32.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace fcs::runtime {
namespace {

struct ImuSnapshot {
    uint64_t timestamp_ns = 0;
    double roll_deg = 0.0, pitch_deg = 0.0, yaw_deg = 0.0, bullet_speed = 0.0;
};

struct DaedalusImu {
    std::shared_ptr<ipc::ShmClient> ipc_device;

    void system(
        talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
        talos::res_mut<ImuSnapshot> imu_out) noexcept {
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_GIMBAL)) {
            const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
            const auto [roll, pitch, yaw] = math_fuxk::rpy(q).rpy();
            imu_out->timestamp_ns         = pose->timestamp_ns;
            imu_out->roll_deg             = roll;
            imu_out->pitch_deg            = pitch;
            imu_out->yaw_deg              = yaw;
            fast_tf::update(
                *tf_buffer->get(),
                fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                    fast_tf::TransformMatrix4d::from_rpy(
                        roll, pitch, yaw, pose->x, pose->y, pose->z)),
                pose->timestamp_ns);
        }
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_ODOM)) {
            fast_tf::update(
                *tf_buffer->get(),
                fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>(
                    fast_tf::TransformMatrix4d::from_translation(pose->x, pose->y, pose->z)),
                pose->timestamp_ns);
        }
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_CAMERA)) {
            const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
            auto [roll, pitch, yaw] = math_fuxk::rpy(q).rpy();
            fast_tf::update(
                *tf_buffer->get(),
                fast_tf::FuckTransform<fast_tf::camera_link_fuxk_frame>(
                    fast_tf::TransformMatrix4d::from_rpy(
                        roll, pitch, yaw, pose->x, pose->y, pose->z)),
                pose->timestamp_ns);
        }
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_MUZZLE)) {
            const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
            const auto [roll, pitch, yaw] = math_fuxk::rpy(q).rpy();
            fast_tf::update(
                *tf_buffer->get(),
                fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>(
                    fast_tf::TransformMatrix4d::from_rpy(
                        roll, pitch, yaw, pose->x, pose->y, pose->z)),
                pose->timestamp_ns);
        }
    }
};

using UsbDevice = talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>;
static talos_gimbal::ReceiveImuData g_imu_data{};

struct UsbImu {
    std::shared_ptr<UsbDevice> device;
    talos_gimbal::ReceiveImuData* imu_data = nullptr;

    void system(
        talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
        talos::res_mut<ImuSnapshot> imu_out) noexcept {
        device->handle_events();
        imu_out->timestamp_ns = fcs::clock::now_ns();
        imu_out->roll_deg     = imu_data->data.roll;
        imu_out->pitch_deg    = imu_data->data.pitch;
        imu_out->yaw_deg      = imu_data->data.yaw;
        fast_tf::update(
            *tf_buffer->get(),
            fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                fast_tf::TransformMatrix4d::from_rpy(
                    imu_out->roll_deg, imu_out->pitch_deg, imu_out->yaw_deg)),
            imu_out->timestamp_ns);
    }
};

struct Imu {
    std::variant<DaedalusImu, UsbImu> impl;

    void system(
        talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
        talos::res_mut<ImuSnapshot> imu_out) noexcept {
        std::visit([&](auto& inner) { inner.system(tf_buffer, imu_out); }, impl);
    }
};

} // namespace

std::expected<L1L2SetupResult, std::string> setup_l1_l2(
    talos::World& world, talos::Scheduler& scheduler, const VisionConfig& vision, bool daedalus,
    const HardwareConfig* hardware) {
    std::shared_ptr<fcs::L1::CameraInterface> camera;
    std::shared_ptr<fcs::L1::OutputInterface> output;
    auto imu = std::make_shared<Imu>();

    if (daedalus) {
        SPDLOG_INFO("connecting to daedalus ipc...");
        auto ipc_ = ipc::ShmClient::connect();
        if (!ipc_) {
            return std::unexpected(fmt::format("connect ipc: {}", ipc::to_string(ipc_.error())));
        }
        auto ipc_device = std::make_shared<ipc::ShmClient>(std::move(*ipc_));

        SPDLOG_INFO("initializing daedalus input interface...");
        auto input_result = fcs::L1::CameraInterface::create_ipc(ipc_device);
        if (!input_result) {
            return std::unexpected(
                fmt::format("init daedalus input: {}", fcs::L1::to_string(input_result.error())));
        }

        SPDLOG_INFO("initializing daedalus output interface...");
        auto out_result = fcs::L1::OutputInterface::create_ipc(ipc_device);
        if (!out_result) {
            return std::unexpected(fmt::format("init daedalus output: {}", out_result.error()));
        }

        camera    = std::make_shared<fcs::L1::CameraInterface>(std::move(*input_result));
        output    = std::make_shared<fcs::L1::OutputInterface>(std::move(*out_result));
        imu->impl = DaedalusImu{ipc_device};
    } else {
        if (hardware == nullptr) {
            return std::unexpected("hardware config is required when daedalus=false");
        }

        talos_gimbal::Stm32Parser::latest_imu = &g_imu_data;

        const auto& config = *hardware;

        auto usb_device = std::make_shared<UsbDevice>();
        if (config.imu_product_id_opt()) {
            SPDLOG_INFO(
                "connecting to USB {:#x}:{:#x}", config.imu_vendor_id, config.imu_product_id);
        } else {
            SPDLOG_INFO("connecting to USB {:#x} (product_id unspecified)", config.imu_vendor_id);
        }

        SPDLOG_INFO("initializing hik input interface...");
        auto input_result = fcs::L1::CameraInterface::create_hik(config.camera);
        if (!input_result) {
            return std::unexpected(
                fmt::format("init hik input: {}", fcs::L1::to_string(input_result.error())));
        }

        SPDLOG_INFO("connecting to USB device...");
        auto connect_result =
            usb_device->connect(config.imu_vendor_id, config.imu_product_id_opt());
        if (!connect_result) {
            return std::unexpected(
                fmt::format(
                    "connect USB device: {}", talos_gimbal::to_string(connect_result.error())));
        }

        camera    = std::make_shared<fcs::L1::CameraInterface>(std::move(*input_result));
        output    = std::make_shared<fcs::L1::OutputInterface>(fcs::L1::UsbOutput(usb_device));
        imu->impl = UsbImu{usb_device, &g_imu_data};
    }

    const auto& cam_info = camera->camera_info();
    world.insert_resource(cam_info);
    world.insert_resource(ImuSnapshot{});

    auto backend_handle_result = fcs::create_detector_backend_handle(vision.at_legacy);
    if (!backend_handle_result) {
        return std::unexpected(
            fmt::format("create detector backend: {}", backend_handle_result.error()));
    }
    SPDLOG_INFO(
        "detector backend: {} (color={})", backend_handle_result->name,
        fcs::to_string(vision.at_legacy.detector.detect_color));

    world.insert_resource(std::move(backend_handle_result->backend));
    world.insert_resource(fcs::create_pnp_solver(cam_info));

    // IMU: 100Hz, silent update (no notify)
    scheduler.add_system<talos::exclusive_silent<1000, 1>>(
        "imu_reader",
        [imu = std::move(imu)](
            talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::res_mut<ImuSnapshot> imu_out) { imu->system(tf_buffer, imu_out); });

    // Camera: 160Hz, notify (triggers compute)
    auto last_seq = std::make_shared<std::atomic<uint64_t>>(0);
    scheduler.add_system<talos::exclusive<250, 1>>(
        "camera_reader", [camera = std::move(camera), last_seq](
                             talos::spmc_mut<AtLegacyImageFrame, AtLegacyImageChannelTag> cam_out) {
            const auto frame = camera->try_recv();
            if (!frame) {
                return;
            }

            const auto seq  = frame->seq;
            const auto prev = last_seq->load(std::memory_order_relaxed);
            if (seq != prev + 1 && prev > 0) {
                SPDLOG_INFO("skip {} frame", seq - prev - 1);
            }
            last_seq->store(seq, std::memory_order_relaxed);

            auto img = frame->image;
            cam_out.write(fcs::AtLegacyImageFrame{std::move(img), frame->timestamp_ns, seq});
        });

    // L1 Output System: Send WeaponCommand to device
    scheduler.add_system<talos::exclusive<250, 1>>(
        "weapon_output",
        [output = std::move(output)](
            talos::spmc<fcs::L5::WeaponCommand, fcs::WeaponCommandChannelTag> cmd_in) {
            auto cmd = cmd_in.read();
            if (!cmd) {
                return;
            }
            output->send(*cmd);
        });

    return L1L2SetupResult{.detector_backend_name = std::move(backend_handle_result->name)};
}

} // namespace fcs::runtime
