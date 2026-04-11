#include "chiral/chiral_collector_system.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/selected_target_snapshot.hpp"
#include "chiral/chrial.hpp"
#include "chiral/talos_triple_buffer_shm.hpp"
#include "core/channel_topics.hpp"
#include "frame.hpp"
#include "scheduler/scheduler.hpp"

#include <magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace fcs::chiral {

namespace {
// Convert Eigen Vector3d to chrial::Vector3d
template <talos::chrial::tag From, talos::chrial::tag To = talos::chrial::untyped>
talos::chrial::Vector3d<From, To> to_chiral_vector3d(const Eigen::Vector3d& v) noexcept {
    return {v.x(), v.y(), v.z()};
}

// Convert FCS TrackerStatus to chrial::TrackerStatus
talos::chrial::TrackerStatus to_chiral_tracker_status(L3::TrackerStatus status) noexcept {
    switch (status) {
    case L3::TrackerStatus::Idle: return talos::chrial::TrackerStatus::Idle;
    case L3::TrackerStatus::Detecting: return talos::chrial::TrackerStatus::Detecting;
    case L3::TrackerStatus::Tracking: return talos::chrial::TrackerStatus::Tracking;
    case L3::TrackerStatus::TempLost: return talos::chrial::TrackerStatus::TempLost;
    }
}

// Convert FCS ArmorName to chrial::ArmorName
talos::chrial::ArmorName to_chrial_armor_name(ArmorName name) noexcept {
    switch (name) {
    case ArmorName::Sentry: return talos::chrial::ArmorName::Sentry;
    case ArmorName::One: return talos::chrial::ArmorName::One;
    case ArmorName::Two: return talos::chrial::ArmorName::Two;
    case ArmorName::Three: return talos::chrial::ArmorName::Three;
    case ArmorName::Four: return talos::chrial::ArmorName::Four;
    case ArmorName::Five: return talos::chrial::ArmorName::Five;
    case ArmorName::Outpost: return talos::chrial::ArmorName::Outpost;
    case ArmorName::Base: return talos::chrial::ArmorName::Base;
    case ArmorName::BaseLarge: return talos::chrial::ArmorName::Invalid;
    case ArmorName::Invalid: return talos::chrial::ArmorName::Invalid;
    }
}

// Convert FCS ArmorColor to chrial::ArmorColor
[[maybe_unused]] talos::chrial::ArmorColor to_chrial_armor_color(ArmorColor color) noexcept {
    switch (color) {
    case ArmorColor::Blue: return talos::chrial::ArmorColor::Blue;
    case ArmorColor::Red: return talos::chrial::ArmorColor::Red;
    case ArmorColor::Neutral: return talos::chrial::ArmorColor::Neutral;
    case ArmorColor::Purple: return talos::chrial::ArmorColor::Purple;
    }
}

} // namespace

void register_chiral_collector_system(talos::Scheduler& scheduler) {
    // Create shared memory writer on initialization
    auto writer = talos::chiral::ipc::TalosDataWriter::create();
    if (!writer) {
        SPDLOG_ERROR(
            "Failed to create TalosData shared memory writer: {}",
            magic_enum::enum_name(writer.error()));
        return;
    }

    scheduler.add_system<talos::pool_compute>(
        "chiral_collector",
        [writer_storage =
             std::make_shared<talos::chiral::ipc::TalosDataWriter>(std::move(*writer))](
            talos::spmc<L4::SelectedTargetSnapshot, SelectedTargetSnapshotChannelTopic>
                selected_target_in,
            talos::res<fast_tf::CoordinateSystem> tf_buffer) mutable {
            if (!selected_target_in.has_new()) {
                return;
            }

            auto selected_target = selected_target_in.read();
            if (!selected_target) {
                return;
            }

            talos::chrial::TalosData talos_data{};
            talos_data.state_kind   = talos::chrial::TargetStateKind::Robot;
            talos_data.state.status = talos::chrial::TrackerStatus::Idle;
            talos_data.state.color  = talos::chrial::ArmorColor::Neutral;
            talos_data.state.name   = talos::chrial::ArmorName::Invalid;

            const bool has_valid_armor_target =
                selected_target->has_target() && selected_target->tracker.is_tracking();
            const L3::TrackerOutput* tracker_output =
                has_valid_armor_target ? std::addressof(selected_target->tracker) : nullptr;

            if (tracker_output) {
                talos_data.state.status = to_chiral_tracker_status(tracker_output->status);
                talos_data.state.color  = to_chrial_armor_color(tracker_output->target_color);
                talos_data.state.name   = to_chrial_armor_name(tracker_output->target_name);
            }

            if (tracker_output && tracker_output->is_robot()) {
                talos_data.state_kind   = talos::chrial::TargetStateKind::Robot;
                const auto* robot_state = tracker_output->robot_state();
                if (robot_state) {
                    talos_data.state.robot.position =
                        to_chiral_vector3d<talos::chrial::odom>(robot_state->position);
                    talos_data.state.robot.velocity =
                        to_chiral_vector3d<talos::chrial::odom>(robot_state->velocity);
                    talos_data.state.robot.yaw     = robot_state->yaw;
                    talos_data.state.robot.v_yaw   = robot_state->v_yaw;
                    talos_data.state.robot.radius0 = robot_state->radius0;
                    talos_data.state.robot.radius1 = robot_state->radius1;
                    talos_data.state.robot.z1      = robot_state->z1;
                    talos_data.state.robot.armor_num =
                        static_cast<uint32_t>(robot_state->armors_num);
                }
            } else if (tracker_output && tracker_output->is_outpost()) {
                talos_data.state_kind     = talos::chrial::TargetStateKind::Outpost;
                const auto* outpost_state = tracker_output->outpost_state();
                if (outpost_state) {
                    talos_data.state.outpost.position.x = outpost_state->position.x();
                    talos_data.state.outpost.position.y = outpost_state->position.y();
                    talos_data.state.outpost.position.z = 0.0; // 2D position
                    talos_data.state.outpost.velocity =
                        to_chiral_vector3d<talos::chrial::odom>(outpost_state->velocity);
                    talos_data.state.outpost.yaw   = outpost_state->yaw;
                    talos_data.state.outpost.v_yaw = outpost_state->v_yaw;
                    talos_data.state.outpost.z     = outpost_state->z;
                }
            }

            const uint64_t current_ns = selected_target->timestamp_ns;
            using namespace fast_tf;

            // Lookup gimbal transform (odom -> gimbal)
            auto gimbal_tf = lookup_clamped<fast_tf::odom, fast_tf::gimbal_pitch_fuxk_frame>(
                *tf_buffer, current_ns);
            if (gimbal_tf) {
                auto translation                     = gimbal_tf.value().translation();
                auto rotation                        = gimbal_tf.value().quaternion();
                talos_data.gimbal_link.translation.x = translation.x();
                talos_data.gimbal_link.translation.y = translation.y();
                talos_data.gimbal_link.translation.z = translation.z();
                talos_data.gimbal_link.rotation.x    = rotation.x();
                talos_data.gimbal_link.rotation.y    = rotation.y();
                talos_data.gimbal_link.rotation.z    = rotation.z();
                talos_data.gimbal_link.rotation.w    = rotation.w();
            } else {
                SPDLOG_WARN("Failed to lookup gimbal transform: {}", gimbal_tf.error());
            }

            // Lookup camera transform (gimbal -> camera)
            auto camera_tf =
                lookup_clamped<fast_tf::gimbal, camera_link_fuxk_frame>(*tf_buffer, current_ns);
            if (camera_tf) {

                auto translation                     = camera_tf.value().translation();
                auto rotation                        = camera_tf.value().quaternion();
                talos_data.camera_link.translation.x = translation.x();
                talos_data.camera_link.translation.y = translation.y();
                talos_data.camera_link.translation.z = translation.z();
                talos_data.camera_link.rotation.x    = rotation.x();
                talos_data.camera_link.rotation.y    = rotation.y();
                talos_data.camera_link.rotation.z    = rotation.z();
                talos_data.camera_link.rotation.w    = rotation.w();
            } else {
                SPDLOG_WARN("Failed to lookup camera transform: {}", camera_tf.error());
            }

            // Lookup muzzle transform (odom -> muzzle)
            auto muzzle_tf =
                lookup_clamped<odom_fuxk_frame, muzzle_link_fuxk_frame>(*tf_buffer, current_ns);
            if (muzzle_tf) {
                auto translation                     = muzzle_tf.value().translation();
                auto rotation                        = muzzle_tf.value().quaternion();
                talos_data.muzzle_link.translation.x = translation.x();
                talos_data.muzzle_link.translation.y = translation.y();
                talos_data.muzzle_link.translation.z = translation.z();
                talos_data.muzzle_link.rotation.x    = rotation.x();
                talos_data.muzzle_link.rotation.y    = rotation.y();
                talos_data.muzzle_link.rotation.z    = rotation.z();
                talos_data.muzzle_link.rotation.w    = rotation.w();
            } else {
                SPDLOG_WARN("Failed to lookup muzzle transform: {}", muzzle_tf.error());
            }

            writer_storage->write(talos_data);
        });
}

} // namespace fcs::chiral
