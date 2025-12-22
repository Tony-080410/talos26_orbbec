#include "tracker.hpp"

#include <numeric>
#include <spdlog/spdlog.h>

namespace auto_aim {

    Tracker::Tracker(const SpVisionTrackerConfig &config) :
        enemy_color_(static_cast<talos::Color>(config.enemy_color)), min_detect_count_(config.min_detect_count),
        normal_temp_lost_count_(config.max_temp_lost_count),
        outpost_max_temp_lost_count_(config.outpost_max_temp_lost_count) {}

    std::string Tracker::state_string(const talos::TrackerSoA &state) {
        switch (state.state.load(std::memory_order_acquire)) {
            case talos::TrackerState::IDLE:
                return "idle";
            case talos::TrackerState::DETECTING:
                return "detecting";
            case talos::TrackerState::TRACKING:
                return "tracking";
            case talos::TrackerState::TEMP_LOST:
                return "temp_lost";
            default:
                return "unknown";
        }
    }

    bool Tracker::track(talos::TrackerSoA &state, Target &target, const talos::ArmorSlots &slots,
                        const std::array<ArmorPose3D, 10> &poses3d,
                        const std::chrono::steady_clock::time_point t) const {
        const auto current_state = state.state.load(std::memory_order_acquire);
        const auto current_ns = t.time_since_epoch().count();
        const auto dt = static_cast<double>(current_ns - state.last_timestamp_ns) / 1e9;
        state.last_timestamp_ns = current_ns;

        // 时间间隔过长，说明可能发生了相机离线
        if (current_state != talos::TrackerState::IDLE && dt > 0.1) {
            spdlog::warn("[Tracker] Large dt: {:.3f}s", dt);
            state.try_reset_idle();
        }

        spdlog::debug("Tracker state: {}", state_string(state));
        bool found;
        if (current_state == talos::TrackerState::IDLE) {
            found = set_target(state, target, slots, poses3d, t);
        } else {
            found = update_target(state, target, slots, poses3d, t);
        }
        spdlog::debug("Tracker found: {}", found ? "yes" : "no");

        state_machine(state, target, found);

        // 发散检测
        if (state.state.load(std::memory_order_acquire) != talos::TrackerState::IDLE && target.diverged()) {
            spdlog::debug("[Tracker] Target diverged!");
            state.try_reset_idle();
            return false;
        }

        // 收敛效果检测
        if (state.state.load(std::memory_order_acquire) != talos::TrackerState::IDLE) {
            const int recent_failures = std::accumulate(target.ekf().recent_nis_failures.begin(),
                                                        target.ekf().recent_nis_failures.end(), 0);
            if (recent_failures >= static_cast<int>(0.4 * target.ekf().window_size)) {
                spdlog::debug("[Tracker] Bad Converge Found!");
                state.try_reset_idle();
                return false;
            }
        }

        return state.state.load(std::memory_order_acquire) != talos::TrackerState::IDLE;
    }

    std::vector<Eigen::Vector4d> Tracker::armor_xyza_list(const talos::TrackerSoA &state, const Target &target) {
        if (state.state.load(std::memory_order_acquire) == talos::TrackerState::IDLE) {
            return {};
        }
        return target.armor_xyza_list();
    }

    void Tracker::state_machine(talos::TrackerSoA &state, const Target &target, const bool found) const {
        const auto current_state = state.state.load(std::memory_order_acquire);

        if (current_state == talos::TrackerState::IDLE) {
            if (!found)
                return;

            // IDLE -> DETECTING
            if (state.try_start_detecting()) {
                state.detecting_count = 1;
            }
        }

        else if (current_state == talos::TrackerState::DETECTING) {
            if (found) {
                state.detecting_count++;
                if (state.detecting_count >= min_detect_count_) {
                    // DETECTING -> TRACKING
                    state.try_confirm_tracking();
                }
            } else {
                // DETECTING -> IDLE
                state.detecting_count = 0;
                state.try_reset_idle();
            }
        }

        else if (current_state == talos::TrackerState::TRACKING) {
            if (found)
                return;

            // TRACKING -> TEMP_LOST
            if (state.try_enter_lost()) {
                state.lost_count = 1;
            }
        }

        else if (current_state == talos::TrackerState::TEMP_LOST) {
            if (found) {
                // TEMP_LOST -> TRACKING
                state.try_restore_tracking();
            } else {
                state.lost_count++;

                // 前哨站特殊处理
                if (target.name == talos::ArmorName::outpost) {
                    state.max_temp_lost_count = outpost_max_temp_lost_count_;
                } else {
                    state.max_temp_lost_count = normal_temp_lost_count_;
                }

                if (state.lost_count > state.max_temp_lost_count) {
                    // TEMP_LOST -> IDLE
                    state.try_reset_idle();
                }
            }
        }
    }

    bool Tracker::set_target(talos::TrackerSoA &state, Target &target, const talos::ArmorSlots &slots,
                             const std::array<ArmorPose3D, 10> &poses3d,
                             const std::chrono::steady_clock::time_point t) const {// 找到第一个有效的敌方装甲板
        spdlog::debug("Setting target...");
        int best_idx = -1;
        float best_distance = 1e10f;

        for (int i = 0; i < 10; ++i) {
            if (!slots.is_valid(i) || !poses3d[i].valid) {
                spdlog::debug("Slot {} is not valid or pose3d is not valid", i);
                continue;
            }

            // 过滤非敌方颜色
            if (const auto &armor = slots.armors[i]; static_cast<talos::Color>(armor.color) != enemy_color_) {
                spdlog::debug("Slot {} is not enemy color", i);
                continue;
            }

            // 选择距离最近的
            if (const float distance = static_cast<float>(poses3d[i].ypd_in_world[2]); distance < best_distance) {
                spdlog::debug("Slot {} has better distance: {}", i, distance);
                best_distance = distance;
                best_idx = i;
            } else {
                spdlog::debug("Slot {} distance {} is worse than best {}", i, distance, best_distance);
            }
        }

        if (best_idx < 0) {
            spdlog::debug("No valid target found");
            return false;
        }
        spdlog::debug("Best target found at slot {}", best_idx);

        const auto &armor = slots.armors[best_idx];
        const auto &pose3d = poses3d[best_idx];
        const auto name = static_cast<talos::ArmorName>(armor.name);
        const auto type = static_cast<talos::ArmorType>(armor.type);

        // 根据兵种优化初始化参数
        const auto is_balance =
                type == talos::ArmorType::big &&
                (name == talos::ArmorName::three || name == talos::ArmorName::four || name == talos::ArmorName::five);

        if (is_balance) {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
            target = Target(armor, pose3d, t, 0.2, 2, P0_dig);
        } else if (name == talos::ArmorName::outpost) {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
            target = Target(armor, pose3d, t, 0.2765, 3, P0_dig);
        } else if (name == talos::ArmorName::base) {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
            target = Target(armor, pose3d, t, 0.3205, 3, P0_dig);
        } else {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
            target = Target(armor, pose3d, t, 0.2, 4, P0_dig);
        }

        // 更新 generation 用于 ABA 防护
        state.next_generation();

        return true;
    }

    bool Tracker::update_target([[maybe_unused]] talos::TrackerSoA &state, Target &target,
                                const talos::ArmorSlots &slots, const Poses3D &poses3d,
                                const std::chrono::steady_clock::time_point t) const {
        target.predict(t);

        int found_count = 0;

        for (int i = 0; i < 10; ++i) {
            if (!slots.is_valid(i) || !poses3d[i].valid) {
                continue;
            }

            const auto &armor = slots.armors[i];
            const auto name = static_cast<talos::ArmorName>(armor.name);
            const auto type = static_cast<talos::ArmorType>(armor.type);

            // 过滤非敌方颜色
            if (static_cast<talos::Color>(armor.color) != enemy_color_) {
                continue;
            }

            // 只更新匹配的目标
            if (name != target.name || type != target.armor_type) {
                continue;
            }

            found_count++;
            target.update(armor, poses3d[i]);
        }

        return found_count > 0;
    }

} // namespace auto_aim
