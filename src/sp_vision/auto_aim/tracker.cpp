#include "auto_aim/tracker.hpp"

#include <numeric>
#include <spdlog/spdlog.h>

namespace auto_aim {

    Tracker::Tracker(const SpVisionTrackerConfig &config, Solver &solver) :
        solver_(solver), enemy_color_(static_cast<spike::Color>(config.enemy_color)),
        min_detect_count_(config.min_detect_count), max_temp_lost_count_(config.max_temp_lost_count), detect_count_(0),
        temp_lost_count_(0), outpost_max_temp_lost_count_(config.outpost_max_temp_lost_count),
        normal_temp_lost_count_(config.max_temp_lost_count), state_(spike::TrackerState::IDLE),
        last_timestamp_(std::chrono::steady_clock::now()) {}

    std::string Tracker::state_string() const {
        switch (state_) {
            case spike::TrackerState::IDLE:
                return "idle";
            case spike::TrackerState::DETECTING:
                return "detecting";
            case spike::TrackerState::TRACKING:
                return "tracking";
            case spike::TrackerState::TEMP_LOST:
                return "temp_lost";
            default:
                return "unknown";
        }
    }

    bool Tracker::track(const spike::ArmorSlots &slots, const std::array<ArmorPose3D, 10> &poses3d,
                        const std::chrono::steady_clock::time_point t) {
        auto dt = std::chrono::duration<double>(t - last_timestamp_).count();
        last_timestamp_ = t;

        // 时间间隔过长，说明可能发生了相机离线
        if (state_ != spike::TrackerState::IDLE && dt > 0.1) {
            spdlog::warn("[Tracker] Large dt: {:.3f}s", dt);
            state_ = spike::TrackerState::IDLE;
        }

        bool found;
        if (state_ == spike::TrackerState::IDLE) {
            found = set_target(slots, poses3d, t);
        } else {
            found = update_target(slots, poses3d, t);
        }

        state_machine(found);

        // 发散检测
        if (state_ != spike::TrackerState::IDLE && target_.diverged()) {
            spdlog::debug("[Tracker] Target diverged!");
            state_ = spike::TrackerState::IDLE;
            return false;
        }

        // 收敛效果检测
        if (state_ != spike::TrackerState::IDLE) {
            const int recent_failures = std::accumulate(target_.ekf().recent_nis_failures.begin(),
                                                  target_.ekf().recent_nis_failures.end(), 0);
            if (recent_failures >= static_cast<int>(0.4 * target_.ekf().window_size)) {
                spdlog::debug("[Tracker] Bad Converge Found!");
                state_ = spike::TrackerState::IDLE;
                return false;
            }
        }

        return state_ != spike::TrackerState::IDLE;
    }

    std::vector<Eigen::Vector4d> Tracker::armor_xyza_list() const {
        if (state_ == spike::TrackerState::IDLE) {
            return {};
        }
        return target_.armor_xyza_list();
    }

    void Tracker::state_machine(const bool found) {
        if (state_ == spike::TrackerState::IDLE) {
            if (!found)
                return;

            state_ = spike::TrackerState::DETECTING;
            detect_count_ = 1;
        }

        else if (state_ == spike::TrackerState::DETECTING) {
            if (found) {
                detect_count_++;
                if (detect_count_ >= min_detect_count_) {
                    state_ = spike::TrackerState::TRACKING;
                }
            } else {
                detect_count_ = 0;
                state_ = spike::TrackerState::IDLE;
            }
        }

        else if (state_ == spike::TrackerState::TRACKING) {
            if (found)
                return;

            temp_lost_count_ = 1;
            state_ = spike::TrackerState::TEMP_LOST;
        }

        else if (state_ == spike::TrackerState::TEMP_LOST) {
            if (found) {
                state_ = spike::TrackerState::TRACKING;
            } else {
                temp_lost_count_++;
                if (target_.name == spike::ArmorName::outpost) {
                    max_temp_lost_count_ = outpost_max_temp_lost_count_;
                } else {
                    max_temp_lost_count_ = normal_temp_lost_count_;
                }

                if (temp_lost_count_ > max_temp_lost_count_) {
                    state_ = spike::TrackerState::IDLE;
                }
            }
        }
    }

    bool Tracker::set_target(const spike::ArmorSlots &slots, const std::array<ArmorPose3D, 10> &poses3d,
                             const std::chrono::steady_clock::time_point t) {
        // 找到第一个有效的敌方装甲板
        int best_idx = -1;
        float best_distance = 1e10f;

        for (int i = 0; i < 10; ++i) {
            if (!slots.is_valid(i) || !poses3d[i].valid) {
                continue;
            }

            const auto &armor = slots.armors[i];

            // 过滤非敌方颜色
            if (static_cast<spike::Color>(armor.color) != enemy_color_) {
                continue;
            }

            // 选择距离最近的
            const float distance = static_cast<float>(poses3d[i].ypd_in_world[2]);
            if (distance < best_distance) {
                best_distance = distance;
                best_idx = i;
            }
        }

        if (best_idx < 0) {
            return false;
        }

        const auto &armor = slots.armors[best_idx];
        const auto &pose3d = poses3d[best_idx];
        const auto name = static_cast<spike::ArmorName>(armor.name);
        const auto type = static_cast<spike::ArmorType>(armor.type);

        // 根据兵种优化初始化参数
        const auto is_balance =
                type == spike::ArmorType::big &&
                (name == spike::ArmorName::three || name == spike::ArmorName::four || name == spike::ArmorName::five);

        if (is_balance) {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
            target_ = Target(armor, pose3d, t, 0.2, 2, P0_dig);
        } else if (name == spike::ArmorName::outpost) {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
            target_ = Target(armor, pose3d, t, 0.2765, 3, P0_dig);
        } else if (name == spike::ArmorName::base) {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
            target_ = Target(armor, pose3d, t, 0.3205, 3, P0_dig);
        } else {
            const Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
            target_ = Target(armor, pose3d, t, 0.2, 4, P0_dig);
        }

        return true;
    }

    bool Tracker::update_target(const spike::ArmorSlots &slots, const std::array<ArmorPose3D, 10> &poses3d,
                                const std::chrono::steady_clock::time_point t) {
        target_.predict(t);

        int found_count = 0;

        for (int i = 0; i < 10; ++i) {
            if (!slots.is_valid(i) || !poses3d[i].valid) {
                continue;
            }

            const auto &armor = slots.armors[i];
            const auto name = static_cast<spike::ArmorName>(armor.name);
            const auto type = static_cast<spike::ArmorType>(armor.type);

            // 过滤非敌方颜色
            if (static_cast<spike::Color>(armor.color) != enemy_color_) {
                continue;
            }

            // 只更新匹配的目标
            if (name != target_.name || type != target_.armor_type) {
                continue;
            }

            found_count++;
            target_.update(armor, poses3d[i]);
        }

        return found_count > 0;
    }

} // namespace auto_aim
