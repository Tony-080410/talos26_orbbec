#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <spdlog/spdlog.h>

#include "solver.hpp" // for math::eulers, math::xyz2ypd

namespace auto_aim {

Tracker::Tracker(const SpVisionTrackerConfig& config, fast_tf::CoordinateSystem<100>& tf_buffer)
    : tf_buffer_(tf_buffer)
    , enemy_color_(static_cast<talos::Color>(config.enemy_color))
    , min_detect_count_(config.min_detect_count)
    , normal_temp_lost_count_(config.max_temp_lost_count)
    , outpost_max_temp_lost_count_(config.outpost_max_temp_lost_count)
    , image_width_(config.image_width)
    , image_height_(config.image_height) {}

bool Tracker::track(
    talos::TrackerSoA& state, Target& target, const talos::ArmorSlots& slots,
    const std::array<ArmorPose3D, 10>& poses3d, const uint64_t timestamp_ns) {
    const auto current_state = state.state.load(std::memory_order_acquire);

    // 时间间隔过长，说明可能发生了相机离线
    if (const auto dt = static_cast<double>(timestamp_ns - target.timestamp_ns_) / 1e9;
        current_state != talos::TrackerState::IDLE && dt > 0.1) {
        spdlog::warn("[Tracker] Large dt: {:.3f}s", dt);
        state.idle();
    }
    // spdlog::info("[Tracker] ts={}, last_ts={}, dt={:.6f}s",timestamp_ns, target.timestamp_ns_,
    // static_cast<double>(timestamp_ns - target.timestamp_ns_) * 1e-9);
    spdlog::debug(
        "Tracker state: {}", talos::tracker_state(state.state.load(std::memory_order_acquire)));
    bool found;
    if (current_state == talos::TrackerState::IDLE) {
        found = set_target(state, target, slots, poses3d, timestamp_ns);
    } else {
        found = update_target(state, target, slots, poses3d, timestamp_ns);
    }
    target.valid = false;
    spdlog::info("Tracker found: {}", found ? "yes" : "no");

    state_machine(state, target, found);

    // 发散检测
    if (state.state.load(std::memory_order_acquire) != talos::TrackerState::IDLE
        && target.name != talos::ArmorName::not_armor && target.diverged()) {
        spdlog::debug("[Tracker] Target diverged!");
        state.idle();
        return false;
    }

    // 收敛效果检测
    if (state.state.load(std::memory_order_acquire) != talos::TrackerState::IDLE) {
        const int recent_failures = std::accumulate(
            target.ekf().recent_nis_failures.begin(), target.ekf().recent_nis_failures.end(), 0);
        if (recent_failures >= static_cast<int>(0.4 * target.ekf().window_size)) {
            spdlog::debug("[Tracker] Bad Converge Found!");
            state.idle();
            return false;
        }
    }

    // 同步 EKF 状态到 TrackerSoA (用于 Foxglove 发布)
    if (target.converged()) {
        const auto& x = target.ekf_x();
        // EKF 状态向量: [x, vx, y, vy, z, vz, yaw, v_yaw, r, l, h]
        //               0   1   2   3   4   5    6     7   8  9  10
        state.pos = decltype(state.pos)(
            {static_cast<float>(x(0)), static_cast<float>(x(2)), static_cast<float>(x(4))});
        state.vel = decltype(state.vel)(
            {static_cast<float>(x(1)), static_cast<float>(x(3)), static_cast<float>(x(5))});

        state.yaw       = decltype(state.yaw)(static_cast<float>(x(6)));   // yaw
        state.v_yaw     = decltype(state.v_yaw)(static_cast<float>(x(7))); // v_yaw
        state.radius[0] = static_cast<float>(x(8));                        // r
        state.radius[1] = static_cast<float>(x(8) + x(9));                 // r + l
    }

    target.valid = state.state.load(std::memory_order_acquire) != talos::TrackerState::IDLE;
    return state.state.load(std::memory_order_acquire) != talos::TrackerState::IDLE;
}

std::vector<Eigen::Vector4d>
    Tracker::armor_xyza_list(const talos::TrackerSoA& state, const Target& target) {
    if (state.state.load(std::memory_order_acquire) == talos::TrackerState::IDLE) {
        return {};
    }
    return target.armor_xyza_list();
}

void Tracker::state_machine(talos::TrackerSoA& state, const Target& target, const bool found) {
    const auto current_state = state.state.load(std::memory_order_acquire);

    if (current_state == talos::TrackerState::IDLE) {
        if (!found) {
            return;
        }
        // IDLE -> DETECTING
        if (state.detect()) {
            state.detecting_count = 1;
        }
        return;
    }

    if (current_state == talos::TrackerState::DETECTING) {
        if (found) {
            state.detecting_count++;
            if (state.detecting_count >= min_detect_count_) {
                // DETECTING -> TRACKING
                state.track();
            }
            return;
        }
        // DETECTING -> IDLE
        state.detecting_count = 0;
        state.idle();
        return;
    }

    if (current_state == talos::TrackerState::TRACKING) {
        if (found) {
            return;
        }

        // TRACKING -> TEMP_LOST
        if (state.temporary_lost()) {
            state.lost_count = 1;
        }
        return;
    }

    if (current_state == talos::TrackerState::TEMP_LOST) {
        if (found) {
            // TEMP_LOST -> TRACKING
            state.track();
            return;
        }
        state.lost_count++;

        // 前哨站特殊处理
        if (target.name == talos::ArmorName::outpost) {
            max_temp_lost_count = outpost_max_temp_lost_count_;
        } else {
            max_temp_lost_count = normal_temp_lost_count_;
        }

        if (state.lost_count > max_temp_lost_count) {
            // TEMP_LOST -> IDLE
            state.idle();
        }
    }
}

bool Tracker::set_target(
    talos::TrackerSoA& state, Target& target, const talos::ArmorSlots& slots,
    const std::array<ArmorPose3D, 10>& poses3d, const uint64_t timestamp_ns) const {
    // ========== 坐标系说明 ==========
    // poses3d中的数据在camera_optical坐标系
    // Target构造时通过tf_manager查询world坐标
    // ================================

    spdlog::debug("Setting target...");

    // 候选装甲板结构
    struct Candidate {
        int idx;
        float center_dist; // 到图像中心的距离
    };
    std::vector<Candidate> candidates;
    candidates.reserve(10);

    // 计算图像中心
    const float cx = static_cast<float>(image_width_) / 2.0f;
    const float cy = static_cast<float>(image_height_) / 2.0f;

    for (int i = 0; i < 10; ++i) {
        if (!slots.is_valid(i) || !poses3d[i].valid) {
            spdlog::debug("Slot {} is not valid or pose3d is not valid", i);
            continue;
        }

        // 过滤非敌方颜色
        const auto& armor = slots.armors[i];
        if (static_cast<talos::Color>(armor.color) != enemy_color_) {
            spdlog::debug("Slot {} is not enemy color", i);
            continue;
        }

        // 计算到图像中心的距离 (对齐 sp_vision_25 的选择逻辑)
        const float dx          = armor.x2d - cx;
        const float dy          = armor.y2d - cy;
        const float center_dist = std::sqrt(dx * dx + dy * dy);

        spdlog::debug("Slot {} center_dist: {:.2f}", i, center_dist);
        candidates.push_back({i, center_dist});
    }

    if (candidates.empty()) {
        spdlog::debug("No valid target found");
        return false;
    }

    // 按图像中心距离排序，选择最靠近图像中心的装甲板 (对齐 sp_vision_25)
    std::ranges::sort(candidates, {}, &Candidate::center_dist);

    const int best_idx = candidates[0].idx;
    spdlog::debug(
        "Best target found at slot {} (center_dist: {:.2f})", best_idx, candidates[0].center_dist);

    const auto& armor  = slots.armors[best_idx];
    const auto& pose3d = poses3d[best_idx];
    const auto name    = static_cast<talos::ArmorName>(armor.name);
    const auto type    = static_cast<talos::ArmorType>(armor.type);

    // 根据兵种优化初始化参数
    const auto is_balance = type == talos::ArmorType::big
                         && (name == talos::ArmorName::three || name == talos::ArmorName::four
                             || name == talos::ArmorName::five);

    if (is_balance) {
        const Eigen::VectorXd P0_dig{
            {1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}
        };
        target = Target(armor, pose3d, tf_buffer_, timestamp_ns, 0.2, 2, P0_dig);
    } else if (name == talos::ArmorName::outpost) {
        const Eigen::VectorXd P0_dig{
            {1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}
        };
        target = Target(armor, pose3d, tf_buffer_, timestamp_ns, 0.2765, 3, P0_dig);
    } else if (name == talos::ArmorName::base) {
        const Eigen::VectorXd P0_dig{
            {1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}
        };
        target = Target(armor, pose3d, tf_buffer_, timestamp_ns, 0.3205, 3, P0_dig);
    } else {
        const Eigen::VectorXd P0_dig{
            {1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}
        };
        target = Target(armor, pose3d, tf_buffer_, timestamp_ns, 0.2, 4, P0_dig);
    }

    // 更新 generation 用于 ABA 防护
    state.next_generation();

    return true;
}

bool Tracker::update_target(
    [[maybe_unused]] talos::TrackerSoA& state, Target& target, const talos::ArmorSlots& slots,
    const Poses3D& poses3d, const uint64_t timestamp_ns) const {
    // ========== 坐标系说明 ==========
    // poses3d中的数据在camera_optical坐标系
    // Target::update时通过tf_manager查询world坐标
    // ================================

    target.predict(timestamp_ns);

    int found_count = 0;

    for (int i = 0; i < 10; ++i) {
        if (!slots.is_valid(i) || !poses3d[i].valid) {
            continue;
        }

        const auto& armor = slots.armors[i];
        const auto name   = static_cast<talos::ArmorName>(armor.name);
        const auto type   = static_cast<talos::ArmorType>(armor.type);

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

// ============ TrackerConcept 方法实现 ============

void Tracker::track(
    const DetectionResult& det, const SolveResult& solve, TrackerOutput& out,
    DebugData& debug) noexcept {
    const auto* slots = get_armor_slots(det);
    const auto* poses = get_poses3d(solve);

    if (!slots || !poses) {
        // 无输入，执行预测
        debug.tracker_state = out.state.load(std::memory_order_relaxed);
        return;
    }
    abort();
    //     // 不再需要坐标变换，直接使用 camera_optical 坐标
    //     // Target 构造/update 时会通过 tf_manager 查询 world 坐标
    //     const auto timestamp = std::chrono::steady_clock::now();
    //     track(out, target_, *slots, *poses, timestamp);
    //
    //     // 更新调试数据
    //     debug.tracker_state = out.state.load(std::memory_order_relaxed);
    //     if (target_.converged()) {
    //         const auto ekf_x    = target_.ekf_x();
    //         debug.predicted_pos = Eigen::Vector3d(ekf_x(0), ekf_x(2), ekf_x(4));
    //     }
}

void Tracker::reset() noexcept {}

} // namespace auto_aim
