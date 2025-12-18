#pragma once

#include <chrono>
#include <string>

#include "detector_state.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tracker_soa.hpp"

namespace auto_aim {

/**
 * @brief 装甲板跟踪器
 *
 * 移植自 sp_vision_25/tasks/auto_aim/tracker
 * 复用 spike::TrackerState 状态机
 * 功能：
 * 1. 状态机管理（IDLE/DETECTING/TRACKING/TEMP_LOST）
 * 2. 目标初始化和更新
 * 3. 发散检测和收敛检测
 */
class Tracker {
public:
    /**
     * @brief 构造跟踪器
     * @param config 跟踪器配置
     * @param solver PnP解算器引用
     */
    Tracker(const SpVisionTrackerConfig &config, Solver &solver);

    /**
     * @brief 获取当前状态
     */
    [[nodiscard]] spike::TrackerState state() const { return state_; }

    /**
     * @brief 获取当前状态字符串
     */
    [[nodiscard]] std::string state_string() const;

    /**
     * @brief 跟踪装甲板
     * @param slots 检测结果
     * @param poses3d 解算得到的3D位姿
     * @param t 时间戳
     * @return 是否有有效目标
     */
    bool track(const spike::ArmorSlots &slots, const std::array<ArmorPose3D, 10> &poses3d,
               std::chrono::steady_clock::time_point t);

    /**
     * @brief 获取当前跟踪目标
     */
    [[nodiscard]] const Target &target() const { return target_; }

    /**
     * @brief 获取当前跟踪目标（可修改）
     */
    [[nodiscard]] Target &target() { return target_; }

    /**
     * @brief 获取目标装甲板列表 (预测位置)
     */
    [[nodiscard]] std::vector<Eigen::Vector4d> armor_xyza_list() const;

private:
    Solver &solver_;
    spike::Color enemy_color_;
    int min_detect_count_;
    int max_temp_lost_count_;
    int detect_count_;
    int temp_lost_count_;
    int outpost_max_temp_lost_count_;
    int normal_temp_lost_count_;
    spike::TrackerState state_;
    Target target_;
    std::chrono::steady_clock::time_point last_timestamp_;

    /**
     * @brief 状态机更新
     */
    void state_machine(bool found);

    /**
     * @brief 设置新目标
     */
    bool set_target(const spike::ArmorSlots &slots, const std::array<ArmorPose3D, 10> &poses3d,
                    std::chrono::steady_clock::time_point t);

    /**
     * @brief 更新现有目标
     */
    bool update_target(const spike::ArmorSlots &slots, const std::array<ArmorPose3D, 10> &poses3d,
                       std::chrono::steady_clock::time_point t);
};

} // namespace auto_aim
