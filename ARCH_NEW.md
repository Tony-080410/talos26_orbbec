# 🎯 RoboMaster FCS 军工级架构设计

基于现有代码和 F-22 火控系统分层理念，完整的架构重构方案。

## 🏗️ 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MISSION COMPUTER (Main Loop)                          │
│                     Orchestration / Mode Management / FSM                    │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────────────────────────────┐
        ▼                                                                      │
┌───────────────────────────────────────────────────────────────┐              │
│              L1: SENSOR MANAGEMENT LAYER                       │              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │              │
│  │ SensorPool  │ │ ScanPolicy  │ │ ResourceAllocator       │  │              │
│  │ (cameras,   │ │ (search,    │ │ (bandwidth, priority,   │  │              │
│  │  IMU, etc.) │ │  track,     │ │  sensor fusion weight)  │  │              │
│  │             │ │  TWS mode)  │ │                         │  │              │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘  │              │
└───────────────────────────┬───────────────────────────────────┘              │
                            │ RawSensorFrame                                   │
                            ▼                                                  │
┌───────────────────────────────────────────────────────────────┐              │
│              L2: PERCEPTION LAYER                              │              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │              │
│  │ Detector    │ │ Classifier  │ │ MeasurementAssociator   │  │              │
│  │ (YOLO/      │ │ (armor ID,  │ │ (gating, confidence     │  │              │
│  │  OpenCV)    │ │  robot type)│ │  scoring, clustering)   │  │              │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘  │              │
└───────────────────────────┬───────────────────────────────────┘              │
                            │ Detection[]                                      │
                            ▼                                                  │
┌───────────────────────────────────────────────────────────────┐              │
│              L3: STATE ESTIMATION LAYER                        │              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │              │
│  │ FilterBank  │ │ MotionModel │ │ UncertaintyQuantifier   │  │              │
│  │ (EKF/UKF/   │ │ (CV, CA,    │ │ (covariance, residual   │  │              │
│  │  ISRCKF)    │ │  CT, IMM)   │ │  chi², track quality)   │  │              │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘  │              │
└───────────────────────────┬───────────────────────────────────┘              │
                            │ TrackState[]                                     │
                            ▼                                                  │
┌───────────────────────────────────────────────────────────────┐              │
│              L4: TARGET MANAGEMENT LAYER                       │              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │              │
│  │ TrackTable  │ │ LockManager │ │ ThreatEvaluator         │  │              │
│  │ (create,    │ │ (STT, TWS,  │ │ (priority, range,       │  │  Feedback    │
│  │  confirm,   │ │  handoff,   │ │  angle, health)         │  │◄─────────────┤
│  │  coast,     │ │  break-lock)│ │                         │  │              │
│  │  delete)    │ │             │ │                         │  │              │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘  │              │
└───────────────────────────┬───────────────────────────────────┘              │
                            │ EngageableTarget                                 │
                            ▼                                                  │
┌───────────────────────────────────────────────────────────────┐              │
│              L5: WEAPON SYSTEM LAYER                           │              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │              │
│  │ Trajectory  │ │ Envelope    │ │ FireControlSolution     │  │              │
│  │ Predictor   │ │ Calculator  │ │ (lead angle, TOF,       │  │              │
│  │ (ballistic, │ │ (Rmax, Rmin,│ │  shoot/no-shoot,        │  │              │
│  │  air drag)  │ │  aspect)    │ │  probability of hit)    │  │              │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘  │              │
└───────────────────────────┬───────────────────────────────────┘              │
                            │ GimbalCommand + TriggerDecision                  │
                            ▼                                                  │
┌───────────────────────────────────────────────────────────────┐              │
│              L6: ACTUATION LAYER                               │              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐  │              │
│  │ GimbalCtrl  │ │ Shooter     │ │ SafetyInterlock         │  │              │
│  │ (servo,     │ │ (trigger,   │ │ (boundary, rate limit,  │──┘              │
│  │  feedfwd)   │ │  burst)     │ │  self-check)            │                 │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘                 │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 重构后的目录结构

```
sp_vision/
├── CMakeLists.txt
├── README.md
├── include/talos/
│   │
│   ├── core/                        # ═══ 核心基础设施 ═══
│   │   ├── types.hpp                # 全局类型定义
│   │   └── time.hpp                 # 高精度时间戳
│   │
│   ├── math/                        # ═══ 数学工具 ═══
│   │   ├── 保持现有的
│   │
│   ├── tf/                          # ═══ 坐标系统 ═══
│   │   ├── frame_graph.hpp          # 坐标系树
│   │   ├── transform.hpp            # SE(3) 变换
│   │   ├── infantry_model.hpp       # 步兵机器人模型
│   │   ├── time_travel.hpp          # 历史变换查询
│   │   └── 大部分保持现有的
│   │
│   │   ┌─────────────────────────────────────────────────────────┐
│   │   │         F-22 FIRE CONTROL SYSTEM LAYERS                 │
│   │   └─────────────────────────────────────────────────────────┘
│   │
│   ├── L1_sensor/                   # ═══ 传感器管理层 ═══
│   │   ├── sensor_manager.hpp       # 传感器池管理
│   │   ├── scan_policy.hpp          # 扫描策略 (Search/Track/TWS)
│   │   ├── resource_allocator.hpp   # 带宽/优先级分配
│   │   ├── interfaces/
│   │   │   ├── sensor_interface.hpp # 传感器抽象接口
│   │   │   ├── camera_driver.hpp    # 相机驱动
│   │   │   └── imu_driver.hpp       # IMU驱动
│   │   ├── fusion/
│   │   │   └── sensor_fusion.hpp    # 多传感器融合
│   │   └── messages/
│   │       └── raw_frame.hpp        # 原始帧数据结构
│   │
│   ├── L2_perception/               # ═══ 感知层 ═══
│   │   ├── perception_pipeline.hpp  # 感知流水线
│   │   ├── detector/
│   │   │   ├── detector_interface.hpp
│   │   │   ├── yolo_detector.hpp    # YOLO检测器
│   │   │   ├── light_bar_detector.hpp # 灯条检测
│   │   │   └── detector_factory.hpp
│   │   ├── classifier/
│   │   │   ├── armor_classifier.hpp # 装甲板分类
│   │   ├── association/
│   │   │   ├── gating.hpp           # 关联门限
│   │   │   ├── hungarian.hpp        # 匈牙利算法
│   │   │   └── measurement_associator.hpp
│   │   └── messages/
│   │       ├── detection.hpp        # 检测结果
│   │       └── measurement.hpp      # 量测数据
│   │
│   ├── L3_estimation/               # ═══ 状态估计层 ═══
│   │   ├── estimator_bank.hpp       # 滤波器组管理
│   │   ├── filters/
│   │   │   ├── filter_interface.hpp
│   │   │   ├── extended_kalman.hpp  # EKF
│   │   │   ├── unscented_kalman.hpp # UKF
│   │   │   ├── isrckf.hpp           # 迭代平方根容积卡尔曼
│   │   │   └── imm_filter.hpp       # 交互多模型
│   │   ├── models/
│   │   │   ├── motion_model.hpp     # 运动模型接口
│   │   │   ├── constant_velocity.hpp
│   │   │   ├── constant_acceleration.hpp
│   │   ├── uncertainty/
│   │   │   ├── covariance_manager.hpp
│   │   │   ├── chi_square_gate.hpp  # χ²检验
│   │   │   └── track_quality.hpp    # 航迹质量评估
│   │   └── messages/
│   │       └── track_state.hpp      # 航迹状态
│   │
│   ├── L4_target/                   # ═══ 目标管理层 ═══
│   │   ├── target_manager.hpp       # 目标管理器
│   │   ├── track_table/
│   │   │   ├── track_table.hpp      # 航迹表
│   │   │   ├── track_lifecycle.hpp  # 生命周期 (tentative/confirmed/coasting/deleted)
│   │   │   └── track_id_allocator.hpp
│   │   ├── lock/
│   │   │   ├── lock_manager.hpp     # 锁定管理器
│   │   │   ├── lock_state.hpp       # STT
│   │   └── messages/
│   │       └── engageable_target.hpp # 可交战目标
│   │
│   ├── L5_weapon/                   # ═══ 武器系统层 ═══
│   │   ├── weapon_controller.hpp    # 武器控制器
│   │   ├── trajectory/
│   │   │   ├── ballistic_solver.hpp # 弹道解算
│   │   │   ├── air_drag_model.hpp   # 空气阻力
│   │   │   ├── gravity_drop.hpp     # 重力补偿
│   │   │   └── lead_angle.hpp       # 提前量计算
│   │   ├── envelope/
│   │   │   ├── engagement_envelope.hpp # 交战包线
│   │   │   ├── range_calculator.hpp    # Rmax/Rmin
│   │   │   └── aspect_angle.hpp        # 姿态角
│   │   ├── solution/
│   │   │   ├── fire_control_solution.hpp # 火控解算
│   │   │   ├── shoot_decision.hpp       # 射击决策
│   │   │   └── hit_probability.hpp      # 命中概率
│   │   └── messages/
│   │       └── weapon_command.hpp   # 武器指令
│   │
│   ├── L6_actuation/                # ═══ 执行层 ═══
│   │   ├── actuation_manager.hpp    # 执行管理器
│   │   ├── gimbal/
│   │   │   ├── gimbal_controller.hpp    # 云台控制
│   │   │   ├── servo_model.hpp          # 伺服模型
│   │   │   ├── feedforward.hpp          # 前馈控制
│   │   │   └── gimbal_limits.hpp        # 限位保护
│   │   ├── shooter/
│   │   │   ├── shooter_controller.hpp   # 射击控制
│   │   │   ├── trigger_logic.hpp        # 触发逻辑
│   │   │   └── burst_controller.hpp     # 连发控制
│   │   ├── safety/
│   │   │   ├── safety_interlock.hpp     # 安全联锁
│   │   │   ├── boundary_check.hpp       # 边界检查
│   │   │   └── self_diagnosis.hpp       # 自检
│   │   └── interfaces/
│   │       ├── actuator_interface.hpp
│   │       └── stm32_protocol.hpp   # STM32通信协议
│   │
│   │   └─────────────────────────────────────────────────────────┘
│   │
│   ├── orchestration/               # ═══ 任务编排 ═══
│   │   ├── mission_computer.hpp     # 任务计算机
│   │   ├── mode_manager.hpp         # 模式管理 (SAFE/SEARCH/TRACK/ENGAGE)
│   │   ├── event_bus.hpp            # 事件总线
│   │   └── health_monitor.hpp       # 健康监控
│   │
│   ├── ipc/                         # ═══ 进程间通信 ═══
│   │   ├── shared_memory.hpp
│   │   ├── triple_buffer.hpp
│   │   └── message_queue.hpp
│   │
│   ├── scheduler/                   # ═══ 调度器 ═══
│   │   ├── real_time_scheduler.hpp
│   │   ├── thread_pool.hpp
│   │   ├── priority_executor.hpp
│   │   └── timing_analysis.hpp
│   │
│   └── telemetry/                   # ═══ 遥测/调试 ═══
│       ├── logger.hpp
│       ├── recorder.hpp             # 数据记录
│       ├── foxglove_bridge.hpp      # 可视化
│       └── performance_probe.hpp
│
├── src/
│   ├── main.cpp                     # 入口点
│   ├── core/
│   ├── L1_sensor/
│   ├── L2_perception/
│   ├── L3_estimation/
│   ├── L4_target/
│   ├── L5_weapon/
│   ├── L6_actuation/
│   ├── orchestration/
│   ├── hal/
│   └── ...
│
├── tests/
│   ├── unit/
│   │   ├── L3_estimation/
│   │   │   └── test_ekf.cpp
│   │   └── L5_weapon/
│   │       └── test_ballistic.cpp
│   ├── integration/
│   │   └── test_full_pipeline.cpp
│   └── simulation/
│       └── hardware_in_loop.cpp
│
└── tools/
    ├── dataset_loader/
    ├── calibration/
    └── tuner/
```

---

## 🔄 数据流与消息定义

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// 层间消息类型定义
// ═══════════════════════════════════════════════════════════════════════════

namespace fcs::msg {

// L1 → L2: 原始传感器帧
struct RawSensorFrame {
    Timestamp timestamp;
    SensorId sensor_id;
    cv::Mat image;
    Pose3D sensor_pose;          // 传感器在世界坐标系中的位姿
    std::optional<ImuData> imu;  // 同步的IMU数据
};

// L2 → L3: 量测数据
struct Measurement {
    Timestamp timestamp;
    Eigen::Vector3d position_camera;  // 相机坐标系下位置
    Eigen::Matrix3d covariance;       // 量测协方差
    float confidence;                  // 检测置信度
    ArmorType armor_type;             // 装甲板类型
    RobotClass robot_class;           // 机器人类别
    cv::Rect2f bbox;                  // 边界框（用于显示）
};

// L3 → L4: 航迹状态
struct TrackState {
    TrackId id;
    Timestamp timestamp;
    
    // 状态向量 [x, y, z, vx, vy, vz, ...]
    Eigen::VectorXd state;
    Eigen::MatrixXd covariance;
    
    // 元数据
    TrackStatus status;               // TENTATIVE/CONFIRMED/COASTING
    uint32_t hit_count;
    uint32_t miss_count;
    float track_quality;              // 0.0 ~ 1.0
    MotionModelType active_model;     // 当前激活的运动模型
};

// L4 → L5: 可交战目标
struct EngageableTarget {
    TrackId track_id;
    Timestamp timestamp;
    
    // 预测状态（在预计射击时刻）
    Eigen::Vector3d predicted_position;
    Eigen::Vector3d predicted_velocity;
    Eigen::Matrix3d position_uncertainty;
    
    // 目标属性
    RobotClass robot_class;
    float threat_level;               // 威胁等级
    LockState lock_state;             // STT/NONE
    
    // 交战窗口
    TimeRange engagement_window;
    bool within_envelope;             // 是否在交战包线内
};

// L5 → L6: 武器指令
struct WeaponCommand {
    Timestamp timestamp;
    
    // 云台指令
    struct GimbalCmd {
        float yaw_angle;              // 目标偏航角
        float pitch_angle;            // 目标俯仰角
        float yaw_rate;               // 角速度（前馈）
        float pitch_rate;
    } gimbal;
    
    // 射击指令
    struct ShootCmd {
        bool fire_enable;             // 允许射击
        ShootMode mode;               // SINGLE/BURST/AUTO
        uint8_t burst_count;          // 连发数量
    } shoot;
    
    // 解算信息
    float time_of_flight;             // 弹丸飞行时间
    float hit_probability;            // 预计命中概率
    SolutionQuality quality;          // 解算质量
};

}  // namespace fcs::msg
```

---

## 🎛️ 核心组件详细设计

### L4: Target Management Layer (航迹管理层)

```cpp
// include/fcs/L4_target/track_table/track_lifecycle.hpp

namespace fcs::L4 {

/**
 * 航迹生命周期状态机
 * 
 *                    ┌─────────────────────────────────────┐
 *                    │                                     │
 *                    ▼                                     │
 *  ┌─────────┐    ┌─────────────┐    ┌───────────┐    ┌────────┐
 *  │ (birth) │───▶│  TENTATIVE  │───▶│ CONFIRMED │───▶│COASTING│
 *  └─────────┘    └─────────────┘    └───────────┘    └────────┘
 *                       │                  │               │
 *                       │ miss > M         │ miss > N      │ coast > T
 *                       ▼                  ▼               ▼
 *                  ┌─────────┐        ┌─────────┐     ┌─────────┐
 *                  │ DELETED │        │ DELETED │     │ DELETED │
 *                  └─────────┘        └─────────┘     └─────────┘
 */
class TrackLifecycle {
public:
    enum class State {
        TENTATIVE,    // 暂定：需要连续命中确认
        CONFIRMED,    // 确认：稳定跟踪中
        COASTING,     // 滑行：短暂丢失，惯性预测
        DELETED       // 删除：待清理
    };

    struct Config {
        uint32_t confirm_hits = 3;      // 确认所需连续命中数
        uint32_t tentative_miss = 2;    // 暂定态允许丢失数
        uint32_t confirmed_miss = 5;    // 确认态允许丢失数
        Duration coast_timeout = 500ms; // 滑行超时时间
    };

    void update(bool measurement_associated);
    State current_state() const { return state_; }
    bool should_delete() const { return state_ == State::DELETED; }

private:
    State state_ = State::TENTATIVE;
    uint32_t consecutive_hits_ = 0;
    uint32_t consecutive_misses_ = 0;
    Timestamp last_update_;
    Config config_;
};

}  // namespace fcs::L4
```

### L5: Weapon System Layer (武器系统层)

```cpp
// include/fcs/L5_weapon/solution/fire_control_solution.hpp

namespace fcs::L5 {

/**
 * 火控解算器
 * 
 * 输入：目标当前状态 + 预测
 * 输出：云台角度 + 射击决策
 */
class FireControlSolution {
public:
    struct Input {
        msg::EngageableTarget target;
        Pose3D gimbal_pose;           // 云台当前位姿
        Eigen::Vector3d muzzle_pos;   // 炮口位置
        float muzzle_velocity;        // 初速度
    };

    struct Output {
        // 瞄准解
        float aim_yaw;                // 云台目标偏航
        float aim_pitch;              // 云台目标俯仰
        Eigen::Vector3d aim_point;    // 瞄准点（世界坐标）
        
        // 射击决策
        ShootDecision decision;       // SHOOT / HOLD / INHIBIT
        float hit_probability;        // 命中概率估计
        float time_of_flight;         // 弹丸飞行时间
        
        // 诊断信息
        SolutionStatus status;        // 解算状态
        std::string reject_reason;    // 拒绝原因（如有）
    };

    Output solve(const Input& input);

private:
    // 迭代求解弹道
    std::optional<BallisticSolution> solve_ballistic(
        const Eigen::Vector3d& target_pos,
        const Eigen::Vector3d& target_vel,
        float muzzle_velocity,
        int max_iterations = 10
    );

    // 射击决策逻辑
    ShootDecision evaluate_shoot_decision(
        const BallisticSolution& ballistic,
        const msg::EngageableTarget& target,
        float hit_prob
    );

    BallisticSolver ballistic_;
    EnvelopeCalculator envelope_;
    HitProbabilityEstimator hit_prob_;
};

/**
 * 射击决策枚举
 */
enum class ShootDecision {
    SHOOT,      // 射击！
    HOLD,       // 等待更好时机
    INHIBIT,    // 禁止射击（安全原因）
    NO_SOLUTION // 无有效解
};

}  // namespace fcs::L5
```

---

## 🔧 Mission Computer (任务计算机)

```cpp
// include/fcs/orchestration/mission_computer.hpp

namespace fcs {

/**
 * 任务计算机 - FCS 主控制器
 * 
 * 职责：
 * 1. 模式管理 (SAFE → SEARCH → TRACK → ENGAGE)
 * 2. 层间数据流编排
 * 3. 系统健康监控
 * 4. 异常处理
 */
class MissionComputer {
public:
    enum class Mode {
        SAFE,       // 安全模式：仅感知，不开火
        SEARCH,     // 搜索模式：广域扫描
        TRACK,      // 跟踪模式：稳定跟踪目标
        ENGAGE      // 交战模式：允许开火
    };

    MissionComputer(const Config& config);

    // 主循环（运行在独立线程）
    void run();
    void request_shutdown();

    // 模式切换
    void set_mode(Mode mode);
    Mode current_mode() const { return mode_; }

    // 外部输入
    void on_operator_command(const OperatorCommand& cmd);

private:
    // 单帧处理流水线
    void process_frame();

    // 各层实例
    std::unique_ptr<L1::SensorManager> sensor_mgr_;
    std::unique_ptr<L2::PerceptionPipeline> perception_;
    std::unique_ptr<L3::EstimatorBank> estimator_;
    std::unique_ptr<L4::TargetManager> target_mgr_;
    std::unique_ptr<L5::WeaponController> weapon_ctrl_;
    std::unique_ptr<L6::ActuationManager> actuation_;

    // 状态
    std::atomic<Mode> mode_{Mode::SAFE};
    std::atomic<bool> shutdown_requested_{false};

    // 事件总线
    EventBus event_bus_;

    // 性能监控
    HealthMonitor health_monitor_;
    TimingAnalyzer timing_;
};

}  // namespace fcs
```

---

## 📊 状态机：模式转换

```
                    ┌─────────────────────────────────────────────────┐
                    │                                                 │
                    │  ┌──────┐         ┌────────┐                    │
     Power On ─────▶│  │ SAFE │◀───────▶│ SEARCH │                    │
                    │  └──┬───┘         └───┬────┘                    │
                    │     │                 │                         │
                    │     │ arm_command     │ target_acquired         │
                    │     │                 ▼                         │
                    │     │            ┌────────┐                     │
                    │     └───────────▶│ TRACK  │◀────────┐           │
                    │                  └───┬────┘         │           │
                    │                      │              │           │
                    │  fire_command &&     │              │ lock_lost │
                    │  in_envelope         ▼              │           │
                    │                 ┌─────────┐         │           │
                    │                 │ ENGAGE  │─────────┘           │
                    │                 └─────────┘                     │
                    │                                                 │
                    │    ANY STATE ──── emergency_stop ───▶ SAFE      │
                    │                                                 │
                    └─────────────────────────────────────────────────┘

Mode Capabilities:
╔══════════╦═════════╦═══════════╦══════════╦═══════════╗
║   Mode   ║ Sensing ║ Tracking  ║ Solution ║ Fire      ║
╠══════════╬═════════╬═══════════╬══════════╬═══════════╣
║ SAFE     ║    ✓    ║     ✗     ║    ✗     ║    ✗      ║
║ SEARCH   ║    ✓    ║     ✓     ║    ✗     ║    ✗      ║
║ TRACK    ║    ✓    ║     ✓     ║    ✓     ║    ✗      ║
║ ENGAGE   ║    ✓    ║     ✓     ║    ✓     ║    ✓      ║
╚══════════╩═════════╩═══════════╩══════════╩═══════════╝
```

---

## ⏱️ 实时调度设计

```cpp
// 线程模型与优先级
// 
// ┌──────────────────────────────────────────────────────────────────┐
// │                        Thread Architecture                        │
// ├──────────────────────────────────────────────────────────────────┤
// │                                                                  │
// │  ┌─────────────────┐    Priority: SCHED_FIFO 99                  │
// │  │  Sensor Thread  │    Period: 1ms (1kHz)                       │
// │  │  (L1)           │    Affinity: CPU 0                          │
// │  └────────┬────────┘                                             │
// │           │ RawFrame                                             │
// │           ▼                                                      │
// │  ┌─────────────────┐    Priority: SCHED_FIFO 90                  │
// │  │ Perception      │    Period: ~8ms (triggered by frame)        │
// │  │ Thread (L2)     │    Affinity: CPU 1                          │
// │  └────────┬────────┘                                             │
// │           │ Detections                                           │
// │           ▼                                                      │
// │  ┌─────────────────┐    Priority: SCHED_FIFO 80                  │
// │  │ Estimation +    │    Period: 1ms (1kHz)                       │
// │  │ Target Thread   │    Affinity: CPU 2                          │
// │  │ (L3 + L4)       │                                             │
// │  └────────┬────────┘                                             │
// │           │ EngageableTarget                                     │
// │           ▼                                                      │
// │  ┌─────────────────┐    Priority: SCHED_FIFO 95                  │
// │  │ Weapon +        │    Period: 1ms (1kHz)                       │
// │  │ Actuation       │    Affinity: CPU 3                          │
// │  │ (L5 + L6)       │    *** HARD REAL-TIME ***                   │
// │  └─────────────────┘                                             │
// │                                                                  │
// │  ┌─────────────────┐    Priority: SCHED_OTHER (normal)           │
// │  │ Telemetry /     │    Non-RT                                   │
// │  │ Logging         │    Affinity: CPU 4-7                        │
// │  └─────────────────┘                                             │
// │                                                                  │
// └──────────────────────────────────────────────────────────────────┘
```

---

## 🔗 与现有代码的映射关系

| 现有模块                           | 新架构位置                                           | 说明            |
|--------------------------------|-------------------------------------------------|---------------|
| `auto_aim/detector.hpp`        | `L2_perception/detector/`                       | 保留核心逻辑，增加接口抽象 |
| `auto_aim/classifier.hpp`      | `L2_perception/classifier/`                     | 同上            |
| `auto_aim/tracker.hpp`         | `L3_estimation/estimator_bank.hpp`              | 拆分为滤波器+数据关联   |
| `auto_aim/at/isrckf.hpp`       | `L3_estimation/filters/isrckf.hpp`              | 直接迁移          |
| `auto_aim/at/motion_model.hpp` | `L3_estimation/models/`                         | 拆分为多个模型类      |
| `auto_aim/target.hpp`          | `L4_target/track_table/`                        | 扩展生命周期管理      |
| `auto_aim/solver.hpp`          | `L5_weapon/trajectory/`                         | 拆分弹道解算        |
| `auto_aim/trajectory.hpp`      | `L5_weapon/trajectory/`                         | 同上            |
| `auto_aim/shooter.hpp`         | `L5_weapon/solution/` + `L6_actuation/shooter/` | 拆分决策与执行       |
| `auto_aim/aimer.hpp`           | `orchestration/mission_computer.hpp`            | 升级为任务计算机      |
| `camera/hik_camera.hpp`        | `hal/camera/`                                   | 移入HAL层        |
| `fast_tf/*`                    | `tf/*`                                          | 保留，增强时间旅行功能   |
| `talos_gimbal/*`               | `L6_actuation/interfaces/`                      | 移入执行层         |
| `scheduler/*`                  | `scheduler/`                                    | 增强实时调度        |
| `ipc/*`                        | `ipc/`                                          | 保留            |

---

## 🚀 实施建议

### Phase 1: 基础设施 (Week 1-2)

```
□ 建立新目录结构
□ 定义核心类型 (types.hpp, result.hpp)
□ 实现消息结构 (msg namespace)
□ 迁移数学工具
```

### Phase 2: 感知管道 (Week 3-4)

```
□ 实现 L1 SensorManager
□ 重构 L2 Detector + Classifier
□ 实现 MeasurementAssociator
```

### Phase 3: 状态估计 (Week 5-6)

```
□ 实现 FilterBank 架构
□ 迁移 EKF/ISRCKF
□ 实现 IMM 滤波器
□ 实现航迹质量评估
```

### Phase 4: 目标管理 (Week 7-8)

```
□ 实现 TrackTable + Lifecycle
□ 实现 LockManager
□ 实现 ThreatEvaluator
```

### Phase 5: 武器系统 (Week 9-10)

```
□ 重构弹道解算器
□ 实现射击决策逻辑
□ 实现命中概率估计
```

### Phase 6: 执行与集成 (Week 11-12)

```
□ 实现 ActuationManager
□ 实现 MissionComputer
□ 全系统集成测试
□ 性能优化
```
