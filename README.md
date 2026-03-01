<div align="center">

# 🤖 Talos

**下一代 RoboMaster 自瞄视觉框架**

_如同守卫克里特岛的青铜巨人，永不疲倦地守护着机器人视觉系统_

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C.svg?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)

</div>

## ✨ 项目亮点

- **🚀 零 ROS 依赖**：告别中间件开销， _新人无需学习 ROS 即可上手_ <sup> ~但需要学talos~ </sup>
- **⚡ 天然无竞态**：DAG 依赖分析 + 通道隔离，初始化期杜绝数据竞争，零锁开销
- **🔒 类型安全**：编译期类型检查，零运行时开销，用类型系统防止误用
- **🔌 模块化设计**：5 级流水线架构，各处理阶段灵活定制，易于独立测试
- **🎯 多种检测后端**：支持传统视觉、OpenVINO、TensorRT，按需选择
- **📊 实时可视化**：集成 Foxglove WebSocket 服务器
- **🎮 模拟器友好**：与 [daedalus](https://github.com/Blackjack200/bevy_robomaster_simulator) 协同工作
- **🖥️ 跨平台支持**：支持 x86_64 PC、NVIDIA Jetson Orin NX、Horizon AX650、Apple Silicon Mac

## 🏗️ 系统架构

### 5 级流水线 (FCS - Fire Control System)

Talos 采用经典的 5 级火控系统架构，各层级职责清晰：

|  阶段  | 模块              | 说明                             |
| :----: | ----------------- | -------------------------------- |
| **L1** | 采集 (Sensor)     | HIK 相机驱动、图像输入接口       |
| **L2** | 检测 (Perception) | 装甲板、能量机关识别             |
| **L3** | 追踪 (Estimation) | 状态估计、整车观测               |
| **L4** | 规划 (Planning)   | 弹道计算、参考轨迹生成、瞄准决策 |
| **L5** | 控制 (Weapon)     | 武器控制系统接口                 |

## 🚀 快速开始

### 环境要求

#### 支持平台

| 平台                      | 架构    |
| ------------------------- | ------- |
| **x86_64 PC**             | x86_64  |
| **NVIDIA Jetson Orin NX** | aarch64 |
| **AX650N**                | aarch64 |
| **Apple Silicon Mac**     | aarch64 |

> **❌ Windows**: 不支持

#### 编译器要求

| 平台          | 编译器          | 标准库      |
| ------------- | --------------- | ----------- |
| **GNU/Linux** | clang-21        | libstdc++13 |
| **macOS**     | Apple Clang 17+ | libSystem   |

#### 核心依赖

| 依赖          | 版本     | 说明                   |
| ------------- | -------- | ---------------------- |
| **CMake**     | >= 3.14  | 构建系统               |
| **Intel TBB** | 2021.9+  | 线程池                 |
| **Eigen3**    | 3.4+     | 线性代数               |
| **OpenCV**    | 4.x      | 图像处理               |
| **Ceres**     | 2.x      | 非线性优化（自动微分） |
| **OpenVINO**  | optional | 推理后端               |
| **TensorRT**  | optional | 推理后端               |

## 🔧 配置管理

Talos 使用 **TOML** 作为统一配置文件格式，按优先级合并：

```
at_vision.toml              # 程序入口配置
config/vision_base.toml     # 视觉基础配置
config/vision/<name>.toml   # 视觉算法特定配置
config/robot/<name>.toml    # 硬件（机器人）配置
```

**合并策略**：具体配置 > 基础配置（覆盖语义）

## 📊 开发状态

| 模块            | 状态 | 说明                                             |
| --------------- | :--: | ------------------------------------------------ |
| **调度系统**    |  ✅  | 高性能多线程调度器，支持 fixed_rate/pool_compute |
| **原语库**      |  ✅  | 锁无关三重缓冲、延迟构造、性能探针               |
| **配置管理**    |  ✅  | TOML 解析 + 合并，版本化资源                     |
| **坐标变换**    |  ✅  | 类型安全坐标变换，替代 tf2                       |
| **相机驱动**    |  ✅  | HIK SDK (x86_64)、USB/MIPI 相机接口 (ARM)        |
| **检测后端**    |  ✅  | 传统视觉、OpenVINO、TensorRT                     |
| **追踪框架**    |  ✅  | EKF + Ceres 自动微分、ISRCKF 能量机关追踪        |
| **弹道模型**    |  ✅  | ideal、linear_drag、quadratic_drag               |
| **瞄准器**      |  ✅  | RobotAimer、OutpostAimer、RuneAimer              |
| **武器系统**    |  ✅  | 武器控制接口                                     |
| **Foxglove**    |  ✅  | 实时可视化                                       |
| **下位机通信**  |  ✅  | gimbal 控制                                      |
| **ROS2 兼容层** |  ✅  | ROS2 风格 API (rclcompat)                        |
| **标定系统**    |  ✅  | 相机内参、手眼标定                               |

## 🙏 致谢

本项目在开发过程中参考了以下开源项目：

- [bevy](https://bevyengine.org/) - 启发了调度器的 System 概念
- [oneTBB](https://github.com/oneapi-src/oneTBB) - Intel 线程构建块
- [toml++](https://marzer.github.io/tomlplusplus/) - 优秀的 TOML 解析库
- [Foxglove](https://foxglove.dev/) - 机器人可视化工具

感谢以下战队的开源分享（排名不分先后）：

- 同济大学 SuperPower 战队 - [sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)
- 华南师范大学 PIONEER 战队 - [rm_vision](https://github.com/chenjunnn/rm_vision)
- 武汉科技大学崇实战队 - [wust_vision](https://github.com/WUST-RM/wust_vision)
- 沈阳航空航天大学 T-UP 战队 - [TUP-InfantryVision-2022](https://github.com/TUP-RM/TUP-InfantryVision-2022)
- 上海交通大学云汉交龙战队 - [CVRM2021-sjtu](https://github.com/Harry-hhj/CVRM2021-sjtu)

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件

<div align="center">

**用 🪓 和 💀 构建**

</div>
