<div align="center">

# 🤖 Talos

**下一代 RoboMaster 自瞄上位机基座**

_如同守卫克里特岛的青铜巨人，以钢铁之躯承载自瞄意志_

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C.svg?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)

</div>

## ✨ 项目亮点

- **🚀 零 ROS 依赖**：摆脱中间件束缚，直接定义系统边界与执行语义
- **⚡ 天然无竞态**：数据流驱动，形式化正确的任务调度
- **🔒 类型安全**：编译期类型检查，零运行时开销，防止误用
- **🔌 模块化设计**：5 级流水线架构，各处理阶段灵活定制，易于独立测试

- **📊 实时可视化**：集成 [Foxglove](https://foxglove.dev) 服务器
- **🎮 模拟器友好**：与 [Daedalus](https://github.com/Blackjack200/bevy_robomaster_simulator) 协同工作
- **🖥️ 跨平台支持**：支持 x86_64 PC、NVIDIA Jetson Orin NX、Axera AX650n、Apple Silicon Mac

## 🏗️ 系统架构

Talos 采用 5 级火控系统架构，各层级职责清晰：

|  阶段  | 模块              | 说明               |
| :----: | ----------------- | ------------------ |
| **L1** | 采集 (Sensor)     | 传感输入与设备接入 |
| **L2** | 检测 (Perception) | 目标识别与观测提取 |
| **L3** | 追踪 (Estimation) | 状态估计与目标建模 |
| **L4** | 规划 (Planning)   | 瞄准决策与解算规划 |
| **L5** | 控制 (Weapon)     | 控制输出与执行接口 |

## 🚀 快速开始

Talos 面向实际部署与跨平台运行，以下为构建与运行所需环境。

### 支持平台

| 平台                      |  架构   | 编译器          | 标准库      |
| ------------------------- | :-----: | --------------- | ----------- |
| **x86_64 PC**             | x86_64  | clang-21        | libstdc++13 |
| **NVIDIA Jetson Orin NX** | aarch64 | clang-21        | libstdc++13 |
| **Axera AX650N**          | aarch64 | clang-21        | libstdc++13 |
| **Mac(Apple Silicon)**    | aarch64 | Apple Clang 17+ | libc++      |

#### 核心依赖

| 依赖                                                  | 版本     | 说明                   |
| ----------------------------------------------------- | -------- | ---------------------- |
| [**CMake**](https://cmake.org)                        | >= 3.14  | 构建系统               |
| [**oneTBB**](https://uxlfoundation.github.io/oneTBB)  | 2021.9+  | 线程池                 |
| [**Eigen3**](https://libeigen.gitlab.io)              | 3.4+     | 线性代数               |
| [**OpenCV**](https://opencv.org)                      | 4.x      | 图像处理               |
| [**Ceres**](https://ceres-solver.org)                 | 2.x      | 非线性优化（自动微分） |
| [**OpenVINO**](https://docs.openvino.ai)              | optional | 推理后端               |
| [**TensorRT**](https://developer.nvidia.com/tensorrt) | optional | 推理后端               |

## 🔧 配置管理

Talos 使用 **[TOML](https://toml.io)** 作为统一配置文件格式，按优先级合并：

```
at_vision.toml              # 程序入口配置
config/vision_base.toml     # 视觉基础配置
config/vision/<name>.toml   # 视觉算法特定配置
config/robot/<name>.toml    # 硬件（机器人）配置
```

**合并策略**：具体配置 > 基础配置（覆盖语义）

## 🙏 致谢

本项目在开发过程中参考了以下开源项目：

- [bevy](https://bevyengine.org/) - 启发了调度器的 System 概念
- [oneTBB](https://uxlfoundation.github.io/oneTBB) - Intel 线程构建块
- [toml++](https://marzer.github.io/tomlplusplus/) - 优秀的 TOML 解析库
- [Foxglove](https://foxglove.dev/) - 机器人可视化工具

感谢以下战队的开源分享（排名不分先后）：

- 同济大学 SuperPower 战队 - [sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)
- 陈君 - [rm_vision](https://github.com/chenjunnn/rm_vision)
- 中南大学 FYT 战队 - [FYT2024_vision](https://github.com/CSU-FYT-Vision/FYT2024_vision)
- 武汉科技大学 崇实战队 - [wust_vision](https://github.com/WUST-RM/wust_vision)
- 沈阳航空航天大学 T-UP 战队 - [TUP-InfantryVision-2022](https://github.com/TUP-RM/TUP-InfantryVision-2022)

---

<div align="center">

**用 🪓 和 💀 构建**

</div>
