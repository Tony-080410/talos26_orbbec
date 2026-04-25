<div align="center">

# 🤖 Talos

**RoboMaster 自瞄系统的下一代上位机基座**

_如同守卫克里特岛的青铜巨人，以钢铁之躯承载自瞄意志。_

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C.svg?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)

</div>

---

## Talos

Talos 是一个面向 RoboMaster 自瞄系统设计的上位机基座。

它将传感输入、目标感知、状态估计、瞄准规划与执行输出组织为一条清晰的火控流水线，并以统一的运行时、配置系统和调度模型承载完整的自瞄应用。

从开发机到车载平台，从模拟器到真实硬件，Talos 提供一致的工程入口。

## 设计理念

Talos 追求三个核心目标：

**清晰。**  
系统被拆分为职责明确的阶段。每个模块只描述自己读取什么、产生什么，以及如何处理数据。

**可靠。**  
运行时根据资源依赖组织任务执行，让数据流自然形成调度边界，减少隐式状态和手写同步带来的复杂度。

**可部署。**  
Talos 面向真实比赛环境设计。配置、硬件接入、可视化、模拟器和跨平台构建都服务于最终部署。

## 火控流水线

Talos 使用五级火控架构：

|  阶段  | 模块              | 说明               |
| :----: | ----------------- | ------------------ |
| **L1** | 采集 (Sensor)     | 传感输入与设备接入 |
| **L2** | 检测 (Perception) | 目标识别与观测提取 |
| **L3** | 追踪 (Estimation) | 状态估计与目标建模 |
| **L4** | 规划 (Planning)   | 瞄准决策与解算规划 |
| **L5** | 控制 (Weapon)     | 开火决策与控制输出 |

## 运行时

Talos 的运行时围绕资源和任务构建。

模块不直接管理全局状态，而是声明自己的输入与输出。调度器根据这些声明建立执行关系，并在运行时驱动整条流水线。

```text
Resource ──read──▶ System ──write──▶ Resource
````

这种模型让模块边界、数据依赖和执行顺序保持一致，也让单元测试、平台迁移和功能裁剪更加直接。

## 平台

| 平台                        |        架构        | 工具链                                  | 状态 |
| ------------------------- | :--------------: | ------------------------------------ | -- |
| **PC**                    |      x86_64      | LLVM clang-21 + libstdc++13          | ✅  |
| **NVIDIA Jetson Orin NX** |      aarch64     | LLVM clang-21 + libstdc++13          | ✅  |
| **Axera AX650N**          |      aarch64     | LLVM clang-21 + libstdc++13          | ✅  |
| **Mac**                   | aarch64 / x86_64 | Apple Clang + libc++                 | ✅  |
| **Android / Termux**      |      aarch64     | LLVM clang-21 + libc++ / bionic libc | 🧪 |

> ✅：已验证支持
> 🧪：实验性支持，主流程可运行，外设、相机链路和厂商 SDK 可能需要额外适配

## 依赖

| 依赖                                                | 版本       | 用途         |
| ------------------------------------------------- | -------- | ---------- |
| [CMake](https://cmake.org)                        | >= 3.14  | 构建系统       |
| [oneTBB](https://uxlfoundation.github.io/oneTBB)  | 2021.9+  | 任务调度与线程池   |
| [Eigen3](https://libeigen.gitlab.io)              | 3.4+     | 线性代数       |
| [OpenCV](https://opencv.org)                      | 4.x      | 图像处理       |
| [Ceres](https://ceres-solver.org)                 | 2.x      | 非线性优化与自动微分 |
| [OpenVINO](https://docs.openvino.ai)              | optional | 可选推理后端     |
| [TensorRT](https://developer.nvidia.com/tensorrt) | optional | 可选推理后端     |

## 快速开始

```bash
bash build.sh
```

Talos 会从入口配置加载运行时、视觉算法和机器人硬件配置。

## 配置

Talos 使用 **[TOML](https://toml.io)** 作为统一配置格式。

```text
at_vision.toml              # 程序入口配置
config/vision_base.toml     # 视觉基础配置
config/vision/<name>.toml   # 视觉算法配置
config/robot/<name>.toml    # 机器人硬件配置
```

配置采用覆盖式合并。越具体的配置拥有越高优先级。

## 可视化

Talos 集成 Foxglove，用于观察图像、识别结果、目标状态、规划信息与运行时调试数据。

在不支持可视化传输的平台上，Talos 仍可保持核心链路运行。

## 模拟器

Talos 可与 Daedalus 协同工作，在模拟环境中完成视觉链路与控制逻辑调试。

[Daedalus](https://github.com/Blackjack200/bevy_robomaster_simulator)

## 状态

Talos 正在快速迭代。

## 致谢

Talos 的开发受到以下项目启发：

* [bevy](https://bevyengine.org/) - System 调度模型
* [oneTBB](https://uxlfoundation.github.io/oneTBB) - 任务调度与并行执行
* [toml++](https://marzer.github.io/tomlplusplus/) - TOML 解析
* [Foxglove](https://foxglove.dev/) - 机器人可视化

感谢以下战队与开发者的开源分享（排名不分先后）：

* 同济大学 SuperPower 战队 - [sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)
* 华南师范大学 PIONEER 陈君 - [rm_vision](https://github.com/chenjunnn/rm_vision)
* 中南大学 FYT 战队 - [FYT2024_vision](https://github.com/CSU-FYT-Vision/FYT2024_vision)
* 武汉科技大学 崇实战队 - [wust_vision](https://github.com/WUST-RM/wust_vision)
* 沈阳航空航天大学 T-UP 战队 - [TUP-InfantryVision-2022](https://github.com/TUP-RM/TUP-InfantryVision-2022)

感谢 RoboMaster 社区长期以来的工程探索与知识传承。

---

<div align="center">

**用 🪓 和 💀 构建**

</div>
