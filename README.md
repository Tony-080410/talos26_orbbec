<div align="center">

# 🤖 Talos

**下一代 RoboMaster 自瞄视觉框架**

_如同守卫克里特岛的青铜巨人，永不疲倦地守护着机器人视觉系统_

[![C++23](https://img.shields.io/badge/C++-23-00599C.svg?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![TBB](https://img.shields.io/badge/Intel-TBB-0071C5.svg?style=for-the-badge&logo=intel)](https://github.com/oneapi-src/oneTBB)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)

</div>

## ✨ 项目亮点

- **无 ROS 依赖**：单二进制部署，告别中间件开销
- **高度可扩展**：模块化设计，灵活定制各个处理阶段
- **全链路性能基准**：正在开发完整的性能分析框架：
  - [x] **全链路追踪**：精确度量每个处理阶段耗时，定位性能瓶颈
  - [ ] **精度评估**：
    - [ ] 自动计算检测准确率
    - [ ] 追踪稳定性
  - [x] **回归测试**：确保算法更新不会影响性能
- **多种检测后端支持**：
  - 传统视觉算法
  - AT 神经网络检测
  - TUP 神经网络检测
- **模拟器友好**：与 [daedalus](https://github.com/Blackjack200/bevy_robomaster_simulator) 协同工作，支持离线开发与测试

## 🏗️ 系统架构

| 阶段 | 模块              | 实现状态    | 说明                       |
| ---- | ----------------- | ----------- | -------------------------- |
| L1   | 采集 (sensor)     | ✅ 已实现   | HIK 相机驱动、图像输入接口 |
| L2   | 检测 (perception) | ✅ 已实现   | 装甲板识别、多种检测后端   |
| L3   | 追踪 (estimation) | ✅ 已实现   | 装甲解算器接               |
| L4   | 解算 (planning)   | 📋 规划中   | 弹道计算、瞄准决策         |
| L5   | 控制 (weapon)     | 📋 接口就绪 | 武器控制系统接口           |

---

## 🗓️ 开发状态

| 模块              | 状态        | 说明                                   |
| ----------------- | ----------- | -------------------------------------- |
| **调度系统**      | ✅ 完成     | 高性能调度器                           |
| **配置管理**      | ✅ 完成     | 支持 TOML 配置文件                     |
| **相机驱动**      | ✅ 完成     | HIK 相机 SDK（Linux x86_64）/ 存根实现 |
| **检测后端**      | ✅ 完成     | 支持 NN、传统算法、TUP 三种实现        |
| **输入接口**      | ✅ 完成     | L1 传感器输入抽象                      |
| **输出接口**      | ✅ 完成     | L1 传感器输出抽象                      |
| **追踪框架**      | 📋 接口就绪 | L3 装甲解算器接口定义                  |
| **武器系统**      | 📋 接口就绪 | L5 武器控制接口定义                    |
| **Foxglove 集成** | ✅ 已集成   | 实时可视化支持                         |
| **串口通信**      | ✅ 已集成   | gimbal 模块接口定义                    |

<div align="center">

**用 🪓 和 💀 构建**

_没有 ROS 的负担，只有纯粹的性能_

</div>
