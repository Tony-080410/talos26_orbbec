<div align="center">

# 🤖 Talos

**下一代 RoboMaster 自瞄视觉框架**

_如同守卫克里特岛的青铜巨人，永不疲倦地守护着机器人视觉系统_

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C.svg?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![TBB](https://img.shields.io/badge/Intel-TBB-0071C5.svg?style=for-the-badge&logo=intel)](https://github.com/oneapi-src/oneTBB)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)

</div>

## ✨ 项目亮点

- **无 ROS 依赖**：单二进制部署，告别中间件开销
- **高度可扩展**：模块化设计，灵活定制各个处理阶段
- **全链路性能基准**：正在开发完整的性能分析框架
- **多种检测后端支持**：传统视觉、AT 神经网络、TUP 神经网络
- **模拟器友好**：与 [daedalus](https://github.com/Blackjack200/bevy_robomaster_simulator) 协同工作，支持离线开发与测试

## 🏗️ 系统架构

| 阶段 | 模块              | 说明                       |
| ---- | ----------------- | -------------------------- |
| L1   | 采集 (sensor)     | HIK 相机驱动、图像输入接口 |
| L2   | 检测 (perception) | 装甲板识别、多种检测后端   |
| L3   | 追踪 (estimation) | 装甲解算器                 |
| L4   | 解算 (planning)   | 弹道计算、瞄准决策         |
| L5   | 控制 (weapon)     | 武器控制系统接口           |

## 🔧 配置管理

Talos 使用 TOML 作为统一配置文件格式。  
所有关键配置均以层级方式组织，示例路径如下：

```
at_vision.toml          # 程序入口配置
config/vision_base.toml  # 视觉基础配置
config/vision/<name>.toml  # 视觉算法特定配置
config/robot/<name>.toml    # 硬件（机器人）配置
```

## 📅 开发状态

| 模块     | 状态 | 说明               |
| -------- | ---- | ------------------ |
| 调度系统 | ✅   | 高性能多线程调度器 |
| 配置管理 | ✅   | TOML 解析 + 合并   |
| 相机驱动 | ✅   | HIK SDK            |
| 检测后端 | ✅   | NN、传统、TUP      |
| 追踪框架 | 📋   | 接口就绪           |
| 武器系统 | 📋   | 接口就绪           |
| Foxglove | ✅   | 实时可视化         |
| 串口通信 | ✅   | gimbal 控制        |

---

<div align="center">

**用 🪓 和 💀 构建**

_没有 ROS 的负担，只有纯粹的性能_

</div>
