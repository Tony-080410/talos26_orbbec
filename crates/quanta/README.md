<div align="center">

# Quanta

**任务图传**

_混沌弥天，一隙渡光。_

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C.svg?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-8.x-007808.svg?style=for-the-badge&logo=ffmpeg)](https://ffmpeg.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)

</div>

---

## Quanta

Quanta 是 Talos 的任务图传。

它以 ~80 kbps 的带宽运行，在不可靠信道上把摄像头画面送到选手端，完成吊射任务。

## 设计

| 组件 | 职责 |
| --- | --- |
| **StreamSender** | 采集、编码、分片、送出 |
| **StreamReceiver** | 收包、重组、解码、替补、恢复 |

```text
Camera ──▶ Sender ──[不可靠信道]──▶ Receiver ──▶ 瞄准管线
```

Sender 不等待完美编码——帧在 hard deadline 前必须送出，
超时的帧直接丢弃，为新帧让路。

Receiver 不等完美信号——分片丢了就用上一帧运动补偿替补，
P 帧坏了就请求 IDR 刷新，画面冻结了就触发链路恢复。

## 工具

| 命令 | 用途 |
| --- | --- |
| `quanta-transcode` | 文件到文件转码，验证编码管线 |
| `quanta-stream-test` | 测试图案编码输出 |
| `quanta-stream-loopback` | 模拟丢包，测量端到端延迟与恢复时间 |
| `quanta-stream-demo` | 源码/解码画面对比窗口 |

## 依赖

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| [FFmpeg](https://ffmpeg.org) | 8.x | HEVC 编解码、滤镜链 |
| [OpenCV](https://opencv.org) | 4.x | 演示窗口（仅 demo） |

## 状态

Quanta 随 Talos 迭代。当前处于验证阶段。

---

<div align="center">

**用 🎯 和 📡 构建**

</div>
