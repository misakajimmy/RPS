# ARM64 Linux (RK3588) 安装指南

## MediaPipe 在 ARM64 上的限制

MediaPipe 官方**不提供 ARM64 Linux 的预编译包**，只支持：
- `manylinux_2_28_x86_64` (x86_64 Linux)
- `macosx_11_0_arm64` (macOS ARM64)
- `win_amd64` (Windows x64)

因此，在 RK3588 等 ARM64 Linux 平台上，项目已实现**自动降级方案**。

## 安装方式

### 方式一：不安装 MediaPipe（推荐，使用 OpenCV 降级方案）

```bash
# 使用 uv
uv sync

# 或使用 pip
pip install -r requirements.txt
```

项目会自动检测到 MediaPipe 不可用，并使用基于 OpenCV 的降级方案。

### 方式二：尝试从源码编译 MediaPipe（高级用户）

如果你需要 MediaPipe 的更高准确度，可以尝试从源码编译：

```bash
# 安装构建依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev

# 克隆 MediaPipe 源码
git clone https://github.com/google/mediapipe.git
cd mediapipe

# 按照 MediaPipe 官方文档编译（需要 Bazel 等工具）
# 注意：这个过程可能很复杂且耗时
```

**注意**：从源码编译 MediaPipe 需要大量依赖和编译时间，不推荐普通用户使用。

## OpenCV 降级方案说明

当 MediaPipe 不可用时，项目会自动使用基于 OpenCV 的简单手势识别：

- **原理**：基于 HSV 颜色空间的肤色检测 + 轮廓分析 + 凸包缺陷检测
- **准确度**：**较低**（约 60-70%），主要用于开发和测试
- **优势**：
  - 无需额外依赖
  - 在 ARM64 上可直接运行
  - 适合验证系统其他部分（机械臂、语音模块等）

### 降级方案的识别逻辑

- **石头（Rock）**：检测到 0 个手指（所有手指弯曲）
- **布（Paper）**：检测到 4+ 个手指（所有手指伸直）
- **剪刀（Scissors）**：检测到 2 个手指（食指和中指伸直）

## 使用建议

1. **开发阶段**：在 ARM64 上使用 OpenCV 降级方案，验证硬件模块（机械臂、语音）是否正常工作
2. **测试阶段**：在 x86_64/Windows/macOS 上安装 MediaPipe，进行准确的手势识别测试
3. **部署阶段**：
   - 如果对准确度要求不高，可以继续使用 OpenCV 降级方案
   - 如果需要高准确度，考虑：
     - 使用 x86_64 平台
     - 或从源码编译 MediaPipe（需要技术支持）

## 检查当前使用的识别方案

运行程序时，查看日志输出：

```
# MediaPipe 可用时
INFO - 使用MediaPipe Tasks API (新版本)
INFO - MediaPipe Tasks API手势识别器初始化完成

# MediaPipe 不可用时（ARM64）
WARNING - MediaPipe未安装或不可用（ARM64 Linux可能不支持）: ...
INFO - 将使用基于OpenCV的简单手势识别降级方案
WARNING - MediaPipe不可用，使用基于OpenCV的简单手势识别降级方案
```

也可以通过 `GestureRecognizer.get_model_info()` 查看：

```python
recognizer = GestureRecognizer()
info = recognizer.get_model_info()
print(info)
# MediaPipe: {'model_type': 'MediaPipe Tasks API - HandLandmarker', ...}
# OpenCV: {'model_type': 'OpenCV Fallback (Contour-based)', ...}
```

## 可选依赖安装

如果你想在支持 MediaPipe 的平台上安装它：

```bash
# 使用 uv
uv sync --extra mediapipe

# 或使用 pip
pip install mediapipe>=0.10.0
```

## 故障排除

### 问题：`uv sync` 报错 MediaPipe 无法安装

**解决方案**：这是正常的，MediaPipe 在 ARM64 上确实无法安装。项目已自动处理，直接运行程序即可。

### 问题：识别准确度很低

**可能原因**：
1. 使用了 OpenCV 降级方案（ARM64 上正常）
2. 光照条件不佳
3. 背景复杂

**解决方案**：
- 改善光照条件
- 使用纯色背景
- 在 x86_64 平台上测试 MediaPipe 版本

### 问题：程序启动时报错

检查是否安装了基础依赖：

```bash
pip install opencv-python numpy Pillow pyyaml
```
