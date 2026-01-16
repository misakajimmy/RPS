# RKNN 模型转换指南

## 概述

本指南介绍如何将 YOLOv8 模型转换为 RKNN 格式，以便在 RK3588 开发板上使用 NPU 加速推理。

## 前置要求

### 1. 安装依赖

```bash
# 基础依赖
pip install ultralytics>=8.0.0

# ONNX 相关（可选，用于验证）
pip install onnx onnxsim

# RKNN Toolkit 2（需要从 Rockchip 官方获取）
# 注意：RKNN Toolkit 2 可能不在 PyPI 上，需要从官方下载
# 下载地址：https://github.com/rockchip-linux/rknn-toolkit2
pip install rknn-toolkit2
```

### 2. 准备模型文件

确保您有 YOLOv8 模型文件（`.pt` 格式），例如：
- `models/best.pt`
- `models/yolov8x-tuned-hand-gestures.pt`

## 转换方法

### 方法一：一键转换（推荐）

使用 `convert_to_rknn.py` 脚本，自动完成 PyTorch -> ONNX -> RKNN 的两步转换：

```bash
# 基本转换
python scripts/convert_to_rknn.py models/best.pt

# 指定输出目录和输入尺寸
python scripts/convert_to_rknn.py models/best.pt -o models/rknn --input-size 640 640

# 禁用量化（使用 FP16，精度更高但速度较慢）
python scripts/convert_to_rknn.py models/best.pt --no-quantization

# 不保留中间 ONNX 文件
python scripts/convert_to_rknn.py models/best.pt --no-keep-onnx
```

### 方法二：分步转换

#### 步骤 1: 导出 ONNX

```bash
# 基本导出
python scripts/export_onnx.py models/best.pt

# 指定输出路径和输入尺寸
python scripts/export_onnx.py models/best.pt -o models/best.onnx --input-size 640 640

# 使用 opset 11（某些 RKNN 版本可能需要）
python scripts/export_onnx.py models/best.pt --opset 11
```

#### 步骤 2: 转换为 RKNN

```bash
# 基本转换
python scripts/convert_rknn.py models/best.onnx

# 指定输出路径和平台
python scripts/convert_rknn.py models/best.onnx -o models/best.rknn --target-platform rk3588

# 禁用量化
python scripts/convert_rknn.py models/best.onnx --no-quantization

# 使用自定义量化数据集（推荐）
python scripts/convert_rknn.py models/best.onnx --quantization-dataset path/to/dataset
```

## 参数说明

### 输入尺寸

YOLOv8 默认使用 640x640 输入。如果您的模型使用其他尺寸，需要指定：

```bash
--input-size 416 416  # 宽度 高度
```

### ONNX Opset 版本

RKNN Toolkit 2 推荐使用 opset 11 或 12：

```bash
--opset 11  # 或 12
```

### 量化

量化可以显著提升推理速度，但可能略微降低精度：

- **启用量化（默认）**：使用 INT8，速度快，精度略低
- **禁用量化**：使用 FP16，速度较慢，精度更高

```bash
--no-quantization  # 禁用量化
```

### 量化校正数据集

量化校正数据集用于校准 INT8 量化。建议使用真实的验证集图像：

```bash
# 使用自定义数据集（推荐）
python scripts/convert_rknn.py models/best.onnx \
    --quantization-dataset path/to/validation/images

# 或使用自动生成的随机数据（不推荐，仅用于测试）
python scripts/convert_rknn.py models/best.onnx \
    --quantization-samples 200
```

## 常见问题

### Q: 转换失败，提示 "ONNX opset 版本不支持"

A: 尝试使用 opset 11：

```bash
python scripts/export_onnx.py models/best.pt --opset 11
```

### Q: 转换后的模型精度下降

A: 尝试禁用量化：

```bash
python scripts/convert_to_rknn.py models/best.pt --no-quantization
```

或使用真实的验证集作为量化校正数据集。

### Q: 转换速度很慢

A: 这是正常的，特别是量化过程。可以：
1. 减少量化校正样本数量（不推荐）
2. 使用更快的硬件进行转换
3. 禁用量化（但会降低推理速度）

### Q: 在 RK3588 上加载 RKNN 模型失败

A: 检查：
1. RKNN 模型是否在正确的平台上转换（`--target-platform rk3588`）
2. RKNN Runtime 版本是否匹配
3. 输入尺寸是否与转换时一致

### Q: 如何验证转换后的模型

A: 可以在转换后使用 RKNN Toolkit 2 的推理功能验证：

```python
from rknn.api import RKNN

rknn = RKNN()
ret = rknn.load_rknn('models/best.rknn')
ret = rknn.init_runtime()
# ... 进行推理测试
```

## 输出文件

转换完成后，您将得到：

- `models/best.onnx` - ONNX 格式模型（如果保留）
- `models/best.rknn` - RKNN 格式模型（用于 RK3588）

## 在 RK3588 上使用

转换完成后，可以在 RK3588 设备上使用 RKNN 模型进行推理。需要：

1. 安装 RKNN Runtime（通常在 RK3588 系统镜像中已包含）
2. 使用 RKNN Runtime API 加载和运行模型

示例代码（需要根据实际项目调整）：

```python
from rknn.api import RKNN
import numpy as np

# 初始化 RKNN
rknn = RKNN()

# 加载模型
ret = rknn.load_rknn('models/best.rknn')
if ret != 0:
    print('Load RKNN model failed!')
    exit(ret)

# 初始化运行时
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime failed!')
    exit(ret)

# 准备输入数据
input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

# 推理
outputs = rknn.inference(inputs=[input_data])

# 处理输出
# ...

# 释放资源
rknn.release()
```

## 性能优化建议

1. **使用量化**：INT8 量化可以显著提升速度
2. **优化输入尺寸**：较小的输入尺寸可以提升速度，但可能降低精度
3. **批量推理**：如果可能，使用批量推理以提高吞吐量
4. **使用真实校正数据集**：使用验证集图像作为量化校正数据集可以提高量化后的精度

## 相关文档

- [RKNN Toolkit 2 官方文档](https://github.com/rockchip-linux/rknn-toolkit2)
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [ONNX 官方文档](https://onnx.ai/)

## 故障排除

如果遇到问题，请检查：

1. **依赖版本**：确保 ultralytics 和 rknn-toolkit2 版本兼容
2. **模型格式**：确保输入的 .pt 文件是有效的 YOLOv8 模型
3. **平台匹配**：确保转换时指定的平台与目标设备匹配
4. **日志信息**：查看详细的日志输出以定位问题
