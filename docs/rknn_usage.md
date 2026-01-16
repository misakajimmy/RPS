# RKNN 手势识别使用指南

## 概述

本项目支持使用 RKNN 模型在 RK3588 开发板上进行 NPU 加速的手势识别。RKNN 识别器与 YOLO 识别器使用相同的接口，可以通过配置文件无缝切换。

## 前置要求

### Windows 上测试（模型转换和模拟推理）

1. **RKNN Toolkit 2**：用于模型转换和模拟推理
   - 可以从 Rockchip 官方获取
   - 支持在 Windows 上进行模型转换和模拟推理
   - 注意：模拟推理速度较慢，主要用于验证模型转换是否正确

### RK3588 设备上使用（实际 NPU 加速）

1. **RK3588 开发板**：需要支持 RKNN Runtime
2. **RKNN 模型文件**：已转换好的 `.rknn` 模型文件
3. **rknn-toolkit-lite2**：在 RK3588 上安装运行时库
   ```bash
   pip install rknn-toolkit-lite2
   # 或
   pip install rknnlite
   ```

**重要**：在 RK3588 设备上运行时，代码会自动检测并使用 `rknnlite`（NPU 运行时），而不是 `rknn-toolkit2`（用于模型转换）。

## 配置使用

### 配置文件设置

在 `config/config.yaml` 中配置使用 RKNN 识别器：

```yaml
game:
  gesture_recognition:
    # 使用 RKNN 识别器
    type: "rknn"  # 识别器类型
    model_path: "models/best.rknn"  # RKNN 模型文件路径
    input_size: [640, 640]  # 模型输入尺寸 [width, height]
    confidence_threshold: 0.7  # 识别置信度阈值
    min_detection_confidence: 0.5  # 检测最小置信度
```

### 切换到 YOLO 识别器

如果需要使用 YOLO 识别器（例如在 PC 上开发），只需修改配置：

```yaml
game:
  gesture_recognition:
    # 使用 YOLO 识别器
    type: "yolo"
    use_huggingface_model: true
    model_path: "lewiswatson/yolov8x-tuned-hand-gestures"
    confidence_threshold: 0.7
    min_detection_confidence: 0.5
    device: null  # 自动检测设备
```

## 代码使用

### 使用工厂模式（推荐）

```python
from src.game.gesture_recognition import RecognizerFactory

# 从配置创建识别器
config = {
    'type': 'rknn',
    'model_path': 'models/best.rknn',
    'input_size': (640, 640),
    'confidence_threshold': 0.7,
    'min_detection_confidence': 0.5
}

recognizer = RecognizerFactory.create_from_config(config)

# 使用识别器（接口与 YOLO 识别器相同）
gesture, confidence, probabilities = recognizer.recognize(image)
```

### 直接创建 RKNN 识别器

```python
from src.game.gesture_recognition import RKNNRecognizer

# 创建识别器
recognizer = RKNNRecognizer(
    model_path='models/best.rknn',
    confidence_threshold=0.7,
    min_detection_confidence=0.5,
    input_size=(640, 640)
)

# 初始化（自动调用）
recognizer.initialize()

# 识别手势
gesture, confidence, probabilities = recognizer.recognize(image)

# 释放资源
recognizer.release()
```

### 使用上下文管理器

```python
from src.game.gesture_recognition import RKNNRecognizer

with RKNNRecognizer('models/best.rknn') as recognizer:
    gesture, confidence, probabilities = recognizer.recognize(image)
    # 自动释放资源
```

## API 接口

RKNN 识别器与 YOLO 识别器使用相同的接口：

### `recognize(image: np.ndarray) -> Tuple[Gesture, float, Dict[str, float]]`

识别图像中的手势。

**参数**：
- `image`: 输入图像（BGR 格式，numpy array）

**返回**：
- `gesture`: 识别的手势（`Gesture.ROCK`, `Gesture.PAPER`, `Gesture.SCISSORS`, 或 `Gesture.UNKNOWN`）
- `confidence`: 置信度（0.0-1.0）
- `probabilities`: 所有类别的概率字典 `{'rock': float, 'paper': float, 'scissors': float}`

### `initialize() -> bool`

初始化 RKNN 运行时。通常在创建识别器后自动调用。

### `release()`

释放 RKNN 运行时资源。

## 性能优化

1. **模型量化**：使用 INT8 量化可以显著提升推理速度
2. **输入尺寸**：较小的输入尺寸可以提升速度，但可能降低精度
3. **批量推理**：如果可能，使用批量推理以提高吞吐量

## 常见问题

### Q: RKNN 识别器初始化失败

A: 检查：
1. RKNN 模型文件是否存在且有效
2. 在 RK3588 上是否安装了 `rknn-toolkit-lite2`：
   ```bash
   pip install rknn-toolkit-lite2
   ```
3. 模型是否在正确的平台上转换（`--target-platform rk3588`）
4. 查看日志输出，确认使用的是 `rknnlite` 还是 `rknn-toolkit2`

### Q: 提示 "rknn-toolkit2 未安装" 但已经安装了

A: 在 RK3588 设备上，应该安装 `rknn-toolkit-lite2`（运行时库），而不是 `rknn-toolkit2`（转换工具）：
```bash
# 在 RK3588 上
pip install rknn-toolkit-lite2

# 在开发机上（用于模型转换）
# 从 Rockchip 官方获取 rknn-toolkit2
```

### Q: 识别结果与 YOLO 不一致

A: 可能原因：
1. 模型转换时的量化可能影响精度
2. 输入预处理可能不一致
3. 建议使用真实的验证集作为量化校正数据集

### Q: 如何切换回 YOLO 识别器

A: 只需修改配置文件中的 `type` 字段为 `"yolo"`，并配置相应的 YOLO 参数。

## 相关文档

- [RKNN 模型转换指南](rknn_conversion.md)
- [手势识别使用说明](game_logic_usage.md)
