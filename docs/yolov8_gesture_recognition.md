# YOLOv8 手势识别使用指南

## 概述

项目支持两种 YOLOv8 模型进行手势识别：

1. **HuggingFace 手势识别模型**（推荐）：`lewiswatson/yolov8x-tuned-hand-gestures`
   - 专门训练用于手势检测和分类
   - 直接输出手势类别（石头/剪刀/布）
   - 准确度更高，使用更简单

2. **YOLOv8-Pose 模型**：基于人体关键点检测
   - 检测人体关键点，然后分析手部区域
   - 需要额外的关键点检测和手指分析
   - 作为备选方案

## 优势

1. **跨平台支持**：YOLOv8 支持 ARM64 Linux（RK3588），无需从源码编译
2. **自动下载模型**：首次运行时会自动下载预训练模型
3. **多种模型大小**：可选择不同大小的模型（nano/small/medium/large/xlarge）平衡速度和准确度
4. **易于部署**：只需安装 `ultralytics` 包即可

## 安装

### 基础安装（支持 YOLOv8-Pose）

```bash
# 使用 uv
uv sync

# 或使用 pip
pip install ultralytics>=8.0.0
```

### 安装 HuggingFace 模型支持（推荐）

```bash
# 使用 uv
uv sync --extra huggingface

# 或使用 pip
pip install ultralyticsplus>=0.0.0
```

**注意**：即使不安装 `ultralyticsplus`，标准 `ultralytics` 也支持从 HuggingFace 加载模型，但 `ultralyticsplus` 提供更好的集成。

## 使用方法

### 使用 HuggingFace 手势识别模型（推荐）

```python
from src.game.gesture_recognition import GestureRecognizer
import cv2

# 创建识别器（使用 HuggingFace 手势识别模型）
recognizer = GestureRecognizer(
    min_detection_confidence=0.5,  # 手势检测最小置信度
    confidence_threshold=0.7,       # 手势识别置信度阈值
    use_huggingface_model=True,     # 使用 HuggingFace 模型
    model_path="lewiswatson/yolov8x-tuned-hand-gestures"  # HuggingFace 模型ID
)
```

### 使用 YOLOv8-Pose 模型（备选）

```python
# 创建识别器（使用 YOLOv8-Pose 模型）
recognizer = GestureRecognizer(
    min_detection_confidence=0.5,  # 人体检测最小置信度
    confidence_threshold=0.7,       # 手势识别置信度阈值
    use_huggingface_model=False,    # 不使用 HuggingFace 模型
    model_size='n'                 # 模型大小：'n', 's', 'm', 'l', 'x'
)

# 从摄像头捕获图像
camera = cv2.VideoCapture(0)
ret, frame = camera.read()

# 识别手势
gesture, confidence, probabilities = recognizer.recognize(frame)

print(f"识别结果: {gesture.value}, 置信度: {confidence:.3f}")
print(f"概率分布: {probabilities}")
```

### 使用自定义模型

```python
# 使用自定义训练的 YOLOv8-Pose 模型
recognizer = GestureRecognizer(
    model_path="models/custom_yolov8_pose.pt",
    confidence_threshold=0.7
)
```

### 配置参数

在 `config/config.yaml` 中配置：

```yaml
game:
  gesture_recognition:
    confidence_threshold: 0.7      # 手势识别置信度阈值
    # 使用 HuggingFace 手势识别模型（推荐）
    use_huggingface_model: true    # 是否使用 HuggingFace 模型
    model_path: "lewiswatson/yolov8x-tuned-hand-gestures"  # HuggingFace 模型ID
    min_detection_confidence: 0.5  # 手势检测最小置信度
    
    # 或使用 YOLOv8-Pose 模型
    # use_huggingface_model: false
    # model_path: null             # YOLOv8-Pose 模型路径，null 则使用默认模型
    # model_size: "n"              # 模型大小：'n', 's', 'm', 'l', 'x'
```

## 工作原理

1. **人体关键点检测**：使用 YOLOv8-Pose 检测人体关键点（17个点，COCO格式）
2. **手部区域提取**：从关键点中提取手腕位置，估算手部区域
3. **手指检测**：在手部区域内使用 OpenCV 轮廓分析检测手指数量
4. **手势判断**：
   - 0 个手指 → 石头（Rock）
   - 2 个手指 → 剪刀（Scissors）
   - 4+ 个手指 → 布（Paper）

## 模型大小选择

| 模型大小 | 参数量 | 速度 | 准确度 | 推荐场景 |
|---------|--------|------|--------|---------|
| nano (n) | 最小 | 最快 | 较低 | ARM64 设备，实时性要求高 |
| small (s) | 小 | 快 | 中等 | 平衡速度和准确度 |
| medium (m) | 中 | 中等 | 较高 | 准确度要求较高 |
| large (l) | 大 | 慢 | 高 | 准确度优先 |
| xlarge (x) | 最大 | 最慢 | 最高 | 离线处理，准确度优先 |

**推荐**：在 RK3588 上使用 `nano` 或 `small` 模型，以获得较好的实时性能。

## 性能优化

### 使用 GPU（如果可用）

修改 `gesture_recognizer.py` 中的 `device` 参数：

```python
results = self.model.predict(
    image,
    conf=self.min_detection_confidence,
    verbose=False,
    device='cuda'  # 改为 'cuda' 如果有 GPU
)
```

### 调整检测置信度

降低 `min_detection_confidence` 可以提高检测率，但可能增加误检：

```python
recognizer = GestureRecognizer(
    min_detection_confidence=0.3,  # 降低阈值，提高检测率
    confidence_threshold=0.7
)
```

## 故障排除

### 问题：模型下载失败

**解决方案**：
1. 检查网络连接
2. 手动下载模型文件到 `~/.ultralytics/weights/` 目录
3. 或使用 `model_path` 参数指定本地模型文件

### 问题：识别准确度低

**可能原因**：
1. 光照条件不佳
2. 背景复杂
3. 手部区域太小
4. 模型大小不合适

**解决方案**：
1. 改善光照条件
2. 使用纯色背景
3. 确保手部在画面中足够大
4. 尝试更大的模型（如 's' 或 'm'）

### 问题：检测不到人体

**解决方案**：
1. 降低 `min_detection_confidence` 阈值
2. 确保整个人体在画面中
3. 改善光照条件

## 与 MediaPipe 对比

| 特性 | YOLOv8-Pose | MediaPipe |
|------|-------------|-----------|
| ARM64 支持 | ✅ 支持 | ❌ 不支持（需从源码编译）|
| 模型大小 | 可调节（n/s/m/l/x） | 固定 |
| 准确度 | 中等-高 | 高 |
| 速度 | 中等-快 | 快 |
| 依赖 | ultralytics | mediapipe |
| 自动下载 | ✅ | ❌ |

## 未来改进

1. **手部关键点细化**：使用专门的手部关键点检测模型（如 MediaPipe Hands）在手部区域内进行更精确的检测
2. **自定义训练**：使用自己的数据集训练专门的手势识别模型
3. **多手检测**：支持同时检测多只手
