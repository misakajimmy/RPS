# USB摄像头模块使用指南

## 概述

USB摄像头模块提供了完整的摄像头驱动封装和图像预处理功能，支持通过工厂模式创建和管理摄像头实例。

## 快速开始

### 1. 直接使用USBCamera类

```python
from src.hardware.implementations.camera import USBCamera

# 创建摄像头实例
camera = USBCamera(device_id=0, width=640, height=480, fps=30)

# 连接摄像头
if camera.connect():
    # 捕获一帧图像
    frame = camera.capture_frame()
    if frame is not None:
        print(f"图像尺寸: {frame.shape}")
    
    # 断开连接
    camera.disconnect()
```

### 2. 使用上下文管理器

```python
from src.hardware.implementations.camera import USBCamera

# 使用上下文管理器自动管理连接
with USBCamera(device_id=0) as camera:
    if camera.is_connected():
        frame = camera.capture_frame()
        # 处理图像...
# 自动断开连接
```

### 3. 使用工厂类创建

```python
from src.hardware.factory.hardware_factory import HardwareFactory

# 配置参数
config = {
    'device_id': 0,
    'width': 640,
    'height': 480,
    'fps': 30
}

# 通过工厂类创建摄像头
camera = HardwareFactory.create_camera('usb_camera', config)

if camera:
    camera.connect()
    frame = camera.capture_frame()
    camera.disconnect()
```

### 4. 使用配置管理器

```python
from src.hardware.config_manager import HardwareConfigManager

# 加载配置文件并创建摄像头
config_manager = HardwareConfigManager('config/config.yaml')
config_manager.load_config()
camera = config_manager.create_camera()

if camera:
    camera.connect()
    frame = camera.capture_frame()
    camera.disconnect()
```

## API参考

### USBCamera类

#### 初始化参数

- `device_id` (int): 摄像头设备ID，默认0
- `width` (int): 图像宽度，默认640
- `height` (int): 图像高度，默认480
- `fps` (int): 帧率，默认30
- `backend` (int, 可选): OpenCV后端，如`cv2.CAP_V4L2`

#### 主要方法

- `connect() -> bool`: 连接摄像头
- `disconnect() -> bool`: 断开摄像头连接
- `is_connected() -> bool`: 检查连接状态
- `capture_frame() -> np.ndarray`: 捕获一帧BGR格式图像
- `capture_frame_rgb() -> np.ndarray`: 捕获一帧RGB格式图像
- `capture_frame_gray() -> np.ndarray`: 捕获一帧灰度图像
- `get_resolution() -> tuple`: 获取当前分辨率
- `set_resolution(width, height) -> bool`: 设置分辨率
- `get_status() -> dict`: 获取摄像头状态信息

### ImageProcessor类

图像预处理工具类，提供以下静态方法：

- `resize()`: 调整图像大小
- `crop()`: 裁剪图像
- `crop_center()`: 从中心裁剪
- `normalize()`: 归一化图像
- `enhance_contrast()`: 增强对比度
- `apply_gaussian_blur()`: 高斯模糊
- `apply_bilateral_filter()`: 双边滤波
- `detect_edges()`: 边缘检测
- `extract_roi()`: 提取感兴趣区域
- `flip()`: 翻转图像
- `rotate()`: 旋转图像
- `adjust_brightness()`: 调整亮度
- `preprocess_for_gesture_recognition()`: 手势识别预处理

## 图像格式说明

- **BGR格式**: OpenCV默认格式，用于大多数图像处理操作
- **RGB格式**: 用于显示和某些深度学习模型
- **灰度格式**: 单通道图像，用于边缘检测等操作

## 配置示例

在`config/config.yaml`中配置摄像头：

```yaml
camera:
  type: "usb_camera"
  device_id: 0
  width: 640
  height: 480
  fps: 30
```

## 注意事项

1. 确保USB摄像头已正确连接
2. 在Linux系统上，可能需要适当的权限访问摄像头设备
3. 某些摄像头可能不支持所有分辨率设置，实际分辨率可能与请求的不同
4. 使用完毕后记得调用`disconnect()`释放资源
5. 推荐使用上下文管理器自动管理资源

## 故障排除

### 问题：无法打开摄像头

- 检查设备ID是否正确
- 检查摄像头是否被其他程序占用
- 在Linux上检查设备权限：`ls -l /dev/video*`

### 问题：分辨率设置不生效

- 某些摄像头不支持动态调整分辨率
- 检查摄像头支持的分辨率列表
- 实际分辨率可能与请求的不同，这是正常现象

### 问题：图像捕获失败

- 检查摄像头连接状态
- 确保摄像头没有被其他程序占用
- 尝试降低帧率或分辨率
