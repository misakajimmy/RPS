# Linux 摄像头设备检测指南

## 方法一：使用 ls 命令（最简单）

```bash
# 列出所有视频设备
ls /dev/video*

# 通常输出类似：
# /dev/video0
# /dev/video1
# /dev/video2
```

**device_id 对应关系**：
- `/dev/video0` → `device_id = 0`
- `/dev/video1` → `device_id = 1`
- `/dev/video2` → `device_id = 2`
- 以此类推...

## 方法二：使用 v4l2-ctl（推荐，信息最详细）

```bash
# 安装 v4l-utils（如果未安装）
sudo apt-get install v4l-utils  # Debian/Ubuntu
# 或
sudo yum install v4l-utils      # CentOS/RHEL

# 列出所有视频设备及其详细信息
v4l2-ctl --list-devices

# 输出示例：
# USB2.0 Camera (usb-0000:01:00.0-1.4):
# 	/dev/video0
# 	/dev/video1
# 
# Integrated Camera (usb-0000:00:14.0-13):
# 	/dev/video2
```

**查看特定设备的详细信息**：
```bash
# 查看 /dev/video0 的详细信息
v4l2-ctl --device=/dev/video0 --all

# 查看支持的格式和分辨率
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

## 方法三：使用 lsusb（查看 USB 摄像头）

```bash
# 列出所有 USB 设备
lsusb

# 输出示例：
# Bus 001 Device 003: ID 0c45:6713 Microdia USB2.0 Camera
# Bus 001 Device 004: ID 13d3:56a2 IMC Networks USB2.0 HD UVC WebCam
```

## 方法四：使用 dmesg 查看系统日志

```bash
# 查看最近的摄像头相关日志
dmesg | grep -i video
dmesg | grep -i camera
dmesg | grep -i usb | grep -i camera

# 查看所有 USB 设备连接日志
dmesg | tail -50
```

## 方法五：使用 Python 脚本自动检测（推荐用于项目）

创建一个测试脚本来检测可用的摄像头：

```python
import cv2
import sys

def detect_cameras(max_devices=10):
    """检测可用的摄像头设备"""
    available_cameras = []
    
    print("正在检测摄像头设备...")
    print("=" * 50)
    
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            # 尝试读取一帧来确认设备可用
            ret, frame = cap.read()
            if ret:
                # 获取设备信息
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                available_cameras.append({
                    'device_id': device_id,
                    'device_path': f'/dev/video{device_id}',
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f"✓ 找到摄像头:")
                print(f"  Device ID: {device_id}")
                print(f"  设备路径: /dev/video{device_id}")
                print(f"  分辨率: {width}x{height}")
                print(f"  帧率: {fps} FPS")
                print()
            cap.release()
    
    if not available_cameras:
        print("✗ 未找到可用的摄像头设备")
        print("\n提示:")
        print("1. 检查摄像头是否已连接")
        print("2. 检查是否有权限访问 /dev/video*")
        print("3. 运行: ls /dev/video* 查看设备文件")
        return None
    
    print(f"共找到 {len(available_cameras)} 个可用摄像头")
    return available_cameras

if __name__ == "__main__":
    cameras = detect_cameras()
    if cameras:
        print("\n推荐配置 (config.yaml):")
        print("camera:")
        print(f"  device_id: {cameras[0]['device_id']}  # 使用第一个检测到的摄像头")
        if len(cameras) > 1:
            print(f"  # 或使用其他摄像头:")
            for cam in cameras[1:]:
                print(f"  # device_id: {cam['device_id']}  # {cam['device_path']}")
```

**使用方法**：
```bash
# 在项目根目录运行
python -c "
import cv2
import sys

def detect_cameras(max_devices=10):
    available_cameras = []
    print('正在检测摄像头设备...')
    print('=' * 50)
    
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                available_cameras.append({
                    'device_id': device_id,
                    'device_path': f'/dev/video{device_id}',
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f'✓ 找到摄像头: Device ID={device_id}, 路径=/dev/video{device_id}, 分辨率={width}x{height}, 帧率={fps} FPS')
            cap.release()
    
    if not available_cameras:
        print('✗ 未找到可用的摄像头设备')
        return None
    
    print(f'共找到 {len(available_cameras)} 个可用摄像头')
    return available_cameras

detect_cameras()
"
```

## 方法六：使用项目自带的测试脚本

项目中的 `tests/test_usb_camera.py` 也可以用来测试摄像头：

```bash
# 运行摄像头测试（会尝试打开 device_id=0）
python tests/test_usb_camera.py

# 或指定 device_id
python -c "from src.hardware.implementations.camera.usb_camera import USBCamera; cam = USBCamera(device_id=0); print('摄像头打开成功' if cam.cap.isOpened() else '摄像头打开失败')"
```

## 常见问题

### Q: 为什么有多个 /dev/video* 设备？

A: 现代摄像头通常会在 `/dev/` 下创建多个设备节点：
- `/dev/video0` - 通常用于视频捕获
- `/dev/video1` - 可能用于元数据或控制接口
- `/dev/video2` - 可能是另一个摄像头或虚拟设备

**建议**：从 `device_id=0` 开始测试，如果无法打开，尝试 `device_id=1`、`device_id=2` 等。

### Q: 提示 "Permission denied" 或无法打开摄像头？

A: 检查权限：
```bash
# 查看设备权限
ls -l /dev/video*

# 将用户添加到 video 组（需要重新登录）
sudo usermod -a -G video $USER

# 或临时修改权限（不推荐，重启后失效）
sudo chmod 666 /dev/video0
```

### Q: 在 RK3588 开发板上如何查看？

A: RK3588 通常使用 MIPI CSI 摄像头，设备路径可能不同：
```bash
# 查看所有视频设备
ls -la /dev/video*

# 使用 v4l2-ctl 查看
v4l2-ctl --list-devices

# 查看 dmesg 日志
dmesg | grep -i video
```

### Q: 如何确认某个 device_id 是否可用？

A: 使用 OpenCV 快速测试：
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('可用' if cap.isOpened() else '不可用'); cap.release()"
```

## 配置示例

检测到摄像头后，在 `config/config.yaml` 中配置：

```yaml
camera:
  type: "usb_camera"
  device_id: 0  # 根据检测结果修改
  width: 640
  height: 480
  fps: 30
```

## 相关文档

- [摄像头使用说明](camera_usage.md)
- [系统集成指南](system_integration.md)
