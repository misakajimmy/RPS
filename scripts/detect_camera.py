#!/usr/bin/env python3
"""
摄像头设备检测脚本
Camera Device Detection Script

用于检测系统中可用的摄像头设备及其 device_id
"""
import cv2
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def detect_cameras(max_devices=10):
    """检测可用的摄像头设备"""
    available_cameras = []
    
    print("=" * 60)
    print("摄像头设备检测")
    print("=" * 60)
    print(f"正在检测 device_id 0 到 {max_devices-1}...")
    print()
    
    for device_id in range(max_devices):
        print(f"检测 device_id={device_id} (/dev/video{device_id})...", end=" ")
        
        try:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                # 尝试读取一帧来确认设备可用
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 获取设备信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    backend = cap.getBackendName()
                    
                    available_cameras.append({
                        'device_id': device_id,
                        'device_path': f'/dev/video{device_id}',
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend
                    })
                    
                    print(f"✓ 可用")
                    print(f"  设备路径: /dev/video{device_id}")
                    print(f"  分辨率: {width}x{height}")
                    print(f"  帧率: {fps} FPS")
                    print(f"  后端: {backend}")
                    print()
                else:
                    print("✗ 无法读取图像")
            else:
                print("✗ 无法打开")
            cap.release()
        except Exception as e:
            print(f"✗ 异常: {e}")
    
    print("=" * 60)
    
    if not available_cameras:
        print("✗ 未找到可用的摄像头设备")
        print()
        print("提示:")
        print("1. 检查摄像头是否已连接")
        print("2. 检查是否有权限访问 /dev/video*")
        print("3. 运行以下命令查看设备文件:")
        print("   ls /dev/video*")
        print("4. 运行以下命令查看详细信息:")
        print("   v4l2-ctl --list-devices")
        print("5. 如果权限不足，尝试:")
        print("   sudo usermod -a -G video $USER")
        print("   (需要重新登录)")
        return None
    
    print(f"✓ 共找到 {len(available_cameras)} 个可用摄像头")
    print()
    print("推荐配置 (config/config.yaml):")
    print("camera:")
    print(f"  type: \"usb_camera\"")
    print(f"  device_id: {available_cameras[0]['device_id']}  # 使用第一个检测到的摄像头")
    
    if len(available_cameras) > 1:
        print()
        print("  # 其他可用摄像头:")
        for cam in available_cameras[1:]:
            print(f"  # device_id: {cam['device_id']}  # {cam['device_path']} ({cam['width']}x{cam['height']})")
    
    print()
    print("=" * 60)
    
    return available_cameras


if __name__ == "__main__":
    try:
        cameras = detect_cameras()
        sys.exit(0 if cameras else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 检测过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
