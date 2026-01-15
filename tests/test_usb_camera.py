"""
USB摄像头测试示例（带UI显示）
USB Camera Test Example with UI Display
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hardware.implementations.camera import USBCamera, ImageProcessor
from src.hardware.factory.hardware_factory import HardwareFactory
from src.hardware.config_manager import HardwareConfigManager


class CameraTestUI:
    """摄像头测试UI类"""
    
    def __init__(self):
        self.camera: Optional[USBCamera] = None
        self.window_name = "USB Camera Test"
        self.running = False
        self.current_test = None
        self.test_results = []
        self.frame_count = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        
    def draw_info_panel(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """
        在图像上绘制信息面板
        
        Args:
            frame: 输入图像
            info: 信息字典
            
        Returns:
            np.ndarray: 绘制了信息的图像
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 创建半透明背景
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # 绘制文本信息
        y_offset = 30
        line_height = 25
        
        # 标题
        cv2.putText(display_frame, "USB Camera Test", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # 状态信息
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(display_frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # 操作提示
        y_offset = h - 80
        hints = [
            "Controls:",
            "  Q/ESC - Quit",
            "  S - Save current frame",
            "  R - Reset test",
        ]
        
        for hint in hints:
            cv2.putText(display_frame, hint, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 15
        
        return display_frame
    
    def calculate_fps(self):
        """计算FPS"""
        self.frame_count += 1
        current_time = datetime.now().timestamp()
        
        if self.fps_start_time is None:
            self.fps_start_time = current_time
        elif current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def test_camera_live(self, device_id: int = 0):
        """实时摄像头测试（带UI）"""
        print("=" * 50)
        print("USB摄像头实时测试（带UI显示）")
        print("=" * 50)
        print("按 'Q' 或 'ESC' 退出")
        print("按 'S' 保存当前帧")
        print("按 'R' 重置测试")
        print()
        
        try:
            # 创建摄像头
            self.camera = USBCamera(device_id=device_id, width=640, height=480, fps=30)
            
            # 连接摄像头
            if not self.camera.connect():
                print("✗ 摄像头连接失败")
                return False
            
            print("✓ 摄像头连接成功")
            
            # 获取摄像头信息
            resolution = self.camera.get_resolution()
            status = self.camera.get_status()
            
            # 创建窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
            
            self.running = True
            saved_count = 0
            
            print("\n开始实时显示...")
            
            while self.running:
                # 捕获帧
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠ 无法捕获图像")
                    break
                
                # 计算FPS
                self.calculate_fps()
                
                # 准备显示信息
                info = {
                    "Status": "Connected" if self.camera.is_connected() else "Disconnected",
                    "Resolution": f"{resolution[0]}x{resolution[1]}",
                    "FPS": f"{self.current_fps:.1f}",
                    "Frames": f"{self.frame_count}",
                    "Saved": f"{saved_count} frames"
                }
                
                # 绘制信息面板
                display_frame = self.draw_info_panel(frame, info)
                
                # 显示图像
                cv2.imshow(self.window_name, display_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q 或 ESC
                    print("\n用户退出")
                    break
                elif key == ord('s'):  # S - 保存当前帧
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"test_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    saved_count += 1
                    print(f"✓ 已保存帧: {filename}")
                elif key == ord('r'):  # R - 重置
                    self.frame_count = 0
                    self.fps_start_time = None
                    self.current_fps = 0.0
                    saved_count = 0
                    print("✓ 测试已重置")
            
            # 清理
            cv2.destroyAllWindows()
            self.camera.disconnect()
            print("✓ 摄像头已断开")
            return True
            
        except KeyboardInterrupt:
            print("\n用户中断")
            return False
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.camera:
                self.camera.disconnect()
            cv2.destroyAllWindows()
    
    def test_image_processing_ui(self, device_id: int = 0):
        """图像处理测试（带UI显示）"""
        print("=" * 50)
        print("图像处理功能测试（带UI显示）")
        print("=" * 50)
        
        try:
            camera = USBCamera(device_id=device_id)
            if not camera.connect():
                print("✗ 摄像头连接失败")
                return False
            
            print("✓ 摄像头连接成功")
            print("按 'Q' 或 'ESC' 退出")
            print("按数字键切换显示模式:")
            print("  1 - 原始图像")
            print("  2 - RGB格式")
            print("  3 - 灰度图像")
            print("  4 - 调整大小")
            print("  5 - 对比度增强")
            print("  6 - 高斯模糊")
            print("  7 - 边缘检测")
            
            window_name = "Image Processing Test"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            display_mode = 1
            running = True
            
            while running:
                # 捕获原始帧
                frame = camera.capture_frame()
                if frame is None:
                    continue
                
                # 根据模式处理图像
                if display_mode == 1:
                    display_frame = frame.copy()
                    mode_text = "Original (BGR)"
                elif display_mode == 2:
                    display_frame = camera.capture_frame_rgb()
                    if display_frame is not None:
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    else:
                        display_frame = frame.copy()
                    mode_text = "RGB Format"
                elif display_mode == 3:
                    gray = camera.capture_frame_gray()
                    if gray is not None:
                        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    else:
                        display_frame = frame.copy()
                    mode_text = "Grayscale"
                elif display_mode == 4:
                    resized = ImageProcessor.resize(frame, 320, 240)
                    display_frame = cv2.resize(resized, (640, 480))
                    mode_text = "Resized (320x240)"
                elif display_mode == 5:
                    display_frame = ImageProcessor.enhance_contrast(frame, alpha=1.5, beta=10)
                    mode_text = "Contrast Enhanced"
                elif display_mode == 6:
                    display_frame = ImageProcessor.apply_gaussian_blur(frame, (15, 15), 0)
                    mode_text = "Gaussian Blur"
                elif display_mode == 7:
                    gray = camera.capture_frame_gray()
                    if gray is not None:
                        edges = ImageProcessor.detect_edges(gray)
                        display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    else:
                        display_frame = frame.copy()
                    mode_text = "Edge Detection"
                else:
                    display_frame = frame.copy()
                    mode_text = "Original"
                
                # 添加模式文本
                cv2.putText(display_frame, f"Mode: {mode_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 1-7 to switch mode, Q to quit", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_name, display_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif ord('1') <= key <= ord('7'):
                    display_mode = key - ord('0')
                    print(f"切换到模式 {display_mode}: {mode_text}")
            
            cv2.destroyAllWindows()
            camera.disconnect()
            print("✓ 测试完成")
            return True
            
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if camera:
                camera.disconnect()
            cv2.destroyAllWindows()


def test_camera_direct():
    """直接使用摄像头类测试"""
    print("=" * 50)
    print("测试1: 直接使用USBCamera类")
    print("=" * 50)
    
    # 创建摄像头实例
    camera = USBCamera(device_id=0, width=640, height=480, fps=30)
    
    # 连接摄像头
    if camera.connect():
        print("✓ 摄像头连接成功")
        
        # 获取分辨率
        resolution = camera.get_resolution()
        print(f"✓ 当前分辨率: {resolution[0]}x{resolution[1]}")
        
        # 捕获一帧
        frame = camera.capture_frame()
        if frame is not None:
            print(f"✓ 成功捕获图像，尺寸: {frame.shape}")
        else:
            print("✗ 捕获图像失败")
        
        # 断开连接
        camera.disconnect()
        print("✓ 摄像头已断开")
    else:
        print("✗ 摄像头连接失败")


def test_camera_factory():
    """使用工厂类创建摄像头测试"""
    print("\n" + "=" * 50)
    print("测试2: 使用工厂类创建摄像头")
    print("=" * 50)
    
    # 检查是否已注册
    if HardwareFactory.is_camera_registered('usb_camera'):
        print("✓ USB摄像头已注册到工厂类")
        
        # 使用工厂类创建摄像头
        config = {
            'device_id': 0,
            'width': 640,
            'height': 480,
            'fps': 30
        }
        
        camera = HardwareFactory.create_camera('usb_camera', config)
        if camera:
            print("✓ 工厂类成功创建摄像头实例")
            
            if camera.connect():
                print("✓ 摄像头连接成功")
                camera.disconnect()
            else:
                print("✗ 摄像头连接失败")
        else:
            print("✗ 工厂类创建摄像头失败")
    else:
        print("✗ USB摄像头未注册到工厂类")


def test_camera_context_manager():
    """使用上下文管理器测试"""
    print("\n" + "=" * 50)
    print("测试3: 使用上下文管理器")
    print("=" * 50)
    
    try:
        with USBCamera(device_id=0) as camera:
            if camera.is_connected():
                print("✓ 上下文管理器：摄像头已连接")
                frame = camera.capture_frame()
                if frame is not None:
                    print(f"✓ 成功捕获图像，尺寸: {frame.shape}")
        print("✓ 上下文管理器：摄像头已自动断开")
    except Exception as e:
        print(f"✗ 上下文管理器测试失败: {e}")


def test_image_processor():
    """测试图像预处理功能"""
    print("\n" + "=" * 50)
    print("测试4: 图像预处理功能")
    print("=" * 50)
    
    # 创建摄像头并捕获图像
    camera = USBCamera(device_id=0)
    if camera.connect():
        frame = camera.capture_frame()
        if frame is not None:
            print(f"✓ 原始图像尺寸: {frame.shape}")
            
            # 测试调整大小
            resized = ImageProcessor.resize(frame, 320, 240)
            print(f"✓ 调整大小后: {resized.shape}")
            
            # 测试转换为RGB
            rgb_frame = camera.capture_frame_rgb()
            if rgb_frame is not None:
                print(f"✓ RGB图像尺寸: {rgb_frame.shape}")
            
            # 测试转换为灰度
            gray_frame = camera.capture_frame_gray()
            if gray_frame is not None:
                print(f"✓ 灰度图像尺寸: {gray_frame.shape}")
            
            # 测试手势识别预处理
            preprocessed = ImageProcessor.preprocess_for_gesture_recognition(frame)
            print(f"✓ 预处理后图像尺寸: {preprocessed.shape}")
            print(f"✓ 预处理后数据类型: {preprocessed.dtype}")
            print(f"✓ 预处理后数值范围: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        
        camera.disconnect()
    else:
        print("✗ 摄像头连接失败，跳过图像处理测试")


def test_config_manager():
    """测试配置管理器"""
    print("\n" + "=" * 50)
    print("测试5: 配置管理器")
    print("=" * 50)
    
    # 注意：这需要有效的配置文件
    # 这里只是展示如何使用
    try:
        config_manager = HardwareConfigManager()
        if config_manager.load_config():
            print("✓ 配置加载成功")
            camera = config_manager.create_camera()
            if camera:
                print("✓ 配置管理器成功创建摄像头")
            else:
                print("✗ 配置管理器创建摄像头失败")
        else:
            print("⚠ 配置文件不存在或加载失败（这是正常的，如果还没有创建配置文件）")
    except Exception as e:
        print(f"⚠ 配置管理器测试跳过: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='USB摄像头测试程序')
    parser.add_argument('--ui', action='store_true', help='启用UI显示模式')
    parser.add_argument('--processing', action='store_true', help='图像处理测试UI')
    parser.add_argument('--device', type=int, default=0, help='摄像头设备ID（默认0）')
    parser.add_argument('--all', action='store_true', help='运行所有测试（包括UI）')
    
    args = parser.parse_args()
    
    print("USB摄像头模块测试")
    print("注意：需要连接USB摄像头才能完整测试")
    print()
    
    ui = CameraTestUI()
    
    if args.ui or args.all:
        # UI模式：实时显示
        ui.test_camera_live(device_id=args.device)
    elif args.processing:
        # 图像处理UI测试
        ui.test_image_processing_ui(device_id=args.device)
    else:
        # 命令行模式：运行所有基础测试
        test_camera_direct()
        test_camera_factory()
        test_camera_context_manager()
        test_image_processor()
        test_config_manager()
        
        print("\n" + "=" * 50)
        print("测试完成")
        print("=" * 50)
        print("\n提示：使用 --ui 参数启用实时UI显示")
        print("     使用 --processing 参数启用图像处理UI测试")
