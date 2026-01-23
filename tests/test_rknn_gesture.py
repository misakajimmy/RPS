#!/usr/bin/env python3
"""
RKNN 手势识别实时测试（带UI显示）
RKNN Gesture Recognition Real-time Test with UI Display
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

from src.game.gesture_recognition import RecognizerFactory

# 尝试导入 RKNN 识别器
try:
    from src.game.gesture_recognition import RKNNRecognizer
    from src.game.gesture_recognition.rknn_recognizer import RKNN_AVAILABLE
except ImportError:
    RKNNRecognizer = None
    RKNN_AVAILABLE = False
from src.game.game_logic.gesture import Gesture
from src.hardware.implementations.camera import USBCamera
from src.utils.logger import setup_logger

logger = setup_logger("RPS.RKNNTest")


class RKNNGestureTestUI:
    """RKNN 手势识别测试UI类"""
    
    def __init__(self):
        self.recognizer: Optional[RKNNRecognizer] = None
        self.camera: Optional[USBCamera] = None
        self.window_name = "RKNN Gesture Recognition Test"
        self.running = False
        self.frame_count = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        self.stats = {
            'rock': 0,
            'paper': 0,
            'scissors': 0,
            'unknown': 0,
            'total': 0
        }
        self.last_result = None
        
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
        cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # 绘制文本信息
        y_offset = 30
        line_height = 25
        
        # 标题
        cv2.putText(display_frame, "RKNN Gesture Recognition", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # 状态信息
        for key, value in info.items():
            text = f"{key}: {value}"
            color = (255, 255, 255)
            if key == "Gesture":
                # 根据手势类型设置颜色
                if value == "ROCK":
                    color = (0, 0, 255)  # 红色
                elif value == "PAPER":
                    color = (0, 255, 0)  # 绿色
                elif value == "SCISSORS":
                    color = (255, 0, 0)  # 蓝色
                elif value == "UNKNOWN":
                    color = (128, 128, 128)  # 灰色
            cv2.putText(display_frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height
        
        # 统计信息
        y_offset += 10
        cv2.putText(display_frame, "Statistics:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        for gesture, count in self.stats.items():
            if gesture != 'total':
                text = f"  {gesture.capitalize()}: {count}"
                cv2.putText(display_frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 18
        
        # 操作提示
        y_offset = h - 100
        hints = [
            "Controls:",
            "  Q/ESC - Quit",
            "  S - Save current frame",
            "  R - Reset statistics",
            "  I - Show model info",
        ]
        
        for hint in hints:
            cv2.putText(display_frame, hint, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 15
        
        return display_frame
    
    def draw_gesture_label(self, frame: np.ndarray, gesture: Gesture, confidence: float):
        """
        在图像上绘制手势标签
        
        Args:
            frame: 输入图像
            gesture: 识别的手势
            confidence: 置信度
        """
        h, w = frame.shape[:2]
        
        # 手势文本
        gesture_text = gesture.value.upper()
        confidence_text = f"{confidence:.2f}"
        
        # 根据手势设置颜色
        if gesture == Gesture.ROCK:
            color = (0, 0, 255)  # 红色
            text = f"ROCK ({confidence_text})"
        elif gesture == Gesture.PAPER:
            color = (0, 255, 0)  # 绿色
            text = f"PAPER ({confidence_text})"
        elif gesture == Gesture.SCISSORS:
            color = (255, 0, 0)  # 蓝色
            text = f"SCISSORS ({confidence_text})"
        else:
            color = (128, 128, 128)  # 灰色
            text = f"UNKNOWN ({confidence_text})"
        
        # 绘制背景
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        )
        cv2.rectangle(frame, (10, h - text_height - 30), 
                     (20 + text_width, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - text_height - 30), 
                     (20 + text_width, h - 10), color, 2)
        
        # 绘制文本
        cv2.putText(frame, text, (15, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    def test_rknn_live(self, 
                      model_path: str = "models/best.rknn",
                      device_id: int = 0,
                      input_size: tuple = (640, 640),
                      confidence_threshold: float = 0.7,
                      image_path: Optional[str] = None):
        """
        实时 RKNN 手势识别测试（带UI）
        
        Args:
            model_path: RKNN 模型文件路径
            device_id: 摄像头设备ID（仅在 image_path 为 None 时使用）
            input_size: 模型输入尺寸
            confidence_threshold: 置信度阈值
            image_path: 图片路径（如果提供，则使用图片推理而不是摄像头）
        """
        print("=" * 60)
        if image_path:
            print("RKNN 手势识别图片测试")
        else:
            print("RKNN 手势识别实时测试（带UI显示）")
        print("=" * 60)
        
        try:
            # 检查 RKNN 是否可用
            if not RKNN_AVAILABLE or RKNNRecognizer is None:
                print("✗ RKNN 运行时未安装，无法使用 RKNN 识别器")
                print("\n提示:")
                print("1. 在 RK3588 设备上安装 rknn-toolkit-lite2 (推荐):")
                print("   pip install rknn-toolkit-lite2")
                print("   或: pip install rknnlite")
                print("2. 在开发机/Windows 上安装 rknn-toolkit2 (用于模型转换):")
                print("   从 Rockchip 官方获取: https://github.com/rockchip-linux/rknn-toolkit2")
                print("   Windows 上可以使用模拟器模式进行测试（速度较慢）")
                print("3. 实际 NPU 加速需要在 RK3588 设备上使用 rknn-toolkit-lite2")
                print("4. 如果只想测试识别功能，可以使用 YOLO 识别器:")
                print("   python tests/test_mediapipe_gesture.py --ui")
                return False
            
            # 创建 RKNN 识别器
            logger.info(f"加载 RKNN 模型: {model_path}")
            self.recognizer = RKNNRecognizer(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                min_detection_confidence=0.5,
                input_size=input_size
            )
            
            if not self.recognizer.initialize():
                print("✗ RKNN 识别器初始化失败")
                return False
            
            print("✓ RKNN 识别器初始化成功")
            
            # 如果提供了图片路径，使用图片模式
            if image_path:
                return self._test_image(image_path, input_size, confidence_threshold)
            
            # 否则使用摄像头实时模式
            print("按 'Q' 或 'ESC' 退出")
            print("按 'S' 保存当前帧")
            print("按 'R' 重置统计")
            print("按 'I' 显示模型信息")
            print()
            
            # 创建摄像头
            self.camera = USBCamera(device_id=device_id, width=640, height=480, fps=30)
            
            if not self.camera.connect():
                print("✗ 摄像头连接失败")
                return False
            
            print("✓ 摄像头连接成功")
            
            # 获取摄像头信息
            resolution = self.camera.get_resolution()
            
            # 创建窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
            
            self.running = True
            saved_count = 0
            
            print("\n开始实时识别...")
            print(f"模型输入尺寸: {input_size[0]}x{input_size[1]}")
            print(f"摄像头分辨率: {resolution[0]}x{resolution[1]}")
            print()
            
            while self.running:
                # 捕获帧
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠ 无法捕获图像")
                    break
                
                # 计算FPS
                self.calculate_fps()
                
                # 识别手势
                start_time = datetime.now()
                gesture, confidence, probabilities = self.recognizer.recognize(frame)
                inference_time = (datetime.now() - start_time).total_seconds() * 1000  # 毫秒
                
                # 更新统计
                if gesture != Gesture.UNKNOWN:
                    self.stats[gesture.value] += 1
                else:
                    self.stats['unknown'] += 1
                self.stats['total'] += 1
                
                self.last_result = (gesture, confidence, probabilities)
                
                # 准备显示信息
                info = {
                    "Status": "Running",
                    "FPS": f"{self.current_fps:.1f}",
                    "Inference": f"{inference_time:.1f} ms",
                    "Gesture": gesture.value.upper(),
                    "Confidence": f"{confidence:.3f}",
                    "Frames": f"{self.stats['total']}",
                    "Saved": f"{saved_count} frames"
                }
                
                # 绘制信息面板
                display_frame = self.draw_info_panel(frame, info)
                
                # 绘制手势标签
                self.draw_gesture_label(display_frame, gesture, confidence)
                
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
                elif key == ord('r'):  # R - 重置统计
                    self.stats = {
                        'rock': 0,
                        'paper': 0,
                        'scissors': 0,
                        'unknown': 0,
                        'total': 0
                    }
                    self.frame_count = 0
                    self.fps_start_time = None
                    self.current_fps = 0.0
                    print("✓ 统计已重置")
                elif key == ord('i'):  # I - 显示模型信息
                    print("\n模型信息:")
                    print(f"  模型路径: {model_path}")
                    print(f"  输入尺寸: {input_size[0]}x{input_size[1]}")
                    print(f"  置信度阈值: {confidence_threshold}")
                    print(f"  当前识别结果:")
                    if self.last_result:
                        g, c, p = self.last_result
                        print(f"    手势: {g.value}")
                        print(f"    置信度: {c:.3f}")
                        print(f"    概率分布: {p}")
            
            # 清理
            cv2.destroyAllWindows()
            if self.camera:
                self.camera.disconnect()
            if self.recognizer:
                self.recognizer.release()
            
            print("\n" + "=" * 60)
            print("测试完成")
            print("=" * 60)
            print("统计结果:")
            for gesture, count in self.stats.items():
                if gesture != 'total':
                    percentage = (count / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
                    print(f"  {gesture.capitalize()}: {count} ({percentage:.1f}%)")
            print(f"  总计: {self.stats['total']} 帧")
            
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
            if self.recognizer:
                self.recognizer.release()
            cv2.destroyAllWindows()
    
    def _test_image(self, image_path: str, input_size: tuple, confidence_threshold: float) -> bool:
        """
        使用图片进行推理测试
        
        Args:
            image_path: 图片文件路径
            input_size: 模型输入尺寸
            confidence_threshold: 置信度阈值
            
        Returns:
            bool: 测试是否成功
        """
        try:
            # 检查图片文件是否存在
            img_path = Path(image_path)
            if not img_path.exists():
                print(f"✗ 图片文件不存在: {img_path}")
                return False
            
            print(f"加载图片: {img_path}")
            
            # 读取图片
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"✗ 无法读取图片: {img_path}")
                return False
            
            print(f"✓ 图片加载成功: {frame.shape[1]}x{frame.shape[0]}")
            print()
            
            # 识别手势
            print("进行推理...")
            start_time = datetime.now()
            gesture, confidence, probabilities = self.recognizer.recognize(frame)
            inference_time = (datetime.now() - start_time).total_seconds() * 1000  # 毫秒
            
            # 显示结果
            print("\n" + "=" * 60)
            print("推理结果:")
            print("=" * 60)
            print(f"手势: {gesture.value.upper()}")
            print(f"置信度: {confidence:.4f}")
            print(f"推理时间: {inference_time:.2f} ms")
            print()
            
            if probabilities:
                print("概率分布:")
                for i, prob in enumerate(probabilities):
                    gesture_name = Gesture(i).value if i < len(Gesture) else f"Class_{i}"
                    print(f"  {gesture_name}: {prob:.4f}")
                print()
            
            # 准备显示信息
            info = {
                "Mode": "Image Test",
                "Image": img_path.name,
                "Inference": f"{inference_time:.1f} ms",
                "Gesture": gesture.value.upper(),
                "Confidence": f"{confidence:.3f}",
            }
            
            # 绘制信息面板
            display_frame = self.draw_info_panel(frame, info)
            
            # 绘制手势标签
            self.draw_gesture_label(display_frame, gesture, confidence)
            
            # 创建窗口并显示
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
            
            print("按任意键关闭窗口...")
            cv2.imshow(self.window_name, display_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print("\n✓ 测试完成")
            return True
            
        except Exception as e:
            print(f"✗ 图片测试异常: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RKNN 手势识别实时测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型和摄像头
  python tests/test_rknn_gesture.py
  
  # 指定模型路径和摄像头设备
  python tests/test_rknn_gesture.py --model models/best.rknn --device 0
  
  # 指定输入尺寸和置信度阈值
  python tests/test_rknn_gesture.py --input-size 640 640 --confidence 0.6
  
  # 使用图片进行推理
  python tests/test_rknn_gesture.py --image path/to/image.jpg
  
  # 使用图片并指定模型和置信度阈值
  python tests/test_rknn_gesture.py --image path/to/image.jpg --model models/best.rknn --confidence 0.6
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/best.rknn',
        help='RKNN 模型文件路径（默认: models/best.rknn）'
    )
    
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='摄像头设备ID（默认: 0）'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('WIDTH', 'HEIGHT'),
        help='模型输入尺寸（默认: 640 640）'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='置信度阈值（默认: 0.7）'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='图片文件路径（如果提供，则使用图片推理而不是摄像头）'
    )
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ 模型文件不存在: {model_path}")
        print("\n提示:")
        print("1. 确保已转换 RKNN 模型")
        print("2. 使用 scripts/convert_to_rknn.py 转换模型")
        print("3. 或指定正确的模型路径: --model path/to/model.rknn")
        sys.exit(1)
    
    # 创建测试UI
    ui = RKNNGestureTestUI()
    
    # 检查图片文件（如果提供）
    image_path = None
    if args.image:
        image_path_obj = Path(args.image)
        if not image_path_obj.exists():
            print(f"✗ 图片文件不存在: {image_path_obj}")
            sys.exit(1)
        image_path = str(image_path_obj)
    
    # 运行测试
    success = ui.test_rknn_live(
        model_path=str(model_path),
        device_id=args.device,
        input_size=tuple(args.input_size),
        confidence_threshold=args.confidence,
        image_path=image_path
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
