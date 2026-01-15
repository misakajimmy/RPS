"""
YOLOv8-Pose手势识别测试示例（带UI显示）
YOLOv8-Pose Gesture Recognition Test Example with UI Display
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from collections import deque

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.game.gesture_recognition import GestureRecognizer
from src.game.game_logic.gesture import Gesture
from src.hardware.implementations.camera import USBCamera


class GestureTestUI:
    """手势识别测试UI类"""
    
    def __init__(self):
        self.camera: Optional[USBCamera] = None
        self.recognizer: Optional[GestureRecognizer] = None
        self.window_name = "Gesture Recognition Test"
        self.running = False
        
        # 统计信息
        self.frame_count = 0
        self.recognition_count = 0
        self.gesture_counts = {Gesture.ROCK: 0, Gesture.PAPER: 0, Gesture.SCISSORS: 0, Gesture.UNKNOWN: 0}
        self.confidence_history = deque(maxlen=30)  # 保存最近30帧的置信度
        
        # FPS计算
        self.fps_start_time = None
        self.current_fps = 0.0
        self.fps_frame_count = 0
        
        # 当前识别结果
        self.current_gesture = Gesture.UNKNOWN
        self.current_confidence = 0.0
        self.current_probabilities = {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}
        self.last_results = None  # YOLOv8结果，用于绘制关键点
        
    def calculate_fps(self):
        """计算FPS"""
        self.fps_frame_count += 1
        current_time = datetime.now().timestamp()
        
        if self.fps_start_time is None:
            self.fps_start_time = current_time
        elif current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
            self.fps_frame_count = 0
            self.fps_start_time = current_time
    
    def draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """
        在图像上绘制信息面板
        
        Args:
            frame: 输入图像
            
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
        line_height = 22
        
        # 标题
        cv2.putText(display_frame, "Gesture Recognition Test", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # 当前识别结果（大字体突出显示）
        gesture_text = f"Gesture: {self.current_gesture.value.upper()}"
        gesture_color = (0, 255, 0) if self.current_gesture != Gesture.UNKNOWN else (0, 0, 255)
        cv2.putText(display_frame, gesture_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        y_offset += line_height + 5
        
        confidence_text = f"Confidence: {self.current_confidence:.2f}"
        cv2.putText(display_frame, confidence_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height + 5
        
        # 概率分布
        prob_text = f"Rock: {self.current_probabilities['rock']:.2f} | "
        prob_text += f"Paper: {self.current_probabilities['paper']:.2f} | "
        prob_text += f"Scissors: {self.current_probabilities['scissors']:.2f}"
        cv2.putText(display_frame, prob_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height + 10
        
        # 统计信息
        cv2.putText(display_frame, "Statistics:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        stats_text = f"  Rock: {self.gesture_counts[Gesture.ROCK]} | "
        stats_text += f"Paper: {self.gesture_counts[Gesture.PAPER]} | "
        stats_text += f"Scissors: {self.gesture_counts[Gesture.SCISSORS]}"
        cv2.putText(display_frame, stats_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += line_height
        
        # 系统信息
        cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Frames: {self.frame_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        # 操作提示
        y_offset = h - 100
        hints = [
            "Controls:",
            "  Q/ESC - Quit",
            "  S - Save current frame",
            "  R - Reset statistics",
            "  L - Toggle landmarks",
        ]
        
        for hint in hints:
            cv2.putText(display_frame, hint, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 15
        
        return display_frame
    
    def draw_confidence_graph(self, frame: np.ndarray) -> np.ndarray:
        """
        在图像上绘制置信度历史图表
        
        Args:
            frame: 输入图像
            
        Returns:
            np.ndarray: 绘制了图表的图像
        """
        if len(self.confidence_history) < 2:
            return frame
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 图表区域
        graph_x, graph_y = w - 220, 20
        graph_w, graph_h = 200, 100
        
        # 绘制背景
        cv2.rectangle(display_frame, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (graph_x, graph_y), 
                     (graph_x + graph_w, graph_y + graph_h), (200, 200, 200), 1)
        
        # 绘制置信度曲线
        if len(self.confidence_history) > 1:
            points = []
            for i, conf in enumerate(self.confidence_history):
                x = graph_x + int((i / (len(self.confidence_history) - 1)) * graph_w)
                y = graph_y + graph_h - int(conf * graph_h)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(display_frame, points[i], points[i + 1], (0, 255, 0), 2)
        
        # 绘制标签
        cv2.putText(display_frame, "Confidence", (graph_x, graph_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display_frame, "1.0", (graph_x + graph_w + 5, graph_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(display_frame, "0.0", (graph_x + graph_w + 5, graph_y + graph_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        return display_frame
    
    def test_gesture_recognition_ui(self, device_id: int = 0, show_landmarks: bool = True):
        """手势识别UI测试"""
        print("=" * 50)
        print("MediaPipe手势识别UI测试")
        print("=" * 50)
        print("请将手放在摄像头前，做出手势（石头、布、剪刀）")
        print("按 'Q' 或 'ESC' 退出")
        print("按 'S' 保存当前帧")
        print("按 'R' 重置统计")
        print("按 'L' 切换关键点显示")
        print()
        
        try:
            # 初始化识别器（使用默认 YOLOv8-Pose nano 模型，首次运行会自动下载）
            self.recognizer = GestureRecognizer(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands=1,
                confidence_threshold=0.7,
                model_size='n'  # 使用 nano 模型（最小最快）
            )
            print("✓ 手势识别器创建成功")
            
            # 初始化摄像头
            self.camera = USBCamera(device_id=device_id, width=640, height=480, fps=30)
            
            if not self.camera.connect():
                print("✗ 摄像头连接失败")
                return False
            
            print("✓ 摄像头连接成功")
            
            # 创建窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1000, 700)
            
            self.running = True
            saved_count = 0
            
            print("\n开始实时识别...")
            
            while self.running:
                # 捕获帧
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # 计算FPS
                self.calculate_fps()
                self.frame_count += 1
                
                # 识别手势（使用带关键点的方法）
                gesture, confidence, probabilities, results = self.recognizer.recognize_with_landmarks(frame)
                
                # 更新当前结果
                self.current_gesture = gesture
                self.current_confidence = confidence
                self.current_probabilities = probabilities
                self.last_results = results
                
                # 更新统计
                if gesture != Gesture.UNKNOWN:
                    self.recognition_count += 1
                    self.gesture_counts[gesture] += 1
                
                # 更新置信度历史
                self.confidence_history.append(confidence)
                
                # 准备显示帧
                display_frame = frame.copy()
                
                # 绘制手部关键点（如果启用）
                # 具体的数据结构由 GestureRecognizer 内部处理，这里不直接访问属性
                if show_landmarks and results:
                    display_frame = self.recognizer.draw_landmarks(display_frame, results)
                
                # 绘制信息面板
                display_frame = self.draw_info_panel(display_frame)
                
                # 绘制置信度图表
                display_frame = self.draw_confidence_graph(display_frame)
                
                # 在图像上绘制手势指示（大图标）
                self._draw_gesture_indicator(display_frame)
                
                # 显示图像
                cv2.imshow(self.window_name, display_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q 或 ESC
                    print("\n用户退出")
                    break
                elif key == ord('s'):  # S - 保存当前帧
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"gesture_{gesture.value}_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    print(f"✓ 已保存: {filename} (手势: {gesture.value}, 置信度: {confidence:.3f})")
                elif key == ord('r'):  # R - 重置统计
                    self.gesture_counts = {Gesture.ROCK: 0, Gesture.PAPER: 0, 
                                          Gesture.SCISSORS: 0, Gesture.UNKNOWN: 0}
                    self.recognition_count = 0
                    self.confidence_history.clear()
                    print("✓ 统计已重置")
                elif key == ord('l'):  # L - 切换关键点显示
                    show_landmarks = not show_landmarks
                    print(f"✓ 关键点显示: {'开启' if show_landmarks else '关闭'}")
            
            # 显示最终统计
            print("\n" + "=" * 50)
            print("识别统计:")
            print(f"  总帧数: {self.frame_count}")
            print(f"  识别次数: {self.recognition_count}")
            print(f"  石头: {self.gesture_counts[Gesture.ROCK]}")
            print(f"  布: {self.gesture_counts[Gesture.PAPER]}")
            print(f"  剪刀: {self.gesture_counts[Gesture.SCISSORS]}")
            print(f"  未知: {self.gesture_counts[Gesture.UNKNOWN]}")
            if self.recognition_count > 0:
                success_rate = (self.recognition_count / self.frame_count) * 100
                print(f"  识别成功率: {success_rate:.1f}%")
            print("=" * 50)
            
            # 清理
            cv2.destroyAllWindows()
            self.camera.disconnect()
            print("✓ 测试完成")
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
    
    def _draw_gesture_indicator(self, frame: np.ndarray):
        """在图像上绘制手势指示图标"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 根据手势类型绘制不同的指示
        if self.current_gesture == Gesture.ROCK:
            # 绘制圆形（石头）
            cv2.circle(frame, (center_x, center_y), 80, (0, 255, 0), 3)
            cv2.putText(frame, "ROCK", (center_x - 40, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.current_gesture == Gesture.PAPER:
            # 绘制矩形（布）
            cv2.rectangle(frame, (center_x - 80, center_y - 60), 
                         (center_x + 80, center_y + 60), (0, 255, 0), 3)
            cv2.putText(frame, "PAPER", (center_x - 50, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.current_gesture == Gesture.SCISSORS:
            # 绘制X形状（剪刀）
            cv2.line(frame, (center_x - 60, center_y - 60), 
                    (center_x + 60, center_y + 60), (0, 255, 0), 3)
            cv2.line(frame, (center_x + 60, center_y - 60), 
                    (center_x - 60, center_y + 60), (0, 255, 0), 3)
            cv2.putText(frame, "SCISSORS", (center_x - 60, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def test_gesture_recognizer():
    """测试手势识别器（命令行模式）"""
    print("=" * 50)
    print("测试YOLOv8-Pose手势识别器")
    print("=" * 50)
    
    # 创建识别器
    recognizer = GestureRecognizer(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
        confidence_threshold=0.7
    )
    
    print("✓ 手势识别器创建成功")
    
    # 测试摄像头
    camera = USBCamera(device_id=0)
    
    if camera.connect():
        print("✓ 摄像头连接成功")
        
        print("\n请将手放在摄像头前，做出手势（石头、布、剪刀）")
        print("按 'q' 键退出，按空格键捕获并识别")
        
        frame_count = 0
        while True:
            frame = camera.capture_frame()
            if frame is None:
                continue
            
            # 识别手势
            gesture, confidence, probabilities = recognizer.recognize(frame)
            
            # 在图像上显示结果
            display_frame = frame.copy()
            
            # 添加文本信息
            text = f"Gesture: {gesture.value}, Confidence: {confidence:.2f}"
            cv2.putText(display_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示概率
            prob_text = f"Rock: {probabilities['rock']:.2f}, "
            prob_text += f"Paper: {probabilities['paper']:.2f}, "
            prob_text += f"Scissors: {probabilities['scissors']:.2f}"
            cv2.putText(display_frame, prob_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 显示提示
            hint_text = "Press SPACE to capture, Q to quit"
            cv2.putText(display_frame, hint_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Gesture Recognition", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # 空格键：打印当前识别结果
                print(f"\n帧 {frame_count}: {gesture.value}, 置信度: {confidence:.3f}")
                print(f"  概率分布: {probabilities}")
            
            frame_count += 1
        
        cv2.destroyAllWindows()
        camera.disconnect()
        print("\n✓ 测试完成")
    else:
        print("✗ 摄像头连接失败，无法进行测试")
        print("提示：请确保摄像头已连接且未被其他程序占用")


def test_gesture_recognizer_with_image():
    """使用测试图像测试手势识别器"""
    print("\n" + "=" * 50)
    print("测试：使用测试图像")
    print("=" * 50)
    
    recognizer = GestureRecognizer()
    
    # 创建一个测试图像（黑色背景）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 测试识别
    gesture, confidence, probabilities = recognizer.recognize(test_image)
    print(f"测试图像识别结果: {gesture.value}, 置信度: {confidence:.3f}")
    print(f"  预期: UNKNOWN（因为没有手部）")


def test_batch_recognition():
    """测试批量识别"""
    print("\n" + "=" * 50)
    print("测试：批量识别")
    print("=" * 50)
    
    recognizer = GestureRecognizer()
    
    # 创建多个测试图像
    test_images = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)
    ]
    
    results = recognizer.recognize_batch(test_images)
    print(f"批量识别完成，共 {len(results)} 个结果")
    
    for i, (gesture, confidence, _) in enumerate(results):
        print(f"  图像 {i+1}: {gesture.value}, 置信度: {confidence:.3f}")


def test_model_info():
    """测试模型信息"""
    print("\n" + "=" * 50)
    print("测试：模型信息")
    print("=" * 50)
    
    recognizer = GestureRecognizer()
    info = recognizer.get_model_info()
    
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MediaPipe手势识别测试程序')
    parser.add_argument('--ui', action='store_true', help='启用UI显示模式')
    parser.add_argument('--device', type=int, default=0, help='摄像头设备ID（默认0）')
    parser.add_argument('--no-landmarks', action='store_true', help='禁用关键点显示')
    parser.add_argument('--all', action='store_true', help='运行所有测试（包括UI）')
    
    args = parser.parse_args()
    
    print("MediaPipe手势识别测试")
    print("注意：需要连接摄像头才能完整测试")
    print()
    
    ui = GestureTestUI()
    
    if args.ui or args.all:
        # UI模式：实时手势识别
        ui.test_gesture_recognition_ui(
            device_id=args.device,
            show_landmarks=not args.no_landmarks
        )
    else:
        # 命令行模式：运行所有基础测试
        test_gesture_recognizer()
        test_gesture_recognizer_with_image()
        test_batch_recognition()
        test_model_info()
        
        print("\n" + "=" * 50)
        print("所有测试完成")
        print("=" * 50)
        print("\n提示：使用 --ui 参数启用实时UI显示")
        print("     使用 --no-landmarks 参数禁用关键点显示")
