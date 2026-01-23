#!/usr/bin/env python3
"""
YOLO 模型实时摄像头测试（显示完整元数据）
YOLO Model Real-time Camera Test with Full Metadata Display
"""
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from collections import deque

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("错误: ultralytics 未安装")
    print("请运行: pip install ultralytics")
    sys.exit(1)

from src.hardware.implementations.camera import USBCamera
from src.utils.logger import setup_logger

logger = setup_logger("RPS.YOLOTest")


def load_yolo_model_safe(model_path: str):
    """
    安全加载 YOLO 模型，处理 PyTorch 2.6+ 的 weights_only 限制
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        YOLO 模型实例
    """
    import torch
    import sys
    import ultralytics.nn.tasks as tasks_module
    
    # 添加 ultralytics 相关的安全全局变量
    try:
        from ultralytics.nn.tasks import DetectionModel
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([DetectionModel])
    except ImportError:
        pass
    
    # 创建兼容性模块映射（处理旧版 ultralytics.yolo 引用）
    class CompatibilityModule:
        """兼容性模块，将旧版 ultralytics.yolo 映射到新版"""
        def __getattr__(self, name):
            # 尝试从新位置导入
            try:
                if name == 'utils':
                    # 特殊处理 utils 子模块
                    import ultralytics.utils as utils
                    return utils
                from ultralytics import YOLO
                return getattr(YOLO, name, None)
            except:
                pass
            # 如果找不到，返回一个占位符模块
            return CompatibilityModule()
    
    # 添加兼容性模块到 sys.modules
    if 'ultralytics.yolo' not in sys.modules:
        sys.modules['ultralytics.yolo'] = CompatibilityModule()
    if 'ultralytics.yolo.utils' not in sys.modules:
        import ultralytics.utils as utils
        sys.modules['ultralytics.yolo.utils'] = utils
    
    # 保存原始函数
    original_torch_load = torch.load
    original_torch_safe_load = tasks_module.torch_safe_load
    
    # 定义 patch 函数
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    def patched_torch_safe_load(weight):
        # 使用 weights_only=False 加载模型
        return torch.load(weight, map_location="cpu", weights_only=False), weight
    
    # 应用 patch
    torch.load = patched_torch_load
    tasks_module.torch_safe_load = patched_torch_safe_load
    
    try:
        model = YOLO(model_path)
    finally:
        # 恢复原始函数
        torch.load = original_torch_load
        tasks_module.torch_safe_load = original_torch_safe_load
    
    return model


class YOLOMetadataTestUI:
    """YOLO 模型元数据测试UI类"""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.camera: Optional[USBCamera] = None
        self.window_name = "YOLO Metadata Test"
        self.running = False
        
        # 统计信息
        self.frame_count = 0
        self.detection_count = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        
        # 最近的结果历史
        self.result_history = deque(maxlen=10)
        
        # 显示选项
        self.show_boxes = True
        self.show_labels = True
        self.show_confidence = True
        self.show_metadata = True
        
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
    
    def parse_detection_results(self, results) -> List[Dict]:
        """
        解析 YOLO 检测结果，提取完整的元数据
        
        Args:
            results: YOLO predict 返回的结果
            
        Returns:
            List[Dict]: 检测结果列表，每个包含完整元数据
        """
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        # results 是 Results 对象列表，通常只有第一个元素
        result = results[0]
        
        # 获取检测框、置信度和类别
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = [result.names[int(cls_id)] for cls_id in class_ids]
            
            for i in range(len(boxes)):
                detection = {
                    'box': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': class_names[i],
                    'center': [
                        float((boxes[i][0] + boxes[i][2]) / 2),
                        float((boxes[i][1] + boxes[i][3]) / 2)
                    ],
                    'width': float(boxes[i][2] - boxes[i][0]),
                    'height': float(boxes[i][3] - boxes[i][1]),
                    'area': float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
                }
                detections.append(detection)
        
        # 如果有关键点（Pose模型）
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # [num_detections, num_keypoints, 2]
            keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            
            for i, detection in enumerate(detections):
                if i < len(keypoints):
                    detection['keypoints'] = keypoints[i].tolist()
                    if keypoints_conf is not None:
                        detection['keypoints_confidence'] = keypoints_conf[i].tolist()
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            
        Returns:
            np.ndarray: 绘制了检测框的图像
        """
        display_frame = frame.copy()
        
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            
            # 绘制检测框
            if self.show_boxes:
                color = (0, 255, 0)  # 绿色
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签和置信度
            if self.show_labels or self.show_confidence:
                label_parts = []
                if self.show_labels:
                    label_parts.append(det['class_name'])
                if self.show_confidence:
                    label_parts.append(f"{det['confidence']:.2f}")
                
                label = " ".join(label_parts)
                
                # 计算文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 绘制文本背景
                cv2.rectangle(
                    display_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    display_frame,
                    label,
                    (x1, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # 绘制关键点（如果有）
            if 'keypoints' in det:
                keypoints = np.array(det['keypoints'])
                for kp in keypoints:
                    if len(kp) >= 2 and not np.isnan(kp[0]) and not np.isnan(kp[1]):
                        cv2.circle(display_frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
        
        return display_frame
    
    def draw_metadata_panel(self, frame: np.ndarray, detections: List[Dict], 
                           inference_time: float, model_info: Dict) -> np.ndarray:
        """
        在图像上绘制元数据面板
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            inference_time: 推理时间（毫秒）
            model_info: 模型信息
            
        Returns:
            np.ndarray: 绘制了元数据的图像
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 计算面板高度（根据检测数量动态调整）
        panel_height = 180 + len(detections) * 120
        panel_height = min(panel_height, h - 20)
        
        # 创建半透明背景
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (w - 400, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
        
        # 绘制文本信息
        y_offset = 30
        line_height = 22
        
        # 标题
        cv2.putText(display_frame, "YOLO Metadata", (w - 390, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # 基本信息
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Inference: {inference_time:.1f} ms",
            f"Detections: {len(detections)}",
            f"Model: {model_info.get('name', 'unknown')}",
            f"Input: {model_info.get('input_size', 'unknown')}",
            "---"
        ]
        
        for line in info_lines:
            cv2.putText(display_frame, line, (w - 390, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # 绘制每个检测的详细信息
        for i, det in enumerate(detections[:3]):  # 最多显示3个检测
            if y_offset + 100 > panel_height:
                break
                
            det_info = [
                f"Detection #{i+1}:",
                f"  Class: {det['class_name']} (id={det['class_id']})",
                f"  Confidence: {det['confidence']:.3f}",
                f"  Box: [{det['box'][0]:.0f}, {det['box'][1]:.0f}, {det['box'][2]:.0f}, {det['box'][3]:.0f}]",
                f"  Center: ({det['center'][0]:.0f}, {det['center'][1]:.0f})",
                f"  Size: {det['width']:.0f}x{det['height']:.0f}",
                f"  Area: {det['area']:.0f} px²",
            ]
            
            if 'keypoints' in det:
                det_info.append(f"  Keypoints: {len(det['keypoints'])} points")
            
            for line in det_info:
                cv2.putText(display_frame, line, (w - 390, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += line_height - 3
            
            y_offset += 5
        
        # 控制提示
        y_offset = h - 120
        cv2.putText(display_frame, "Controls:", (w - 390, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        controls = [
            "Q/ESC: Exit",
            "B: Toggle boxes",
            "L: Toggle labels",
            "M: Toggle metadata",
            "S: Save frame"
        ]
        for line in controls:
            cv2.putText(display_frame, line, (w - 390, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset += line_height - 3
        
        return display_frame
    
    def test_yolo_live(self,
                      model_path: str = "models/best.pt",
                      device_id: int = 0,
                      confidence: float = 0.25,
                      device: Optional[str] = None):
        """
        实时测试 YOLO 模型
        
        Args:
            model_path: YOLO 模型文件路径
            device_id: 摄像头设备ID
            confidence: 检测置信度阈值
            device: 计算设备 ('cuda', 'cpu', 'mps' 或 None 自动检测)
        """
        print("=" * 60)
        print("YOLO 模型实时摄像头测试（完整元数据）")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"摄像头设备: {device_id}")
        print(f"置信度阈值: {confidence}")
        print()
        
        # 检查模型文件
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"✗ 模型文件不存在: {model_path}")
            print("\n提示:")
            print("1. 确保模型文件路径正确")
            print("2. 默认路径: models/best.pt")
            return False
        
        try:
            # 加载模型（使用安全加载方法处理 PyTorch 2.6+ 的 weights_only 问题）
            print("加载 YOLO 模型...")
            self.model = load_yolo_model_safe(str(model_path_obj))
            
            # 自动检测设备
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 'cuda'
                        print(f"✓ 检测到 GPU: {torch.cuda.get_device_name(0)}")
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = 'mps'
                        print("✓ 检测到 Apple Silicon GPU (MPS)")
                    else:
                        device = 'cpu'
                        print("使用 CPU")
                except ImportError:
                    device = 'cpu'
                    print("使用 CPU")
            
            print(f"✓ 模型加载成功，使用设备: {device}")
            print(f"✓ 模型类别: {list(self.model.names.values())}")
            
            # 创建摄像头
            print("连接摄像头...")
            self.camera = USBCamera(device_id=device_id, width=640, height=480, fps=30)
            
            if not self.camera.connect():
                print("✗ 摄像头连接失败")
                return False
            
            print("✓ 摄像头连接成功")
            
            # 获取摄像头信息
            resolution = self.camera.get_resolution()
            
            # 创建窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            
            self.running = True
            saved_count = 0
            
            print("\n开始实时检测...")
            print(f"摄像头分辨率: {resolution[0]}x{resolution[1]}")
            print(f"模型输入尺寸: {self.model.overrides.get('imgsz', 'auto')}")
            print()
            print("控制:")
            print("  Q 或 ESC: 退出")
            print("  B: 切换检测框显示")
            print("  L: 切换标签显示")
            print("  M: 切换元数据面板显示")
            print("  S: 保存当前帧")
            print()
            
            model_info = {
                'name': model_path_obj.name,
                'input_size': str(self.model.overrides.get('imgsz', 'auto'))
            }
            
            while self.running:
                # 捕获帧
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠ 无法捕获图像")
                    break
                
                # 计算FPS
                self.calculate_fps()
                
                # 推理
                start_time = datetime.now()
                results = self.model.predict(
                    frame,
                    conf=confidence,
                    device=device,
                    verbose=False
                )
                inference_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # 解析结果
                detections = self.parse_detection_results(results)
                
                if detections:
                    self.detection_count += len(detections)
                    self.result_history.append({
                        'timestamp': datetime.now(),
                        'detections': detections,
                        'inference_time': inference_time
                    })
                
                # 绘制检测结果
                if self.show_boxes or self.show_labels:
                    frame = self.draw_detections(frame, detections)
                
                # 绘制元数据面板
                if self.show_metadata:
                    frame = self.draw_metadata_panel(frame, detections, inference_time, model_info)
                else:
                    # 至少显示FPS
                    cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Detections: {len(detections)}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow(self.window_name, frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q 或 ESC
                    break
                elif key == ord('b'):  # 切换检测框
                    self.show_boxes = not self.show_boxes
                    print(f"检测框显示: {'开启' if self.show_boxes else '关闭'}")
                elif key == ord('l'):  # 切换标签
                    self.show_labels = not self.show_labels
                    print(f"标签显示: {'开启' if self.show_labels else '关闭'}")
                elif key == ord('m'):  # 切换元数据面板
                    self.show_metadata = not self.show_metadata
                    print(f"元数据面板: {'开启' if self.show_metadata else '关闭'}")
                elif key == ord('s'):  # 保存帧
                    saved_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"yolo_test_frame_{timestamp}_{saved_count:04d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"保存帧: {filename}")
            
            print("\n测试完成")
            print(f"总帧数: {self.frame_count}")
            print(f"总检测数: {self.detection_count}")
            if self.frame_count > 0:
                print(f"平均FPS: {self.current_fps:.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.camera:
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='YOLO 模型实时摄像头测试（显示完整元数据）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型和摄像头
  python tests/test_yolo_metadata.py
  
  # 指定模型路径和摄像头设备
  python tests/test_yolo_metadata.py --model models/best.pt --device-id 0
  
  # 指定置信度阈值和设备
  python tests/test_yolo_metadata.py --confidence 0.3 --device cuda
  
  # 使用CPU
  python tests/test_yolo_metadata.py --device cpu
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        default="models/best.pt",
        help='YOLO 模型文件路径（默认: models/best.pt）'
    )
    parser.add_argument(
        '--device-id',
        type=int,
        default=0,
        help='摄像头设备ID（默认: 0）'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='检测置信度阈值（默认: 0.25）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'mps'],
        help='计算设备（默认: 自动检测）'
    )
    
    args = parser.parse_args()
    
    # 检查 YOLO 是否可用
    if not YOLO_AVAILABLE:
        print("错误: ultralytics 未安装")
        print("请运行: pip install ultralytics")
        sys.exit(1)
    
    ui = YOLOMetadataTestUI()
    success = ui.test_yolo_live(
        model_path=args.model,
        device_id=args.device_id,
        confidence=args.confidence,
        device=args.device
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
