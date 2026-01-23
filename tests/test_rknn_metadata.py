#!/usr/bin/env python3
"""
RKNN 模型实时摄像头测试（显示完整元数据）
RKNN Model Real-time Camera Test with Full Metadata Display
"""
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from collections import deque
import platform

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入 RKNN Lite（优先，用于 RK3588 运行时推理）
RKNN_AVAILABLE = False
RKNN_CLASS = None
USE_RKNN_LITE = False

try:
    from rknnlite.api import RKNNLite
    RKNN_CLASS = RKNNLite
    RKNN_AVAILABLE = True
    USE_RKNN_LITE = True
    print("✓ 检测到 rknnlite (rknn-toolkit-lite2)，将使用 NPU 运行时")
except ImportError:
    # 如果 rknnlite 不可用，尝试 rknn-toolkit2（用于开发/转换）
    try:
        from rknn.api import RKNN
        RKNN_CLASS = RKNN
        RKNN_AVAILABLE = True
        USE_RKNN_LITE = False
        print("✓ 检测到 rknn-toolkit2，将使用开发模式（可能使用模拟器）")
    except ImportError:
        RKNN_AVAILABLE = False
        RKNN_CLASS = None
        print("⚠ rknnlite 和 rknn-toolkit2 均未安装")

from src.hardware.implementations.camera import USBCamera
from src.utils.logger import setup_logger

logger = setup_logger("RPS.RKNNMetadataTest")


class RKNNMetadataTestUI:
    """RKNN 模型元数据测试UI类"""
    
    def __init__(self):
        self.rknn = None  # 类型: Optional[RKNNLite] 或 Optional[RKNN]
        self.camera: Optional[USBCamera] = None
        self.window_name = "RKNN Metadata Test"
        self.running = False
        self._initialized = False
        
        # 模型信息
        self.model_path: Optional[Path] = None
        self.input_size: tuple = (640, 640)
        
        # 统计信息
        self.frame_count = 0
        self.inference_count = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        
        # 最近的结果历史
        self.result_history = deque(maxlen=10)
        
        # 显示选项
        self.show_metadata = True
        self.show_input_info = True
        self.show_detections = True
        
        # 模型输出信息
        self.output_shape: Optional[tuple] = None
        self.output_dtype: Optional[str] = None
        self.num_classes: int = 3  # 默认3类：rock, paper, scissors
        
        # 检测模型相关参数
        self.is_detection_model = False
        self.detection_confidence_threshold = 0.25
        self.nms_iou_threshold = 0.4
        self.class_names: List[str] = []
        
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
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以适配模型输入
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            np.ndarray: 预处理后的图像（NCHW格式，归一化到[0,1]）
        """
        # 调整大小
        resized = cv2.resize(image, self.input_size)
        
        # BGR 转 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # HWC 转 CHW
        chw = rgb.transpose(2, 0, 1)
        
        # 归一化到 [0, 1]
        normalized = chw.astype(np.float32) / 255.0
        
        # 添加 batch 维度
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def parse_rknn_output(self, output: np.ndarray) -> Dict:
        """
        解析 RKNN 推理输出，提取完整的元数据
        
        Args:
            output: RKNN 推理输出
            
        Returns:
            Dict: 包含完整元数据的字典
        """
        # 记录输出形状和类型（首次运行时）
        if self.output_shape is None:
            self.output_shape = output.shape
            self.output_dtype = str(output.dtype)
            logger.info(f"模型输出形状: {self.output_shape}, 数据类型: {self.output_dtype}")
        
        # 展平输出
        output_flat = output.flatten()
        
        # 尝试解析不同类型的输出
        metadata = {
            'raw_output_shape': list(output.shape),
            'raw_output_dtype': str(output.dtype),
            'output_size': int(output.size),
            'output_min': float(np.min(output_flat)),
            'output_max': float(np.max(output_flat)),
            'output_mean': float(np.mean(output_flat)),
            'output_std': float(np.std(output_flat)),
        }
        
        # 如果是分类输出（通常是 [1, num_classes] 或 [num_classes]）
        if len(output.shape) == 1 or (len(output.shape) == 2 and output.shape[0] == 1):
            if len(output.shape) == 2:
                probs = output[0]
            else:
                probs = output
            
            # 应用 softmax（如果模型没有应用）
            probs = np.exp(probs - np.max(probs))
            probs = probs / np.sum(probs)
            
            # 获取最大概率的类别
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            
            # 类别名称映射
            class_names = ['rock', 'paper', 'scissors']
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f'class_{class_id}'
            
            metadata.update({
                'type': 'classification',
                'num_classes': len(probs),
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'probabilities': [float(p) for p in probs],
                'class_names': class_names[:len(probs)]
            })
        # 检测输出（YOLOv8 检测模型：shape 为 (1, 25, 8400) 或类似）
        elif len(output.shape) == 3 and output.shape[2] == 8400:
            # YOLOv8 检测模型
            num_classes = output.shape[1] - 4
            metadata.update({
                'type': 'yolo_detection',
                'num_anchors': output.shape[2],  # 8400
                'num_classes': num_classes,
                'coord_dim': 4,
                'detection_format': 'yolov8'
            })
        elif len(output.shape) >= 2:
            metadata.update({
                'type': 'detection',
                'num_detections': output.shape[1] if len(output.shape) >= 2 else 1,
                'detection_dim': output.shape[2] if len(output.shape) >= 3 else output.shape[1],
                'raw_values': output[0].tolist() if len(output.shape) >= 2 else output.tolist()
            })
        else:
            metadata['type'] = 'unknown'
        
        return metadata
    
    def postprocess_detection_output(self, output: np.ndarray,
                                      img_width: int, img_height: int) -> List[Dict]:
        """
        后处理 RKNN YOLOv8 检测模型输出
        
        Args:
            output: 模型输出，形状为 (1, 25, 8400) 或 (25, 8400)
            img_width: 输入图像宽度
            img_height: 输入图像高度
            
        Returns:
            List[Dict]: 检测结果列表，每个包含 box, confidence, class_id, class_name
        """
        # 移除 batch 维度
        if len(output.shape) == 3:
            output = output[0]  # shape: (25, 8400)
        
        # 提取坐标和类别分数
        bbox_coords = output[0:4, :]  # shape: (4, 8400) - [cx, cy, w, h] (相对坐标)
        class_logits = output[4:, :]  # shape: (21, 8400) - 类别分数
        
        # 应用 sigmoid 激活
        bbox_coords_sigmoid = 1 / (1 + np.exp(-bbox_coords))
        class_probs = 1 / (1 + np.exp(-class_logits))
        
        # 找到每个检测点的最高类别分数
        max_class_probs = np.max(class_probs, axis=0)  # shape: (8400,)
        class_ids = np.argmax(class_probs, axis=0)      # shape: (8400,)
        
        # YOLOv8 的特征图尺度（多尺度检测）
        # 8400 = 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400
        strides = [8, 16, 32]  # 对应的步长
        grid_sizes = [(80, 80), (40, 40), (20, 20)]
        
        detections = []
        
        # 过滤和转换每个检测点
        for i in range(8400):
            confidence = float(max_class_probs[i])
            
            # 过滤低置信度
            if confidence < self.detection_confidence_threshold:
                continue
            
            class_id = int(class_ids[i])
            cx_rel, cy_rel, w_rel, h_rel = bbox_coords_sigmoid[:, i]
            
            # 确定当前检测点属于哪个特征图尺度
            if i < 6400:
                # 第一个尺度：80x80
                stride = strides[0]
                grid_h, grid_w = grid_sizes[0]
                grid_idx = i
            elif i < 6400 + 1600:
                # 第二个尺度：40x40
                stride = strides[1]
                grid_h, grid_w = grid_sizes[1]
                grid_idx = i - 6400
            else:
                # 第三个尺度：20x20
                stride = strides[2]
                grid_h, grid_w = grid_sizes[2]
                grid_idx = i - 8000
            
            # 计算网格位置
            grid_y = grid_idx // grid_w
            grid_x = grid_idx % grid_w
            
            # 转换为绝对坐标
            cx = (grid_x + cx_rel) * stride
            cy = (grid_y + cy_rel) * stride
            w = w_rel * img_width
            h = h_rel * img_height
            
            # 转换为 x1, y1, x2, y2
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(img_width, int(cx + w / 2))
            y2 = min(img_height, int(cy + h / 2))
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'Class_{class_id}',
                'center': [cx, cy],
                'width': w,
                'height': h
            })
        
        # NMS（非极大值抑制）
        if detections:
            boxes = [det['box'] for det in detections]
            scores = [det['confidence'] for det in detections]
            
            # 使用 OpenCV 的 NMSBoxes
            indices = cv2.dnn.NMSBoxes(
                boxes, scores,
                self.detection_confidence_threshold,
                self.nms_iou_threshold
            )
            
            if len(indices) > 0:
                # 如果 indices 是嵌套数组，展平
                if isinstance(indices[0], np.ndarray):
                    indices = [idx[0] for idx in indices]
                else:
                    indices = list(indices.flatten())
                
                final_detections = [detections[i] for i in indices]
                return final_detections
        
        return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测框和标签（只绘制置信度最高的一个）
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            
        Returns:
            np.ndarray: 绘制了检测框的图像
        """
        display_frame = frame.copy()
        
        # 如果没有检测结果，直接返回
        if not detections:
            return display_frame
        
        # 找到置信度最高的检测结果
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        x1, y1, x2, y2 = best_detection['box']
        confidence = best_detection['confidence']
        class_name = best_detection['class_name']
        
        # 绘制检测框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签和置信度
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 绘制文本背景
        cv2.rectangle(
            display_frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
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
        
        return display_frame
    
    def draw_metadata_panel(self, frame: np.ndarray, metadata: Dict,
                           inference_time: float, model_info: Dict) -> np.ndarray:
        """
        在图像上绘制元数据面板
        
        Args:
            frame: 输入图像
            metadata: 推理元数据
            inference_time: 推理时间（毫秒）
            model_info: 模型信息
            
        Returns:
            np.ndarray: 绘制了元数据的图像
        """
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 计算面板高度（根据内容动态调整）
        panel_height = 450
        panel_height = min(panel_height, h - 20)
        
        # 创建半透明背景
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (w - 500, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
        
        # 绘制文本信息
        y_offset = 30
        line_height = 20
        
        # 标题
        cv2.putText(display_frame, "RKNN Metadata", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # 基本信息
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Inference: {inference_time:.2f} ms",
            f"Runtime: {'rknnlite' if USE_RKNN_LITE else 'rknn-toolkit2'}",
            f"Model: {model_info.get('name', 'unknown')}",
            f"Input: {self.input_size[0]}x{self.input_size[1]}",
            "---"
        ]
        
        for line in info_lines:
            cv2.putText(display_frame, line, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # 输出形状信息
        if metadata.get('raw_output_shape'):
            shape_text = f"Output Shape: {metadata['raw_output_shape']}"
            cv2.putText(display_frame, shape_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height
            
            dtype_text = f"Output Dtype: {metadata['raw_output_dtype']}"
            cv2.putText(display_frame, dtype_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height + 5
        
        # 输出统计信息
        if 'output_min' in metadata:
            stats_lines = [
                f"Output Stats:",
                f"  Min: {metadata['output_min']:.4f}",
                f"  Max: {metadata['output_max']:.4f}",
                f"  Mean: {metadata['output_mean']:.4f}",
                f"  Std: {metadata['output_std']:.4f}",
            ]
            for line in stats_lines:
                cv2.putText(display_frame, line, (w - 490, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset += line_height - 3
            y_offset += 3
        
        # 分类结果
        if metadata.get('type') == 'classification':
            class_name = metadata.get('class_name', 'unknown')
            confidence = metadata.get('confidence', 0.0)
            
            # 高亮显示分类结果
            cv2.putText(display_frame, "Classification Result:", (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
            
            result_text = f"  Class: {class_name.upper()} ({metadata['class_id']})"
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.putText(display_frame, result_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += line_height
            
            conf_text = f"  Confidence: {confidence:.3f}"
            cv2.putText(display_frame, conf_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height + 3
            
            # 所有类别的概率
            if 'probabilities' in metadata and 'class_names' in metadata:
                cv2.putText(display_frame, "Probabilities:", (w - 490, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset += line_height - 3
                for cls_name, prob in zip(metadata['class_names'], metadata['probabilities']):
                    prob_text = f"  {cls_name}: {prob:.3f}"
                    cv2.putText(display_frame, prob_text, (w - 490, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                    y_offset += line_height - 5
        
        # 检测结果
        elif metadata.get('type') == 'yolo_detection':
            det_text = f"YOLOv8 Detection: {metadata.get('num_anchors', 0)} anchors"
            cv2.putText(display_frame, det_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            y_offset += line_height
            if 'num_detections' in metadata:
                det_count_text = f"  Detections: {metadata['num_detections']}"
                cv2.putText(display_frame, det_count_text, (w - 490, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset += line_height - 3
        elif metadata.get('type') == 'detection':
            det_text = f"Detections: {metadata.get('num_detections', 0)}"
            cv2.putText(display_frame, det_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            y_offset += line_height
        
        # 控制提示
        y_offset = h - 120
        cv2.putText(display_frame, "Controls:", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        controls = [
            "Q/ESC: Exit",
            "M: Toggle metadata",
            "D: Toggle detections",
            "S: Save frame"
        ]
        for line in controls:
            cv2.putText(display_frame, line, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset += line_height - 3
        
        return display_frame
    
    def initialize_rknn(self, model_path: str, input_size: tuple) -> bool:
        """
        初始化 RKNN 运行时
        
        Args:
            model_path: RKNN 模型文件路径
            input_size: 输入尺寸 (width, height)
            
        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            logger.warning("RKNN 运行时已经初始化")
            return True
        
        if not RKNN_AVAILABLE:
            logger.error("RKNN 运行时未安装")
            return False
        
        try:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"RKNN 模型文件不存在: {self.model_path}")
            
            self.input_size = input_size
            
            # 创建 RKNN 对象
            if USE_RKNN_LITE:
                self.rknn = RKNN_CLASS()
                logger.info("使用 rknnlite (NPU 运行时)")
            else:
                self.rknn = RKNN_CLASS(verbose=False)
                logger.info("使用 rknn-toolkit2 (开发模式)")
            
            # 加载 RKNN 模型
            logger.info("加载 RKNN 模型...")
            ret = self.rknn.load_rknn(str(self.model_path))
            if ret != 0:
                raise RuntimeError(f"加载 RKNN 模型失败，错误代码: {ret}")
            logger.info(f"✓ RKNN 模型加载成功: {self.model_path}")
            
            # 初始化运行时（不指定 target，让系统自动检测）
            logger.info("初始化 RKNN 运行时（自动检测平台）...")
            ret = self.rknn.init_runtime()  # 不指定 target 参数
            
            if ret != 0:
                raise RuntimeError(f"初始化 RKNN 运行时失败，错误代码: {ret}")
            
            self._initialized = True
            logger.info("✓ RKNN 运行时初始化成功")
            
            # 检测是否是 YOLOv8 检测模型（通过首次推理来检测）
            # 这里先设置默认值，实际检测在首次推理时进行
            self.is_detection_model = False
            self.class_names = [
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L',
                'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'
            ]  # 默认21个类别
            
            return True
            
        except Exception as e:
            logger.error(f"初始化 RKNN 运行时失败: {e}")
            if self.rknn:
                try:
                    self.rknn.release()
                except:
                    pass
                self.rknn = None
            return False
    
    def test_rknn_live(self,
                      model_path: str = "models/best.rknn",
                      device_id: int = 0,
                      input_size: tuple = (640, 640)):
        """
        实时测试 RKNN 模型
        
        Args:
            model_path: RKNN 模型文件路径
            device_id: 摄像头设备ID
            input_size: 模型输入尺寸 (width, height)
        """
        print("=" * 60)
        print("RKNN 模型实时摄像头测试（完整元数据）")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"摄像头设备: {device_id}")
        print(f"输入尺寸: {input_size[0]}x{input_size[1]}")
        print()
        
        # 检查 RKNN 是否可用
        if not RKNN_AVAILABLE:
            print("✗ RKNN 运行时未安装，无法使用 RKNN 推理")
            print("\n提示:")
            print("1. 在 RK3588 设备上安装 rknn-toolkit-lite2 (推荐):")
            print("   pip install rknn-toolkit-lite2")
            print("   或: pip install rknnlite")
            print("2. 在开发机/Windows 上安装 rknn-toolkit2 (用于模型转换):")
            print("   从 Rockchip 官方获取: https://github.com/rockchip-linux/rknn-toolkit2")
            return False
        
        try:
            # 初始化 RKNN
            print("初始化 RKNN...")
            if not self.initialize_rknn(model_path, input_size):
                print("✗ RKNN 初始化失败")
                return False
            
            print("✓ RKNN 初始化成功")
            
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
            
            print("\n开始实时推理...")
            print(f"模型输入尺寸: {input_size[0]}x{input_size[1]}")
            print(f"摄像头分辨率: {resolution[0]}x{resolution[1]}")
            print()
            print("控制:")
            print("  Q 或 ESC: 退出")
            print("  M: 切换元数据面板显示")
            print("  D: 切换检测框显示（如果是检测模型）")
            print("  S: 保存当前帧")
            print()
            
            model_info = {
                'name': self.model_path.name,
                'runtime': 'rknnlite' if USE_RKNN_LITE else 'rknn-toolkit2'
            }
            
            while self.running:
                # 捕获帧
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠ 无法捕获图像")
                    break
                
                # 计算FPS
                self.calculate_fps()
                
                # 获取原始图像尺寸（用于坐标转换）
                orig_h, orig_w = frame.shape[:2]
                
                # 预处理
                preprocessed = self.preprocess_image(frame)
                
                # 推理
                start_time = datetime.now()
                outputs = self.rknn.inference(inputs=[preprocessed])
                inference_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # 检测结果
                detections = []
                
                if outputs:
                    self.inference_count += 1
                    output = outputs[0]
                    
                    # 首次运行时检测模型类型
                    if self.output_shape is None:
                        if len(output.shape) == 3 and output.shape[2] == 8400:
                            self.is_detection_model = True
                            num_classes = output.shape[1] - 4
                            logger.info(f"检测到 YOLOv8 检测模型: {num_classes} 个类别")
                    
                    # 如果是检测模型，进行后处理
                    if self.is_detection_model and len(output.shape) == 3 and output.shape[2] == 8400:
                        detections = self.postprocess_detection_output(output, self.input_size[0], self.input_size[1])
                        # 缩放检测框到原始图像尺寸
                        scale_x = orig_w / self.input_size[0]
                        scale_y = orig_h / self.input_size[1]
                        for det in detections:
                            box = det['box']
                            det['box'] = [
                                int(box[0] * scale_x),
                                int(box[1] * scale_y),
                                int(box[2] * scale_x),
                                int(box[3] * scale_y)
                            ]
                    
                    # 解析输出元数据
                    metadata = self.parse_rknn_output(output)
                    
                    # 如果是检测模型，添加检测结果到元数据
                    if self.is_detection_model and detections:
                        metadata['detections'] = detections
                        metadata['num_detections'] = len(detections)
                    
                    # 保存到历史
                    self.result_history.append({
                        'timestamp': datetime.now(),
                        'metadata': metadata,
                        'inference_time': inference_time
                    })
                    
                    # 绘制检测框
                    if self.is_detection_model and self.show_detections and detections:
                        frame = self.draw_detections(frame, detections)
                    
                    # 绘制元数据面板
                    if self.show_metadata:
                        frame = self.draw_metadata_panel(frame, metadata, inference_time, model_info)
                    else:
                        # 至少显示FPS和检测数量
                        det_text = f" | Detections: {len(detections)}" if detections else ""
                        cv2.putText(frame, f"FPS: {self.current_fps:.1f} | Inference: {inference_time:.2f} ms{det_text}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow(self.window_name, frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q 或 ESC
                    break
                elif key == ord('m'):  # 切换元数据面板
                    self.show_metadata = not self.show_metadata
                    print(f"元数据面板: {'开启' if self.show_metadata else '关闭'}")
                elif key == ord('d'):  # 切换检测框显示
                    if self.is_detection_model:
                        self.show_detections = not self.show_detections
                        print(f"检测框: {'开启' if self.show_detections else '关闭'}")
                elif key == ord('s'):  # 保存帧
                    saved_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"rknn_test_frame_{timestamp}_{saved_count:04d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"保存帧: {filename}")
            
            print("\n测试完成")
            print(f"总帧数: {self.frame_count}")
            print(f"总推理次数: {self.inference_count}")
            if self.frame_count > 0:
                print(f"平均FPS: {self.current_fps:.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.rknn and self._initialized:
                try:
                    self.rknn.release()
                except:
                    pass
            if self.camera:
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='RKNN 模型实时摄像头测试（显示完整元数据）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型和摄像头
  python tests/test_rknn_metadata.py
  
  # 指定模型路径和摄像头设备
  python tests/test_rknn_metadata.py --model models/best.rknn --device-id 0
  
  # 指定输入尺寸
  python tests/test_rknn_metadata.py --input-size 640 640
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        default="models/best.rknn",
        help='RKNN 模型文件路径（默认: models/best.rknn）'
    )
    parser.add_argument(
        '--device-id',
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
    
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ 模型文件不存在: {model_path}")
        print("\n提示:")
        print("1. 确保已转换 RKNN 模型")
        print("2. 使用 scripts/convert_to_rknn.py 转换模型")
        print("3. 或指定正确的模型路径: --model path/to/model.rknn")
        sys.exit(1)
    
    ui = RKNNMetadataTestUI()
    success = ui.test_rknn_live(
        model_path=args.model,
        device_id=args.device_id,
        input_size=tuple(args.input_size)
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
