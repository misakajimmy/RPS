#!/usr/bin/env python3
"""
ONNX 模型实时摄像头测试（显示完整元数据）
ONNX Model Real-time Camera Test with Full Metadata Display
"""
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import deque

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("错误: onnxruntime 未安装")
    print("请运行: pip install onnxruntime")
    print("或使用 GPU 版本: pip install onnxruntime-gpu")
    sys.exit(1)

from src.hardware.implementations.camera import USBCamera
from src.utils.logger import setup_logger

logger = setup_logger("RPS.ONNXTest")


class ONNXMetadataTestUI:
    """ONNX 模型元数据测试UI类"""
    
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.camera: Optional[USBCamera] = None
        self.window_name = "ONNX Metadata Test"
        self.running = False
        
        # 模型信息
        self.model_path: Optional[Path] = None
        self.input_name: str = ""
        self.output_names: List[str] = []
        self.input_shape: Tuple[int, ...] = ()
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}
        self.input_dtype: np.dtype = np.float32
        self.output_dtypes: Dict[str, np.dtype] = {}
        
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
        
        # 执行提供者（CPU/GPU）
        self.providers: List[str] = []
        
        # 检测模型相关参数
        self.is_detection_model = False
        self.detection_confidence_threshold = 0.25
        self.nms_iou_threshold = 0.4
        self.class_names: List[str] = []
        self.input_image_size = (640, 640)  # 模型输入尺寸
        
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
    
    def detect_available_providers(self) -> List[str]:
        """
        检测可用的执行提供者
        
        Returns:
            List[str]: 可用的提供者列表
        """
        available = ['CPUExecutionProvider']  # CPU 总是可用
        
        # 检测 CUDA 支持
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            try:
                # 尝试创建临时会话测试 CUDA
                test_session = ort.InferenceSession(
                    self.model_path,
                    providers=['CUDAExecutionProvider'],
                    sess_options=ort.SessionOptions()
                )
                test_session.run(None, {})  # 这会失败，但会初始化提供者
                available.insert(0, 'CUDAExecutionProvider')
                test_session = None
            except:
                pass
        
        # 检测 TensorRT 支持
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            available.insert(0, 'TensorrtExecutionProvider')
        
        return available
    
    def load_onnx_model(self, model_path: str) -> bool:
        """
        加载 ONNX 模型
        
        Args:
            model_path: ONNX 模型文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"ONNX 模型文件不存在: {self.model_path}")
            
            # 检测可用的执行提供者
            self.providers = self.detect_available_providers()
            logger.info(f"可用的执行提供者: {self.providers}")
            
            # 创建推理会话
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.providers
            )
            
            # 获取输入输出信息
            meta = self.session.get_modelmeta()
            
            # 输入信息
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_shape = tuple(input_meta.shape)
            # ONNX 输入形状可能包含动态维度（用字符串或 -1 表示）
            # 我们需要处理这种情况
            self.input_shape = tuple(
                dim if isinstance(dim, int) and dim > 0 else 1 
                for dim in self.input_shape
            )
            
            # 尝试从类型信息获取数据类型
            if hasattr(input_meta, 'type') and hasattr(input_meta.type, 'tensor_type'):
                tensor_type = input_meta.type.tensor_type
                if tensor_type.elem_type:
                    type_map = {
                        1: np.float32,  # FLOAT
                        2: np.uint8,    # UINT8
                        3: np.int8,     # INT8
                        4: np.uint16,   # UINT16
                        5: np.int16,    # INT16
                        6: np.int32,    # INT32
                        7: np.int64,    # INT64
                        8: np.str_,     # STRING
                        9: np.bool_,    # BOOL
                        10: np.float16, # FLOAT16
                        11: np.float64, # DOUBLE
                        12: np.uint32,  # UINT32
                        13: np.uint64,  # UINT64
                    }
                    self.input_dtype = type_map.get(tensor_type.elem_type, np.float32)
            else:
                self.input_dtype = np.float32  # 默认
            
            # 输出信息
            self.output_names = [output.name for output in self.session.get_outputs()]
            for output in self.session.get_outputs():
                output_shape = tuple(
                    dim if isinstance(dim, int) and dim > 0 else 1
                    for dim in output.shape
                )
                self.output_shapes[output.name] = output_shape
                # 获取输出数据类型（类似输入）
                if hasattr(output, 'type') and hasattr(output.type, 'tensor_type'):
                    tensor_type = output.type.tensor_type
                    if tensor_type.elem_type:
                        type_map = {
                            1: np.float32, 2: np.uint8, 3: np.int8,
                            4: np.uint16, 5: np.int16, 6: np.int32,
                            7: np.int64, 8: np.str_, 9: np.bool_,
                            10: np.float16, 11: np.float64,
                            12: np.uint32, 13: np.uint64,
                        }
                        self.output_dtypes[output.name] = type_map.get(
                            tensor_type.elem_type, np.float32
                        )
                    else:
                        self.output_dtypes[output.name] = np.float32
                else:
                    self.output_dtypes[output.name] = np.float32
            
            # 检测是否是 YOLOv8 检测模型
            self._detect_model_type()
            
            logger.info(f"模型加载成功")
            logger.info(f"输入: {self.input_name}, 形状: {self.input_shape}, 类型: {self.input_dtype}")
            logger.info(f"输出: {self.output_names}, 形状: {self.output_shapes}")
            logger.info(f"模型类型: {'检测模型' if self.is_detection_model else '分类模型'}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载 ONNX 模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_model_type(self):
        """检测模型类型（检测模型或分类模型）"""
        # 检查输出形状，如果是 (1, 25, 8400) 或类似格式，是检测模型
        for output_name, output_shape in self.output_shapes.items():
            if len(output_shape) == 3 and output_shape[2] == 8400:
                # YOLOv8 检测模型的特征：第三维度是 8400
                self.is_detection_model = True
                # 计算类别数量：25 - 4 (坐标) = 21
                num_classes = output_shape[1] - 4
                # 默认类别名称（可以根据模型调整）
                self.class_names = [
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L',
                    'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'
                ]
                if num_classes != len(self.class_names):
                    # 如果类别数量不匹配，使用通用名称
                    self.class_names = [f'Class_{i}' for i in range(num_classes)]
                logger.info(f"检测到 YOLOv8 检测模型: {num_classes} 个类别")
                return
        
        self.is_detection_model = False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以适配模型输入
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 从输入形状推断图像尺寸
        # 通常 ONNX 输入形状是 [batch, channels, height, width] 或 [batch, height, width, channels]
        if len(self.input_shape) == 4:
            # [N, C, H, W] 或 [N, H, W, C]
            # 通常 YOLO 模型是 [1, 3, 640, 640]
            if self.input_shape[1] == 3:  # NCHW
                h, w = self.input_shape[2], self.input_shape[3]
            else:  # NHWC
                h, w = self.input_shape[1], self.input_shape[2]
        else:
            # 默认使用 640x640
            h, w = 640, 640
        
        # 调整大小
        resized = cv2.resize(image, (w, h))
        
        # BGR 转 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 检查输入格式
        if len(self.input_shape) == 4:
            if self.input_shape[1] == 3:  # NCHW: [1, 3, H, W]
                # HWC 转 CHW
                chw = rgb.transpose(2, 0, 1)
                # 归一化到 [0, 1]
                normalized = chw.astype(np.float32) / 255.0
                # 添加 batch 维度
                batched = np.expand_dims(normalized, axis=0)
                return batched
            else:  # NHWC: [1, H, W, 3]
                # 归一化到 [0, 1]
                normalized = rgb.astype(np.float32) / 255.0
                # 添加 batch 维度
                batched = np.expand_dims(normalized, axis=0)
                return batched
        else:
            # 简单处理：转换为 [1, H, W, 3]
            normalized = rgb.astype(np.float32) / 255.0
            return np.expand_dims(normalized, axis=0)
    
    def postprocess_detection_output(self, output: np.ndarray, 
                                    img_width: int, img_height: int) -> List[Dict]:
        """
        后处理 YOLOv8 检测模型输出
        
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
            
            # 转换为绝对坐标（简化版，实际YOLOv8更复杂）
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
    
    def parse_onnx_output(self, outputs: Dict[str, np.ndarray]) -> Dict:
        """
        解析 ONNX 推理输出，提取完整的元数据
        
        Args:
            outputs: ONNX 推理输出字典
            
        Returns:
            Dict: 包含完整元数据的字典
        """
        metadata = {
            'num_outputs': len(outputs),
            'output_names': list(outputs.keys()),
            'outputs': {}
        }
        
        for output_name, output in outputs.items():
            output_flat = output.flatten()
            
            output_meta = {
                'shape': list(output.shape),
                'dtype': str(output.dtype),
                'size': int(output.size),
                'min': float(np.min(output_flat)),
                'max': float(np.max(output_flat)),
                'mean': float(np.mean(output_flat)),
                'std': float(np.std(output_flat)),
            }
            
            # 尝试解析分类输出
            if len(output.shape) == 1 or (len(output.shape) == 2 and output.shape[0] == 1):
                if len(output.shape) == 2:
                    probs = output[0]
                else:
                    probs = output
                
                # 应用 softmax
                probs = np.exp(probs - np.max(probs))
                probs = probs / np.sum(probs)
                
                class_id = int(np.argmax(probs))
                confidence = float(probs[class_id])
                
                class_names = ['rock', 'paper', 'scissors']
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f'class_{class_id}'
                
                output_meta.update({
                    'type': 'classification',
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
                output_meta.update({
                    'type': 'yolo_detection',
                    'num_anchors': output.shape[2],  # 8400
                    'num_classes': num_classes,
                    'coord_dim': 4,
                    'detection_format': 'yolov8'
                })
            elif len(output.shape) >= 2:
                output_meta.update({
                    'type': 'detection',
                    'num_detections': output.shape[1] if len(output.shape) >= 2 else 1,
                    'detection_dim': output.shape[2] if len(output.shape) >= 3 else output.shape[1]
                })
            else:
                output_meta['type'] = 'unknown'
            
            metadata['outputs'][output_name] = output_meta
        
        return metadata
    
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
        panel_height = 500
        panel_height = min(panel_height, h - 20)
        
        # 创建半透明背景
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (w - 500, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
        
        # 绘制文本信息
        y_offset = 30
        line_height = 18
        
        # 标题
        cv2.putText(display_frame, "ONNX Metadata", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # 基本信息
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Inference: {inference_time:.2f} ms",
            f"Provider: {self.providers[0] if self.providers else 'Unknown'}",
            f"Model: {model_info.get('name', 'unknown')}",
            f"Input: {self.input_shape}",
            "---"
        ]
        
        for line in info_lines:
            cv2.putText(display_frame, line, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += line_height
        
        # 输入信息
        if self.show_input_info:
            cv2.putText(display_frame, f"Input: {self.input_name}", (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            y_offset += line_height
            cv2.putText(display_frame, f"  Shape: {list(self.input_shape)}", (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += line_height
            cv2.putText(display_frame, f"  Dtype: {self.input_dtype}", (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += line_height + 3
        
        # 输出信息
        for output_name, output_meta in metadata['outputs'].items():
            if y_offset + 150 > panel_height:
                break
                
            cv2.putText(display_frame, f"Output: {output_name}", (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)
            y_offset += line_height
            
            shape_text = f"  Shape: {output_meta['shape']}"
            cv2.putText(display_frame, shape_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += line_height - 2
            
            dtype_text = f"  Dtype: {output_meta['dtype']}"
            cv2.putText(display_frame, dtype_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += line_height - 2
            
            # 统计信息
            stats_text = f"  Min: {output_meta['min']:.4f} Max: {output_meta['max']:.4f}"
            cv2.putText(display_frame, stats_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y_offset += line_height - 2
            
            mean_std_text = f"  Mean: {output_meta['mean']:.4f} Std: {output_meta['std']:.4f}"
            cv2.putText(display_frame, mean_std_text, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            y_offset += line_height + 2
            
            # 分类结果
            if output_meta.get('type') == 'classification':
                class_name = output_meta.get('class_name', 'unknown')
                confidence = output_meta.get('confidence', 0.0)
                
                result_text = f"  Class: {class_name.upper()} ({output_meta['class_id']})"
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.putText(display_frame, result_text, (w - 490, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += line_height - 2
                
                conf_text = f"  Confidence: {confidence:.3f}"
                cv2.putText(display_frame, conf_text, (w - 490, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += line_height - 2
                
                # 概率（简化显示，只显示主要类别）
                if 'probabilities' in output_meta and 'class_names' in output_meta:
                    for cls_name, prob in zip(output_meta['class_names'][:3], 
                                            output_meta['probabilities'][:3]):
                        if prob > 0.1:  # 只显示概率大于 0.1 的类别
                            prob_text = f"    {cls_name}: {prob:.3f}"
                            cv2.putText(display_frame, prob_text, (w - 490, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                            y_offset += line_height - 4
                y_offset += 3
        
        # 控制提示
        y_offset = h - 100
        cv2.putText(display_frame, "Controls:", (w - 490, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        controls = [
            "Q/ESC: Exit",
            "M: Toggle metadata",
            "I: Toggle input info",
            "D: Toggle detections",
            "S: Save frame"
        ]
        for line in controls:
            cv2.putText(display_frame, line, (w - 490, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset += line_height - 3
        
        return display_frame
    
    def test_onnx_live(self,
                      model_path: str = "models/best.onnx",
                      device_id: int = 0):
        """
        实时测试 ONNX 模型
        
        Args:
            model_path: ONNX 模型文件路径
            device_id: 摄像头设备ID
        """
        print("=" * 60)
        print("ONNX 模型实时摄像头测试（完整元数据）")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"摄像头设备: {device_id}")
        print()
        
        # 检查 onnxruntime 是否可用
        if not ONNXRUNTIME_AVAILABLE:
            print("✗ onnxruntime 未安装")
            print("请运行: pip install onnxruntime")
            print("或使用 GPU 版本: pip install onnxruntime-gpu")
            return False
        
        try:
            # 加载模型
            print("加载 ONNX 模型...")
            if not self.load_onnx_model(model_path):
                print("✗ ONNX 模型加载失败")
                return False
            
            print("✓ ONNX 模型加载成功")
            
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
            print(f"模型输入形状: {self.input_shape}")
            print(f"模型输出: {self.output_names}")
            print(f"摄像头分辨率: {resolution[0]}x{resolution[1]}")
            print(f"执行提供者: {self.providers[0] if self.providers else 'Unknown'}")
            print()
            print("控制:")
            print("  Q 或 ESC: 退出")
            print("  M: 切换元数据面板显示")
            print("  I: 切换输入信息显示")
            print("  S: 保存当前帧")
            print()
            
            model_info = {
                'name': self.model_path.name
            }
            
            while self.running:
                # 捕获帧
                frame = self.camera.capture_frame()
                if frame is None:
                    print("⚠ 无法捕获图像")
                    break
                
                # 计算FPS
                self.calculate_fps()
                
                # 获取原始图像尺寸
                orig_h, orig_w = frame.shape[:2]
                self.input_image_size = (orig_w, orig_h)
                
                # 预处理
                preprocessed = self.preprocess_image(frame)
                
                # 获取模型输入尺寸（用于坐标转换）
                if len(self.input_shape) == 4:
                    if self.input_shape[1] == 3:  # NCHW
                        model_h, model_w = self.input_shape[2], self.input_shape[3]
                    else:  # NHWC
                        model_h, model_w = self.input_shape[1], self.input_shape[2]
                else:
                    model_w, model_h = 640, 640
                
                # 推理
                start_time = datetime.now()
                outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
                inference_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # 将输出转换为字典
                outputs_dict = {name: output for name, output in zip(self.output_names, outputs)}
                
                # 检测结果
                detections = []
                
                if outputs_dict:
                    self.inference_count += 1
                    
                    # 如果是检测模型，进行后处理
                    if self.is_detection_model:
                        for output_name, output in outputs_dict.items():
                            if len(output.shape) == 3 and output.shape[2] == 8400:
                                detections = self.postprocess_detection_output(output, model_w, model_h)
                                # 缩放检测框到原始图像尺寸
                                scale_x = orig_w / model_w
                                scale_y = orig_h / model_h
                                for det in detections:
                                    box = det['box']
                                    det['box'] = [
                                        int(box[0] * scale_x),
                                        int(box[1] * scale_y),
                                        int(box[2] * scale_x),
                                        int(box[3] * scale_y)
                                    ]
                                break
                    
                    # 解析输出元数据
                    metadata = self.parse_onnx_output(outputs_dict)
                    
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
                elif key == ord('i'):  # 切换输入信息
                    self.show_input_info = not self.show_input_info
                    print(f"输入信息: {'开启' if self.show_input_info else '关闭'}")
                elif key == ord('d'):  # 切换检测框显示
                    if self.is_detection_model:
                        self.show_detections = not self.show_detections
                        print(f"检测框: {'开启' if self.show_detections else '关闭'}")
                elif key == ord('s'):  # 保存帧
                    saved_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"onnx_test_frame_{timestamp}_{saved_count:04d}.jpg"
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
            if self.camera:
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='ONNX 模型实时摄像头测试（显示完整元数据）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认模型和摄像头
  python tests/test_onnx_metadata.py
  
  # 指定模型路径和摄像头设备
  python tests/test_onnx_metadata.py --model models/best.onnx --device-id 0
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        default="models/best.onnx",
        help='ONNX 模型文件路径（默认: models/best.onnx）'
    )
    parser.add_argument(
        '--device-id',
        type=int,
        default=0,
        help='摄像头设备ID（默认: 0）'
    )
    
    args = parser.parse_args()
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ 模型文件不存在: {model_path}")
        print("\n提示:")
        print("1. 确保模型文件路径正确")
        print("2. 默认路径: models/best.onnx")
        print("3. 可以使用 scripts/export_onnx.py 从 PyTorch 模型导出 ONNX")
        sys.exit(1)
    
    ui = ONNXMetadataTestUI()
    success = ui.test_onnx_live(
        model_path=args.model,
        device_id=args.device_id
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
