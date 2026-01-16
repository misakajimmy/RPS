"""
RKNN 手势识别器
RKNN Gesture Recognizer for RK3588 NPU acceleration
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
from ...utils.logger import setup_logger
from ..game_logic.gesture import Gesture

logger = setup_logger("RPS.RKNNRecognizer")

# 尝试导入 RKNN Lite（优先，用于 RK3588 运行时推理）
RKNN_AVAILABLE = False
RKNN_CLASS = None
USE_RKNN_LITE = False

try:
    from rknnlite.api import RKNNLite
    RKNN_CLASS = RKNNLite
    RKNN_AVAILABLE = True
    USE_RKNN_LITE = True
    logger.info("✓ 检测到 rknnlite (rknn-toolkit-lite2)，将使用 NPU 运行时")
except ImportError:
    # 如果 rknnlite 不可用，尝试 rknn-toolkit2（用于开发/转换）
    try:
        from rknn.api import RKNN
        RKNN_CLASS = RKNN
        RKNN_AVAILABLE = True
        USE_RKNN_LITE = False
        logger.info("✓ 检测到 rknn-toolkit2，将使用开发模式（可能使用模拟器）")
    except ImportError:
        RKNN_AVAILABLE = False
        RKNN_CLASS = None
        logger.warning("rknnlite 和 rknn-toolkit2 均未安装，RKNN 推理不可用")
        logger.warning("在 RK3588 上请安装: pip install rknn-toolkit-lite2")
        logger.warning("在开发机上请安装: rknn-toolkit2 (需从 Rockchip 官方获取)")


class RKNNRecognizer:
    """基于 RKNN 的手势识别器（用于 RK3588 NPU 加速）"""
    
    # 手势类别映射（根据 HuggingFace 模型输出）
    GESTURE_CLASSES = {
        0: Gesture.ROCK,      # rock / 石头
        1: Gesture.PAPER,     # paper / 布
        2: Gesture.SCISSORS,  # scissors / 剪刀
    }
    
    def __init__(self,
                 model_path: str,
                 confidence_threshold: float = 0.7,
                 min_detection_confidence: float = 0.5,
                 input_size: Tuple[int, int] = (640, 640)):
        """
        初始化 RKNN 手势识别器
        
        Args:
            model_path: RKNN 模型文件路径（.rknn）
            confidence_threshold: 手势识别置信度阈值
            min_detection_confidence: 检测最小置信度
            input_size: 模型输入尺寸 (width, height)
        """
        # 先初始化属性，确保即使后续出错也能正确清理
        self.rknn = None
        self._initialized = False
        
        if not RKNN_AVAILABLE:
            raise ImportError(
                "RKNN 运行时未安装，无法使用 RKNN 推理。\n"
                "在 RK3588 设备上请安装: pip install rknn-toolkit-lite2\n"
                "在开发机上请安装: rknn-toolkit2 (需从 Rockchip 官方获取)"
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN 模型文件不存在: {self.model_path}")
        
        self.confidence_threshold = confidence_threshold
        self.min_detection_confidence = min_detection_confidence
        self.input_size = input_size
        
        logger.info(f"初始化 RKNN 手势识别器: {self.model_path}")
        logger.info(f"输入尺寸: {input_size[0]}x{input_size[1]}")
        logger.info(f"置信度阈值: {confidence_threshold}")
    
    def initialize(self, target: str = None) -> bool:
        """
        初始化 RKNN 运行时
        
        Args:
            target: 目标平台，'rk3588' 或 None（自动检测）
                   None 时在 Windows 上会使用模拟器模式
        
        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            logger.warning("RKNN 运行时已经初始化")
            return True
        
        try:
            # 创建 RKNN 对象
            if USE_RKNN_LITE:
                # rknnlite 不需要 verbose 参数
                self.rknn = RKNN_CLASS()
                logger.info("使用 rknnlite (NPU 运行时)")
            else:
                # rknn-toolkit2 支持 verbose 参数
                self.rknn = RKNN_CLASS(verbose=False)
                logger.info("使用 rknn-toolkit2 (开发模式)")
            
            # 加载 RKNN 模型
            logger.info("加载 RKNN 模型...")
            ret = self.rknn.load_rknn(str(self.model_path))
            if ret != 0:
                raise RuntimeError(f"加载 RKNN 模型失败，错误代码: {ret}")
            logger.info(f"✓ RKNN 模型加载成功: {self.model_path}")
            
            # 初始化运行时
            logger.info("初始化 RKNN 运行时...")
            
            # 确定目标平台
            if target is None:
                # 自动检测：优先使用 rk3588
                target = 'rk3588'
            
            # 初始化运行时
            # rknnlite 和 rknn-toolkit2 的 init_runtime API 相同
            if USE_RKNN_LITE:
                # rknnlite 在 RK3588 上直接使用 NPU
                ret = self.rknn.init_runtime(target=target)
            else:
                # rknn-toolkit2 可能使用模拟器模式
                ret = self.rknn.init_runtime(target=target)
            
            if ret != 0:
                # 如果初始化失败，尝试其他方式
                import platform
                if platform.system() == 'Windows' and not USE_RKNN_LITE:
                    logger.warning("无法连接到 RK3588 设备，尝试使用模拟器模式...")
                    logger.warning("注意：模拟器模式速度较慢，仅用于验证模型转换是否正确")
                    ret = self.rknn.init_runtime(target='rk3588', perf_debug=True)
                    if ret != 0:
                        raise RuntimeError(f"初始化 RKNN 运行时失败（包括模拟器模式），错误代码: {ret}")
                else:
                    raise RuntimeError(f"初始化 RKNN 运行时失败，错误代码: {ret}")
            
            self._initialized = True
            
            # 检测是否使用模拟器模式
            try:
                # 尝试获取设备信息来判断是否使用模拟器
                if hasattr(self.rknn, 'get_sdk_version'):
                    ret, sdk_version = self.rknn.get_sdk_version()
                    if ret == 0:
                        logger.info(f"✓ RKNN 运行时初始化成功 (SDK: {sdk_version})")
                    else:
                        logger.info("✓ RKNN 运行时初始化成功")
                else:
                    logger.info("✓ RKNN 运行时初始化成功")
            except:
                logger.info("✓ RKNN 运行时初始化成功")
            
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
    
    def recognize(self, image: np.ndarray) -> Tuple[Gesture, float, Dict[str, float]]:
        """
        识别图像中的手势
        
        Args:
            image: 输入图像（BGR 格式，numpy array）
        
        Returns:
            Tuple[Gesture, float, Dict]: (识别的手势, 置信度, 所有类别的概率字典)
        """
        if not self._initialized:
            if not self.initialize():
                return Gesture.UNKNOWN, 0.0, self._get_empty_probabilities()
        
        try:
            # 预处理图像
            preprocessed = self._preprocess_image(image)
            
            # 推理
            outputs = self.rknn.inference(inputs=[preprocessed])
            
            # 后处理
            gesture, confidence, probabilities = self._postprocess_outputs(outputs[0])
            
            return gesture, confidence, probabilities
            
        except Exception as e:
            logger.error(f"手势识别失败: {e}")
            return Gesture.UNKNOWN, 0.0, self._get_empty_probabilities()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以适配模型输入
        
        Args:
            image: 输入图像（BGR 格式）
        
        Returns:
            np.ndarray: 预处理后的图像（NCHW 格式，归一化到 [0, 1]）
        """
        # 调整大小
        resized = cv2.resize(image, self.input_size)
        
        # BGR 转 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # HWC 转 CHW
        chw = rgb.transpose(2, 0, 1)
        
        # 归一化到 [0, 1]（YOLOv8 通常使用 [0, 255] 范围，但这里归一化）
        normalized = chw.astype(np.float32) / 255.0
        
        # 添加 batch 维度
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def _postprocess_outputs(self, output: np.ndarray) -> Tuple[Gesture, float, Dict[str, float]]:
        """
        后处理模型输出
        
        Args:
            output: 模型输出（形状取决于模型，通常是 [batch, num_classes] 或检测结果）
        
        Returns:
            Tuple[Gesture, float, Dict]: (识别的手势, 置信度, 所有类别的概率字典)
        """
        # 根据模型输出格式处理
        # HuggingFace YOLOv8x 手势识别模型通常输出检测结果
        # 格式可能是: [batch, num_detections, 6] (x, y, w, h, conf, class)
        # 或者: [batch, num_classes] (分类结果)
        
        output_shape = output.shape
        logger.debug(f"模型输出形状: {output_shape}")
        
        # 尝试不同的输出格式
        if len(output_shape) == 2:
            # 分类输出: [batch, num_classes]
            if output_shape[1] >= 3:
                # 取第一个 batch 的结果
                probs = output[0]
                # 应用 softmax（如果模型没有应用）
                probs = np.exp(probs - np.max(probs))
                probs = probs / np.sum(probs)
                
                # 获取最大概率的类别
                class_id = np.argmax(probs)
                confidence = float(probs[class_id])
                
                # 映射到手势
                gesture = self.GESTURE_CLASSES.get(class_id, Gesture.UNKNOWN)
                
                # 构建概率字典
                probabilities = {
                    'rock': float(probs[0]) if len(probs) > 0 else 0.0,
                    'paper': float(probs[1]) if len(probs) > 1 else 0.0,
                    'scissors': float(probs[2]) if len(probs) > 2 else 0.0,
                }
                
                return gesture, confidence, probabilities
        
        elif len(output_shape) == 3:
            # 检测输出: [batch, num_detections, 6] 或类似格式
            # 取第一个 batch
            detections = output[0]
            
            if len(detections) == 0:
                return Gesture.UNKNOWN, 0.0, self._get_empty_probabilities()
            
            # 找到置信度最高的检测结果
            # 假设格式是 [x, y, w, h, conf, class] 或 [conf, class, ...]
            best_detection = None
            best_conf = 0.0
            
            for det in detections:
                if len(det) >= 6:
                    # 格式: [x, y, w, h, conf, class]
                    conf = float(det[4])
                    class_id = int(det[5])
                elif len(det) >= 2:
                    # 格式: [conf, class]
                    conf = float(det[0])
                    class_id = int(det[1])
                else:
                    continue
                
                if conf > best_conf and conf >= self.min_detection_confidence:
                    best_conf = conf
                    best_detection = (class_id, conf)
            
            if best_detection:
                class_id, confidence = best_detection
                gesture = self.GESTURE_CLASSES.get(class_id, Gesture.UNKNOWN)
                
                # 构建概率字典（简化版，只设置检测到的类别）
                probabilities = self._get_empty_probabilities()
                if gesture == Gesture.ROCK:
                    probabilities['rock'] = confidence
                elif gesture == Gesture.PAPER:
                    probabilities['paper'] = confidence
                elif gesture == Gesture.SCISSORS:
                    probabilities['scissors'] = confidence
                
                return gesture, confidence, probabilities
        
        # 未知输出格式
        logger.warning(f"未知的模型输出格式: {output_shape}")
        return Gesture.UNKNOWN, 0.0, self._get_empty_probabilities()
    
    def _get_empty_probabilities(self) -> Dict[str, float]:
        """获取空的概率字典"""
        return {
            'rock': 0.0,
            'paper': 0.0,
            'scissors': 0.0
        }
    
    def release(self):
        """释放 RKNN 运行时资源"""
        if hasattr(self, 'rknn') and self.rknn is not None and self._initialized:
            try:
                self.rknn.release()
                self._initialized = False
                logger.info("RKNN 运行时资源已释放")
            except Exception as e:
                logger.warning(f"释放 RKNN 资源时出错: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()
    
    def __del__(self):
        """析构函数"""
        self.release()
