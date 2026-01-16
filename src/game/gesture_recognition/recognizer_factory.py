"""
手势识别器工厂
Gesture Recognizer Factory

支持创建不同类型的识别器（YOLO、RKNN）
"""
from typing import Optional, Union
from pathlib import Path
from ...utils.logger import setup_logger
from .gesture_recognizer import GestureRecognizer
from .rknn_recognizer import RKNNRecognizer, RKNN_AVAILABLE

logger = setup_logger("RPS.RecognizerFactory")


class RecognizerFactory:
    """手势识别器工厂类"""
    
    @staticmethod
    def create_recognizer(
        recognizer_type: str = "yolo",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        min_detection_confidence: float = 0.5,
        use_huggingface_model: bool = True,
        model_size: str = "n",
        device: Optional[str] = None,
        input_size: tuple = (640, 640),
        **kwargs
    ) -> Union[GestureRecognizer, RKNNRecognizer]:
        """
        创建手势识别器
        
        Args:
            recognizer_type: 识别器类型 ("yolo" 或 "rknn")
            model_path: 模型文件路径
            confidence_threshold: 置信度阈值
            min_detection_confidence: 最小检测置信度
            use_huggingface_model: 是否使用 HuggingFace 模型（仅 YOLO）
            model_size: 模型大小（仅 YOLO）
            device: 计算设备（仅 YOLO）
            input_size: 输入尺寸（仅 RKNN）
            **kwargs: 其他参数
        
        Returns:
            GestureRecognizer 或 RKNNRecognizer 实例
        
        Raises:
            ValueError: 不支持的识别器类型
            ImportError: 必要的依赖未安装
        """
        recognizer_type = recognizer_type.lower()
        
        if recognizer_type == "yolo":
            logger.info("创建 YOLO 手势识别器")
            return GestureRecognizer(
                min_detection_confidence=min_detection_confidence,
                confidence_threshold=confidence_threshold,
                model_path=model_path,
                model_size=model_size,
                use_huggingface_model=use_huggingface_model,
                device=device
            )
        
        elif recognizer_type == "rknn":
            if not RKNN_AVAILABLE:
                raise ImportError(
                    "rknn-toolkit2 未安装，无法创建 RKNN 识别器。\n"
                    "请安装: pip install rknn-toolkit2\n"
                    "注意：RKNN Toolkit 2 可能需要从 Rockchip 官方获取"
                )
            
            if model_path is None:
                # 尝试查找默认的 RKNN 模型
                project_root = Path(__file__).parent.parent.parent.parent
                default_paths = [
                    project_root / "models" / "best.rknn",
                    project_root / "models" / "yolov8x-tuned-hand-gestures.rknn",
                ]
                
                for default_path in default_paths:
                    if default_path.exists():
                        model_path = str(default_path)
                        logger.info(f"找到默认 RKNN 模型: {model_path}")
                        break
                
                if model_path is None:
                    raise FileNotFoundError(
                        "未指定 RKNN 模型路径，且未找到默认模型。\n"
                        "请指定 model_path 参数，或确保 models/best.rknn 存在"
                    )
            
            logger.info(f"创建 RKNN 手势识别器: {model_path}")
            recognizer = RKNNRecognizer(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                min_detection_confidence=min_detection_confidence,
                input_size=input_size
            )
            
            # 自动初始化
            if not recognizer.initialize():
                raise RuntimeError("RKNN 识别器初始化失败")
            
            return recognizer
        
        else:
            raise ValueError(
                f"不支持的识别器类型: {recognizer_type}\n"
                f"支持的类型: 'yolo', 'rknn'"
            )
    
    @staticmethod
    def create_from_config(config: dict) -> Union[GestureRecognizer, RKNNRecognizer]:
        """
        从配置字典创建识别器
        
        Args:
            config: 配置字典，包含识别器相关配置
        
        Returns:
            GestureRecognizer 或 RKNNRecognizer 实例
        """
        # 从配置中提取参数
        recognizer_type = config.get('type', 'yolo').lower()
        model_path = config.get('model_path')
        confidence_threshold = config.get('confidence_threshold', 0.7)
        min_detection_confidence = config.get('min_detection_confidence', 0.5)
        
        if recognizer_type == "yolo":
            use_huggingface_model = config.get('use_huggingface_model', True)
            model_size = config.get('model_size', 'n')
            device = config.get('device')
            
            return RecognizerFactory.create_recognizer(
                recognizer_type="yolo",
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                min_detection_confidence=min_detection_confidence,
                use_huggingface_model=use_huggingface_model,
                model_size=model_size,
                device=device
            )
        
        elif recognizer_type == "rknn":
            input_size = config.get('input_size', (640, 640))
            if isinstance(input_size, list):
                input_size = tuple(input_size)
            
            return RecognizerFactory.create_recognizer(
                recognizer_type="rknn",
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                min_detection_confidence=min_detection_confidence,
                input_size=input_size
            )
        
        else:
            raise ValueError(f"不支持的识别器类型: {recognizer_type}")
