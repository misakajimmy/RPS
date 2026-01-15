"""
手势识别模块
Gesture Recognition Module using YOLOv8-Pose
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from ...utils.logger import setup_logger
from ..game_logic.gesture import Gesture

logger = setup_logger("RPS.GestureRecognizer")

# 修复 PyTorch 2.6+ 的 weights_only 问题
# 需要在导入 ultralytics 之前设置
try:
    import torch
    # 添加 ultralytics 相关的安全全局变量，允许加载模型类
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
        except ImportError:
            pass  # 如果无法导入，会在加载模型时处理
except ImportError:
    pass  # PyTorch 未安装，会在后续处理

# 导入 YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLOv8 可用")
except ImportError as e:
    logger.error(f"YOLOv8 未安装: {e}")
    logger.error("请运行: pip install ultralytics>=8.0.0")
    YOLO_AVAILABLE = False
    YOLO = None

# 尝试导入 ultralyticsplus（用于 HuggingFace 模型）
try:
    from ultralyticsplus import YOLO as YOLOPlus
    YOLO_PLUS_AVAILABLE = True
    logger.info("ultralyticsplus 可用，支持 HuggingFace 模型")
except ImportError:
    YOLO_PLUS_AVAILABLE = False
    YOLOPlus = None

# 尝试导入 huggingface_hub（用于下载 HuggingFace 模型）
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
    logger.info("huggingface_hub 可用，支持下载 HuggingFace 模型")
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None


class GestureRecognizer:
    """手势识别器类（基于 YOLOv8-Pose）"""
    
    # YOLOv8-Pose 关键点索引（COCO 格式）
    # 手部关键点从人体关键点中提取
    # COCO Pose 关键点：0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳,
    # 5-左肩, 6-右肩, 7-左肘, 8-右肘, 9-左手腕, 10-右手腕,
    # 11-左髋, 12-右髋, 13-左膝, 14-右膝, 15-左脚踝, 16-右脚踝
    WRIST_LEFT = 9
    WRIST_RIGHT = 10
    ELBOW_LEFT = 7
    ELBOW_RIGHT = 8
    SHOULDER_LEFT = 5
    SHOULDER_RIGHT = 6
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 1,
                 confidence_threshold: float = 0.7,
                 model_path: Optional[str] = None,
                 model_size: str = "n",
                 use_huggingface_model: bool = True,
                 device: Optional[str] = None):  # 是否使用 HuggingFace 手势识别模型
        """
        初始化手势识别器
        
        Args:
            min_detection_confidence: 手势检测的最小置信度
            min_tracking_confidence: 未使用（保留兼容性）
            max_num_hands: 最大检测手部数量（保留兼容性）
            confidence_threshold: 手势识别置信度阈值
            model_path: 模型路径（.pt 文件或 HuggingFace 模型ID），如果为 None 则使用默认模型
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')，仅在 use_huggingface_model=False 时使用
            use_huggingface_model: 是否使用 HuggingFace 手势识别模型（推荐）
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 未安装，请运行: pip install ultralytics>=8.0.0")
        
        self.confidence_threshold = confidence_threshold
        self.min_detection_confidence = min_detection_confidence
        self.use_huggingface_model = use_huggingface_model
        
        # 设备配置（CPU/GPU）
        if device is None:
            # 自动检测设备
            self.device = self._detect_device()
        else:
            self.device = device
        logger.info(f"使用设备: {self.device}")
        
        # 确定模型路径
        if model_path is None:
            if use_huggingface_model:
                # 使用 HuggingFace 手势识别模型
                model_id = "lewiswatson/yolov8x-tuned-hand-gestures"
                logger.info(f"使用 HuggingFace 手势识别模型: {model_id}")
                
                # 首先检查本地 models 目录
                project_root = Path(__file__).parent.parent.parent.parent
                local_model_paths = [
                    project_root / "models" / "yolov8x-tuned-hand-gestures.pt",
                    project_root / "models" / "best.pt",
                    project_root / "models" / "weights" / "best.pt",
                ]
                
                model_file = None
                for local_path in local_model_paths:
                    if local_path.exists():
                        model_file = str(local_path)
                        logger.info(f"找到本地模型文件: {model_file}")
                        break
                
                # 如果本地没有，尝试从 HuggingFace 下载
                if not model_file:
                    model_file = self._download_huggingface_model(model_id)
                    if model_file:
                        logger.info(f"从 HuggingFace 下载模型: {model_file}")
                
                if model_file:
                    # 使用安全加载方法处理 PyTorch 2.6+ 的 weights_only 问题
                    self.model = self._load_yolo_model_safe(model_file)
                    logger.info(f"成功加载模型: {model_file}")
                else:
                    raise ImportError(
                        "无法加载 HuggingFace 模型。请选择以下方式之一：\n"
                        "  1. 手动下载模型到 models/ 目录（推荐）\n"
                        "     下载地址: https://huggingface.co/lewiswatson/yolov8x-tuned-hand-gestures/resolve/main/weights/best.pt\n"
                        "     保存为: models/yolov8x-tuned-hand-gestures.pt 或 models/best.pt\n"
                        "  2. 安装 huggingface_hub 自动下载:\n"
                        "     pip install huggingface_hub\n"
                        "  3. 安装 ultralyticsplus:\n"
                        "     pip install ultralyticsplus"
                    )
                self.model_type = "huggingface_gesture_detection"
            else:
                # 使用默认的 YOLOv8-Pose 模型（会自动下载）
                model_name = f"yolov8{model_size}-pose.pt"
                logger.info(f"使用默认 YOLOv8-Pose 模型: {model_name}")
                self.model = YOLO(model_name)
                self.model_type = "yolov8_pose"
        else:
            # 检查是否是 HuggingFace 模型ID（包含 / 字符且不是文件路径）
            model_path_obj = Path(model_path)
            
            # 如果是相对路径，尝试从项目根目录查找
            if not model_path_obj.is_absolute() and not model_path_obj.exists():
                project_root = Path(__file__).parent.parent.parent.parent
                potential_path = project_root / model_path
                if potential_path.exists():
                    model_path_obj = potential_path
                    model_path = str(potential_path)
            
            if '/' in model_path and not model_path_obj.exists() and not model_path.startswith('models/'):
                # HuggingFace 模型ID
                logger.info(f"加载 HuggingFace 模型: {model_path}")
                # 使用 huggingface_hub 下载模型文件，然后加载
                model_file = self._download_huggingface_model(model_path)
                if model_file:
                    # 使用安全加载方法处理 PyTorch 2.6+ 的 weights_only 问题
                    self.model = self._load_yolo_model_safe(model_file)
                    logger.info(f"从 HuggingFace 下载并加载模型: {model_file}")
                else:
                    raise ImportError(
                        f"无法加载 HuggingFace 模型: {model_path}\n"
                        "请手动下载模型到 models/ 目录，或安装 huggingface_hub:\n"
                        "  pip install huggingface_hub"
                    )
                self.model_type = "huggingface_gesture_detection"
            elif model_path_obj.exists():
                # 本地模型文件
                logger.info(f"加载本地模型: {model_path}")
                self.model = self._load_yolo_model_safe(str(model_path_obj))
                # 根据模型名称判断类型
                if 'pose' in model_path.lower():
                    self.model_type = "yolov8_pose"
                else:
                    self.model_type = "yolov8_detection"
            else:
                # 文件不存在，尝试从默认位置查找
                logger.warning(f"模型文件不存在: {model_path}，尝试从默认位置查找")
                project_root = Path(__file__).parent.parent.parent.parent
                default_paths = [
                    project_root / "models" / "yolov8x-tuned-hand-gestures.pt",
                    project_root / "models" / "best.pt",
                    project_root / "models" / "weights" / "best.pt",
                ]
                
                model_file = None
                for default_path in default_paths:
                    if default_path.exists():
                        model_file = str(default_path)
                        logger.info(f"找到默认模型文件: {model_file}")
                        break
                
                if model_file:
                    self.model = self._load_yolo_model_safe(model_file)
                    self.model_type = "huggingface_gesture_detection"
                else:
                    raise FileNotFoundError(
                        f"模型文件不存在: {model_path}\n"
                        "请手动下载模型到 models/ 目录:\n"
                        "  下载地址: https://huggingface.co/lewiswatson/yolov8x-tuned-hand-gestures/resolve/main/weights/best.pt\n"
                        "  保存为: models/yolov8x-tuned-hand-gestures.pt"
                    )
        
        logger.info(f"手势识别器初始化完成，模型类型: {self.model_type}")
        
        # 将模型移动到正确的设备（如果使用GPU）
        # YOLO 模型在 predict 时会自动使用指定设备，但我们可以显式设置
        if self.device != 'cpu':
            try:
                import torch
                if self.device == 'cuda' and torch.cuda.is_available():
                    # YOLO 会自动处理设备，但我们可以显式设置
                    logger.info(f"模型将使用 GPU 设备: {self.device}")
                elif self.device == 'mps':
                    logger.info(f"模型将使用 Apple Silicon GPU: {self.device}")
            except Exception as e:
                logger.warning(f"设置设备 {self.device} 失败: {e}，将使用 CPU")
                self.device = 'cpu'
    
    def _detect_device(self) -> str:
        """
        自动检测可用的计算设备（GPU/CPU）
        
        Returns:
            str: 设备名称 ('cuda', 'mps', 'cpu')
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"检测到 GPU: {device_name}")
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("检测到 Apple Silicon GPU (MPS)")
                return 'mps'
            else:
                logger.info("未检测到 GPU，使用 CPU")
                return 'cpu'
        except ImportError:
            logger.warning("PyTorch 未安装，使用 CPU")
            return 'cpu'
    
    def _load_yolo_model_safe(self, model_path: str):
        """
        安全加载 YOLO 模型，处理 PyTorch 2.6+ 的 weights_only 限制
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            YOLO 模型实例
        """
        return self._load_model_with_patch(model_path, YOLO)
    
    def _load_yolo_plus_model_safe(self, model_id: str):
        """
        安全加载 YOLOPlus 模型，处理 PyTorch 2.6+ 的 weights_only 限制
        
        Args:
            model_id: HuggingFace 模型ID或路径
            
        Returns:
            YOLOPlus 模型实例
        """
        return self._load_model_with_patch(model_id, YOLOPlus)
    
    def _load_model_with_patch(self, model_path_or_id: str, model_class):
        """
        使用 monkey patch 安全加载模型，处理 PyTorch 2.6+ 的 weights_only 限制
        
        Args:
            model_path_or_id: 模型路径或 HuggingFace 模型ID
            model_class: 模型类（YOLO 或 YOLOPlus）
            
        Returns:
            模型实例
        """
        import torch
        import sys
        import ultralytics.nn.tasks as tasks_module
        
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
            # 根据设备配置加载模型到正确的设备
            # 注意：这里使用 "cpu" 作为默认值，实际设备会在模型初始化后设置
            # YOLO 会自动将模型移动到正确的设备
            return torch.load(weight, map_location="cpu", weights_only=False), weight
        
        # 应用 patch
        torch.load = patched_torch_load
        tasks_module.torch_safe_load = patched_torch_safe_load
        
        try:
            model = model_class(model_path_or_id)
        finally:
            # 恢复原始函数
            torch.load = original_torch_load
            tasks_module.torch_safe_load = original_torch_safe_load
            # 清理兼容性模块（可选）
            # if 'ultralytics.yolo' in sys.modules:
            #     del sys.modules['ultralytics.yolo']
        
        return model
    
    def _download_huggingface_model(self, model_id: str) -> Optional[str]:
        """
        从 HuggingFace 下载模型文件
        
        Args:
            model_id: HuggingFace 模型ID（如 "lewiswatson/yolov8x-tuned-hand-gestures"）
            
        Returns:
            Optional[str]: 下载的模型文件路径，失败返回 None
        """
        if not HF_HUB_AVAILABLE:
            logger.error("huggingface_hub 未安装，无法下载模型")
            logger.error("请运行: pip install huggingface_hub")
            return None
        
        try:
            logger.info(f"正在从 HuggingFace 下载模型: {model_id}")
            
            # 尝试下载模型文件（通常是 .pt 文件）
            # HuggingFace 上的 YOLOv8 模型通常文件名是 "model.pt" 或 "best.pt"
            model_files_to_try = ["model.pt", "best.pt", "yolov8x.pt", "weights.pt"]
            
            for filename in model_files_to_try:
                try:
                    model_path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        cache_dir=None  # 使用默认缓存目录
                    )
                    logger.info(f"成功下载模型文件: {model_path}")
                    return model_path
                except Exception as e:
                    logger.debug(f"尝试下载 {filename} 失败: {e}")
                    continue
            
            # 如果所有文件名都失败，尝试列出仓库文件
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id=model_id)
                logger.info(f"模型仓库中的文件: {files}")
                
                # 查找 .pt 文件
                pt_files = [f for f in files if f.endswith('.pt')]
                if pt_files:
                    model_path = hf_hub_download(
                        repo_id=model_id,
                        filename=pt_files[0],
                        cache_dir=None
                    )
                    logger.info(f"成功下载模型文件: {model_path}")
                    return model_path
            except Exception as e:
                logger.warning(f"列出仓库文件失败: {e}")
            
            logger.error(f"无法从 HuggingFace 下载模型: {model_id}")
            logger.error("请检查模型ID是否正确，或手动下载模型文件")
            return None
            
        except Exception as e:
            logger.error(f"下载 HuggingFace 模型时发生错误: {e}")
            return None
    
    def _extract_hand_region(self, keypoints: np.ndarray, confidences: np.ndarray, image_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        从人体关键点中提取手部区域（改进版，使用更大的区域）
        
        Args:
            keypoints: 关键点坐标数组 (17, 2) - COCO Pose 格式
            confidences: 关键点置信度数组 (17,)
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            Optional[Tuple[int, int, int, int]]: (x, y, width, height) 手部区域，如果检测不到返回 None
        """
        img_h, img_w = image_shape[:2]
        
        # 优先使用右手腕（右手更常见）
        if confidences[self.WRIST_RIGHT] > 0.3:  # 降低置信度阈值
            wrist = keypoints[self.WRIST_RIGHT]
            elbow = keypoints[self.ELBOW_RIGHT] if confidences[self.ELBOW_RIGHT] > 0.3 else None
            shoulder = keypoints[self.SHOULDER_RIGHT] if confidences[self.SHOULDER_RIGHT] > 0.3 else None
            
            # 使用更大的手部区域，提高检测准确度
            if elbow is not None:
                # 根据手腕到肘部的距离估算手部大小
                arm_length = np.linalg.norm(wrist - elbow)
                hand_size = int(max(arm_length * 0.8, 100))  # 增大手部区域，最小100像素
            else:
                hand_size = 120  # 增大默认手部大小
            
            # 确保手部区域不会超出图像边界
            x = max(0, int(wrist[0] - hand_size))
            y = max(0, int(wrist[1] - hand_size))
            w = min(hand_size * 2, img_w - x)
            h = min(hand_size * 2, img_h - y)
            
            # 确保区域有效
            if w > 50 and h > 50:
                return (x, y, w, h)
        
        # 如果没有右手腕，尝试左手腕
        if confidences[self.WRIST_LEFT] > 0.3:
            wrist = keypoints[self.WRIST_LEFT]
            elbow = keypoints[self.ELBOW_LEFT] if confidences[self.ELBOW_LEFT] > 0.3 else None
            
            if elbow is not None:
                arm_length = np.linalg.norm(wrist - elbow)
                hand_size = int(max(arm_length * 0.8, 100))
            else:
                hand_size = 120
            
            x = max(0, int(wrist[0] - hand_size))
            y = max(0, int(wrist[1] - hand_size))
            w = min(hand_size * 2, img_w - x)
            h = min(hand_size * 2, img_h - y)
            
            if w > 50 and h > 50:
                return (x, y, w, h)
        
        return None
    
    def _recognize_gesture_direct(self, image: np.ndarray) -> Tuple[Gesture, float, Dict[str, float]]:
        """
        直接在整个图像中识别手势（当无法检测到人体关键点时使用）
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            Tuple[Gesture, float, Dict]: (识别的手势, 置信度, 所有类别的概率)
        """
        # 使用整个图像进行手势识别
        gesture, confidence = self._detect_fingers_in_hand_region(image, (0, 0, image.shape[1], image.shape[0]))
        
        # 构建概率字典
        prob_dict = {
            'rock': 0.0,
            'paper': 0.0,
            'scissors': 0.0
        }
        
        if gesture != Gesture.UNKNOWN and confidence >= self.confidence_threshold:
            prob_dict[gesture.value] = confidence
            remaining = (1.0 - confidence) / 2.0
            for key in prob_dict:
                if key != gesture.value:
                    prob_dict[key] = remaining
        else:
            gesture = Gesture.UNKNOWN
        
        return gesture, confidence, prob_dict
    
    def _detect_hand_keypoints(self, hand_roi: np.ndarray) -> Optional[Dict]:
        """
        检测手部关键点（模拟21点结构）
        
        Args:
            hand_roi: 手部区域图像（BGR格式）
            
        Returns:
            Optional[Dict]: 包含关键点信息的字典，格式类似 MediaPipe Hands 的21点结构
        """
        if hand_roi.size == 0:
            return None
        
        h, w = hand_roi.shape[:2]
        
        # 转换为HSV进行肤色检测
        hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        min_area = max(500, (w * h) // 50)
        if cv2.contourArea(largest_contour) < min_area:
            return None
        
        # 计算凸包和缺陷
        hull = cv2.convexHull(largest_contour, returnPoints=True)
        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull_indices)
        
        # 找到手腕中心（轮廓底部中心）
        moments = cv2.moments(largest_contour)
        if moments['m00'] == 0:
            return None
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # 找到轮廓最底部的点作为手腕
        bottom_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
        wrist = np.array([bottom_point[0], bottom_point[1]], dtype=np.float32)
        
        # 检测手指指尖（凸包的最高点）
        finger_tips = []
        if len(hull) > 0:
            # 找到凸包上的高点（可能是指尖）
            hull_points = hull.reshape(-1, 2)
            # 按y坐标排序，取前5个最高点
            top_points = sorted(hull_points, key=lambda p: p[1])[:5]
            
            # 过滤掉太靠近的点
            filtered_tips = []
            for tip in top_points:
                if len(filtered_tips) == 0:
                    filtered_tips.append(tip)
                else:
                    # 检查是否与已有点太近
                    too_close = False
                    for existing in filtered_tips:
                        dist = np.linalg.norm(tip - existing)
                        if dist < w * 0.15:  # 至少相距手部宽度的15%
                            too_close = True
                            break
                    if not too_close:
                        filtered_tips.append(tip)
            
            finger_tips = filtered_tips[:5]  # 最多5个指尖
        
        # 构建21点关键点结构（简化版）
        # MediaPipe Hands 21点结构：
        # 0: 手腕, 1-4: 拇指, 5-8: 食指, 9-12: 中指, 13-16: 无名指, 17-20: 小指
        keypoints = {}
        
        # 0: 手腕
        keypoints[0] = {'x': wrist[0] / w, 'y': wrist[1] / h, 'z': 0.0}
        
        # 根据检测到的指尖数量分配手指关键点
        # MediaPipe Hands 21点索引：
        # 0: 手腕
        # 1-4: 拇指 (MCP, IP, DIP, TIP)
        # 5-8: 食指 (MCP, PIP, DIP, TIP)
        # 9-12: 中指 (MCP, PIP, DIP, TIP)
        # 13-16: 无名指 (MCP, PIP, DIP, TIP)
        # 17-20: 小指 (MCP, PIP, DIP, TIP)
        
        finger_base_indices = [1, 5, 9, 13, 17]  # 每个手指的起始索引
        
        for i, tip in enumerate(finger_tips[:5]):  # 最多5个指尖
            finger_base_idx = finger_base_indices[i] if i < len(finger_base_indices) else finger_base_indices[-1]
            
            # 指尖 (TIP)
            keypoints[finger_base_idx + 3] = {
                'x': tip[0] / w,
                'y': tip[1] / h,
                'z': 0.0
            }
            
            # MCP (指根，靠近手腕)
            mcp_x = wrist[0] + (tip[0] - wrist[0]) * 0.3
            mcp_y = wrist[1] + (tip[1] - wrist[1]) * 0.3
            keypoints[finger_base_idx] = {'x': mcp_x / w, 'y': mcp_y / h, 'z': 0.0}
            
            # PIP (中间关节)
            pip_x = wrist[0] + (tip[0] - wrist[0]) * 0.6
            pip_y = wrist[1] + (tip[1] - wrist[1]) * 0.6
            keypoints[finger_base_idx + 1] = {'x': pip_x / w, 'y': pip_y / h, 'z': 0.0}
            
            # DIP (远端关节)
            dip_x = wrist[0] + (tip[0] - wrist[0]) * 0.85
            dip_y = wrist[1] + (tip[1] - wrist[1]) * 0.85
            keypoints[finger_base_idx + 2] = {'x': dip_x / w, 'y': dip_y / h, 'z': 0.0}
        
        return {
            'keypoints': keypoints,
            'finger_count': len(finger_tips),
            'contour': largest_contour,
            'hull': hull
        }
    
    def _detect_fingers_in_hand_region(self, image: np.ndarray, hand_region: Tuple[int, int, int, int]) -> Tuple[Gesture, float]:
        """
        在手部区域内检测手指状态并识别手势（使用改进的21点关键点检测）
        
        Args:
            image: 原始图像（BGR格式）
            hand_region: 手部区域 (x, y, width, height)
            
        Returns:
            Tuple[Gesture, float]: (识别的手势, 置信度)
        """
        x, y, w, h = hand_region
        
        # 裁剪手部区域
        hand_roi = image[y:y+h, x:x+w]
        
        if hand_roi.size == 0:
            return Gesture.UNKNOWN, 0.0
        
        # 检测手部关键点
        hand_data = self._detect_hand_keypoints(hand_roi)
        
        if hand_data is None:
            return Gesture.UNKNOWN, 0.0
        
        finger_count = hand_data['finger_count']
        keypoints = hand_data['keypoints']
        
        # 根据检测到的关键点判断手指状态
        extended_fingers = 0
        
        # 检查每个手指是否伸直（通过关键点位置判断）
        # 拇指 (1-4)
        if 4 in keypoints and 3 in keypoints:
            thumb_tip = keypoints[4]
            thumb_mcp = keypoints[3]
            if thumb_tip['y'] < thumb_mcp['y']:  # 指尖在指根上方
                extended_fingers += 1
        
        # 食指 (5-8)
        if 8 in keypoints and 5 in keypoints:
            index_tip = keypoints[8]
            index_mcp = keypoints[5]
            if index_tip['y'] < index_mcp['y']:
                extended_fingers += 1
        
        # 中指 (9-12)
        if 12 in keypoints and 9 in keypoints:
            middle_tip = keypoints[12]
            middle_mcp = keypoints[9]
            if middle_tip['y'] < middle_mcp['y']:
                extended_fingers += 1
        
        # 无名指 (13-16)
        if 16 in keypoints and 13 in keypoints:
            ring_tip = keypoints[16]
            ring_mcp = keypoints[13]
            if ring_tip['y'] < ring_mcp['y']:
                extended_fingers += 1
        
        # 小指 (17-20)
        if 20 in keypoints and 17 in keypoints:
            pinky_tip = keypoints[20]
            pinky_mcp = keypoints[17]
            if pinky_tip['y'] < pinky_mcp['y']:
                extended_fingers += 1
        
        # 如果关键点检测不完整，使用指尖数量作为备选
        if extended_fingers == 0:
            extended_fingers = finger_count
        
        # 根据伸直的手指数量判断手势
        if extended_fingers == 0:
            return Gesture.ROCK, 0.85
        elif extended_fingers >= 4:
            return Gesture.PAPER, 0.85
        elif extended_fingers == 2:
            # 检查是否是食指和中指
            if 8 in keypoints and 12 in keypoints:
                return Gesture.SCISSORS, 0.85
            else:
                return Gesture.SCISSORS, 0.7
        else:
            return Gesture.UNKNOWN, 0.5
    
    def _map_class_name_to_gesture(self, class_name: str) -> Optional[Gesture]:
        """
        将模型输出的类别名称映射到 Gesture 枚举
        
        Args:
            class_name: 模型输出的类别名称
            
        Returns:
            Optional[Gesture]: 对应的手势，如果无法映射返回 None
        """
        class_name_lower = class_name.lower()
        
        # 石头相关
        if any(keyword in class_name_lower for keyword in ['rock', 'fist', 'closed', '0', 'zero']):
            return Gesture.ROCK
        
        # 布相关
        if any(keyword in class_name_lower for keyword in ['paper', 'open', 'palm', '5', 'five']):
            return Gesture.PAPER
        
        # 剪刀相关
        if any(keyword in class_name_lower for keyword in ['scissors', 'scissor', 'two', '2', 'peace', 'victory']):
            return Gesture.SCISSORS
        
        return None
    
    def recognize(self, image: np.ndarray) -> Tuple[Gesture, float, Dict[str, float]]:
        """
        识别手势
        
        Args:
            image: 输入图像（BGR格式，numpy数组）
            
        Returns:
            Tuple[Gesture, float, Dict]: (识别的手势, 置信度, 所有类别的概率)
        """
        try:
            # 使用 YOLOv8 检测
            results = self.model.predict(
                image,
                conf=self.min_detection_confidence,
                verbose=False,
                device=self.device  # 使用配置的设备（自动检测或手动指定）
            )
            
            # 检查结果是否有效
            if len(results) == 0:
                logger.debug("未检测到任何目标")
                return Gesture.UNKNOWN, 0.0, {
                    'rock': 0.0,
                    'paper': 0.0,
                    'scissors': 0.0
                }
            
            # 如果是 HuggingFace 手势识别模型，直接从检测结果中提取手势类别
            if self.model_type == "huggingface_gesture_detection":
                return self._recognize_from_detection_results(results)
            
            # 如果是 YOLOv8-Pose 模型，使用关键点检测方法
            elif self.model_type == "yolov8_pose":
                return self._recognize_from_pose_results(results, image)
            
            # 其他情况，尝试检测结果
            else:
                # 尝试从检测结果中识别
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    return self._recognize_from_detection_results(results)
                else:
                    # 回退到关键点方法
                    return self._recognize_from_pose_results(results, image)
            
        except Exception as e:
            logger.error(f"手势识别异常: {e}", exc_info=True)
            return Gesture.UNKNOWN, 0.0, {
                'rock': 0.0,
                'paper': 0.0,
                'scissors': 0.0
            }
    
    def _recognize_from_detection_results(self, results) -> Tuple[Gesture, float, Dict[str, float]]:
        """
        从检测结果中识别手势（用于 HuggingFace 手势识别模型）
        
        Args:
            results: YOLOv8 检测结果
            
        Returns:
            Tuple[Gesture, float, Dict]: (识别的手势, 置信度, 所有类别的概率)
        """
        if not hasattr(results[0], 'boxes') or results[0].boxes is None:
            logger.debug("检测结果中没有 boxes")
            return Gesture.UNKNOWN, 0.0, {
                'rock': 0.0,
                'paper': 0.0,
                'scissors': 0.0
            }
        
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            logger.debug("未检测到任何手势")
            return Gesture.UNKNOWN, 0.0, {
                'rock': 0.0,
                'paper': 0.0,
                'scissors': 0.0
            }
        
        # 获取置信度最高的检测结果
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        # 获取模型类别名称
        class_names = results[0].names if hasattr(results[0], 'names') else {}
        
        # 找到置信度最高的检测
        max_idx = np.argmax(confidences)
        best_confidence = float(confidences[max_idx])
        best_class_id = class_ids[max_idx]
        best_class_name = class_names.get(best_class_id, f"class_{best_class_id}")
        
        logger.debug(f"检测到类别: {best_class_name}, 置信度: {best_confidence:.3f}")
        
        # 映射到我们的手势类型
        gesture = self._map_class_name_to_gesture(best_class_name)
        
        if gesture is None:
            logger.debug(f"无法映射类别 '{best_class_name}' 到手势")
            return Gesture.UNKNOWN, 0.0, {
                'rock': 0.0,
                'paper': 0.0,
                'scissors': 0.0
            }
        
        # 如果置信度低于阈值，返回UNKNOWN
        if best_confidence < self.confidence_threshold:
            logger.debug(f"置信度 {best_confidence:.3f} 低于阈值 {self.confidence_threshold}")
            return Gesture.UNKNOWN, 0.0, {
                'rock': 0.0,
                'paper': 0.0,
                'scissors': 0.0
            }
        
        # 构建概率字典
        prob_dict = {
            'rock': 0.0,
            'paper': 0.0,
            'scissors': 0.0
        }
        
        prob_dict[gesture.value] = best_confidence
        
        # 计算其他类别的概率（基于所有检测结果）
        for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
            if i == max_idx:
                continue
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            mapped_gesture = self._map_class_name_to_gesture(cls_name)
            if mapped_gesture and mapped_gesture != gesture:
                # 累加相同手势的概率
                prob_dict[mapped_gesture.value] += conf * 0.3  # 降低次要检测的权重
        
        # 归一化概率
        total_prob = sum(prob_dict.values())
        if total_prob > 0:
            for key in prob_dict:
                prob_dict[key] /= total_prob
        
        logger.debug(f"识别结果: {gesture.value}, 置信度: {best_confidence:.3f}")
        
        return gesture, best_confidence, prob_dict
    
    def _recognize_from_pose_results(self, results, image: np.ndarray) -> Tuple[Gesture, float, Dict[str, float]]:
        """
        从姿态检测结果中识别手势（用于 YOLOv8-Pose 模型）
        
        Args:
            results: YOLOv8-Pose 检测结果
            image: 输入图像
            
        Returns:
            Tuple[Gesture, float, Dict]: (识别的手势, 置信度, 所有类别的概率)
        """
        # 检查关键点是否存在
        if results[0].keypoints is None:
            logger.debug("未检测到人体关键点")
            return self._recognize_gesture_direct(image)
        
        # 检查关键点数据是否为空
        if results[0].keypoints.data is None or len(results[0].keypoints.data) == 0:
            logger.debug("关键点数据为空")
            return self._recognize_gesture_direct(image)
        
        # 获取第一个检测到的人体关键点
        try:
            keypoints_data = results[0].keypoints.data[0]  # (17, 3) - (x, y, confidence)
        except (IndexError, AttributeError) as e:
            logger.debug(f"无法获取关键点数据: {e}")
            return self._recognize_gesture_direct(image)
        
        if keypoints_data is None or len(keypoints_data) < 17:
            logger.debug("关键点数据不完整")
            return self._recognize_gesture_direct(image)
        
        # 提取关键点坐标和置信度
        try:
            keypoints = keypoints_data[:, :2].cpu().numpy()  # (17, 2) - (x, y)
            confidences = keypoints_data[:, 2].cpu().numpy()  # (17,) - confidence
        except Exception as e:
            logger.debug(f"提取关键点失败: {e}")
            return self._recognize_gesture_direct(image)
        
        # 提取手部区域（传入图像尺寸用于边界检查）
        hand_region = self._extract_hand_region(keypoints, confidences, image.shape)
        
        if hand_region is None:
            logger.debug("无法提取手部区域，使用直接检测")
            return self._recognize_gesture_direct(image)
        
        # 在手部区域内识别手势
        gesture, confidence = self._detect_fingers_in_hand_region(image, hand_region)
        
        # 如果置信度低于阈值，返回UNKNOWN
        if confidence < self.confidence_threshold:
            gesture = Gesture.UNKNOWN
            logger.debug(f"置信度 {confidence:.3f} 低于阈值 {self.confidence_threshold}")
        
        # 构建概率字典（基于置信度分配）
        prob_dict = {
            'rock': 0.0,
            'paper': 0.0,
            'scissors': 0.0
        }
        
        if gesture != Gesture.UNKNOWN:
            prob_dict[gesture.value] = confidence
            # 其他类别分配剩余概率
            remaining = (1.0 - confidence) / 2.0
            for key in prob_dict:
                if key != gesture.value:
                    prob_dict[key] = remaining
        
        logger.debug(f"识别结果: {gesture.value}, 置信度: {confidence:.3f}")
        
        return gesture, confidence, prob_dict
    
    def recognize_batch(self, images: list) -> list:
        """
        批量识别手势
        
        Args:
            images: 图像列表
            
        Returns:
            list: 识别结果列表，每个元素为(gesture, confidence, prob_dict)
        """
        results = []
        for image in images:
            result = self.recognize(image)
            results.append(result)
        return results
    
    def recognize_with_landmarks(self, image: np.ndarray) -> Tuple[Gesture, float, Dict[str, float], any]:
        """
        识别手势并返回检测结果（用于绘制）
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            Tuple[Gesture, float, Dict, any]: (识别的手势, 置信度, 概率字典, 检测结果)
        """
        # 使用 YOLOv8 检测
        results = self.model.predict(
            image,
            conf=self.min_detection_confidence,
            verbose=False,
            device=self.device  # 使用配置的设备
        )
        
        # 识别手势
        gesture, confidence, prob_dict = self.recognize(image)
        
        # 提取手部关键点数据（仅用于 Pose 模型）
        hand_keypoints_data = None
        hand_region = None
        
        if self.model_type == "yolov8_pose":
            try:
                if (len(results) > 0 and results[0].keypoints is not None and 
                    results[0].keypoints.data is not None and len(results[0].keypoints.data) > 0):
                    keypoints_data = results[0].keypoints.data[0]
                    keypoints = keypoints_data[:, :2].cpu().numpy()
                    confidences = keypoints_data[:, 2].cpu().numpy()
                    hand_region = self._extract_hand_region(keypoints, confidences, image.shape)
                    
                    if hand_region is not None:
                        x, y, w, h = hand_region
                        hand_roi = image[y:y+h, x:x+w]
                        hand_data = self._detect_hand_keypoints(hand_roi)
                        if hand_data:
                            # 将相对坐标转换为绝对坐标
                            hand_keypoints_data = {}
                            for idx, kp in hand_data['keypoints'].items():
                                hand_keypoints_data[idx] = {
                                    'x': kp['x'] * w + x,
                                    'y': kp['y'] * h + y,
                                    'z': kp['z']
                                }
            except Exception as e:
                logger.debug(f"提取手部关键点数据失败: {e}")
        
        return gesture, confidence, prob_dict, {
            'yolo_results': results,
            'hand_keypoints': hand_keypoints_data,
            'hand_region': hand_region,
            'model_type': self.model_type
        }
    
    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        在图像上绘制检测结果（手势检测框或关键点）
        
        Args:
            image: 输入图像（BGR格式）
            results: 检测结果（来自 recognize_with_landmarks）
            
        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        annotated_image = image.copy()
        
        if results is None:
            return annotated_image
        
        # 获取模型类型
        model_type = results.get('model_type', self.model_type) if isinstance(results, dict) else self.model_type
        
        # 获取 YOLOv8 结果
        if isinstance(results, dict) and 'yolo_results' in results:
            yolo_results = results['yolo_results']
        elif hasattr(results, '__iter__') and len(results) > 0:
            yolo_results = results
        else:
            yolo_results = None
        
        if yolo_results is not None:
            try:
                # 使用 YOLOv8 的绘图功能（会自动绘制检测框和标签）
                annotated_image = yolo_results[0].plot()
                
                # 如果是 Pose 模型，绘制手部21点关键点
                if model_type == "yolov8_pose" and isinstance(results, dict) and 'hand_keypoints' in results:
                    hand_keypoints = results['hand_keypoints']
                    if hand_keypoints:
                        # MediaPipe Hands 连接关系
                        hand_connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
                            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
                            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
                            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
                        ]
                        
                        # 绘制关键点
                        for idx, kp in hand_keypoints.items():
                            x, y = int(kp['x']), int(kp['y'])
                            if 0 <= x < annotated_image.shape[1] and 0 <= y < annotated_image.shape[0]:
                                if idx == 0:
                                    cv2.circle(annotated_image, (x, y), 8, (0, 255, 255), -1)
                                else:
                                    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
                        
                        # 绘制连接线
                        for start_idx, end_idx in hand_connections:
                            if start_idx in hand_keypoints and end_idx in hand_keypoints:
                                start = hand_keypoints[start_idx]
                                end = hand_keypoints[end_idx]
                                start_pt = (int(start['x']), int(start['y']))
                                end_pt = (int(end['x']), int(end['y']))
                                
                                if (0 <= start_pt[0] < annotated_image.shape[1] and 
                                    0 <= start_pt[1] < annotated_image.shape[0] and
                                    0 <= end_pt[0] < annotated_image.shape[1] and 
                                    0 <= end_pt[1] < annotated_image.shape[0]):
                                    cv2.line(annotated_image, start_pt, end_pt, (0, 0, 255), 2)
            except Exception as e:
                logger.debug(f"绘制检测结果失败: {e}")
        
        return annotated_image
    
    def set_confidence_threshold(self, threshold: float):
        """
        设置置信度阈值
        
        Args:
            threshold: 新的阈值（0.0-1.0）
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"置信度阈值已更新: {self.confidence_threshold}")
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息字典
        """
        model_name = 'unknown'
        if hasattr(self.model, 'model_name'):
            model_name = self.model.model_name
        elif hasattr(self.model, 'ckpt_path'):
            model_name = str(self.model.ckpt_path)
        
        return {
            'model_type': self.model_type,
            'model_name': model_name,
            'confidence_threshold': self.confidence_threshold,
            'min_detection_confidence': self.min_detection_confidence,
            'device': self.device,
            'use_huggingface_model': self.use_huggingface_model,
        }
    
    def __del__(self):
        """析构函数，释放资源"""
        # YOLOv8 模型会自动管理资源
        pass
