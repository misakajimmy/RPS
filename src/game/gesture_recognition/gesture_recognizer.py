"""
手势识别模块
Gesture Recognition Module using MediaPipe
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from ...utils.logger import setup_logger
from ..game_logic.gesture import Gesture

logger = setup_logger("RPS.GestureRecognizer")

# 导入MediaPipe，兼容新旧版本
try:
    import mediapipe as mp
    
    # 尝试新版本Tasks API (0.10.31+)
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, HandLandmarksConnections
        from mediapipe.tasks.python import BaseOptions
        from mediapipe import Image as MPImage
        MP_USE_TASKS_API = True
        MP_VISION = vision
        MP_HAND_LANDMARKER = HandLandmarker
        MP_HAND_LANDMARKER_OPTIONS = HandLandmarkerOptions
        MP_HAND_CONNECTIONS = HandLandmarksConnections
        MP_BASE_OPTIONS = BaseOptions
        logger.info("使用MediaPipe Tasks API (新版本)")
    except ImportError:
        MP_USE_TASKS_API = False
        # 尝试旧版本API
        try:
            from mediapipe.python.solutions import hands
            from mediapipe.python.solutions import drawing_utils
            mp_hands = hands
            mp_drawing = drawing_utils
            MP_API_TYPE = "python_solutions"
            logger.info("使用MediaPipe旧API: mediapipe.python.solutions")
        except ImportError:
            try:
                mp_hands = mp.solutions.hands
                mp_drawing = mp.solutions.drawing_utils
                MP_API_TYPE = "old_solutions"
                logger.info("使用MediaPipe旧API: mp.solutions")
            except AttributeError:
                raise ImportError("无法导入MediaPipe，请检查安装")
        
except ImportError as e:
    raise ImportError(f"MediaPipe未正确安装: {e}")


class GestureRecognizer:
    """手势识别器类（基于MediaPipe）"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 1,
                 confidence_threshold: float = 0.7,
                 model_path: Optional[str] = None):
        """
        初始化手势识别器
        
        Args:
            min_detection_confidence: 手部检测的最小置信度
            min_tracking_confidence: 手部跟踪的最小置信度
            max_num_hands: 最大检测手部数量
            confidence_threshold: 手势识别置信度阈值
            model_path: MediaPipe Tasks 手势模型路径（.task），默认为 models/hand_landmarker.task
        """
        self.confidence_threshold = confidence_threshold
        
        # 模型路径（仅在 Tasks API 下使用）
        if model_path is None:
            model_path = "models/hand_landmarker.task"
        self.model_path = model_path
        
        if MP_USE_TASKS_API:
            # 使用新的Tasks API
            base_options = MP_BASE_OPTIONS(model_asset_path=self.model_path)
            options = MP_HAND_LANDMARKER_OPTIONS(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.hand_landmarker = MP_HAND_LANDMARKER.create_from_options(options)
            self.use_tasks_api = True
            logger.info("MediaPipe Tasks API手势识别器初始化完成")
        else:
            # 使用旧API
            self.mp_hands = mp_hands
            self.mp_drawing = mp_drawing
            self.mp_api_type = MP_API_TYPE
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.use_tasks_api = False
            logger.info(f"MediaPipe旧API手势识别器初始化完成 (API类型: {MP_API_TYPE})")
    
    def _calculate_finger_state(self, landmarks):
        """
        计算手指状态（伸直或弯曲）
        
        Args:
            landmarks: MediaPipe手部关键点列表（21个点）
            
        Returns:
            list: 5个手指的状态，True表示伸直，False表示弯曲
        """
        finger_states = []
        
        # 拇指（特殊处理，需要比较x坐标）
        if landmarks[4].x > landmarks[3].x:  # 右手
            thumb_up = landmarks[4].y < landmarks[3].y
        else:  # 左手
            thumb_up = landmarks[4].y < landmarks[3].y
        
        finger_states.append(thumb_up)
        
        # 其他四指（比较y坐标，指尖低于指根表示伸直）
        finger_tips = [8, 12, 16, 20]  # 食指、中指、无名指、小指的指尖
        finger_pips = [6, 10, 14, 18]  # 对应的指根
        
        for tip, pip in zip(finger_tips, finger_pips):
            finger_states.append(landmarks[tip].y < landmarks[pip].y)
        
        return finger_states
    
    def _recognize_gesture_from_landmarks(self, landmarks) -> Tuple[Gesture, float]:
        """
        从手部关键点识别手势
        
        Args:
            landmarks: MediaPipe手部关键点列表
            
        Returns:
            Tuple[Gesture, float]: (识别的手势, 置信度)
        """
        finger_states = self._calculate_finger_state(landmarks)
        
        # 统计伸直的手指数量
        extended_fingers = sum(finger_states)
        
        # 手势判断逻辑
        # 石头：所有手指弯曲（0个伸直）
        # 布：所有手指伸直（5个伸直）
        # 剪刀：食指和中指伸直，其他弯曲（2个伸直，且是食指和中指）
        
        if extended_fingers == 0:
            # 所有手指弯曲 -> 石头
            return Gesture.ROCK, 0.9
        
        elif extended_fingers == 5:
            # 所有手指伸直 -> 布
            return Gesture.PAPER, 0.9
        
        elif extended_fingers == 2:
            # 检查是否是食指和中指
            if finger_states[1] and finger_states[2] and not any(finger_states[3:]):
                # 食指和中指伸直，其他弯曲 -> 剪刀
                return Gesture.SCISSORS, 0.9
            else:
                # 其他两根手指的组合，可能是误识别
                return Gesture.UNKNOWN, 0.3
        
        else:
            # 其他情况，可能是过渡状态
            return Gesture.UNKNOWN, 0.5
    
    def recognize(self, image: np.ndarray) -> Tuple[Gesture, float, Dict[str, float]]:
        """
        识别手势
        
        Args:
            image: 输入图像（BGR格式，numpy数组）
            
        Returns:
            Tuple[Gesture, float, Dict]: (识别的手势, 置信度, 所有类别的概率)
        """
        try:
            if self.use_tasks_api:
                # 使用新Tasks API
                # 转换为RGB
                if len(image.shape) == 3:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # 创建MediaPipe Image对象
                mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # 检测手部关键点
                detection_result = self.hand_landmarker.detect(mp_image)
                
                # 检查是否检测到手部
                if not detection_result.hand_landmarks:
                    logger.debug("未检测到手部")
                    return Gesture.UNKNOWN, 0.0, {
                        'rock': 0.0,
                        'paper': 0.0,
                        'scissors': 0.0
                    }
                
                # 使用第一只检测到的手
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # 转换为landmark对象列表（兼容旧API格式）
                landmarks = [type('Landmark', (), {'x': lm.x, 'y': lm.y, 'z': lm.z})() 
                            for lm in hand_landmarks]
                
                # 识别手势
                gesture, confidence = self._recognize_gesture_from_landmarks(landmarks)
                
            else:
                # 使用旧API
                # 转换为RGB
                if len(image.shape) == 3:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # 处理图像
                results = self.hands.process(rgb_image)
                
                # 检查是否检测到手部
                if not results.multi_hand_landmarks:
                    logger.debug("未检测到手部")
                    return Gesture.UNKNOWN, 0.0, {
                        'rock': 0.0,
                        'paper': 0.0,
                        'scissors': 0.0
                    }
                
                # 使用第一只检测到的手
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # 识别手势
                gesture, confidence = self._recognize_gesture_from_landmarks(hand_landmarks.landmark)
            
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
            
        except Exception as e:
            logger.error(f"手势识别异常: {e}", exc_info=True)
            return Gesture.UNKNOWN, 0.0, {
                'rock': 0.0,
                'paper': 0.0,
                'scissors': 0.0
            }
    
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
        识别手势并返回MediaPipe结果（用于绘制关键点）
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            Tuple[Gesture, float, Dict, any]: (识别的手势, 置信度, 概率字典, MediaPipe结果)
        """
        # 转换为RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.use_tasks_api:
            # 新API
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            results = self.hand_landmarker.detect(mp_image)
        else:
            # 旧API
            results = self.hands.process(rgb_image)
        
        # 识别手势
        gesture, confidence, prob_dict = self.recognize(image)
        
        return gesture, confidence, prob_dict, results
    
    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        在图像上绘制手部关键点（用于调试和可视化）
        
        Args:
            image: 输入图像（BGR格式）
            results: MediaPipe处理结果
            
        Returns:
            np.ndarray: 绘制了关键点的图像
        """
        annotated_image = image.copy()
        
        if self.use_tasks_api:
            # 新Tasks API绘制
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # 绘制关键点
                    for landmark in hand_landmarks:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
                    
                    # 绘制连接线
                    connections = MP_HAND_CONNECTIONS.HAND_CONNECTIONS
                    for connection in connections:
                        start_idx = connection.start
                        end_idx = connection.end
                        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                            start_point = (
                                int(hand_landmarks[start_idx].x * image.shape[1]),
                                int(hand_landmarks[start_idx].y * image.shape[0])
                            )
                            end_point = (
                                int(hand_landmarks[end_idx].x * image.shape[1]),
                                int(hand_landmarks[end_idx].y * image.shape[0])
                            )
                            cv2.line(annotated_image, start_point, end_point, (0, 0, 255), 2)
        else:
            # 旧API绘制
            if results.multi_hand_landmarks:
                # 获取HAND_CONNECTIONS
                try:
                    if self.mp_api_type == "python_solutions":
                        from mediapipe.python.solutions import hands_connections
                        hand_connections = hands_connections.HAND_CONNECTIONS
                    else:
                        hand_connections = self.mp_hands.HAND_CONNECTIONS
                except AttributeError:
                    try:
                        import mediapipe as mp
                        hand_connections = mp.solutions.hands.HAND_CONNECTIONS
                    except:
                        hand_connections = None
                
                for hand_landmarks in results.multi_hand_landmarks:
                    if hand_connections is not None:
                        self.mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            hand_connections,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                    else:
                        self.mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
        
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
        if self.use_tasks_api:
            return {
                'model_type': 'MediaPipe Tasks API - HandLandmarker',
                'api_type': 'tasks',
                'confidence_threshold': self.confidence_threshold,
            }
        else:
            return {
                'model_type': 'MediaPipe Hands',
                'api_type': self.mp_api_type,
                'confidence_threshold': self.confidence_threshold,
                'min_detection_confidence': self.hands.min_detection_confidence,
                'min_tracking_confidence': self.hands.min_tracking_confidence,
                'max_num_hands': self.hands.max_num_hands
            }
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        elif hasattr(self, 'hands'):
            self.hands.close()
