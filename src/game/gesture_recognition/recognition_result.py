"""
手势识别结果处理
Gesture Recognition Result Processing
"""
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from ..game_logic.gesture import Gesture
from ...utils.logger import setup_logger

logger = setup_logger("RPS.RecognitionResult")


@dataclass
class RecognitionResult:
    """手势识别结果数据类"""
    gesture: Gesture
    confidence: float
    probabilities: dict
    timestamp: datetime
    image_shape: Optional[tuple] = None
    
    def is_valid(self, min_confidence: float = 0.7) -> bool:
        """
        检查识别结果是否有效
        
        Args:
            min_confidence: 最小置信度阈值
            
        Returns:
            bool: 是否有效
        """
        return (self.gesture != Gesture.UNKNOWN and 
                self.confidence >= min_confidence)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'gesture': self.gesture.value,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'timestamp': self.timestamp.isoformat(),
            'image_shape': self.image_shape
        }


class RecognitionResultProcessor:
    """识别结果处理器"""
    
    def __init__(self, smoothing_window: int = 5, min_confidence: float = 0.7):
        """
        初始化结果处理器
        
        Args:
            smoothing_window: 平滑窗口大小（用于时间序列平滑）
            min_confidence: 最小置信度阈值
        """
        self.smoothing_window = smoothing_window
        self.min_confidence = min_confidence
        self.result_history: List[RecognitionResult] = []
    
    def add_result(self, result: RecognitionResult):
        """
        添加识别结果到历史记录
        
        Args:
            result: 识别结果
        """
        self.result_history.append(result)
        
        # 保持历史记录在窗口大小内
        if len(self.result_history) > self.smoothing_window:
            self.result_history.pop(0)
    
    def get_smoothed_result(self) -> Optional[Gesture]:
        """
        获取平滑后的识别结果（使用多数投票）
        
        Returns:
            Optional[Gesture]: 平滑后的手势，如果历史记录为空返回None
        """
        if not self.result_history:
            return None
        
        # 过滤有效结果
        valid_results = [r for r in self.result_history if r.is_valid(self.min_confidence)]
        
        if not valid_results:
            return Gesture.UNKNOWN
        
        # 统计各手势的出现次数
        gesture_counts = {}
        for result in valid_results:
            gesture = result.gesture
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # 返回出现次数最多的手势
        if gesture_counts:
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            return most_common[0]
        
        return Gesture.UNKNOWN
    
    def get_average_confidence(self) -> float:
        """
        获取平均置信度
        
        Returns:
            float: 平均置信度
        """
        if not self.result_history:
            return 0.0
        
        valid_results = [r for r in self.result_history if r.is_valid(self.min_confidence)]
        if not valid_results:
            return 0.0
        
        return sum(r.confidence for r in valid_results) / len(valid_results)
    
    def clear_history(self):
        """清空历史记录"""
        self.result_history.clear()
        logger.debug("识别结果历史记录已清空")
    
    def get_recent_results(self, count: int = 5) -> List[RecognitionResult]:
        """
        获取最近的识别结果
        
        Args:
            count: 返回结果数量
            
        Returns:
            List[RecognitionResult]: 最近的识别结果列表
        """
        return self.result_history[-count:]
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        Returns:
            dict: 统计信息字典
        """
        if not self.result_history:
            return {
                'total_count': 0,
                'valid_count': 0,
                'average_confidence': 0.0,
                'gesture_distribution': {}
            }
        
        valid_results = [r for r in self.result_history if r.is_valid(self.min_confidence)]
        
        gesture_distribution = {}
        for result in self.result_history:
            gesture = result.gesture.value
            gesture_distribution[gesture] = gesture_distribution.get(gesture, 0) + 1
        
        return {
            'total_count': len(self.result_history),
            'valid_count': len(valid_results),
            'average_confidence': self.get_average_confidence(),
            'gesture_distribution': gesture_distribution,
            'smoothing_window': self.smoothing_window
        }
