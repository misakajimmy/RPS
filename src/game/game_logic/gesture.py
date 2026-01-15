"""
手势枚举类型
Gesture Enumeration
"""
from enum import Enum


class Gesture(Enum):
    """手势类型枚举"""
    ROCK = "rock"          # 石头
    PAPER = "paper"        # 布
    SCISSORS = "scissors"  # 剪刀
    UNKNOWN = "unknown"    # 未知/未识别
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, value: str):
        """
        从字符串创建手势枚举
        
        Args:
            value: 手势字符串（rock, paper, scissors）
            
        Returns:
            Gesture: 手势枚举值
        """
        value_lower = value.lower()
        for gesture in cls:
            if gesture.value == value_lower:
                return gesture
        return cls.UNKNOWN
