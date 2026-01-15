"""
机械臂抽象基类
Robot Arm Base Class
"""
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum


class GestureType(Enum):
    """手势类型枚举"""
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"


class RobotArmBase(ABC):
    """机械臂抽象基类，定义所有机械臂必须实现的接口"""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接机械臂
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开机械臂连接
        
        Returns:
            bool: 断开是否成功
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查机械臂是否已连接
        
        Returns:
            bool: 连接状态
        """
        pass
    
    @abstractmethod
    def move_to_rock(self) -> bool:
        """
        移动到石头手势位置
        
        Returns:
            bool: 动作是否成功
        """
        pass
    
    @abstractmethod
    def move_to_paper(self) -> bool:
        """
        移动到布手势位置
        
        Returns:
            bool: 动作是否成功
        """
        pass
    
    @abstractmethod
    def move_to_scissors(self) -> bool:
        """
        移动到剪刀手势位置
        
        Returns:
            bool: 动作是否成功
        """
        pass
    
    @abstractmethod
    def reset_position(self) -> bool:
        """
        重置到初始位置
        
        Returns:
            bool: 动作是否成功
        """
        pass
    
    @abstractmethod
    def move_to_gesture(self, gesture: GestureType) -> bool:
        """
        移动到指定手势位置
        
        Args:
            gesture: 手势类型
            
        Returns:
            bool: 动作是否成功
        """
        pass
    
    def get_status(self) -> dict:
        """
        获取机械臂状态信息（可选实现）
        
        Returns:
            dict: 状态信息字典
        """
        return {
            "connected": self.is_connected(),
            "type": self.__class__.__name__
        }