"""
摄像头抽象基类
Camera Base Class
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CameraBase(ABC):
    """摄像头抽象基类，定义所有摄像头必须实现的接口"""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接摄像头
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开摄像头连接
        
        Returns:
            bool: 断开是否成功
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查摄像头是否已连接
        
        Returns:
            bool: 连接状态
        """
        pass
    
    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像
        
        Returns:
            Optional[np.ndarray]: 图像数据，失败返回None
        """
        pass
    
    @abstractmethod
    def get_resolution(self) -> tuple:
        """
        获取摄像头分辨率
        
        Returns:
            tuple: (width, height)
        """
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置摄像头分辨率
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            bool: 设置是否成功
        """
        pass
    
    def get_status(self) -> dict:
        """
        获取摄像头状态信息（可选实现）
        
        Returns:
            dict: 状态信息字典
        """
        resolution = self.get_resolution() if self.is_connected() else (0, 0)
        return {
            "connected": self.is_connected(),
            "resolution": resolution,
            "type": self.__class__.__name__
        }
