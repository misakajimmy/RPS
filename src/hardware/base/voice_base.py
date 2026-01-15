"""
语音模块抽象基类
Voice Module Base Class
"""
from abc import ABC, abstractmethod
from typing import Optional


class VoiceBase(ABC):
    """语音模块抽象基类，定义所有语音模块必须实现的接口"""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接语音模块
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开语音模块连接
        
        Returns:
            bool: 断开是否成功
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查语音模块是否已连接
        
        Returns:
            bool: 连接状态
        """
        pass
    
    @abstractmethod
    def recognize_speech(self, timeout: float = 5.0) -> Optional[str]:
        """
        语音识别
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            Optional[str]: 识别的文本，失败返回None
        """
        pass
    
    @abstractmethod
    def synthesize_speech(self, text: str) -> bool:
        """
        语音合成（TTS）
        
        Args:
            text: 要合成的文本
            
        Returns:
            bool: 合成是否成功
        """
        pass
    
    @abstractmethod
    def play_audio(self, audio_data: bytes) -> bool:
        """
        播放音频数据
        
        Args:
            audio_data: 音频数据
            
        Returns:
            bool: 播放是否成功
        """
        pass
    
    def get_status(self) -> dict:
        """
        获取语音模块状态信息（可选实现）
        
        Returns:
            dict: 状态信息字典
        """
        return {
            "connected": self.is_connected(),
            "type": self.__class__.__name__
        }
