"""
硬件工厂类
Hardware Factory Class
"""
from typing import Optional, Dict, Any
from ..base.robot_arm_base import RobotArmBase
from ..base.camera_base import CameraBase
from ..base.voice_base import VoiceBase


class HardwareFactory:
    """硬件工厂类，负责创建各种硬件实例"""
    
    _robot_arm_classes: Dict[str, type] = {}
    _camera_classes: Dict[str, type] = {}
    _voice_classes: Dict[str, type] = {}
    
    @classmethod
    def register_robot_arm(cls, name: str, arm_class: type):
        """
        注册机械臂类
        
        Args:
            name: 机械臂名称（如 'uhand2.0'）
            arm_class: 机械臂类（必须继承自RobotArmBase）
        """
        if not issubclass(arm_class, RobotArmBase):
            raise TypeError(f"{arm_class} must be a subclass of RobotArmBase")
        cls._robot_arm_classes[name.lower()] = arm_class
    
    @classmethod
    def register_camera(cls, name: str, camera_class: type):
        """
        注册摄像头类
        
        Args:
            name: 摄像头名称（如 'usb_camera'）
            camera_class: 摄像头类（必须继承自CameraBase）
        """
        if not issubclass(camera_class, CameraBase):
            raise TypeError(f"{camera_class} must be a subclass of CameraBase")
        cls._camera_classes[name.lower()] = camera_class
    
    @classmethod
    def register_voice(cls, name: str, voice_class: type):
        """
        注册语音模块类
        
        Args:
            name: 语音模块名称（如 'ci1302'）
            voice_class: 语音模块类（必须继承自VoiceBase）
        """
        if not issubclass(voice_class, VoiceBase):
            raise TypeError(f"{voice_class} must be a subclass of VoiceBase")
        cls._voice_classes[name.lower()] = voice_class
    
    @classmethod
    def create_robot_arm(cls, name: str, config: Dict[str, Any]) -> Optional[RobotArmBase]:
        """
        创建机械臂实例
        
        Args:
            name: 机械臂名称
            config: 配置参数
            
        Returns:
            Optional[RobotArmBase]: 机械臂实例，失败返回None
        """
        name_lower = name.lower()
        if name_lower not in cls._robot_arm_classes:
            raise ValueError(f"Unknown robot arm: {name}")
        
        arm_class = cls._robot_arm_classes[name_lower]
        return arm_class(**config)
    
    @classmethod
    def create_camera(cls, name: str, config: Dict[str, Any]) -> Optional[CameraBase]:
        """
        创建摄像头实例
        
        Args:
            name: 摄像头名称
            config: 配置参数
            
        Returns:
            Optional[CameraBase]: 摄像头实例，失败返回None
        """
        name_lower = name.lower()
        if name_lower not in cls._camera_classes:
            raise ValueError(f"Unknown camera: {name}")
        
        camera_class = cls._camera_classes[name_lower]
        return camera_class(**config)
    
    @classmethod
    def create_voice(cls, name: str, config: Dict[str, Any]) -> Optional[VoiceBase]:
        """
        创建语音模块实例
        
        Args:
            name: 语音模块名称
            config: 配置参数
            
        Returns:
            Optional[VoiceBase]: 语音模块实例，失败返回None
        """
        name_lower = name.lower()
        if name_lower not in cls._voice_classes:
            raise ValueError(f"Unknown voice module: {name}")
        
        voice_class = cls._voice_classes[name_lower]
        return voice_class(**config)
    
    @classmethod
    def list_robot_arms(cls) -> list:
        """
        列出所有已注册的机械臂类型
        
        Returns:
            list: 机械臂类型名称列表
        """
        return list(cls._robot_arm_classes.keys())
    
    @classmethod
    def list_cameras(cls) -> list:
        """
        列出所有已注册的摄像头类型
        
        Returns:
            list: 摄像头类型名称列表
        """
        return list(cls._camera_classes.keys())
    
    @classmethod
    def list_voice_modules(cls) -> list:
        """
        列出所有已注册的语音模块类型
        
        Returns:
            list: 语音模块类型名称列表
        """
        return list(cls._voice_classes.keys())
    
    @classmethod
    def is_robot_arm_registered(cls, name: str) -> bool:
        """
        检查机械臂类型是否已注册
        
        Args:
            name: 机械臂名称
            
        Returns:
            bool: 是否已注册
        """
        return name.lower() in cls._robot_arm_classes
    
    @classmethod
    def is_camera_registered(cls, name: str) -> bool:
        """
        检查摄像头类型是否已注册
        
        Args:
            name: 摄像头名称
            
        Returns:
            bool: 是否已注册
        """
        return name.lower() in cls._camera_classes
    
    @classmethod
    def is_voice_registered(cls, name: str) -> bool:
        """
        检查语音模块类型是否已注册
        
        Args:
            name: 语音模块名称
            
        Returns:
            bool: 是否已注册
        """
        return name.lower() in cls._voice_classes
