"""
硬件配置管理模块
Hardware Configuration Manager
"""
from typing import Dict, Any, Optional
from pathlib import Path
from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger
from .factory.hardware_factory import HardwareFactory
from .base.robot_arm_base import RobotArmBase
from .base.camera_base import CameraBase
from .base.voice_base import VoiceBase

logger = setup_logger("RPS.HardwareConfigManager")


class HardwareConfigManager:
    """硬件配置管理器，负责加载配置和创建硬件实例"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化硬件配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = str(config_path)
        self.config: Dict[str, Any] = {}
        self._robot_arm: Optional[RobotArmBase] = None
        self._camera: Optional[CameraBase] = None
        self._voice: Optional[VoiceBase] = None
    
    def load_config(self) -> bool:
        """
        加载配置文件
        
        Returns:
            bool: 加载是否成功
        """
        try:
            self.config = ConfigLoader.load_config(self.config_path)
            logger.info("硬件配置加载成功")
            return True
        except Exception as e:
            logger.error(f"硬件配置加载失败: {e}")
            return False
    
    def create_robot_arm(self) -> Optional[RobotArmBase]:
        """
        根据配置创建机械臂实例
        
        Returns:
            Optional[RobotArmBase]: 机械臂实例，失败返回None
        """
        robot_arm_config = ConfigLoader.get_hardware_config(self.config, 'robot_arm')
        if not robot_arm_config:
            logger.error("配置文件中未找到机械臂配置")
            return None
        
        arm_type = robot_arm_config.get('type')
        if not arm_type:
            logger.error("机械臂配置中未指定类型")
            return None
        
        try:
            # 移除type字段，因为它不是硬件实例的参数
            arm_config = {k: v for k, v in robot_arm_config.items() if k != 'type'}
            self._robot_arm = HardwareFactory.create_robot_arm(arm_type, arm_config)
            logger.info(f"成功创建机械臂实例: {arm_type}")
            return self._robot_arm
        except Exception as e:
            logger.error(f"创建机械臂实例失败: {e}")
            return None
    
    def create_camera(self) -> Optional[CameraBase]:
        """
        根据配置创建摄像头实例
        
        Returns:
            Optional[CameraBase]: 摄像头实例，失败返回None
        """
        camera_config = ConfigLoader.get_hardware_config(self.config, 'camera')
        if not camera_config:
            logger.error("配置文件中未找到摄像头配置")
            return None
        
        camera_type = camera_config.get('type')
        if not camera_type:
            logger.error("摄像头配置中未指定类型")
            return None
        
        try:
            # 移除type字段，因为它不是硬件实例的参数
            cam_config = {k: v for k, v in camera_config.items() if k != 'type'}
            self._camera = HardwareFactory.create_camera(camera_type, cam_config)
            logger.info(f"成功创建摄像头实例: {camera_type}")
            return self._camera
        except Exception as e:
            logger.error(f"创建摄像头实例失败: {e}")
            return None
    
    def create_voice(self) -> Optional[VoiceBase]:
        """
        根据配置创建语音模块实例
        
        Returns:
            Optional[VoiceBase]: 语音模块实例，失败返回None
        """
        voice_config = ConfigLoader.get_hardware_config(self.config, 'voice')
        if not voice_config:
            logger.error("配置文件中未找到语音模块配置")
            return None
        
        voice_type = voice_config.get('type')
        if not voice_type:
            logger.error("语音模块配置中未指定类型")
            return None
        
        try:
            # 移除type字段，因为它不是硬件实例的参数
            voice_cfg = {k: v for k, v in voice_config.items() if k != 'type'}
            self._voice = HardwareFactory.create_voice(voice_type, voice_cfg)
            logger.info(f"成功创建语音模块实例: {voice_type}")
            return self._voice
        except Exception as e:
            logger.error(f"创建语音模块实例失败: {e}")
            return None
    
    def create_all_hardware(self) -> bool:
        """
        创建所有硬件实例
        
        Returns:
            bool: 是否全部创建成功
        """
        success = True
        
        if not self.create_robot_arm():
            success = False
        
        if not self.create_camera():
            success = False
        
        if not self.create_voice():
            success = False
        
        return success
    
    def get_robot_arm(self) -> Optional[RobotArmBase]:
        """获取机械臂实例"""
        return self._robot_arm
    
    def get_camera(self) -> Optional[CameraBase]:
        """获取摄像头实例"""
        return self._camera
    
    def get_voice(self) -> Optional[VoiceBase]:
        """获取语音模块实例"""
        return self._voice
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config.copy()
