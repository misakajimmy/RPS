"""
自定义异常类
Custom Exception Classes
"""
from typing import Optional


class HardwareException(Exception):
    """硬件相关异常基类"""
    def __init__(self, message: str, hardware_type: Optional[str] = None):
        super().__init__(message)
        self.hardware_type = hardware_type
        self.message = message


class RobotArmException(HardwareException):
    """机械臂异常"""
    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message, hardware_type="robot_arm")
        self.error_code = error_code


class CameraException(HardwareException):
    """摄像头异常"""
    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message, hardware_type="camera")
        self.error_code = error_code


class VoiceException(HardwareException):
    """语音模块异常"""
    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message, hardware_type="voice")
        self.error_code = error_code


class RecognitionException(Exception):
    """手势识别异常"""
    def __init__(self, message: str, confidence: Optional[float] = None):
        super().__init__(message)
        self.confidence = confidence
        self.message = message


class GameException(Exception):
    """游戏逻辑异常"""
    def __init__(self, message: str, game_state: Optional[str] = None):
        super().__init__(message)
        self.game_state = game_state
        self.message = message


class ConfigurationException(Exception):
    """配置异常"""
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message)
        self.config_key = config_key
        self.message = message
