"""
错误处理工具模块
Error Handler Utility Module
"""
import traceback
from typing import Optional, Callable
from .exceptions import (
    HardwareException, RobotArmException, CameraException, VoiceException,
    RecognitionException, GameException, ConfigurationException
)
from .logger import setup_logger

logger = setup_logger("RPS.ErrorHandler")


class ErrorHandler:
    """错误处理器类"""
    
    def __init__(self):
        """初始化错误处理器"""
        self.error_callbacks: dict = {}
        self.setup_default_handlers()
    
    def setup_default_handlers(self):
        """设置默认错误处理函数"""
        self.error_callbacks[HardwareException] = self._handle_hardware_error
        self.error_callbacks[RobotArmException] = self._handle_robot_arm_error
        self.error_callbacks[CameraException] = self._handle_camera_error
        self.error_callbacks[VoiceException] = self._handle_voice_error
        self.error_callbacks[RecognitionException] = self._handle_recognition_error
        self.error_callbacks[GameException] = self._handle_game_error
        self.error_callbacks[ConfigurationException] = self._handle_config_error
    
    def register_handler(self, exception_type: type, handler: Callable):
        """
        注册错误处理函数
        
        Args:
            exception_type: 异常类型
            handler: 处理函数
        """
        self.error_callbacks[exception_type] = handler
        logger.debug(f"注册错误处理函数: {exception_type.__name__}")
    
    def handle(self, exception: Exception, context: Optional[str] = None) -> bool:
        """
        处理异常
        
        Args:
            exception: 异常对象
            context: 上下文信息
            
        Returns:
            bool: 是否成功处理
        """
        exception_type = type(exception)
        
        # 记录错误
        error_msg = f"异常发生"
        if context:
            error_msg += f" (上下文: {context})"
        error_msg += f": {str(exception)}"
        
        logger.error(error_msg, exc_info=True)
        
        # 查找处理函数
        handler = None
        for exc_type, handler_func in self.error_callbacks.items():
            if issubclass(exception_type, exc_type):
                handler = handler_func
                break
        
        if handler:
            try:
                handler(exception, context)
                return True
            except Exception as e:
                logger.error(f"错误处理函数执行异常: {e}", exc_info=True)
                return False
        else:
            # 使用默认处理
            self._handle_generic_error(exception, context)
            return False
    
    def _handle_hardware_error(self, exception: HardwareException, context: Optional[str]):
        """处理硬件错误"""
        logger.error(f"硬件错误 [{exception.hardware_type}]: {exception.message}")
        # 可以在这里添加硬件重连逻辑
    
    def _handle_robot_arm_error(self, exception: RobotArmException, context: Optional[str]):
        """处理机械臂错误"""
        logger.error(f"机械臂错误 [代码: {exception.error_code}]: {exception.message}")
    
    def _handle_camera_error(self, exception: CameraException, context: Optional[str]):
        """处理摄像头错误"""
        logger.error(f"摄像头错误 [代码: {exception.error_code}]: {exception.message}")
    
    def _handle_voice_error(self, exception: VoiceException, context: Optional[str]):
        """处理语音模块错误"""
        logger.error(f"语音模块错误 [代码: {exception.error_code}]: {exception.message}")
    
    def _handle_recognition_error(self, exception: RecognitionException, context: Optional[str]):
        """处理识别错误"""
        logger.warning(f"手势识别错误 [置信度: {exception.confidence}]: {exception.message}")
    
    def _handle_game_error(self, exception: GameException, context: Optional[str]):
        """处理游戏逻辑错误"""
        logger.error(f"游戏逻辑错误 [状态: {exception.game_state}]: {exception.message}")
    
    def _handle_config_error(self, exception: ConfigurationException, context: Optional[str]):
        """处理配置错误"""
        logger.error(f"配置错误 [键: {exception.config_key}]: {exception.message}")
    
    def _handle_generic_error(self, exception: Exception, context: Optional[str]):
        """处理通用错误"""
        logger.error(f"未处理的异常: {type(exception).__name__}: {str(exception)}")
        logger.debug(traceback.format_exc())


# 全局错误处理器实例
global_error_handler = ErrorHandler()
