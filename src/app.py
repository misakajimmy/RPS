"""
应用程序主类
Application Main Class
"""
import sys
import time
import signal
from pathlib import Path
from typing import Optional
from .hardware.config_manager import HardwareConfigManager
from .hardware.base.robot_arm_base import RobotArmBase
from .hardware.base.camera_base import CameraBase
from .hardware.base.voice_base import VoiceBase
from .game import GameController
from .game.gesture_recognition import GestureRecognizer
from .game.state_machine import GameState
from .utils.logger import setup_logger_from_config
from .utils.config_loader import ConfigLoader
from .utils.error_handler import global_error_handler
from .utils.exceptions import (
    HardwareException, RecognitionException, GameException, ConfigurationException
)

logger = None  # 将在初始化时设置


class Application:
    """应用程序主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化应用程序
        
        Args:
            config_path: 配置文件路径
        """
        global logger
        
        # 设置日志
        if config_path:
            try:
                config = ConfigLoader.load_config(config_path)
                logging_config = ConfigLoader.get_logging_config(config)
                logger = setup_logger_from_config(logging_config, "RPS.App")
            except Exception as e:
                print(f"加载日志配置失败: {e}，使用默认配置")
                from .utils.logger import setup_logger
                logger = setup_logger("RPS.App")
        else:
            from .utils.logger import setup_logger
            logger = setup_logger("RPS.App")
        
        self.config_path = config_path or str(Path(__file__).parent.parent / "config" / "config.yaml")
        self.config = {}
        
        # 硬件组件
        self.hardware_manager: Optional[HardwareConfigManager] = None
        self.robot_arm: Optional[RobotArmBase] = None
        self.camera: Optional[CameraBase] = None
        self.voice: Optional[VoiceBase] = None
        
        # 游戏组件
        self.gesture_recognizer: Optional[GestureRecognizer] = None
        self.game_controller: Optional[GameController] = None
        
        # 运行状态
        self.is_running = False
        self.should_exit = False
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("应用程序初始化完成")
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"收到信号 {signum}，准备退出")
        self.should_exit = True
    
    def initialize(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("=" * 50)
            logger.info("开始初始化应用程序")
            logger.info("=" * 50)
            
            # 1. 加载配置
            if not self._load_config():
                return False
            
            # 2. 初始化硬件
            if not self._initialize_hardware():
                return False
            
            # 3. 初始化手势识别器
            if not self._initialize_gesture_recognizer():
                return False
            
            # 4. 初始化游戏控制器
            if not self._initialize_game_controller():
                return False
            
            logger.info("=" * 50)
            logger.info("应用程序初始化成功")
            logger.info("=" * 50)
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            global_error_handler.handle(e, "初始化")
            return False
    
    def _load_config(self) -> bool:
        """加载配置文件"""
        try:
            self.config = ConfigLoader.load_config(self.config_path)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return True
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.config_path}")
            logger.info("请复制 config/config.yaml.example 到 config/config.yaml 并配置")
            return False
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            global_error_handler.handle(ConfigurationException(str(e)), "加载配置")
            return False
    
    def _initialize_hardware(self) -> bool:
        """初始化硬件"""
        try:
            logger.info("初始化硬件模块...")
            
            self.hardware_manager = HardwareConfigManager(self.config_path)
            if not self.hardware_manager.load_config():
                logger.error("硬件配置加载失败")
                return False
            
            # 创建硬件实例
            self.robot_arm = self.hardware_manager.create_robot_arm()
            self.camera = self.hardware_manager.create_camera()
            self.voice = self.hardware_manager.create_voice()
            
            # 连接硬件
            success_count = 0
            
            if self.camera:
                if self.camera.connect():
                    logger.info("✓ 摄像头连接成功")
                    success_count += 1
                else:
                    logger.warning("✗ 摄像头连接失败")
            else:
                logger.warning("摄像头未配置或创建失败")
            
            if self.robot_arm:
                if self.robot_arm.connect():
                    logger.info("✓ 机械臂连接成功")
                    success_count += 1
                else:
                    logger.warning("✗ 机械臂连接失败")
            else:
                logger.warning("机械臂未配置或创建失败")
            
            if self.voice:
                if self.voice.connect():
                    logger.info("✓ 语音模块连接成功")
                    success_count += 1
                else:
                    logger.warning("✗ 语音模块连接失败")
            else:
                logger.warning("语音模块未配置或创建失败")
            
            logger.info(f"硬件初始化完成，成功连接 {success_count} 个设备")
            
            # 至少需要摄像头
            if not self.camera or not self.camera.is_connected():
                logger.error("摄像头未连接，无法继续")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"硬件初始化异常: {e}", exc_info=True)
            global_error_handler.handle(HardwareException(str(e)), "硬件初始化")
            return False
    
    def _initialize_gesture_recognizer(self) -> bool:
        """初始化手势识别器"""
        try:
            logger.info("初始化手势识别器...")
            
            # 从配置获取识别器参数
            game_config = ConfigLoader.get_game_config(self.config)
            recognition_config = game_config.get('gesture_recognition', {})
            confidence_threshold = recognition_config.get('confidence_threshold', 0.7)
            use_huggingface_model = recognition_config.get('use_huggingface_model', True)  # 默认使用 HuggingFace 模型
            model_path = recognition_config.get('model_path', None)
            model_size = recognition_config.get('model_size', 'n')
            min_detection_confidence = recognition_config.get('min_detection_confidence', 0.5)
            device = recognition_config.get('device', None)  # None 则自动检测
            
            self.gesture_recognizer = GestureRecognizer(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,  # 保留兼容性
                max_num_hands=1,  # 保留兼容性
                confidence_threshold=confidence_threshold,
                model_path=model_path,
                model_size=model_size,
                use_huggingface_model=use_huggingface_model,
                device=device
            )
            
            logger.info("✓ 手势识别器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"手势识别器初始化失败: {e}", exc_info=True)
            global_error_handler.handle(RecognitionException(str(e)), "初始化识别器")
            return False
    
    def _initialize_game_controller(self) -> bool:
        """初始化游戏控制器"""
        try:
            logger.info("初始化游戏控制器...")
            
            # 从配置获取游戏参数
            game_config = ConfigLoader.get_game_config(self.config)
            max_rounds = game_config.get('max_rounds', 5)
            countdown_seconds = 3  # 可以添加到配置
            
            self.game_controller = GameController(
                gesture_recognizer=self.gesture_recognizer,
                robot_arm=self.robot_arm,
                camera=self.camera,
                voice=self.voice,
                max_rounds=max_rounds,
                countdown_seconds=countdown_seconds
            )
            
            # 注册回调
            self.game_controller.on_state_changed = self._on_game_state_changed
            self.game_controller.on_round_result = self._on_round_result
            self.game_controller.on_game_over = self._on_game_over
            
            logger.info("✓ 游戏控制器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"游戏控制器初始化失败: {e}", exc_info=True)
            global_error_handler.handle(GameException(str(e)), "初始化游戏控制器")
            return False
    
    def _on_game_state_changed(self, state: GameState):
        """游戏状态改变回调"""
        logger.debug(f"游戏状态改变: {state}")
    
    def _on_round_result(self, round_result):
        """回合结果回调"""
        logger.info(f"回合 {round_result.round_number}: "
                   f"玩家={round_result.player_gesture.value}, "
                   f"机器人={round_result.robot_gesture.value}, "
                   f"结果={round_result.result.value}")
    
    def _on_game_over(self, statistics):
        """游戏结束回调"""
        logger.info("游戏结束")
        logger.info(f"统计信息: 总回合={statistics.total_rounds}, "
                   f"玩家胜={statistics.player_wins}, "
                   f"机器人胜={statistics.robot_wins}, "
                   f"平局={statistics.draws}, "
                   f"胜率={statistics.get_win_rate():.2%}")
    
    def run(self):
        """运行应用程序主循环"""
        if not self.is_running:
            logger.error("应用程序未初始化，无法运行")
            return
        
        logger.info("=" * 50)
        logger.info("应用程序主循环启动")
        logger.info("=" * 50)
        
        try:
            # 启动游戏
            self.game_controller.start_game()
            
            # 主循环
            while not self.should_exit:
                state = self.game_controller.get_current_state()
                
                # 根据状态执行相应操作
                if state == GameState.WAITING_GESTURE:
                    # 自动触发识别
                    self.game_controller.trigger_recognize()
                
                elif state == GameState.GAME_OVER:
                    # 游戏结束，等待一段时间后可以重新开始
                    time.sleep(3)
                    # 可以选择自动重新开始或退出
                    # self.game_controller.reset_game()
                    # self.game_controller.start_game()
                    break
                
                elif state == GameState.ERROR:
                    # 错误状态，尝试恢复
                    logger.warning("游戏进入错误状态，尝试恢复")
                    time.sleep(2)
                    self.game_controller.reset_game()
                    self.game_controller.start_game()
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        except Exception as e:
            logger.error(f"主循环异常: {e}", exc_info=True)
            global_error_handler.handle(e, "主循环")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        logger.info("开始清理资源...")
        
        try:
            # 停止游戏
            if self.game_controller:
                self.game_controller.stop_game()
            
            # 断开硬件连接
            if self.robot_arm and self.robot_arm.is_connected():
                self.robot_arm.disconnect()
                logger.info("✓ 机械臂已断开")
            
            if self.camera and self.camera.is_connected():
                self.camera.disconnect()
                logger.info("✓ 摄像头已断开")
            
            if self.voice and self.voice.is_connected():
                self.voice.disconnect()
                logger.info("✓ 语音模块已断开")
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源时发生异常: {e}", exc_info=True)
    
    def start(self) -> bool:
        """
        启动应用程序
        
        Returns:
            bool: 启动是否成功
        """
        if self.initialize():
            self.is_running = True
            self.run()
            return True
        else:
            logger.error("应用程序启动失败")
            return False
