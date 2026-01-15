"""
游戏控制器
Game Controller - 整合所有游戏逻辑
"""
import time
from typing import Optional, Callable
from .state_machine import GameState, GameStateMachine
from .game_logic import GameManager, Gesture, GameResult
from .gesture_recognition import GestureRecognizer, RecognitionResult, RecognitionResultProcessor
from ..hardware.base.robot_arm_base import RobotArmBase, GestureType
from ..hardware.base.camera_base import CameraBase
from ..hardware.base.voice_base import VoiceBase
from ..utils.logger import setup_logger

logger = setup_logger("RPS.GameController")


class GameController:
    """游戏控制器类，整合所有游戏组件"""
    
    def __init__(self, 
                 gesture_recognizer: GestureRecognizer,
                 robot_arm: Optional[RobotArmBase] = None,
                 camera: Optional[CameraBase] = None,
                 voice: Optional[VoiceBase] = None,
                 max_rounds: int = 5,
                 countdown_seconds: int = 3):
        """
        初始化游戏控制器
        
        Args:
            gesture_recognizer: 手势识别器
            robot_arm: 机械臂（可选）
            camera: 摄像头（可选）
            voice: 语音模块（可选）
            max_rounds: 最大回合数
            countdown_seconds: 倒计时秒数
        """
        self.gesture_recognizer = gesture_recognizer
        self.robot_arm = robot_arm
        self.camera = camera
        self.voice = voice
        self.max_rounds = max_rounds
        self.countdown_seconds = countdown_seconds
        
        # 初始化游戏管理器
        self.game_manager = GameManager(max_rounds=max_rounds)
        
        # 初始化状态机
        self.state_machine = GameStateMachine(initial_state=GameState.IDLE)
        self._setup_state_handlers()
        
        # 识别结果处理器
        self.result_processor = RecognitionResultProcessor()
        
        # 回调函数
        self.on_state_changed: Optional[Callable] = None
        self.on_round_result: Optional[Callable] = None
        self.on_game_over: Optional[Callable] = None
        
        logger.info("游戏控制器初始化完成")
    
    def _setup_state_handlers(self):
        """设置状态处理函数"""
        # 空闲状态
        self.state_machine.register_state_handler(GameState.IDLE, self._handle_idle)
        
        # 等待开始
        self.state_machine.register_state_handler(GameState.WAITING_START, self._handle_waiting_start)
        
        # 倒计时
        self.state_machine.register_state_handler(GameState.COUNTDOWN, self._handle_countdown)
        
        # 等待手势
        self.state_machine.register_state_handler(GameState.WAITING_GESTURE, self._handle_waiting_gesture)
        
        # 识别中
        self.state_machine.register_state_handler(GameState.RECOGNIZING, self._handle_recognizing)
        
        # 机械臂移动
        self.state_machine.register_state_handler(GameState.ROBOT_MOVING, self._handle_robot_moving)
        
        # 显示结果
        self.state_machine.register_state_handler(GameState.RESULT_SHOWING, self._handle_result_showing)
        
        # 回合结束
        self.state_machine.register_state_handler(GameState.ROUND_END, self._handle_round_end)
        
        # 游戏结束
        self.state_machine.register_state_handler(GameState.GAME_OVER, self._handle_game_over)
    
    def start_game(self):
        """开始游戏"""
        logger.info("开始游戏")
        self.game_manager.start_game()
        self.state_machine.transition_to(GameState.WAITING_START)
        self._notify_state_changed()
    
    def stop_game(self):
        """停止游戏"""
        logger.info("停止游戏")
        self.game_manager.end_game()
        self.state_machine.transition_to(GameState.IDLE)
        self._notify_state_changed()
    
    def reset_game(self):
        """重置游戏"""
        logger.info("重置游戏")
        self.game_manager.reset()
        self.result_processor.clear_history()
        self.state_machine.reset(GameState.IDLE)
        self._notify_state_changed()
    
    def _handle_idle(self):
        """处理空闲状态"""
        logger.debug("进入空闲状态")
        if self.voice:
            try:
                self.voice.synthesize_speech("游戏已就绪，等待开始")
            except:
                pass
    
    def _handle_waiting_start(self):
        """处理等待开始状态"""
        logger.debug("进入等待开始状态")
        if self.voice:
            try:
                self.voice.synthesize_speech("请准备开始游戏")
            except:
                pass
    
    def _handle_countdown(self):
        """处理倒计时状态"""
        logger.debug("进入倒计时状态")
        
        if self.voice:
            try:
                for i in range(self.countdown_seconds, 0, -1):
                    self.voice.synthesize_speech(str(i))
                    time.sleep(1)
                self.voice.synthesize_speech("开始")
            except:
                # 如果语音失败，使用时间延迟
                time.sleep(self.countdown_seconds)
        
        self.state_machine.transition_to(GameState.WAITING_GESTURE)
        self._notify_state_changed()
    
    def _handle_waiting_gesture(self):
        """处理等待手势状态"""
        logger.debug("进入等待手势状态")
        
        if self.voice:
            try:
                self.voice.synthesize_speech("请做出手势")
            except:
                pass
    
    def _handle_recognizing(self):
        """处理识别中状态"""
        logger.debug("进入识别中状态")
        
        if not self.camera:
            logger.error("摄像头未初始化")
            self.state_machine.transition_to(GameState.ERROR)
            return
        
        # 捕获图像并识别
        frame = self.camera.capture_frame()
        if frame is None:
            logger.warning("无法捕获图像")
            self.state_machine.transition_to(GameState.WAITING_GESTURE)
            return
        
        # 识别手势
        gesture, confidence, probabilities = self.gesture_recognizer.recognize(frame)
        
        # 创建识别结果
        from datetime import datetime
        result = RecognitionResult(
            gesture=gesture,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.now(),
            image_shape=frame.shape
        )
        
        self.result_processor.add_result(result)
        
        # 获取平滑后的结果
        smoothed_gesture = self.result_processor.get_smoothed_result()
        
        if smoothed_gesture and smoothed_gesture != Gesture.UNKNOWN:
            # 识别成功，进入机械臂移动状态
            self._player_gesture = smoothed_gesture
            self.state_machine.transition_to(GameState.ROBOT_MOVING)
        else:
            # 识别失败，继续等待
            logger.debug("手势识别失败或置信度不足")
            self.state_machine.transition_to(GameState.WAITING_GESTURE)
        
        self._notify_state_changed()
    
    def _handle_robot_moving(self):
        """处理机械臂移动状态"""
        logger.debug("进入机械臂移动状态")
        
        # 确定机器人手势（这里可以添加策略，比如随机或智能选择）
        robot_gesture = self._select_robot_gesture()
        self._robot_gesture = robot_gesture
        
        # 移动机械臂
        if self.robot_arm:
            try:
                gesture_type = GestureType[robot_gesture.name]
                self.robot_arm.move_to_gesture(gesture_type)
            except Exception as e:
                logger.error(f"机械臂移动失败: {e}")
        
        # 进入结果显示状态
        self.state_machine.transition_to(GameState.RESULT_SHOWING)
        self._notify_state_changed()
    
    def _handle_result_showing(self):
        """处理结果显示状态"""
        logger.debug("进入结果显示状态")
        
        # 判断结果
        round_result = self.game_manager.play_round(
            self._player_gesture,
            self._robot_gesture
        )
        
        # 语音反馈
        if self.voice and round_result:
            try:
                result_text = self._get_result_text(round_result.result)
                self.voice.synthesize_speech(result_text)
            except:
                pass
        
        # 通知回调
        if self.on_round_result:
            try:
                self.on_round_result(round_result)
            except Exception as e:
                logger.error(f"回合结果回调异常: {e}")
        
        # 检查游戏是否结束
        if self.game_manager.is_game_over():
            self.state_machine.transition_to(GameState.GAME_OVER)
        else:
            self.state_machine.transition_to(GameState.ROUND_END)
        
        self._notify_state_changed()
    
    def _handle_round_end(self):
        """处理回合结束状态"""
        logger.debug("进入回合结束状态")
        
        # 等待一段时间后继续下一回合
        time.sleep(2)
        
        if not self.game_manager.is_game_over():
            self.state_machine.transition_to(GameState.WAITING_GESTURE)
        else:
            self.state_machine.transition_to(GameState.GAME_OVER)
        
        self._notify_state_changed()
    
    def _handle_game_over(self):
        """处理游戏结束状态"""
        logger.debug("进入游戏结束状态")
        
        stats = self.game_manager.get_statistics()
        
        if self.voice:
            try:
                result_text = f"游戏结束。你赢了{stats.player_wins}局，机器人赢了{stats.robot_wins}局"
                self.voice.synthesize_speech(result_text)
            except:
                pass
        
        # 通知回调
        if self.on_game_over:
            try:
                self.on_game_over(stats)
            except Exception as e:
                logger.error(f"游戏结束回调异常: {e}")
        
        self._notify_state_changed()
    
    def _select_robot_gesture(self) -> Gesture:
        """
        选择机器人手势（简单策略：随机选择）
        
        Returns:
            Gesture: 机器人手势
        """
        import random
        gestures = [Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS]
        return random.choice(gestures)
    
    def _get_result_text(self, result: GameResult) -> str:
        """
        获取结果文本
        
        Args:
            result: 游戏结果
            
        Returns:
            str: 结果文本
        """
        if result == GameResult.PLAYER_WIN:
            return "你赢了"
        elif result == GameResult.ROBOT_WIN:
            return "我赢了"
        elif result == GameResult.DRAW:
            return "平局"
        else:
            return "无效"
    
    def _notify_state_changed(self):
        """通知状态改变"""
        if self.on_state_changed:
            try:
                self.on_state_changed(self.state_machine.get_current_state())
            except Exception as e:
                logger.error(f"状态改变回调异常: {e}")
    
    def trigger_recognize(self):
        """触发识别（从外部调用）"""
        if self.state_machine.is_in_state(GameState.WAITING_GESTURE):
            self.state_machine.transition_to(GameState.RECOGNIZING)
            self._notify_state_changed()
    
    def get_current_state(self) -> GameState:
        """获取当前状态"""
        return self.state_machine.get_current_state()
    
    def get_game_statistics(self):
        """获取游戏统计信息"""
        return self.game_manager.get_statistics()
