"""
游戏状态机
Game State Machine
"""
from typing import Optional, Callable, Dict
from enum import Enum
from .game_state import GameState
from ...utils.logger import setup_logger

logger = setup_logger("RPS.GameStateMachine")


class GameStateMachine:
    """游戏状态机类"""
    
    # 状态转换规则
    VALID_TRANSITIONS: Dict[GameState, list] = {
        GameState.IDLE: [GameState.WAITING_START, GameState.ERROR],
        GameState.WAITING_START: [GameState.COUNTDOWN, GameState.IDLE, GameState.ERROR],
        GameState.COUNTDOWN: [GameState.WAITING_GESTURE, GameState.ERROR],
        GameState.WAITING_GESTURE: [GameState.RECOGNIZING, GameState.ERROR],
        GameState.RECOGNIZING: [GameState.ROBOT_MOVING, GameState.WAITING_GESTURE, GameState.ERROR],
        GameState.ROBOT_MOVING: [GameState.RESULT_SHOWING, GameState.ERROR],
        GameState.RESULT_SHOWING: [GameState.ROUND_END, GameState.ERROR],
        GameState.ROUND_END: [GameState.WAITING_GESTURE, GameState.GAME_OVER, GameState.IDLE, GameState.ERROR],
        GameState.GAME_OVER: [GameState.IDLE, GameState.WAITING_START, GameState.ERROR],
        GameState.ERROR: [GameState.IDLE, GameState.ERROR]
    }
    
    def __init__(self, initial_state: GameState = GameState.IDLE):
        """
        初始化状态机
        
        Args:
            initial_state: 初始状态
        """
        self.current_state = initial_state
        self.previous_state: Optional[GameState] = None
        self.state_handlers: Dict[GameState, Callable] = {}
        self.transition_handlers: Dict[tuple, Callable] = {}
        
        logger.info(f"游戏状态机初始化，初始状态: {self.current_state}")
    
    def register_state_handler(self, state: GameState, handler: Callable):
        """
        注册状态处理函数
        
        Args:
            state: 状态
            handler: 处理函数
        """
        self.state_handlers[state] = handler
        logger.debug(f"注册状态处理函数: {state}")
    
    def register_transition_handler(self, from_state: GameState, to_state: GameState, 
                                   handler: Callable):
        """
        注册状态转换处理函数
        
        Args:
            from_state: 源状态
            to_state: 目标状态
            handler: 处理函数
        """
        key = (from_state, to_state)
        self.transition_handlers[key] = handler
        logger.debug(f"注册转换处理函数: {from_state} -> {to_state}")
    
    def transition_to(self, new_state: GameState, force: bool = False) -> bool:
        """
        转换到新状态
        
        Args:
            new_state: 新状态
            force: 是否强制转换（忽略转换规则）
            
        Returns:
            bool: 转换是否成功
        """
        # 检查是否是有效转换
        if not force and new_state not in self.VALID_TRANSITIONS.get(self.current_state, []):
            logger.warning(f"无效的状态转换: {self.current_state} -> {new_state}")
            return False
        
        # 如果状态相同，不执行转换
        if self.current_state == new_state:
            logger.debug(f"状态未改变: {self.current_state}")
            return True
        
        # 执行转换
        old_state = self.current_state
        self.previous_state = old_state
        self.current_state = new_state
        
        logger.info(f"状态转换: {old_state} -> {new_state}")
        
        # 调用转换处理函数
        transition_key = (old_state, new_state)
        if transition_key in self.transition_handlers:
            try:
                self.transition_handlers[transition_key]()
            except Exception as e:
                logger.error(f"转换处理函数执行异常: {e}")
        
        # 调用新状态的处理函数
        if new_state in self.state_handlers:
            try:
                self.state_handlers[new_state]()
            except Exception as e:
                logger.error(f"状态处理函数执行异常: {e}")
        
        return True
    
    def get_current_state(self) -> GameState:
        """获取当前状态"""
        return self.current_state
    
    def get_previous_state(self) -> Optional[GameState]:
        """获取上一个状态"""
        return self.previous_state
    
    def can_transition_to(self, state: GameState) -> bool:
        """
        检查是否可以转换到指定状态
        
        Args:
            state: 目标状态
            
        Returns:
            bool: 是否可以转换
        """
        return state in self.VALID_TRANSITIONS.get(self.current_state, [])
    
    def reset(self, state: GameState = GameState.IDLE):
        """
        重置状态机
        
        Args:
            state: 重置后的状态
        """
        self.previous_state = self.current_state
        self.current_state = state
        logger.info(f"状态机已重置到: {state}")
    
    def is_in_state(self, state: GameState) -> bool:
        """
        检查是否在指定状态
        
        Args:
            state: 目标状态
            
        Returns:
            bool: 是否在指定状态
        """
        return self.current_state == state
