"""
游戏规则实现
Game Rules Implementation
"""
from typing import Tuple, Optional
from enum import Enum
from .gesture import Gesture
from ...utils.logger import setup_logger

logger = setup_logger("RPS.GameRules")


class GameResult(Enum):
    """游戏结果枚举"""
    PLAYER_WIN = "player_win"      # 玩家获胜
    ROBOT_WIN = "robot_win"        # 机器人获胜
    DRAW = "draw"                  # 平局
    INVALID = "invalid"            # 无效（未识别等）


class GameRules:
    """游戏规则类"""
    
    # 胜负规则：key胜value
    WIN_RULES = {
        Gesture.ROCK: Gesture.SCISSORS,      # 石头胜剪刀
        Gesture.PAPER: Gesture.ROCK,         # 布胜石头
        Gesture.SCISSORS: Gesture.PAPER       # 剪刀胜布
    }
    
    @staticmethod
    def judge(player_gesture: Gesture, robot_gesture: Gesture) -> GameResult:
        """
        判断游戏结果
        
        Args:
            player_gesture: 玩家手势
            robot_gesture: 机器人手势
            
        Returns:
            GameResult: 游戏结果
        """
        # 检查无效手势
        if player_gesture == Gesture.UNKNOWN or robot_gesture == Gesture.UNKNOWN:
            logger.debug(f"无效手势: 玩家={player_gesture}, 机器人={robot_gesture}")
            return GameResult.INVALID
        
        # 平局
        if player_gesture == robot_gesture:
            logger.debug(f"平局: {player_gesture}")
            return GameResult.DRAW
        
        # 判断胜负
        if GameRules.WIN_RULES.get(player_gesture) == robot_gesture:
            logger.debug(f"玩家获胜: {player_gesture} 胜 {robot_gesture}")
            return GameResult.PLAYER_WIN
        else:
            logger.debug(f"机器人获胜: {robot_gesture} 胜 {player_gesture}")
            return GameResult.ROBOT_WIN
    
    @staticmethod
    def get_winning_gesture(gesture: Gesture) -> Optional[Gesture]:
        """
        获取能战胜指定手势的手势
        
        Args:
            gesture: 目标手势
            
        Returns:
            Optional[Gesture]: 能战胜目标的手势，如果输入无效返回None
        """
        if gesture == Gesture.UNKNOWN:
            return None
        
        # 找到能战胜gesture的手势
        for winner, loser in GameRules.WIN_RULES.items():
            if loser == gesture:
                return winner
        
        return None
    
    @staticmethod
    def get_losing_gesture(gesture: Gesture) -> Optional[Gesture]:
        """
        获取会被指定手势战胜的手势
        
        Args:
            gesture: 目标手势
            
        Returns:
            Optional[Gesture]: 会被目标战胜的手势，如果输入无效返回None
        """
        return GameRules.WIN_RULES.get(gesture)
    
    @staticmethod
    def is_valid_gesture(gesture: Gesture) -> bool:
        """
        检查手势是否有效
        
        Args:
            gesture: 手势
            
        Returns:
            bool: 是否有效
        """
        return gesture != Gesture.UNKNOWN and gesture in [
            Gesture.ROCK, Gesture.PAPER, Gesture.SCISSORS
        ]
