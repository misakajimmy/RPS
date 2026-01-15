"""
游戏管理器
Game Manager
"""
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from .gesture import Gesture
from .game_rules import GameRules, GameResult
from ...utils.logger import setup_logger

logger = setup_logger("RPS.GameManager")


@dataclass
class RoundResult:
    """回合结果数据类"""
    round_number: int
    player_gesture: Gesture
    robot_gesture: Gesture
    result: GameResult
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'round_number': self.round_number,
            'player_gesture': self.player_gesture.value,
            'robot_gesture': self.robot_gesture.value,
            'result': self.result.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class GameStatistics:
    """游戏统计信息"""
    total_rounds: int = 0
    player_wins: int = 0
    robot_wins: int = 0
    draws: int = 0
    invalid_rounds: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def get_win_rate(self) -> float:
        """
        获取玩家胜率
        
        Returns:
            float: 胜率（0.0-1.0）
        """
        valid_rounds = self.total_rounds - self.invalid_rounds
        if valid_rounds == 0:
            return 0.0
        return self.player_wins / valid_rounds
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'total_rounds': self.total_rounds,
            'player_wins': self.player_wins,
            'robot_wins': self.robot_wins,
            'draws': self.draws,
            'invalid_rounds': self.invalid_rounds,
            'win_rate': self.get_win_rate(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class GameManager:
    """游戏管理器类"""
    
    def __init__(self, max_rounds: int = 5):
        """
        初始化游戏管理器
        
        Args:
            max_rounds: 最大回合数
        """
        self.max_rounds = max_rounds
        self.current_round = 0
        self.statistics = GameStatistics()
        self.round_history: List[RoundResult] = []
        self.is_game_active = False
        
        logger.info(f"游戏管理器初始化，最大回合数: {max_rounds}")
    
    def start_game(self):
        """开始游戏"""
        if self.is_game_active:
            logger.warning("游戏已在进行中")
            return
        
        self.is_game_active = True
        self.current_round = 0
        self.statistics = GameStatistics()
        self.statistics.start_time = datetime.now()
        self.round_history.clear()
        
        logger.info("游戏开始")
    
    def end_game(self):
        """结束游戏"""
        if not self.is_game_active:
            logger.warning("游戏未在进行中")
            return
        
        self.is_game_active = False
        self.statistics.end_time = datetime.now()
        
        logger.info(f"游戏结束，总回合数: {self.statistics.total_rounds}, "
                   f"玩家胜: {self.statistics.player_wins}, "
                   f"机器人胜: {self.statistics.robot_wins}, "
                   f"平局: {self.statistics.draws}")
    
    def play_round(self, player_gesture: Gesture, robot_gesture: Gesture) -> RoundResult:
        """
        进行一回合游戏
        
        Args:
            player_gesture: 玩家手势
            robot_gesture: 机器人手势
            
        Returns:
            RoundResult: 回合结果
        """
        if not self.is_game_active:
            logger.warning("游戏未开始，无法进行回合")
            return None
        
        self.current_round += 1
        
        # 判断结果
        result = GameRules.judge(player_gesture, robot_gesture)
        
        # 更新统计
        self.statistics.total_rounds += 1
        if result == GameResult.PLAYER_WIN:
            self.statistics.player_wins += 1
        elif result == GameResult.ROBOT_WIN:
            self.statistics.robot_wins += 1
        elif result == GameResult.DRAW:
            self.statistics.draws += 1
        else:
            self.statistics.invalid_rounds += 1
        
        # 创建回合结果
        round_result = RoundResult(
            round_number=self.current_round,
            player_gesture=player_gesture,
            robot_gesture=robot_gesture,
            result=result
        )
        
        self.round_history.append(round_result)
        
        # 检查是否达到最大回合数
        if self.current_round >= self.max_rounds:
            self.end_game()
        
        logger.info(f"回合 {self.current_round}: {result.value}")
        
        return round_result
    
    def get_current_round(self) -> int:
        """获取当前回合数"""
        return self.current_round
    
    def get_remaining_rounds(self) -> int:
        """获取剩余回合数"""
        return max(0, self.max_rounds - self.current_round)
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        return not self.is_game_active or self.current_round >= self.max_rounds
    
    def get_statistics(self) -> GameStatistics:
        """获取游戏统计信息"""
        return self.statistics
    
    def get_round_history(self) -> List[RoundResult]:
        """获取回合历史"""
        return self.round_history.copy()
    
    def get_last_round_result(self) -> Optional[RoundResult]:
        """获取上一回合结果"""
        if self.round_history:
            return self.round_history[-1]
        return None
    
    def reset(self):
        """重置游戏"""
        self.is_game_active = False
        self.current_round = 0
        self.statistics = GameStatistics()
        self.round_history.clear()
        logger.info("游戏已重置")
