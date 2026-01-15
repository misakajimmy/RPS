"""
游戏状态枚举
Game State Enumeration
"""
from enum import Enum, auto


class GameState(Enum):
    """游戏状态枚举"""
    IDLE = auto()              # 空闲状态
    WAITING_START = auto()     # 等待开始
    COUNTDOWN = auto()         # 倒计时
    WAITING_GESTURE = auto()   # 等待玩家手势
    RECOGNIZING = auto()       # 识别中
    ROBOT_MOVING = auto()      # 机械臂移动中
    RESULT_SHOWING = auto()    # 显示结果
    ROUND_END = auto()         # 回合结束
    GAME_OVER = auto()         # 游戏结束
    ERROR = auto()             # 错误状态
    
    def __str__(self):
        return self.name
