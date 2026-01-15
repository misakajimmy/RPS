# 游戏核心逻辑使用指南

## 概述

阶段四实现了完整的游戏核心逻辑，包括手势识别（使用PyTorch）、游戏状态机、游戏规则和用户交互逻辑。

## 主要组件

### 1. 手势识别模块

#### GestureRecognizer（手势识别器）

使用MediaPipe进行手势识别：

```python
from src.game.gesture_recognition import GestureRecognizer
import numpy as np

# 创建识别器
recognizer = GestureRecognizer(
    min_detection_confidence=0.5,  # 手部检测最小置信度
    min_tracking_confidence=0.5,   # 手部跟踪最小置信度
    max_num_hands=1,               # 最大检测手部数量
    confidence_threshold=0.7        # 手势识别置信度阈值
)

# 识别手势（输入BGR格式图像）
frame = camera.capture_frame()  # numpy数组（BGR格式）
gesture, confidence, probabilities = recognizer.recognize(frame)

print(f"识别结果: {gesture.value}, 置信度: {confidence:.3f}")
```

#### RecognitionResultProcessor（识别结果处理器）

用于平滑识别结果，减少抖动：

```python
from src.game.gesture_recognition import RecognitionResult, RecognitionResultProcessor
from datetime import datetime

processor = RecognitionResultProcessor(
    smoothing_window=5,  # 平滑窗口大小
    min_confidence=0.7
)

# 添加识别结果
result = RecognitionResult(
    gesture=gesture,
    confidence=confidence,
    probabilities=probabilities,
    timestamp=datetime.now()
)
processor.add_result(result)

# 获取平滑后的结果
smoothed_gesture = processor.get_smoothed_result()
```

### 2. 游戏状态机

#### GameStateMachine（游戏状态机）

管理游戏状态转换：

```python
from src.game.state_machine import GameState, GameStateMachine

# 创建状态机
state_machine = GameStateMachine(initial_state=GameState.IDLE)

# 注册状态处理函数
def handle_waiting_gesture():
    print("等待玩家手势")

state_machine.register_state_handler(GameState.WAITING_GESTURE, handle_waiting_gesture)

# 状态转换
state_machine.transition_to(GameState.WAITING_START)

# 检查当前状态
current_state = state_machine.get_current_state()
```

#### 游戏状态列表

- `IDLE`: 空闲状态
- `WAITING_START`: 等待开始
- `COUNTDOWN`: 倒计时
- `WAITING_GESTURE`: 等待玩家手势
- `RECOGNIZING`: 识别中
- `ROBOT_MOVING`: 机械臂移动中
- `RESULT_SHOWING`: 显示结果
- `ROUND_END`: 回合结束
- `GAME_OVER`: 游戏结束
- `ERROR`: 错误状态

### 3. 游戏规则

#### GameRules（游戏规则）

实现胜负判断逻辑：

```python
from src.game.game_logic import GameRules, GameResult, Gesture

# 判断游戏结果
result = GameRules.judge(Gesture.ROCK, Gesture.SCISSORS)
# 返回: GameResult.PLAYER_WIN

# 获取能战胜指定手势的手势
winning_gesture = GameRules.get_winning_gesture(Gesture.SCISSORS)
# 返回: Gesture.ROCK

# 检查手势是否有效
is_valid = GameRules.is_valid_gesture(Gesture.ROCK)
# 返回: True
```

#### GameManager（游戏管理器）

管理游戏流程和统计：

```python
from src.game.game_logic import GameManager, Gesture

# 创建游戏管理器
game_manager = GameManager(max_rounds=5)

# 开始游戏
game_manager.start_game()

# 进行一回合
round_result = game_manager.play_round(
    player_gesture=Gesture.ROCK,
    robot_gesture=Gesture.SCISSORS
)

# 获取统计信息
stats = game_manager.get_statistics()
print(f"玩家胜: {stats.player_wins}, 机器人胜: {stats.robot_wins}")

# 检查游戏是否结束
if game_manager.is_game_over():
    game_manager.end_game()
```

### 4. 游戏控制器（整合所有组件）

#### GameController（游戏控制器）

整合所有游戏组件，提供完整的游戏流程控制：

```python
from src.game import GameController
from src.game.gesture_recognition import GestureRecognizer
from src.hardware.config_manager import HardwareConfigManager

# 初始化硬件
config_manager = HardwareConfigManager()
config_manager.load_config()
camera = config_manager.get_camera()
robot_arm = config_manager.get_robot_arm()
voice = config_manager.get_voice()

# 初始化手势识别器
recognizer = GestureRecognizer()

# 创建游戏控制器
controller = GameController(
    gesture_recognizer=recognizer,
    robot_arm=robot_arm,
    camera=camera,
    voice=voice,
    max_rounds=5,
    countdown_seconds=3
)

# 注册回调函数
def on_round_result(round_result):
    print(f"回合 {round_result.round_number}: {round_result.result.value}")

def on_game_over(stats):
    print(f"游戏结束，玩家胜率: {stats.get_win_rate():.2%}")

controller.on_round_result = on_round_result
controller.on_game_over = on_game_over

# 开始游戏
controller.start_game()

# 触发识别（在适当的时候调用）
controller.trigger_recognize()

# 停止游戏
controller.stop_game()
```

## 完整使用示例

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game import GameController
from src.game.gesture_recognition import GestureRecognizer
from src.hardware.config_manager import HardwareConfigManager
from src.hardware.implementations.camera import USBCamera

def main():
    # 1. 初始化硬件
    camera = USBCamera(device_id=0)
    camera.connect()
    
    # 2. 初始化手势识别器
    recognizer = GestureRecognizer(
        model_path=None,  # 使用默认模型
        confidence_threshold=0.7
    )
    
    # 3. 创建游戏控制器
    controller = GameController(
        gesture_recognizer=recognizer,
        camera=camera,
        max_rounds=5
    )
    
    # 4. 开始游戏
    controller.start_game()
    
    # 5. 游戏循环
    import time
    while controller.get_current_state() != GameState.GAME_OVER:
        state = controller.get_current_state()
        
        if state == GameState.WAITING_GESTURE:
            # 触发识别
            controller.trigger_recognize()
        
        time.sleep(0.1)
    
    # 6. 显示统计信息
    stats = controller.get_game_statistics()
    print(f"\n游戏统计:")
    print(f"总回合数: {stats.total_rounds}")
    print(f"玩家胜: {stats.player_wins}")
    print(f"机器人胜: {stats.robot_wins}")
    print(f"平局: {stats.draws}")
    print(f"胜率: {stats.get_win_rate():.2%}")
    
    # 7. 清理
    camera.disconnect()

if __name__ == "__main__":
    main()
```

## MediaPipe手势识别说明

当前实现使用MediaPipe Hands进行手势识别，无需训练模型即可使用：

1. **MediaPipe Hands**：自动检测手部关键点
2. **手势判断**：基于手指状态判断手势类型
   - **石头**：所有手指弯曲（0个伸直）
   - **布**：所有手指伸直（5个伸直）
   - **剪刀**：食指和中指伸直，其他弯曲（2个伸直）

### 可视化调试

可以使用`draw_landmarks`方法在图像上绘制手部关键点：

```python
import cv2

# 识别手势
frame = camera.capture_frame()
gesture, confidence, _ = recognizer.recognize(frame)

# 处理图像以获取关键点（用于绘制）
rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = recognizer.hands.process(rgb_image)

# 绘制关键点
annotated_image = recognizer.draw_landmarks(frame, results)
cv2.imshow("Hand Landmarks", annotated_image)
```

## 注意事项

1. **模型要求**：输入图像应为224x224的RGB格式
2. **置信度阈值**：根据实际模型性能调整`confidence_threshold`
3. **状态转换**：状态机有严格的转换规则，不能随意跳转
4. **资源管理**：使用完毕后记得断开硬件连接
5. **错误处理**：所有组件都包含错误处理和日志记录

## 扩展建议

1. **智能策略**：在`GameController._select_robot_gesture()`中实现更智能的机器人策略
2. **模型优化**：使用更先进的网络架构（如ResNet、MobileNet等）
3. **数据增强**：在训练时使用数据增强提高模型鲁棒性
4. **多线程**：将识别和机械臂控制放在不同线程中提高响应速度
