# 系统集成使用指南

## 概述

阶段五完成了完整的系统集成，包括主程序框架、异常处理机制、日志系统和各模块的集成。

## 快速开始

### 1. 配置文件准备

首先，复制配置文件示例并修改：

```bash
cp config/config.yaml.example config/config.yaml
```

然后编辑 `config/config.yaml`，配置你的硬件参数。

### 2. 运行程序

#### 方式一：使用主程序入口

```bash
python src/main.py
```

或指定配置文件：

```bash
python src/main.py --config /path/to/config.yaml
```

#### 方式二：使用Application类

```python
from src.app import Application

app = Application(config_path="config/config.yaml")
app.start()
```

## 系统架构

### Application类

`Application` 类是系统的主入口，负责：

1. **配置加载**：从YAML文件加载配置
2. **硬件初始化**：创建和连接所有硬件设备
3. **游戏组件初始化**：初始化手势识别器和游戏控制器
4. **主循环管理**：运行游戏主循环
5. **资源清理**：程序退出时清理所有资源

### 初始化流程

```
Application.start()
  ├─ _load_config()           # 加载配置文件
  ├─ _initialize_hardware()    # 初始化硬件
  │   ├─ 创建摄像头
  │   ├─ 创建机械臂
  │   └─ 创建语音模块
  ├─ _initialize_gesture_recognizer()  # 初始化手势识别器
  └─ _initialize_game_controller()     # 初始化游戏控制器
```

### 主循环流程

```
主循环
  ├─ 检查游戏状态
  ├─ 根据状态执行相应操作
  │   ├─ WAITING_GESTURE: 触发识别
  │   ├─ GAME_OVER: 等待或退出
  │   └─ ERROR: 尝试恢复
  └─ 循环继续
```

## 异常处理

### 异常类型

系统定义了以下异常类型：

- `HardwareException`: 硬件相关异常基类
  - `RobotArmException`: 机械臂异常
  - `CameraException`: 摄像头异常
  - `VoiceException`: 语音模块异常
- `RecognitionException`: 手势识别异常
- `GameException`: 游戏逻辑异常
- `ConfigurationException`: 配置异常

### 使用错误处理器

```python
from src.utils.error_handler import global_error_handler
from src.utils.exceptions import CameraException

try:
    # 可能抛出异常的代码
    camera.capture_frame()
except CameraException as e:
    # 自动处理异常
    global_error_handler.handle(e, "捕获图像")
```

### 注册自定义错误处理

```python
from src.utils.error_handler import ErrorHandler
from src.utils.exceptions import CameraException

def my_camera_error_handler(exception, context):
    print(f"自定义处理: {exception.message}")

error_handler = ErrorHandler()
error_handler.register_handler(CameraException, my_camera_error_handler)
```

## 日志系统

### 配置日志

在 `config/config.yaml` 中配置：

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "logs/rps.log"  # 日志文件路径
```

### 使用日志

```python
from src.utils.logger import setup_logger

logger = setup_logger("MyModule")
logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
```

### 从配置加载日志

```python
from src.utils.logger import setup_logger_from_config
from src.utils.config_loader import ConfigLoader

config = ConfigLoader.load_config("config/config.yaml")
logging_config = ConfigLoader.get_logging_config(config)
logger = setup_logger_from_config(logging_config, "MyModule")
```

## 集成测试

### 运行集成测试

```bash
python tests/test_integration.py
```

测试包括：

1. **摄像头与手势识别集成测试**
   - 测试摄像头连接
   - 测试手势识别功能
   - 可视化识别结果

2. **游戏控制器集成测试**
   - 测试游戏控制器创建
   - 测试状态转换
   - 测试游戏流程

3. **硬件配置管理器测试**
   - 测试配置加载
   - 测试硬件创建

4. **错误处理机制测试**
   - 测试异常处理
   - 测试错误回调

5. **完整系统测试**
   - 测试完整初始化流程
   - 测试所有组件集成

## 故障排除

### 问题：配置文件加载失败

**解决方案**：
- 检查配置文件路径是否正确
- 检查YAML格式是否正确
- 查看日志文件获取详细错误信息

### 问题：硬件连接失败

**解决方案**：
- 检查硬件是否已连接
- 检查串口/设备ID是否正确
- 检查硬件权限（Linux系统）
- 查看日志获取详细错误信息

### 问题：手势识别失败

**解决方案**：
- 确保摄像头正常工作
- 确保手部在摄像头视野内
- 调整识别置信度阈值
- 检查光照条件

### 问题：游戏状态卡住

**解决方案**：
- 查看日志了解当前状态
- 检查是否有异常发生
- 尝试重置游戏：`controller.reset_game()`

## 扩展开发

### 添加新的硬件类型

1. 实现硬件基类接口
2. 在工厂类中注册
3. 在配置文件中添加配置项

### 添加新的游戏功能

1. 在 `GameController` 中添加新的状态处理
2. 更新状态机转换规则
3. 添加相应的回调函数

### 自定义错误处理

1. 定义新的异常类型（继承自相应基类）
2. 注册错误处理函数
3. 在相应位置抛出异常

## 性能优化建议

1. **异步处理**：将硬件操作放在独立线程中
2. **缓存识别结果**：减少重复识别
3. **资源池**：复用硬件连接
4. **日志级别**：生产环境使用INFO或WARNING级别

## 部署注意事项

1. **权限**：确保程序有访问硬件的权限
2. **依赖**：安装所有必需的Python包
3. **配置文件**：确保配置文件路径正确
4. **日志目录**：确保日志目录可写
5. **资源清理**：确保程序退出时正确清理资源
