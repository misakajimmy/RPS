# 剪刀石头布游戏 (Rock Paper Scissors Game)

一个基于RK3588 ARM Linux平台的智能剪刀石头布游戏，使用机械臂、摄像头和语音模块实现人机交互。

## 项目简介

本项目实现了一个完整的剪刀石头布游戏系统，通过摄像头识别玩家的手势，机械臂做出相应的手势回应，并通过语音模块提供交互反馈。

## 硬件平台

- **运行平台**: 瑞芯微RK3588 ARM Linux
- **机械臂**: 幻尔uHand2.0
- **摄像头**: USB相机
- **语音模块**: 亚博智能AI语音交互模块CI1302

## 技术架构

- **开发语言**: Python 3.8+
- **架构模式**: 工厂模式（支持硬件模块和平台的灵活替换）
- **项目结构**: 模块化设计，硬件抽象层与业务逻辑分离

## 项目结构

```
RPS/
├── src/                           # 源代码目录
│   ├── hardware/                  # 硬件抽象层
│   │   ├── base/                  # 抽象基类接口
│   │   ├── factory/               # 工厂类
│   │   └── implementations/       # 具体硬件实现
│   │       ├── robot_arm/         # 机械臂实现
│   │       ├── camera/            # 摄像头实现
│   │       └── voice/             # 语音模块实现
│   ├── game/                      # 游戏逻辑
│   │   ├── gesture_recognition/   # 手势识别
│   │   ├── game_logic/            # 游戏规则
│   │   └── state_machine/         # 状态机
│   ├── utils/                     # 工具类
│   └── main.py                    # 主程序入口
├── config/                        # 配置文件
├── tests/                         # 测试文件
├── docs/                          # 文档
├── pyproject.toml                # 项目配置（uv/pip）
├── requirements.txt              # Python依赖（传统pip方式）
└── PLAN.md                       # 项目开发计划
```

## 快速开始

### 环境要求

- Python 3.8 或更高版本
- RK3588 ARM Linux 系统（或其他支持的系统）q
- 已连接的硬件设备（机械臂、摄像头、语音模块）

**注意**：项目使用 **YOLOv8-Pose** 进行手势识别，支持 ARM64 Linux（RK3588）。详见 [YOLOv8 手势识别指南](docs/yolov8_gesture_recognition.md)。

### 安装步骤

#### 方式一：使用 uv（推荐）

[uv](https://github.com/astral-sh/uv) 是一个快速的 Python 包管理器和项目管理工具。

1. 安装 uv
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 pip
pip install uv
```

2. 克隆项目
```bash
git clone <repository-url>
cd RPS
```

3. 使用 uv 创建虚拟环境并安装依赖
```bash
# 创建虚拟环境并安装依赖（包括 YOLOv8）
uv sync

# 推荐：安装 HuggingFace 模型支持（用于更好的手势识别）
uv sync --extra huggingface

# 如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch（用于 GPU 加速）
# 注意：需要手动安装，因为需要从 PyTorch 官方索引安装
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 或安装开发依赖
uv sync --extra dev

# 或安装所有可选依赖（包括串口通信等）
uv sync --extra all
```

**ARM64 用户注意**：YOLOv8 完全支持 ARM64 Linux。推荐使用 HuggingFace 手势识别模型，首次运行时会自动下载模型文件。

**Windows/Linux GPU 用户注意**：如果您的系统有 NVIDIA GPU，需要安装支持 CUDA 的 PyTorch 版本才能使用 GPU 加速。默认安装的是 CPU 版本。请参考 [PyTorch CUDA 安装指南](docs/pytorch_cuda_setup.md) 安装 CUDA 版本的 PyTorch。

**RK3588 用户注意**：如果要在 RK3588 上使用 NPU 加速，需要将模型转换为 RKNN 格式。请参考 [RKNN 模型转换指南](docs/rknn_conversion.md) 进行转换。

4. 配置硬件参数
```bash
# 复制配置文件示例
cp config/config.yaml.example config/config.yaml
# 编辑 config/config.yaml 配置硬件连接参数
```

5. 运行程序
```bash
# 使用 uv 运行（推荐）
uv run python src/main.py

# 或激活虚拟环境后运行
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
python src/main.py
```

#### 方式二：使用传统 pip

1. 克隆项目
```bash
git clone <repository-url>
cd RPS
```

2. 创建虚拟环境（推荐）
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置硬件参数
```bash
# 复制配置文件示例
cp config/config.yaml.example config/config.yaml
# 编辑 config/config.yaml 配置硬件连接参数
```

5. 运行程序
```bash
python src/main.py
```

## 开发计划

详细开发计划请参见 [PLAN.md](PLAN.md)

## 设计理念

### 工厂模式

项目采用工厂模式设计，通过抽象基类定义硬件接口，使用工厂类创建具体硬件实例。这种设计使得：

- 支持不同品牌/型号硬件的灵活替换
- 便于单元测试（可使用模拟硬件）
- 代码结构清晰，易于维护和扩展

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

[待定]

## 联系方式

[待定]
