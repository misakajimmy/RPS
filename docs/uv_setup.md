# uv 环境管理指南

## 什么是 uv？

[uv](https://github.com/astral-sh/uv) 是由 Astral 开发的极速 Python 包管理器和项目管理工具，用 Rust 编写，比传统的 pip 快 10-100 倍。

## 安装 uv

### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后，将 `~/.cargo/bin` 添加到 PATH 环境变量中。

### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 使用 pip 安装

```bash
pip install uv
```

### 验证安装

```bash
uv --version
```

## 项目配置

项目使用 `pyproject.toml` 进行依赖管理，支持：

- **基础依赖**：自动安装（opencv-python, numpy, Pillow, mediapipe）
- **开发依赖**：`--extra dev`（pytest, mypy 等）
- **串口通信**：`--extra serial`（pyserial）
- **所有依赖**：`--extra all`

## 基本使用

### 初始化项目

```bash
# 创建虚拟环境并安装所有依赖
uv sync

# 安装开发依赖
uv sync --extra dev

# 安装所有可选依赖
uv sync --extra all
```

### 运行程序

```bash
# 使用 uv 运行（自动激活虚拟环境）
uv run python src/main.py

# 或手动激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
python src/main.py
```

### 添加依赖

```bash
# 添加新依赖
uv add package-name

# 添加开发依赖
uv add --dev package-name

# 添加可选依赖组
uv add --optional serial pyserial
```

### 更新依赖

```bash
# 更新所有依赖
uv sync --upgrade

# 更新特定依赖
uv sync --upgrade-package package-name
```

### 查看依赖

```bash
# 列出所有依赖
uv tree

# 查看项目信息
uv pip list
```

### 运行脚本

```bash
# 运行 Python 脚本
uv run python script.py

# 运行测试
uv run pytest

# 运行类型检查
uv run mypy src/
```

## 虚拟环境管理

### 创建虚拟环境

```bash
# uv 会自动创建 .venv 目录
uv sync
```

### 激活虚拟环境

```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 删除虚拟环境

```bash
# 删除 .venv 目录
rm -rf .venv  # Linux/Mac
rmdir /s .venv  # Windows
```

## 配置文件

### pyproject.toml

项目的主要配置文件，包含：
- 项目元数据
- 依赖声明
- 构建配置
- 工具配置（pytest, mypy, ruff）

### .uvrc（可选）

uv 的配置文件，可以设置：
- Python 版本
- 虚拟环境位置
- 包索引 URL
- 超时时间等

## 优势

相比传统 pip + venv：

1. **速度更快**：安装包的速度快 10-100 倍
2. **统一管理**：依赖和虚拟环境统一管理
3. **更好的依赖解析**：更快的依赖解析和冲突检测
4. **跨平台**：Windows、Linux、macOS 统一体验
5. **兼容性好**：完全兼容 pip 和 PyPI

## 常见问题

### Q: uv 和 pip 可以混用吗？

A: 可以，但建议统一使用 uv 或 pip，避免依赖冲突。

### Q: 如何迁移现有项目到 uv？

A: 
1. 创建 `pyproject.toml`（本项目已创建）
2. 运行 `uv sync` 安装依赖
3. 删除旧的 `venv` 目录

### Q: uv.lock 文件需要提交吗？

A: 建议提交，确保团队成员使用相同的依赖版本。

### Q: 如何在 CI/CD 中使用 uv？

A:
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 运行测试
uv run pytest
```

## 更多信息

- [uv 官方文档](https://docs.astral.sh/uv/)
- [uv GitHub](https://github.com/astral-sh/uv)
