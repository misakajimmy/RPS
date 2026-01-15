# PyTorch CUDA 安装指南

## 概述

本项目默认安装的是 PyTorch CPU 版本。如果您有 NVIDIA GPU 并希望使用 GPU 加速，需要安装支持 CUDA 的 PyTorch 版本。

## 检查系统

首先检查您的系统是否支持 CUDA：

```bash
# 检查 NVIDIA 驱动和 CUDA 版本
nvidia-smi

# 检查当前 PyTorch 版本
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
```

## 安装 CUDA 版本的 PyTorch

### 方式1：使用 uv（推荐）

```bash
# 1. 卸载 CPU 版本的 PyTorch（如果已安装）
uv pip uninstall torch torchvision torchaudio

# 2. 安装 CUDA 12.1 版本的 PyTorch（兼容 CUDA 11.8-13.x）
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 方式2：使用 pip

```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
# 或
source .venv/bin/activate  # Linux/Mac

# 卸载 CPU 版本
pip uninstall torch torchvision torchaudio

# 安装 CUDA 版本
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## CUDA 版本选择

根据您的 CUDA 版本选择合适的 PyTorch 版本：

| CUDA 版本 | PyTorch 索引 URL | 说明 |
|-----------|------------------|------|
| CUDA 12.1+ | `https://download.pytorch.org/whl/cu121` | 推荐，兼容 CUDA 11.8-13.x |
| CUDA 11.8 | `https://download.pytorch.org/whl/cu118` | 适用于较旧的 CUDA 版本 |
| CUDA 11.7 | `https://download.pytorch.org/whl/cu117` | 适用于较旧的 CUDA 版本 |

**注意**：PyTorch 的 CUDA 版本向后兼容，例如 cu121 可以在 CUDA 11.8-13.x 上运行。

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU 名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

如果看到：
- `CUDA 可用: True`
- `CUDA 版本: 12.1`（或类似）
- `GPU 名称: NVIDIA GeForce RTX XXX`

说明安装成功！

## 配置项目使用 GPU

安装 CUDA 版本的 PyTorch 后，程序会自动检测并使用 GPU。您也可以在配置文件中手动指定：

```yaml
game:
  gesture_recognition:
    device: "cuda"  # 强制使用 GPU
    # 或
    device: null    # 自动检测（推荐）
```

## 常见问题

### Q: 安装后仍然显示 "未检测到 GPU"

A: 检查以下几点：
1. 确认已安装 NVIDIA 驱动：`nvidia-smi`
2. 确认 PyTorch 版本包含 CUDA：`python -c "import torch; print(torch.__version__)"` 应该显示类似 `2.x.x+cu121`
3. 确认 CUDA 版本兼容：PyTorch 的 CUDA 版本应该与系统 CUDA 版本兼容

### Q: 如何切换回 CPU 版本？

A: 
```bash
uv pip uninstall torch torchvision torchaudio
uv sync  # 这会安装 CPU 版本（从 PyPI）
```

### Q: 安装失败怎么办？

A: 
1. 检查网络连接
2. 尝试使用不同的 CUDA 版本索引 URL
3. 查看 PyTorch 官方安装页面：https://pytorch.org/get-started/locally/

## 更多信息

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [CUDA 兼容性说明](https://pytorch.org/get-started/previous-versions/)
