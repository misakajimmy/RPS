# 安装支持 CUDA 的 PyTorch

## 当前情况
- 您的系统：NVIDIA GeForce RTX 3080 Ti
- CUDA 版本：13.1
- 当前 PyTorch：2.9.1+cpu（CPU 版本，不支持 GPU）

## 安装步骤

### 方式1：使用 uv（推荐）

```bash
# 1. 卸载 CPU 版本的 PyTorch
uv pip uninstall torch torchvision torchaudio

# 2. 安装支持 CUDA 12.1 的 PyTorch（兼容 CUDA 13.1）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**注意**：下载大小约 2.3GB，可能需要一些时间。

### 方式2：使用 pip（如果 uv 不可用）

```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
# 或
source .venv/bin/activate  # Linux/Mac

# 卸载 CPU 版本
pip uninstall torch torchvision torchaudio

# 安装 CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU 名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

如果看到：
- `CUDA 可用: True`
- `CUDA 版本: 12.1`（或类似）
- `GPU 名称: NVIDIA GeForce RTX 3080 Ti`

说明安装成功！

## CUDA 版本选择

根据您的 CUDA 13.1，可以选择：
- **cu121**（CUDA 12.1）- 推荐，向后兼容 CUDA 13.1
- **cu118**（CUDA 11.8）- 如果 cu121 不可用

## 其他 CUDA 版本

如果需要其他 CUDA 版本，访问：
https://pytorch.org/get-started/locally/

选择您的配置（Windows、CUDA 12.1、Python 3.11）获取安装命令。
