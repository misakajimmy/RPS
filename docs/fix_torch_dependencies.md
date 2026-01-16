# 修复 PyTorch 依赖版本兼容性问题

## 问题描述

在导入 `ultralytics` 时可能遇到以下错误：

```
RuntimeError: operator torchvision::nms does not exist
```

这通常是由于 `torch` 和 `torchvision` 版本不兼容导致的。

## 解决方案

### 方法一：使用修复脚本（推荐）

```bash
# 在项目根目录运行
chmod +x scripts/fix_torch_dependencies.sh
./scripts/fix_torch_dependencies.sh
```

### 方法二：手动修复

#### 1. 卸载现有版本

```bash
pip uninstall torch torchvision torchaudio
```

#### 2. 安装兼容版本

**ARM64 系统（如 RK3588）：**

```bash
# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**x86_64 系统：**

```bash
# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或 CUDA 版本（如果有 NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. 验证安装

```bash
python3 -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}')"
```

#### 4. 测试 ultralytics

```bash
python3 -c "from ultralytics import YOLO; print('✓ 导入成功')"
```

### 方法三：使用 uv（如果使用 uv 管理依赖）

```bash
# 卸载
uv pip uninstall torch torchvision torchaudio

# 重新安装
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 版本兼容性参考

### PyTorch 和 torchvision 版本对应关系

| PyTorch 版本 | torchvision 版本 | 说明 |
|-------------|-----------------|------|
| 2.0.x | 0.15.x | |
| 2.1.x | 0.16.x | |
| 2.2.x | 0.17.x | |
| 2.3.x | 0.18.x | |
| 2.4.x | 0.19.x | |

**注意**：建议使用最新稳定版本，并确保 `torch` 和 `torchvision` 版本匹配。

### 查看当前版本

```bash
python3 -c "import torch; import torchvision; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}')"
```

## 常见问题

### Q: 修复后仍然报错

A: 尝试：
1. 完全卸载并重新安装
2. 清除 pip 缓存：`pip cache purge`
3. 使用虚拟环境隔离依赖

### Q: ARM64 系统安装很慢

A: ARM64 的预编译 wheel 可能较少，安装时间较长。可以考虑：
1. 使用预编译的 wheel（如果可用）
2. 从源码编译（需要更多时间）
3. 使用 conda（如果有 conda 环境）

### Q: 如何确认版本是否兼容

A: 运行以下命令测试：

```bash
python3 << EOF
try:
    import torch
    import torchvision
    from ultralytics import YOLO
    print("✓ 所有依赖导入成功")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  torchvision: {torchvision.__version__}")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)
EOF
```

## 预防措施

1. **使用虚拟环境**：避免系统级依赖冲突
2. **固定版本**：在 `requirements.txt` 或 `pyproject.toml` 中固定版本
3. **定期更新**：保持依赖版本更新，但要注意兼容性

## 相关文档

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [ultralytics 官方文档](https://docs.ultralytics.com/)
