# RKNN Toolkit 2 安装指南

## 概述

RKNN Toolkit 2 是 Rockchip 提供的用于模型转换和推理的工具包。**注意：它不在 PyPI 上**，需要从 Rockchip 官方获取。

## Windows 安装

### 步骤 1: 确定 Python 版本

```bash
python --version
```

记录您的 Python 版本（如 Python 3.10）。

### 步骤 2: 获取 RKNN Toolkit 2

#### 方法 A: 从 GitHub 获取（推荐）

1. 访问：https://github.com/rockchip-linux/rknn-toolkit2
2. 查看 Releases 或下载页面
3. 下载对应您 Python 版本的 wheel 文件：
   - Python 3.8: `cp38`
   - Python 3.9: `cp39`
   - Python 3.10: `cp310`
   - Python 3.11: `cp311`
   - Windows 64位: `win_amd64`

示例文件名：`rknn_toolkit2-1.6.0+81f21f4f-cp310-cp310-win_amd64.whl`

#### 方法 B: 从 Rockchip 官方文档获取

1. 访问 Rockchip 开发者文档
2. 查找 "RKNN Toolkit 2" 或 "NPU 开发工具"
3. 下载 Windows 版本

### 步骤 3: 安装

```bash
# 进入下载目录
cd path/to/download

# 安装 wheel 文件
pip install rknn_toolkit2-*.whl

# 或指定完整文件名
pip install rknn_toolkit2-1.6.0+81f21f4f-cp310-cp310-win_amd64.whl
```

### 步骤 4: 验证安装

```bash
python -c "from rknn.api import RKNN; print('✓ RKNN Toolkit 2 安装成功')"
```

如果成功，会显示 "✓ RKNN Toolkit 2 安装成功"。

## Linux (RK3588) 安装

### 方法一：使用 pip 安装预编译包

```bash
# 如果有预编译的 wheel 文件
pip install rknn_toolkit2-*.whl
```

### 方法二：从源码编译

```bash
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
cd rknn-toolkit2
pip install -r requirements.txt
python setup.py install
```

## 依赖要求

RKNN Toolkit 2 通常需要：

- Python 3.8-3.11
- NumPy
- OpenCV (opencv-python)
- ONNX (用于模型转换)
- 其他依赖（见 requirements.txt）

## 故障排除

### 问题 1: 找不到对应 Python 版本的 wheel

**解决方案**：
1. 检查 Python 版本是否匹配
2. 尝试使用不同 Python 版本
3. 从源码编译（需要更多依赖）

### 问题 2: 安装后无法导入

**解决方案**：
```bash
# 检查是否在正确的环境中
which python  # Linux/Mac
where python  # Windows

# 确认安装位置
pip show rknn-toolkit2

# 重新安装
pip uninstall rknn-toolkit2
pip install rknn_toolkit2-*.whl
```

### 问题 3: 缺少依赖

**解决方案**：
```bash
# 安装常见依赖
pip install numpy opencv-python onnx
```

## 替代方案

如果无法安装 RKNN Toolkit 2，可以：

1. **使用 YOLO 识别器进行开发**：
   ```bash
   python tests/test_mediapipe_gesture.py --ui
   ```

2. **在 RK3588 设备上直接测试**：
   - 在设备上安装 RKNN Toolkit 2
   - 或使用预装的环境

3. **使用 Docker 容器**（如果 Rockchip 提供）：
   ```bash
   docker pull rockchip/rknn-toolkit2
   ```

## 相关资源

- [RKNN Toolkit 2 GitHub](https://github.com/rockchip-linux/rknn-toolkit2)
- [Rockchip 官方文档](https://www.rock-chips.com/)
- [RKNN 模型转换指南](rknn_conversion.md)
- [RKNN Windows 测试指南](rknn_windows_testing.md)

## 注意事项

1. **版本兼容性**：确保 RKNN Toolkit 2 版本与 RKNN Runtime 版本兼容
2. **平台限制**：Windows 上只能使用模拟器模式，实际 NPU 加速需要在 RK3588 上
3. **许可证**：注意 Rockchip 的许可证要求
