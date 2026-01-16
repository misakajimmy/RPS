# Windows 下使用 RKNN Toolkit 2 进行测试

## 概述

RKNN Toolkit 2 可以在 Windows 上安装和使用，支持：
1. **模型转换**：将 ONNX 模型转换为 RKNN 格式
2. **模拟推理**：使用模拟器模式进行推理测试（速度较慢，但可以验证模型转换是否正确）

## 安装 RKNN Toolkit 2

**注意**：`rknn-toolkit2` **不在 PyPI 上**，需要从 Rockchip 官方获取。

### 方法一：从 Rockchip GitHub 获取（推荐）

1. **访问 GitHub 仓库**：
   - 主仓库：https://github.com/rockchip-linux/rknn-toolkit2
   - 或搜索 "rknn-toolkit2" 获取最新版本

2. **下载 Windows 版本**：
   - 查找 Releases 页面
   - 下载对应 Python 版本的 wheel 文件（如 `rknn_toolkit2-1.x.x-cp310-cp310-win_amd64.whl`）
   - 注意 Python 版本匹配（Python 3.10 对应 cp310）

3. **安装**：
   ```bash
   pip install rknn_toolkit2-1.x.x-cp310-cp310-win_amd64.whl
   ```

### 方法二：从 Rockchip 官方文档/论坛获取

1. 访问 Rockchip 官方文档或开发者论坛
2. 查找 RKNN Toolkit 2 下载页面
3. 下载 Windows 版本的安装包
4. 按照官方说明安装

### 方法三：使用 conda（如果可用）

某些 Rockchip 提供的 conda 环境可能包含预装的 rknn-toolkit2。

### 验证安装

安装后验证：

```bash
python -c "from rknn.api import RKNN; print('RKNN Toolkit 2 安装成功')"
```

如果成功，会显示 "RKNN Toolkit 2 安装成功"；如果失败，会显示 ImportError。

### 常见问题

**Q: 找不到对应 Python 版本的 wheel 文件**

A: 
- 检查 Python 版本：`python --version`
- 尝试使用不同 Python 版本
- 或从源码编译（较复杂）

**Q: 安装后仍然无法导入**

A:
- 确认安装到了正确的 Python 环境
- 检查是否在虚拟环境中
- 尝试重新安装

## Windows 上的限制

1. **模拟器模式**：在 Windows 上，RKNN Toolkit 2 使用模拟器模式进行推理
   - 速度较慢（使用 CPU 模拟 NPU）
   - 主要用于验证模型转换是否正确
   - 不能获得实际的 NPU 加速效果

2. **实际 NPU 加速**：需要在 RK3588 设备上进行
   - 需要 RKNN Runtime
   - 可以获得真正的 NPU 加速

## 在 Windows 上测试

### 1. 转换模型

```bash
# 将 PyTorch 模型转换为 RKNN
python scripts/convert_to_rknn.py models/best.pt
```

### 2. 运行测试

```bash
# 使用 RKNN 模型进行实时测试（模拟器模式）
python tests/test_rknn_gesture.py --model models/best.rknn --device 0
```

### 3. 预期行为

- **成功情况**：模型加载成功，可以进行推理（虽然速度较慢）
- **警告信息**：可能会显示 "使用模拟器模式" 的警告
- **性能**：推理速度会比实际 NPU 慢很多，但可以验证：
  - 模型转换是否正确
  - 输入输出格式是否正确
  - 识别结果是否合理

## 性能对比

| 平台 | 推理速度 | 用途 |
|------|---------|------|
| Windows (模拟器) | 慢（CPU 模拟） | 验证模型转换、调试 |
| RK3588 (NPU) | 快（NPU 加速） | 实际部署、生产环境 |

## 常见问题

### Q: Windows 上推理速度很慢

A: 这是正常的，因为使用的是模拟器模式（CPU 模拟 NPU）。实际部署时在 RK3588 上会快很多。

### Q: 如何判断是否使用模拟器模式？

A: 查看日志输出，如果显示 "可能使用模拟器模式" 或类似的警告，说明正在使用模拟器。

### Q: 模拟器模式的结果准确吗？

A: 模拟器模式的结果应该与实际 NPU 推理结果一致，但速度会慢很多。

### Q: 可以在 Windows 上获得 NPU 加速吗？

A: 不可以。NPU 加速只能在 RK3588 设备上获得。Windows 上只能使用模拟器模式进行测试。

## 建议的工作流程

1. **在 Windows 上**：
   - 转换模型（PyTorch -> ONNX -> RKNN）
   - 使用模拟器模式验证模型转换是否正确
   - 调试和优化模型

2. **在 RK3588 上**：
   - 部署转换好的 RKNN 模型
   - 使用实际的 NPU 加速进行推理
   - 进行性能测试和优化

## 相关文档

- [RKNN 模型转换指南](rknn_conversion.md)
- [RKNN 使用指南](rknn_usage.md)
