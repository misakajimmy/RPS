#!/bin/bash
# 修复 PyTorch 和 torchvision 版本兼容性问题
# Fix PyTorch and torchvision version compatibility issues

echo "=========================================="
echo "修复 PyTorch 依赖版本兼容性"
echo "=========================================="
echo ""

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "警告: 未检测到虚拟环境"
    echo "建议在虚拟环境中运行此脚本"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "步骤 1: 卸载现有的 torch 和 torchvision..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo "步骤 2: 安装兼容的 PyTorch 版本..."
echo "注意: 根据您的系统选择合适的版本"

# 检测系统架构
ARCH=$(uname -m)
OS=$(uname -s)

echo "检测到系统: $OS $ARCH"

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "ARM64 架构检测到"
    echo "安装 CPU 版本的 PyTorch（ARM64）..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "$ARCH" = "x86_64" ]; then
    echo "x86_64 架构检测到"
    echo "安装 CPU 版本的 PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "未知架构，尝试安装默认版本..."
    pip install torch torchvision torchaudio
fi

echo ""
echo "步骤 3: 验证安装..."
python3 -c "import torch; import torchvision; print(f'PyTorch 版本: {torch.__version__}'); print(f'torchvision 版本: {torchvision.__version__}')" || {
    echo "验证失败，请手动检查"
    exit 1
}

echo ""
echo "步骤 4: 测试 ultralytics 导入..."
python3 -c "from ultralytics import YOLO; print('✓ ultralytics 导入成功')" || {
    echo "✗ ultralytics 导入失败"
    echo "请检查错误信息并手动修复"
    exit 1
}

echo ""
echo "=========================================="
echo "✓ 修复完成！"
echo "=========================================="
