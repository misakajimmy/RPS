#!/usr/bin/env python3
"""
下载 HuggingFace 手势识别模型到本地 models/ 目录
Download HuggingFace gesture recognition model to local models/ directory
"""
import urllib.request
import os
from pathlib import Path

def download_model():
    """下载模型文件"""
    # 模型下载地址
    url = "https://huggingface.co/lewiswatson/yolov8x-tuned-hand-gestures/resolve/main/weights/best.pt"
    
    # 输出路径
    models_dir = Path("models")
    output_path = models_dir / "yolov8x-tuned-hand-gestures.pt"
    
    # 创建 models 目录
    models_dir.mkdir(exist_ok=True)
    
    # 检查文件是否已存在
    if output_path.exists():
        print(f"模型文件已存在: {output_path}")
        response = input("是否重新下载？(y/N): ")
        if response.lower() != 'y':
            print("跳过下载")
            return
    
    print(f"正在下载模型...")
    print(f"  来源: {url}")
    print(f"  目标: {output_path}")
    print(f"  文件大小: 约 200-300 MB")
    print(f"  请稍候...")
    
    try:
        # 下载文件
        urllib.request.urlretrieve(url, output_path)
        print(f"\n[OK] 模型下载成功: {output_path}")
        print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        print(f"\n请手动下载:")
        print(f"  1. 访问: {url}")
        print(f"  2. 保存到: {output_path}")
        raise

if __name__ == "__main__":
    download_model()
