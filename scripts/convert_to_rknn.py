#!/usr/bin/env python3
"""
一键转换脚本：PyTorch (.pt) -> ONNX -> RKNN
One-step conversion script: PyTorch (.pt) -> ONNX -> RKNN

这个脚本会自动执行两步转换：
1. 将 PyTorch 模型导出为 ONNX
2. 将 ONNX 模型转换为 RKNN
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger("RPS.ConvertToRKNN")

# 导入子脚本的功能
# 使用相对导入或直接导入
import importlib.util

def load_module_from_file(module_name, file_path):
    """从文件路径加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 加载 export_onnx 模块
export_onnx_path = Path(__file__).parent / "export_onnx.py"
if export_onnx_path.exists():
    export_onnx_module = load_module_from_file("export_onnx", str(export_onnx_path))
    export_to_onnx = export_onnx_module.export_to_onnx
else:
    logger.error(f"无法找到 export_onnx.py: {export_onnx_path}")
    sys.exit(1)

# 加载 convert_rknn 模块
convert_rknn_path = Path(__file__).parent / "convert_rknn.py"
if convert_rknn_path.exists():
    convert_rknn_module = load_module_from_file("convert_rknn", str(convert_rknn_path))
    convert_onnx_to_rknn = convert_rknn_module.convert_onnx_to_rknn
else:
    logger.error(f"无法找到 convert_rknn.py: {convert_rknn_path}")
    sys.exit(1)


def convert_pt_to_rknn(pt_model_path: str,
                       output_dir: str = None,
                       keep_onnx: bool = True,
                       input_size: tuple = (640, 640),
                       opset_version: int = 12,
                       target_platform: str = 'rk3588',
                       do_quantization: bool = True,
                       **kwargs):
    """
    一键转换：PyTorch -> ONNX -> RKNN
    
    Args:
        pt_model_path: 输入的 .pt 模型文件路径
        output_dir: 输出目录（如果为 None，则使用模型文件所在目录）
        keep_onnx: 是否保留中间 ONNX 文件
        input_size: 输入图像尺寸 (width, height)
        opset_version: ONNX opset 版本
        target_platform: 目标平台
        do_quantization: 是否执行量化
        **kwargs: 其他传递给 convert_onnx_to_rknn 的参数
    
    Returns:
        tuple: (onnx_path, rknn_path)
    """
    pt_model_path = Path(pt_model_path)
    if not pt_model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {pt_model_path}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = pt_model_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("一键转换：PyTorch -> ONNX -> RKNN")
    logger.info("=" * 60)
    logger.info(f"输入模型: {pt_model_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"输入尺寸: {input_size[0]}x{input_size[1]}")
    logger.info(f"目标平台: {target_platform}")
    logger.info(f"量化: {do_quantization}")
    logger.info("")
    
    # 步骤 1: 导出 ONNX
    logger.info("步骤 1/2: 导出 ONNX 模型...")
    onnx_path = output_dir / f"{pt_model_path.stem}.onnx"
    
    try:
        onnx_path = export_to_onnx(
            model_path=str(pt_model_path),
            output_path=str(onnx_path),
            input_size=input_size,
            opset_version=opset_version,
            simplify=True,
            dynamic=False
        )
        logger.info(f"✓ ONNX 导出完成: {onnx_path}")
    except Exception as e:
        logger.error(f"✗ ONNX 导出失败: {e}")
        raise
    
    # 步骤 2: 转换为 RKNN
    logger.info("")
    logger.info("步骤 2/2: 转换为 RKNN 模型...")
    rknn_path = output_dir / f"{pt_model_path.stem}.rknn"
    
    try:
        rknn_path = convert_onnx_to_rknn(
            onnx_model_path=str(onnx_path),
            rknn_model_path=str(rknn_path),
            target_platform=target_platform,
            input_size=input_size,
            do_quantization=do_quantization,
            **kwargs
        )
        logger.info(f"✓ RKNN 转换完成: {rknn_path}")
    except Exception as e:
        logger.error(f"✗ RKNN 转换失败: {e}")
        # 如果转换失败，询问是否删除 ONNX 文件
        if not keep_onnx:
            logger.info(f"删除中间文件: {onnx_path}")
            Path(onnx_path).unlink()
        raise
    
    # 清理中间文件（如果需要）
    if not keep_onnx:
        logger.info(f"删除中间文件: {onnx_path}")
        Path(onnx_path).unlink()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ 转换完成！")
    logger.info("=" * 60)
    logger.info(f"ONNX 模型: {onnx_path}")
    logger.info(f"RKNN 模型: {rknn_path}")
    
    return str(onnx_path), str(rknn_path)


def main():
    parser = argparse.ArgumentParser(
        description='一键转换：PyTorch (.pt) -> ONNX -> RKNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换（使用默认参数）
  python scripts/convert_to_rknn.py models/best.pt
  
  # 指定输出目录和输入尺寸
  python scripts/convert_to_rknn.py models/best.pt -o models/rknn --input-size 640 640
  
  # 禁用量化
  python scripts/convert_to_rknn.py models/best.pt --no-quantization
  
  # 不保留中间 ONNX 文件
  python scripts/convert_to_rknn.py models/best.pt --no-keep-onnx

注意:
  - 需要安装 ultralytics 和 rknn-toolkit2
  - 转换过程可能需要几分钟时间
  - 量化校正数据集会自动生成（建议使用真实验证集）
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='输入的 .pt 模型文件路径'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认：与输入文件同目录）'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('WIDTH', 'HEIGHT'),
        help='输入图像尺寸（默认: 640 640）'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='ONNX opset 版本（默认: 12）'
    )
    
    parser.add_argument(
        '--target-platform',
        type=str,
        default='rk3588',
        choices=['rk3588', 'rk3566', 'rk3568', 'rk3562', 'rv1109', 'rv1126'],
        help='目标平台（默认: rk3588）'
    )
    
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        help='禁用 INT8 量化（使用 FP16）'
    )
    
    parser.add_argument(
        '--quantization-samples',
        type=int,
        default=100,
        help='量化校正样本数量（默认: 100）'
    )
    
    parser.add_argument(
        '--no-keep-onnx',
        action='store_true',
        help='不保留中间 ONNX 文件（默认保留）'
    )
    
    args = parser.parse_args()
    
    try:
        onnx_path, rknn_path = convert_pt_to_rknn(
            pt_model_path=args.model_path,
            output_dir=args.output_dir,
            keep_onnx=not args.no_keep_onnx,
            input_size=tuple(args.input_size),
            opset_version=args.opset,
            target_platform=args.target_platform,
            do_quantization=not args.no_quantization,
            num_quantization_samples=args.quantization_samples
        )
        
        print("\n" + "=" * 60)
        print("✓ 转换完成！")
        print("=" * 60)
        print(f"ONNX 模型: {onnx_path}")
        print(f"RKNN 模型: {rknn_path}")
        print("\n下一步：在 RK3588 设备上使用 RKNN 模型进行推理")
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
