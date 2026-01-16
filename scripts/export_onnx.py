#!/usr/bin/env python3
"""
将 YOLOv8 模型导出为 ONNX 格式
Export YOLOv8 model to ONNX format

用于后续转换为 RKNN 格式以在 RK3588 上使用 NPU 加速
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger("RPS.ExportONNX")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    logger.error(f"ultralytics 未安装: {e}")
    logger.error("请运行: pip install ultralytics>=8.0.0")
    sys.exit(1)
except RuntimeError as e:
    logger.error(f"导入 ultralytics 时发生运行时错误: {e}")
    logger.error("这通常是由于 torch 和 torchvision 版本不兼容导致的")
    logger.error("请尝试:")
    logger.error("  1. 重新安装兼容的版本:")
    logger.error("     pip install --upgrade torch torchvision")
    logger.error("  2. 或者使用预编译的 wheel 文件")
    logger.error("  3. 检查 PyTorch 官方文档以获取兼容版本信息")
    sys.exit(1)
except Exception as e:
    logger.error(f"导入 ultralytics 时发生未知错误: {e}")
    logger.error("请检查依赖是否正确安装")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def export_to_onnx(model_path: str, 
                   output_path: str = None,
                   input_size: tuple = (640, 640),
                   opset_version: int = 12,
                   simplify: bool = True,
                   dynamic: bool = False):
    """
    将 YOLOv8 模型导出为 ONNX 格式
    
    Args:
        model_path: 输入的 .pt 模型文件路径
        output_path: 输出的 ONNX 文件路径（如果为 None，则自动生成）
        input_size: 输入图像尺寸 (width, height)，默认 (640, 640)
        opset_version: ONNX opset 版本，RKNN 推荐使用 11 或 12
        simplify: 是否简化 ONNX 模型（推荐）
        dynamic: 是否使用动态 batch size（不推荐用于 RKNN）
    
    Returns:
        str: 导出的 ONNX 文件路径
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    logger.info(f"加载模型: {model_path}")
    
    # 加载 YOLOv8 模型
    model = YOLO(str(model_path))
    
    # 确定输出路径
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    else:
        output_path = Path(output_path)
    
    logger.info(f"导出 ONNX 模型到: {output_path}")
    logger.info(f"输入尺寸: {input_size[0]}x{input_size[1]}")
    logger.info(f"ONNX opset 版本: {opset_version}")
    logger.info(f"简化模型: {simplify}")
    logger.info(f"动态 batch: {dynamic}")
    
    # 使用 YOLOv8 的 export 方法导出 ONNX
    # 注意：YOLOv8 的 export 方法会自动处理模型转换
    try:
        exported_path = model.export(
            format='onnx',
            imgsz=input_size,  # (height, width) 或单个整数
            opset=opset_version,
            simplify=simplify,
            dynamic=dynamic,
            half=False,  # 不使用 FP16（RKNN 可能不支持）
        )
        
        # 如果指定了输出路径，重命名文件
        if str(exported_path) != str(output_path):
            import shutil
            shutil.move(str(exported_path), str(output_path))
            logger.info(f"模型已重命名到: {output_path}")
        
        logger.info(f"✓ ONNX 模型导出成功: {output_path}")
        
        # 验证导出的模型
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX 模型验证通过")
            
            # 打印模型信息
            logger.info(f"模型输入: {[input.name for input in onnx_model.graph.input]}")
            logger.info(f"模型输出: {[output.name for output in onnx_model.graph.output]}")
            
        except ImportError:
            logger.warning("onnx 未安装，跳过模型验证")
        except Exception as e:
            logger.warning(f"ONNX 模型验证失败: {e}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"导出 ONNX 模型失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='将 YOLOv8 模型导出为 ONNX 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出默认模型
  python scripts/export_onnx.py models/best.pt
  
  # 指定输出路径和输入尺寸
  python scripts/export_onnx.py models/best.pt -o models/best.onnx --input-size 640 640
  
  # 使用 opset 11（某些 RKNN 版本可能需要）
  python scripts/export_onnx.py models/best.pt --opset 11
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='输入的 .pt 模型文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出的 ONNX 文件路径（默认：与输入文件同目录，扩展名为 .onnx）'
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
        help='ONNX opset 版本（默认: 12，RKNN 推荐 11 或 12）'
    )
    
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='不简化 ONNX 模型（默认会简化）'
    )
    
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='使用动态 batch size（不推荐用于 RKNN）'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = export_to_onnx(
            model_path=args.model_path,
            output_path=args.output,
            input_size=tuple(args.input_size),
            opset_version=args.opset,
            simplify=not args.no_simplify,
            dynamic=args.dynamic
        )
        
        print("\n" + "=" * 60)
        print("✓ 导出完成！")
        print("=" * 60)
        print(f"ONNX 模型: {output_path}")
        print("\n下一步：使用 scripts/convert_rknn.py 将 ONNX 转换为 RKNN 格式")
        print("示例命令:")
        print(f"  python scripts/convert_rknn.py {output_path}")
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
