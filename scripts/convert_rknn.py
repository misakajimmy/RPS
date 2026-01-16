#!/usr/bin/env python3
"""
将 ONNX 模型转换为 RKNN 格式
Convert ONNX model to RKNN format for RK3588 NPU acceleration

需要安装 RKNN Toolkit 2:
  pip install rknn-toolkit2
"""
import sys
import argparse
import numpy as np
import tempfile
import shutil
import cv2
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger("RPS.ConvertRKNN")

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
except ImportError:
    logger.error("rknn-toolkit2 未安装")
    logger.error("请安装: pip install rknn-toolkit2")
    logger.error("注意：RKNN Toolkit 2 可能需要从 Rockchip 官方获取")
    sys.exit(1)


def create_calibration_dataset(dataset_path: str = None, 
                               num_samples: int = 100,
                               input_size: tuple = (640, 640)):
    """
    创建量化校正数据集
    
    Args:
        dataset_path: 数据集保存路径（如果为 None，则创建临时目录）
        num_samples: 样本数量
        input_size: 输入图像尺寸 (width, height)
    
    Returns:
        str: 数据集文件夹路径（RKNN Toolkit 2 需要路径字符串）
    """
    logger.info(f"创建量化校正数据集: {num_samples} 个样本，尺寸 {input_size}")
    
    # 确定数据集保存路径
    if dataset_path is None:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='rknn_calib_')
        dataset_path = temp_dir
        logger.info(f"使用临时目录: {dataset_path}")
    else:
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)
        dataset_path = str(dataset_path)
    
    # 生成随机图像数据并保存为文件
    # RKNN Toolkit 2 需要图像文件路径列表文件（.txt）
    image_paths = []
    for i in range(num_samples):
        # 生成随机图像数据 [0, 255]
        img = np.random.randint(0, 256, (input_size[1], input_size[0], 3), dtype=np.uint8)
        # BGR 格式（OpenCV 默认）
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
        
        # 保存为图像文件
        img_path = Path(dataset_path) / f"calib_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img_bgr)
        image_paths.append(img_path.resolve())
    
    # 创建数据集列表文件（.txt），每行一个图像路径
    # RKNN Toolkit 2 需要文本文件路径，而不是目录或列表
    dataset_txt_path = Path(dataset_path) / "dataset.txt"
    with open(dataset_txt_path, 'w') as f:
        for img_path in image_paths:
            # 写入绝对路径
            f.write(str(img_path) + '\n')
    
    logger.info(f"✓ 创建了 {num_samples} 个校正样本")
    logger.info(f"✓ 数据集列表文件: {dataset_txt_path}")
    
    # 返回数据集列表文件路径（RKNN Toolkit 2 需要 .txt 文件路径）
    return str(dataset_txt_path)


def convert_onnx_to_rknn(onnx_model_path: str,
                         rknn_model_path: str = None,
                         target_platform: str = 'rk3588',
                         input_size: tuple = (640, 640),
                         mean_values: list = None,
                         std_values: list = None,
                         do_quantization: bool = True,
                         quantization_dataset: str = None,  # 可以是目录路径、文本文件路径或图像路径列表
                         num_quantization_samples: int = 100,
                         output_dir: str = None):
    """
    将 ONNX 模型转换为 RKNN 格式
    
    Args:
        onnx_model_path: 输入的 ONNX 模型文件路径
        rknn_model_path: 输出的 RKNN 文件路径（如果为 None，则自动生成）
        target_platform: 目标平台，默认 'rk3588'
        input_size: 输入图像尺寸 (width, height)
        mean_values: 输入预处理中的均值（RGB 顺序），默认 [0, 0, 0]
        std_values: 输入预处理中的标准差（RGB 顺序），默认 [1, 1, 1]
        do_quantization: 是否执行 INT8 量化（推荐）
        quantization_dataset: 量化校正数据集路径（字符串，如果为 None，则自动生成）
        num_quantization_samples: 量化校正样本数量
        output_dir: 输出目录（如果为 None，则使用 ONNX 文件所在目录）
    
    Returns:
        str: 导出的 RKNN 文件路径
    """
    onnx_model_path = Path(onnx_model_path)
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_model_path}")
    
    # 确定输出路径
    if rknn_model_path is None:
        rknn_model_path = onnx_model_path.parent / f"{onnx_model_path.stem}.rknn"
    else:
        rknn_model_path = Path(rknn_model_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rknn_model_path = output_dir / rknn_model_path.name
    
    logger.info("=" * 60)
    logger.info("ONNX 转 RKNN 转换")
    logger.info("=" * 60)
    logger.info(f"输入 ONNX: {onnx_model_path}")
    logger.info(f"输出 RKNN: {rknn_model_path}")
    logger.info(f"目标平台: {target_platform}")
    logger.info(f"输入尺寸: {input_size[0]}x{input_size[1]}")
    logger.info(f"量化: {do_quantization}")
    
    # 默认预处理参数（YOLOv8 通常不需要归一化，直接使用 [0, 255]）
    if mean_values is None:
        mean_values = [0, 0, 0]  # YOLOv8 不使用均值归一化
    if std_values is None:
        std_values = [255, 255, 255]  # YOLOv8 使用 [0, 255] 范围
    
    logger.info(f"均值: {mean_values}")
    logger.info(f"标准差: {std_values}")
    
    # 创建 RKNN 对象
    rknn = RKNN(verbose=True)
    
    try:
        # 配置 RKNN
        logger.info("配置 RKNN...")
        config_dict = {
            'target_platform': target_platform,
            'mean_values': [mean_values],  # 需要是列表的列表
            'std_values': [std_values],     # 需要是列表的列表
        }
        
        ret = rknn.config(**config_dict)
        if ret != 0:
            raise RuntimeError(f"RKNN 配置失败，错误代码: {ret}")
        logger.info("✓ RKNN 配置完成")
        
        # 加载 ONNX 模型
        logger.info("加载 ONNX 模型...")
        ret = rknn.load_onnx(model=str(onnx_model_path))
        if ret != 0:
            raise RuntimeError(f"加载 ONNX 模型失败，错误代码: {ret}")
        logger.info("✓ ONNX 模型加载完成")
        
        # 准备量化数据集
        # RKNN Toolkit 2 的 dataset 参数需要是包含图像路径的文本文件（.txt）
        # 文本文件格式：每行一个图像文件的绝对路径
        dataset_txt_path = quantization_dataset
        if do_quantization and dataset_txt_path is None:
            logger.info("创建量化校正数据集...")
            dataset_txt_path = create_calibration_dataset(
                num_samples=num_quantization_samples,
                input_size=input_size
            )
        elif do_quantization:
            # 如果提供了数据集路径，需要确保是文本文件
            dataset_path_obj = Path(dataset_txt_path)
            if dataset_path_obj.is_dir():
                # 如果是目录，创建数据集列表文件
                logger.info(f"从目录创建数据集列表文件: {dataset_path_obj}")
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(dataset_path_obj.glob(f'*{ext}')))
                    image_files.extend(list(dataset_path_obj.glob(f'*{ext.upper()}')))
                
                # 限制数量
                image_files = image_files[:num_quantization_samples]
                
                # 创建数据集列表文件
                dataset_txt_path = dataset_path_obj / "dataset.txt"
                with open(dataset_txt_path, 'w') as f:
                    for img_file in image_files:
                        f.write(str(img_file.resolve()) + '\n')
                dataset_txt_path = str(dataset_txt_path)
                logger.info(f"创建数据集列表文件，包含 {len(image_files)} 个图像路径")
            elif not dataset_path_obj.suffix == '.txt':
                # 如果不是 .txt 文件，尝试创建列表文件
                logger.warning(f"数据集路径不是 .txt 文件: {dataset_txt_path}")
                logger.warning("尝试将其作为单个图像文件处理...")
                # 创建临时列表文件
                temp_txt = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                temp_txt.write(str(Path(dataset_txt_path).resolve()) + '\n')
                temp_txt.close()
                dataset_txt_path = temp_txt.name
            else:
                # 已经是 .txt 文件
                dataset_txt_path = str(dataset_path_obj.resolve())
                logger.info(f"使用数据集列表文件: {dataset_txt_path}")
        
        # 构建 RKNN 模型
        logger.info("构建 RKNN 模型...")
        # RKNN Toolkit 2 的 dataset 参数应该是包含图像路径的文本文件（.txt）路径
        ret = rknn.build(
            do_quantization=do_quantization,
            dataset=dataset_txt_path if do_quantization else None
        )
        if ret != 0:
            raise RuntimeError(f"构建 RKNN 模型失败，错误代码: {ret}")
        logger.info("✓ RKNN 模型构建完成")
        
        # 导出 RKNN 模型
        logger.info("导出 RKNN 模型...")
        ret = rknn.export_rknn(export_path=str(rknn_model_path))
        if ret != 0:
            raise RuntimeError(f"导出 RKNN 模型失败，错误代码: {ret}")
        logger.info(f"✓ RKNN 模型导出成功: {rknn_model_path}")
        
        # 获取模型信息
        try:
            logger.info("获取模型信息...")
            ret, sdk_version = rknn.get_sdk_version()
            if ret == 0:
                logger.info(f"SDK 版本: {sdk_version}")
        except:
            pass
        
        # 清理临时数据集目录（如果是临时创建的）
        if do_quantization and dataset_txt_path:
            try:
                dataset_txt_path_obj = Path(dataset_txt_path)
                dataset_dir = dataset_txt_path_obj.parent
                # 检查是否是临时目录
                temp_dir = tempfile.gettempdir()
                if str(dataset_dir).startswith(temp_dir) or 'rknn_calib_' in str(dataset_dir):
                    logger.info(f"清理临时数据集目录: {dataset_dir}")
                    shutil.rmtree(dataset_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
        
        return str(rknn_model_path)
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 释放资源
        rknn.release()
        logger.info("RKNN 资源已释放")


def main():
    parser = argparse.ArgumentParser(
        description='将 ONNX 模型转换为 RKNN 格式（用于 RK3588 NPU 加速）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换（使用默认参数）
  python scripts/convert_rknn.py models/best.onnx
  
  # 指定输出路径和输入尺寸
  python scripts/convert_rknn.py models/best.onnx -o models/best.rknn --input-size 640 640
  
  # 禁用量化（使用 FP16）
  python scripts/convert_rknn.py models/best.onnx --no-quantization
  
  # 使用自定义量化数据集
  python scripts/convert_rknn.py models/best.onnx --quantization-dataset path/to/dataset

注意:
  - 需要安装 rknn-toolkit2
  - 量化校正数据集建议使用真实的验证集图像
  - 如果转换失败，可以尝试禁用量化或调整输入尺寸
        """
    )
    
    parser.add_argument(
        'onnx_model',
        type=str,
        help='输入的 ONNX 模型文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出的 RKNN 文件路径（默认：与输入文件同目录，扩展名为 .rknn）'
    )
    
    parser.add_argument(
        '--target-platform',
        type=str,
        default='rk3588',
        choices=['rk3588', 'rk3566', 'rk3568', 'rk3562', 'rv1109', 'rv1126'],
        help='目标平台（默认: rk3588）'
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
        '--mean',
        type=float,
        nargs=3,
        default=None,
        metavar=('R', 'G', 'B'),
        help='输入预处理均值（RGB 顺序，默认: 0 0 0）'
    )
    
    parser.add_argument(
        '--std',
        type=float,
        nargs=3,
        default=None,
        metavar=('R', 'G', 'B'),
        help='输入预处理标准差（RGB 顺序，默认: 255 255 255）'
    )
    
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        help='禁用 INT8 量化（使用 FP16，速度较慢但精度更高）'
    )
    
    parser.add_argument(
        '--quantization-samples',
        type=int,
        default=100,
        help='量化校正样本数量（默认: 100）'
    )
    
    parser.add_argument(
        '--quantization-dataset',
        type=str,
        default=None,
        help='量化校正数据集路径（如果未指定，则自动生成随机数据）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（如果指定，RKNN 文件将保存到此目录）'
    )
    
    args = parser.parse_args()
    
    try:
        rknn_path = convert_onnx_to_rknn(
            onnx_model_path=args.onnx_model,
            rknn_model_path=args.output,
            target_platform=args.target_platform,
            input_size=tuple(args.input_size),
            mean_values=args.mean,
            std_values=args.std,
            do_quantization=not args.no_quantization,
            quantization_dataset=None,  # 可以扩展支持从文件加载
            num_quantization_samples=args.quantization_samples,
            output_dir=args.output_dir
        )
        
        print("\n" + "=" * 60)
        print("✓ 转换完成！")
        print("=" * 60)
        print(f"RKNN 模型: {rknn_path}")
        print("\n下一步：在 RK3588 设备上使用 RKNN 模型进行推理")
        print("参考: docs/rknn_inference.md（如果存在）")
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
