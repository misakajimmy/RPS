"""
图像预处理工具模块
Image Processing Utility Module
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from ....utils.logger import setup_logger

logger = setup_logger("RPS.ImageProcessor")


class ImageProcessor:
    """图像预处理工具类"""
    
    @staticmethod
    def resize(image: np.ndarray, width: int, height: int, 
               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            width: 目标宽度
            height: 目标高度
            interpolation: 插值方法
            
        Returns:
            np.ndarray: 调整后的图像
        """
        return cv2.resize(image, (width, height), interpolation=interpolation)
    
    @staticmethod
    def crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        裁剪图像
        
        Args:
            image: 输入图像
            x: 起始x坐标
            y: 起始y坐标
            width: 裁剪宽度
            height: 裁剪高度
            
        Returns:
            np.ndarray: 裁剪后的图像
        """
        return image[y:y+height, x:x+width]
    
    @staticmethod
    def crop_center(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        从中心裁剪图像
        
        Args:
            image: 输入图像
            width: 裁剪宽度
            height: 裁剪高度
            
        Returns:
            np.ndarray: 裁剪后的图像
        """
        h, w = image.shape[:2]
        x = (w - width) // 2
        y = (h - height) // 2
        return ImageProcessor.crop(image, x, y, width, height)
    
    @staticmethod
    def normalize(image: np.ndarray, mean: Optional[Tuple[float, float, float]] = None,
                  std: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        归一化图像
        
        Args:
            image: 输入图像
            mean: 均值（RGB三个通道）
            std: 标准差（RGB三个通道）
            
        Returns:
            np.ndarray: 归一化后的图像
        """
        if mean is None:
            mean = (0.485, 0.456, 0.406)  # ImageNet均值
        if std is None:
            std = (0.229, 0.224, 0.225)   # ImageNet标准差
        
        image = image.astype(np.float32) / 255.0
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        else:
            image = (image - mean[0]) / std[0]
        
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        增强对比度
        
        Args:
            image: 输入图像
            alpha: 对比度控制（1.0-3.0）
            beta: 亮度控制（0-100）
            
        Returns:
            np.ndarray: 增强后的图像
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                           sigma_x: float = 0) -> np.ndarray:
        """
        应用高斯模糊
        
        Args:
            image: 输入图像
            kernel_size: 核大小
            sigma_x: X方向标准差
            
        Returns:
            np.ndarray: 模糊后的图像
        """
        return cv2.GaussianBlur(image, kernel_size, sigma_x)
    
    @staticmethod
    def apply_bilateral_filter(image: np.ndarray, d: int = 9, 
                              sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        应用双边滤波（保边去噪）
        
        Args:
            image: 输入图像
            d: 邻域直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def detect_edges(image: np.ndarray, low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """
        边缘检测（Canny）
        
        Args:
            image: 输入图像（灰度图）
            low_threshold: 低阈值
            high_threshold: 高阈值
            
        Returns:
            np.ndarray: 边缘图像
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(image, low_threshold, high_threshold)
    
    @staticmethod
    def extract_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        提取感兴趣区域（Region of Interest）
        
        Args:
            image: 输入图像
            roi: (x, y, width, height) 感兴趣区域
            
        Returns:
            np.ndarray: ROI图像
        """
        x, y, w, h = roi
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def flip(image: np.ndarray, flip_code: int = 1) -> np.ndarray:
        """
        翻转图像
        
        Args:
            image: 输入图像
            flip_code: 翻转代码（0=垂直，1=水平，-1=垂直+水平）
            
        Returns:
            np.ndarray: 翻转后的图像
        """
        return cv2.flip(image, flip_code)
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None,
               scale: float = 1.0) -> np.ndarray:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            center: 旋转中心，None则使用图像中心
            scale: 缩放比例
            
        Returns:
            np.ndarray: 旋转后的图像
        """
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
        """
        调整亮度
        
        Args:
            image: 输入图像
            value: 亮度调整值（-100到100）
            
        Returns:
            np.ndarray: 调整后的图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def preprocess_for_gesture_recognition(image: np.ndarray, 
                                          target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        为手势识别预处理图像
        
        Args:
            image: 输入图像（BGR格式）
            target_size: 目标尺寸
            
        Returns:
            np.ndarray: 预处理后的图像（RGB格式，归一化）
        """
        # 转换为RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 调整大小
        resized = ImageProcessor.resize(rgb_image, target_size[0], target_size[1])
        
        # 归一化
        normalized = ImageProcessor.normalize(resized)
        
        return normalized
