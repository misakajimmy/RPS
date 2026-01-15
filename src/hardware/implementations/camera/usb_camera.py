"""
USB摄像头实现
USB Camera Implementation
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from ...base.camera_base import CameraBase
from ....utils.logger import setup_logger

logger = setup_logger("RPS.USBCamera")


class USBCamera(CameraBase):
    """USB摄像头实现类"""
    
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, 
                 fps: int = 30, backend: Optional[int] = None):
        """
        初始化USB摄像头
        
        Args:
            device_id: 摄像头设备ID（默认0）
            width: 图像宽度（默认640）
            height: 图像高度（默认480）
            fps: 帧率（默认30）
            backend: OpenCV后端（可选，如cv2.CAP_V4L2）
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._current_resolution = (width, height)
        
        logger.info(f"初始化USB摄像头: device_id={device_id}, resolution={width}x{height}, fps={fps}")
    
    def connect(self) -> bool:
        """
        连接摄像头
        
        Returns:
            bool: 连接是否成功
        """
        if self._connected:
            logger.warning("摄像头已经连接")
            return True
        
        try:
            # 创建VideoCapture对象
            if self.backend is not None:
                self._cap = cv2.VideoCapture(self.device_id, self.backend)
            else:
                self._cap = cv2.VideoCapture(self.device_id)
            
            if not self._cap.isOpened():
                logger.error(f"无法打开摄像头设备: {self.device_id}")
                return False
            
            # 设置分辨率
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 验证实际分辨率
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._current_resolution = (actual_width, actual_height)
            
            # 读取一帧测试连接
            ret, frame = self._cap.read()
            if not ret:
                logger.error("摄像头连接测试失败：无法读取图像")
                self._cap.release()
                self._cap = None
                return False
            
            self._connected = True
            logger.info(f"摄像头连接成功: device_id={self.device_id}, "
                       f"实际分辨率={actual_width}x{actual_height}")
            return True
            
        except Exception as e:
            logger.error(f"摄像头连接异常: {e}")
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            return False
    
    def disconnect(self) -> bool:
        """
        断开摄像头连接
        
        Returns:
            bool: 断开是否成功
        """
        if not self._connected:
            logger.warning("摄像头未连接")
            return True
        
        try:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            
            self._connected = False
            logger.info(f"摄像头已断开: device_id={self.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"断开摄像头连接异常: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        检查摄像头是否已连接
        
        Returns:
            bool: 连接状态
        """
        if not self._connected or self._cap is None:
            return False
        
        # 检查摄像头是否仍然可用
        if not self._cap.isOpened():
            self._connected = False
            return False
        
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像
        
        Returns:
            Optional[np.ndarray]: 图像数据（BGR格式），失败返回None
        """
        if not self.is_connected():
            logger.warning("摄像头未连接，无法捕获图像")
            return None
        
        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("捕获图像失败")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"捕获图像异常: {e}")
            return None
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        获取摄像头分辨率
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        if not self.is_connected():
            return self._current_resolution
        
        try:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._current_resolution = (width, height)
            return self._current_resolution
        except Exception as e:
            logger.error(f"获取分辨率异常: {e}")
            return self._current_resolution
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置摄像头分辨率
        
        Args:
            width: 宽度
            height: 高度
            
        Returns:
            bool: 设置是否成功
        """
        if not self.is_connected():
            logger.warning("摄像头未连接，无法设置分辨率")
            return False
        
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 验证实际设置的分辨率
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._current_resolution = (actual_width, actual_height)
            
            if actual_width != width or actual_height != height:
                logger.warning(f"分辨率设置不完全匹配: 请求{width}x{height}, "
                             f"实际{actual_width}x{actual_height}")
            
            self.width = actual_width
            self.height = actual_height
            logger.info(f"分辨率已设置: {actual_width}x{actual_height}")
            return True
            
        except Exception as e:
            logger.error(f"设置分辨率异常: {e}")
            return False
    
    def capture_frame_rgb(self) -> Optional[np.ndarray]:
        """
        捕获一帧RGB格式图像
        
        Returns:
            Optional[np.ndarray]: 图像数据（RGB格式），失败返回None
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        
        # BGR转RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def capture_frame_gray(self) -> Optional[np.ndarray]:
        """
        捕获一帧灰度图像
        
        Returns:
            Optional[np.ndarray]: 灰度图像数据，失败返回None
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        
        # 转为灰度图
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
    
    def __del__(self):
        """析构函数"""
        if self._connected:
            self.disconnect()
