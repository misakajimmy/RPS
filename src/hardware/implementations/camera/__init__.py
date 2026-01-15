"""
摄像头实现模块
Camera Implementation
"""
from .usb_camera import USBCamera
from .image_processor import ImageProcessor
from ...factory.hardware_factory import HardwareFactory

# 自动注册USB摄像头到工厂类
HardwareFactory.register_camera('usb_camera', USBCamera)

__all__ = ['USBCamera', 'ImageProcessor']
