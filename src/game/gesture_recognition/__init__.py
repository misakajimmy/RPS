"""
手势识别模块
Gesture Recognition Module
"""
from .gesture_recognizer import GestureRecognizer
from .recognition_result import RecognitionResult, RecognitionResultProcessor
from .recognizer_factory import RecognizerFactory

# 尝试导入 RKNN 识别器（可选）
try:
    from .rknn_recognizer import RKNNRecognizer
    __all__ = [
        'GestureRecognizer',
        'RKNNRecognizer',
        'RecognitionResult',
        'RecognitionResultProcessor',
        'RecognizerFactory'
    ]
except ImportError:
    __all__ = [
        'GestureRecognizer',
        'RecognitionResult',
        'RecognitionResultProcessor',
        'RecognizerFactory'
    ]