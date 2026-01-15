"""
日志工具模块
Logger Utility Module
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def get_log_level(level_str: str) -> int:
    """
    从字符串获取日志级别
    
    Args:
        level_str: 日志级别字符串（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        
    Returns:
        int: 日志级别
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return level_map.get(level_str.upper(), logging.INFO)


def setup_logger(
    name: str = "RPS",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        format_string: 日志格式字符串（可选）
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_logger_from_config(config: Dict[str, Any], name: str = "RPS") -> logging.Logger:
    """
    从配置字典设置日志记录器
    
    Args:
        config: 配置字典（包含level和file键）
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    level_str = config.get('level', 'INFO')
    level = get_log_level(level_str)
    log_file = config.get('file')
    
    return setup_logger(name=name, log_file=log_file, level=level)


# 默认日志记录器
default_logger = setup_logger()
