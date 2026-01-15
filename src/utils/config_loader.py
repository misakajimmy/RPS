"""
配置加载工具模块
Configuration Loader Utility
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from ..utils.logger import setup_logger

logger = setup_logger("RPS.ConfigLoader")


class ConfigLoader:
    """配置加载器类"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                logger.warning(f"配置文件为空: {config_path}")
                return {}
            
            logger.info(f"成功加载配置文件: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """
        保存配置到YAML文件
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
            
        Returns:
            bool: 保存是否成功
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            
            logger.info(f"成功保存配置文件: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    @staticmethod
    def get_hardware_config(config: Dict[str, Any], hardware_type: str) -> Optional[Dict[str, Any]]:
        """
        从配置中获取指定硬件的配置
        
        Args:
            config: 完整配置字典
            hardware_type: 硬件类型 ('robot_arm', 'camera', 'voice')
            
        Returns:
            Optional[Dict[str, Any]]: 硬件配置字典，不存在返回None
        """
        return config.get(hardware_type)
    
    @staticmethod
    def get_game_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        从配置中获取游戏配置
        
        Args:
            config: 完整配置字典
            
        Returns:
            Dict[str, Any]: 游戏配置字典
        """
        return config.get('game', {})
    
    @staticmethod
    def get_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        从配置中获取日志配置
        
        Args:
            config: 完整配置字典
            
        Returns:
            Dict[str, Any]: 日志配置字典
        """
        return config.get('logging', {})
