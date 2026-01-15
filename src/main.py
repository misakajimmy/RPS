"""
剪刀石头布游戏主程序入口
Rock Paper Scissors Game Main Entry
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app import Application
from src.utils.logger import setup_logger

logger = setup_logger("RPS.Main")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='剪刀石头布游戏')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（默认: config/config.yaml）'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("剪刀石头布游戏启动")
    logger.info("Rock Paper Scissors Game Starting")
    logger.info("=" * 50)
    
    # 创建并启动应用程序
    app = Application(config_path=args.config)
    
    try:
        success = app.start()
        if not success:
            logger.error("应用程序启动失败")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序异常退出: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("程序退出")
        logger.info("Program Exited")


if __name__ == "__main__":
    main()
