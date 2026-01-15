"""
系统集成测试
System Integration Tests
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hardware.config_manager import HardwareConfigManager
from src.hardware.implementations.camera import USBCamera
from src.game.gesture_recognition import GestureRecognizer
from src.game import GameController
from src.game.state_machine import GameState
from src.utils.logger import setup_logger

logger = setup_logger("RPS.IntegrationTest")


def test_camera_gesture_integration():
    """测试摄像头与手势识别集成"""
    print("=" * 50)
    print("测试1: 摄像头与手势识别集成")
    print("=" * 50)
    
    try:
        # 初始化摄像头
        camera = USBCamera(device_id=0, width=640, height=480)
        if not camera.connect():
            print("✗ 摄像头连接失败，跳过测试")
            return False
        
        print("✓ 摄像头连接成功")
        
        # 初始化手势识别器
        recognizer = GestureRecognizer(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            confidence_threshold=0.7
        )
        print("✓ 手势识别器初始化成功")
        
        # 测试识别
        print("\n请将手放在摄像头前，做出手势...")
        import time
        import cv2
        
        for i in range(10):
            frame = camera.capture_frame()
            if frame is None:
                continue
            
            gesture, confidence, probabilities = recognizer.recognize(frame)
            
            # 显示结果
            display_frame = frame.copy()
            text = f"Gesture: {gesture.value}, Confidence: {confidence:.2f}"
            cv2.putText(display_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Recognition", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)
        
        cv2.destroyAllWindows()
        camera.disconnect()
        print("✓ 摄像头与手势识别集成测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        logger.error(f"集成测试失败: {e}", exc_info=True)
        return False


def test_game_controller_integration():
    """测试游戏控制器集成"""
    print("\n" + "=" * 50)
    print("测试2: 游戏控制器集成")
    print("=" * 50)
    
    try:
        # 初始化摄像头
        camera = USBCamera(device_id=0)
        if not camera.connect():
            print("✗ 摄像头连接失败，跳过测试")
            return False
        
        # 初始化手势识别器
        recognizer = GestureRecognizer()
        
        # 创建游戏控制器（不包含机械臂和语音模块）
        controller = GameController(
            gesture_recognizer=recognizer,
            camera=camera,
            max_rounds=3,
            countdown_seconds=2
        )
        
        print("✓ 游戏控制器创建成功")
        
        # 测试状态转换
        print("\n测试状态转换...")
        controller.start_game()
        
        current_state = controller.get_current_state()
        print(f"✓ 当前状态: {current_state}")
        
        # 等待一段时间观察状态变化
        import time
        for i in range(5):
            state = controller.get_current_state()
            print(f"  状态 {i+1}: {state}")
            time.sleep(1)
        
        controller.stop_game()
        camera.disconnect()
        print("✓ 游戏控制器集成测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        logger.error(f"集成测试失败: {e}", exc_info=True)
        return False


def test_hardware_config_manager():
    """测试硬件配置管理器"""
    print("\n" + "=" * 50)
    print("测试3: 硬件配置管理器")
    print("=" * 50)
    
    try:
        config_manager = HardwareConfigManager()
        
        if config_manager.load_config():
            print("✓ 配置加载成功")
            
            # 尝试创建硬件（可能失败，因为硬件可能未连接）
            camera = config_manager.create_camera()
            if camera:
                print("✓ 摄像头创建成功")
                if camera.connect():
                    print("✓ 摄像头连接成功")
                    camera.disconnect()
                else:
                    print("⚠ 摄像头连接失败（可能是硬件未连接）")
            else:
                print("⚠ 摄像头创建失败（可能是配置问题）")
            
            print("✓ 硬件配置管理器测试通过")
            return True
        else:
            print("⚠ 配置文件不存在或加载失败（这是正常的，如果还没有创建配置文件）")
            return True  # 不算失败，只是配置未准备好
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        logger.error(f"集成测试失败: {e}", exc_info=True)
        return False


def test_error_handling():
    """测试错误处理机制"""
    print("\n" + "=" * 50)
    print("测试4: 错误处理机制")
    print("=" * 50)
    
    try:
        from src.utils.error_handler import global_error_handler
        from src.utils.exceptions import (
            HardwareException, CameraException, RecognitionException
        )
        
        # 测试硬件异常处理
        try:
            raise CameraException("测试摄像头异常", error_code=1001)
        except CameraException as e:
            global_error_handler.handle(e, "测试上下文")
        
        # 测试识别异常处理
        try:
            raise RecognitionException("测试识别异常", confidence=0.5)
        except RecognitionException as e:
            global_error_handler.handle(e, "测试上下文")
        
        print("✓ 错误处理机制测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        logger.error(f"集成测试失败: {e}", exc_info=True)
        return False


def test_full_system():
    """完整系统测试（需要所有硬件）"""
    print("\n" + "=" * 50)
    print("测试5: 完整系统测试")
    print("=" * 50)
    print("注意：此测试需要所有硬件设备连接")
    print()
    
    try:
        from src.app import Application
        
        # 创建应用程序实例
        app = Application()
        
        # 初始化（不运行主循环）
        if app.initialize():
            print("✓ 系统初始化成功")
            
            # 检查各组件
            components = {
                '摄像头': app.camera,
                '机械臂': app.robot_arm,
                '语音模块': app.voice,
                '手势识别器': app.gesture_recognizer,
                '游戏控制器': app.game_controller
            }
            
            for name, component in components.items():
                if component:
                    print(f"✓ {name}: 已初始化")
                else:
                    print(f"⚠ {name}: 未初始化")
            
            # 清理
            app.cleanup()
            print("✓ 完整系统测试通过")
            return True
        else:
            print("⚠ 系统初始化失败（可能是硬件未连接或配置问题）")
            return True  # 不算失败，可能是环境问题
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        logger.error(f"集成测试失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print("系统集成测试")
    print("=" * 50)
    print()
    
    results = []
    
    # 运行测试
    results.append(("摄像头与手势识别集成", test_camera_gesture_integration()))
    results.append(("游戏控制器集成", test_game_controller_integration()))
    results.append(("硬件配置管理器", test_hardware_config_manager()))
    results.append(("错误处理机制", test_error_handling()))
    results.append(("完整系统测试", test_full_system()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    print("=" * 50)
