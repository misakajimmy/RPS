"""
CI1302 语音模块实现（基于参考的树莓派 I2C 示例协议）
CI1302 Voice Module Implementation (I2C via GPIO BitBang)

说明：
- 协议参考你提供的树莓派 smbus 示例：
  - I2C 总线：/dev/i2c-1
  - 从机地址：0x2b
  - 写寄存器：0x03，用于发送播报命令（例如 0x67 表示“初始化完成”）
  - 读寄存器：0x64，用于读取识别结果（颜色编号等）
"""
from typing import Optional, Dict, Any

from ...base.voice_base import VoiceBase
from ...i2c_base import BitBangI2C, I2CInterfaceBase
from ...factory.hardware_factory import HardwareFactory
from ...utils.logger import setup_logger

logger = setup_logger("RPS.CI1302")


class CI1302Voice(VoiceBase):
    """
    CI1302 语音模块驱动（基于 I2C 寄存器协议）

    配置示例（config.yaml 中）：

    voice:
      type: "ci1302"
      i2c:
        sda_pin: 25
        scl_pin: 26
        address: 0x2b      # 与树莓派示例一致
        freq_hz: 100000
    """

    # 协议常量（根据树莓派示例整理）
    DEFAULT_I2C_ADDRESS = 0x2B
    REG_COMMAND = 0x03
    REG_RESULT = 0x64

    # 一些已知的播报命令码（根据示例定义，具体含义待确认/补充）
    CMD_THIS_RED = 0x60
    CMD_THIS_GREEN = 0x61
    CMD_THIS_YELLOW = 0x62
    CMD_RECOG_YELLOW = 0x63
    CMD_RECOG_GREEN = 0x64
    CMD_RECOG_BLUE = 0x65
    CMD_RECOG_RED = 0x66
    CMD_INIT_DONE = 0x67  # “初始化完成”

    def __init__(
        self,
        i2c: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            i2c: I2C 配置字典，包含 sda_pin/scl_pin/address/freq_hz 等
            kwargs: 预留扩展字段
        """
        self._connected = False

        i2c_cfg = i2c or {}
        self._address: int = int(i2c_cfg.get("address", self.DEFAULT_I2C_ADDRESS))
        sda_pin: int = int(i2c_cfg.get("sda_pin", 25))
        scl_pin: int = int(i2c_cfg.get("scl_pin", 26))
        freq_hz: int = int(i2c_cfg.get("freq_hz", 100_000))

        self._i2c: I2CInterfaceBase = BitBangI2C(
            sda_pin=sda_pin,
            scl_pin=scl_pin,
            freq_hz=freq_hz,
            name="CI1302-I2C",
        )

        logger.info(
            "CI1302Voice 初始化完成: addr=0x%02X, SDA=%s, SCL=%s, freq=%dHz",
            self._address,
            sda_pin,
            scl_pin,
            freq_hz,
        )

    # --- 内部 I2C 工具方法 ---

    def _write_register(self, reg: int, value: int) -> bool:
        """向寄存器写入单字节"""
        data = bytes([reg, value & 0xFF])
        logger.debug(
            "CI1302 写寄存器: addr=0x%02X reg=0x%02X value=0x%02X",
            self._address,
            reg,
            value,
        )
        return self._i2c.write(self._address, data)

    def _read_register(self, reg: int, length: int = 1) -> bytes:
        """从寄存器读取数据（先写寄存器地址，再读）"""
        logger.debug(
            "CI1302 读寄存器: addr=0x%02X reg=0x%02X len=%d",
            self._address,
            reg,
            length,
        )
        return self._i2c.write_read(self._address, bytes([reg]), length)

    def _play_command(self, cmd: int) -> bool:
        """发送播报命令"""
        logger.info("CI1302 播放命令: 0x%02X", cmd)
        return self._write_register(self.REG_COMMAND, cmd)

    # --- VoiceBase 接口实现 ---

    def connect(self) -> bool:
        """
        连接语音模块，并播放一次“初始化完成”提示（与参考示例一致）
        """
        logger.info("尝试连接 CI1302 语音模块 (I2C addr=0x%02X)...", self._address)
        self._connected = True

        # 发送初始化完成播报
        if not self._play_command(self.CMD_INIT_DONE):
            logger.warning("CI1302 初始化播报命令发送失败")
        else:
            logger.info("CI1302 已发送初始化完成播报命令 (0x%02X)", self.CMD_INIT_DONE)

        return self._connected

    def disconnect(self) -> bool:
        logger.info("断开 CI1302 语音模块连接")
        try:
            self._i2c.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("关闭 I2C 时发生异常（可忽略）: %s", exc)
        self._connected = False
        return True

    def is_connected(self) -> bool:
        return self._connected

    def recognize_speech(self, timeout: float = 5.0) -> Optional[str]:
        """
        语音识别 / 结果读取

        当前根据参考示例，只是读取寄存器 0x64 的一个字节，
        实际含义需要结合你后续拿到的协议文档来解析。
        """
        logger.info("CI1302 读取识别结果（寄存器 0x%02X，timeout=%.1fs）", self.REG_RESULT, timeout)

        # 这里简单读取一次，不轮询；你可以仿照示例在上层循环调用。
        data = self._read_register(self.REG_RESULT, length=1)
        if not data:
            logger.warning("CI1302 读取结果失败（空数据）")
            return None

        value = data[0]
        logger.info("CI1302 读取到结果值: 0x%02X (%d)", value, value)

        # TODO: 根据协议将 value 映射到具体含义，目前返回十六进制字符串
        return f"0x{value:02X}"

    def synthesize_speech(self, text: str) -> bool:
        """
        语音合成 / 固定播报

        由于当前知道的是固定命令码，这里做一个简单映射：
        - 文本里包含特定关键词时，发送对应命令（例如“初始化”、“红色”、“绿色”等）。
        """
        logger.info("CI1302 合成/播报请求：text=%s", text)
        t = text.lower()

        if "init" in t or "初始化" in text:
            cmd = self.CMD_INIT_DONE
        elif "this red" in t or "当前红" in text or "红色" in text:
            cmd = self.CMD_THIS_RED
        elif "this green" in t or "当前绿" in text or "绿色" in text:
            cmd = self.CMD_THIS_GREEN
        elif "this yellow" in t or "当前黄" in text or "黄色" in text:
            cmd = self.CMD_THIS_YELLOW
        elif "recognize yellow" in t or "识别黄" in text:
            cmd = self.CMD_RECOG_YELLOW
        elif "recognize green" in t or "识别绿" in text:
            cmd = self.CMD_RECOG_GREEN
        elif "recognize blue" in t or "识别蓝" in text:
            cmd = self.CMD_RECOG_BLUE
        elif "recognize red" in t or "识别红" in text:
            cmd = self.CMD_RECOG_RED
        else:
            logger.warning("CI1302 暂不支持的文本指令，无法映射到固定命令：%s", text)
            return False

        return self._play_command(cmd)

    def play_audio(self, audio_data: bytes) -> bool:
        """
        播放原始音频数据

        当前协议信息不足，这里仍保持占位实现，只记录日志，
        并通过 I2C 写入一个简单前缀用于调试。
        """
        logger.info("CI1302 播放音频数据（占位），长度=%d 字节", len(audio_data))
        # 这里只写入一个占位前缀，不代表真实协议
        dummy_len = min(8, len(audio_data))
        payload = bytes([0x00]) + audio_data[:dummy_len]
        logger.debug(
            "CI1302 占位发送音频前缀: addr=0x%02X, len=%d",
            self._address,
            len(payload),
        )
        return self._i2c.write(self._address, payload)

    def get_status(self) -> dict:
        base_status = super().get_status()
        base_status.update(
            {
                "address": hex(self._address),
                "i2c_type": type(self._i2c).__name__,
                "note": "CI1302 使用 I2C 寄存器协议，底层通过 BitBangI2C（需在 RK3588 上实现真实 GPIO）。",
            }
        )
        return base_status


# 自动注册到硬件工厂
HardwareFactory.register_voice("ci1302", CI1302Voice)

__all__ = ["CI1302Voice"]

