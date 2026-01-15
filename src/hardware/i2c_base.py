"""
I2C 抽象层
I2C Abstraction Layer

说明：
- 本模块提供 I2C 抽象接口和基于 GPIO 的软 I2C（BitBangI2C）骨架实现。
- 目前 BitBangI2C 仅做占位和日志输出，方便在 PC 上开发、在 RK3588 上再替换为真实 GPIO 实现。
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from ..utils.logger import setup_logger

logger = setup_logger("RPS.I2C")


class I2CInterfaceBase(ABC):
    """I2C 接口抽象基类"""

    @abstractmethod
    def write(self, address: int, data: bytes) -> bool:
        """
        向指定从机地址写入数据

        Args:
            address: I2C 从机地址
            data: 要写入的数据

        Returns:
            bool: 写入是否成功
        """

    @abstractmethod
    def read(self, address: int, length: int) -> bytes:
        """
        从指定从机地址读取数据

        Args:
            address: I2C 从机地址
            length: 读取长度

        Returns:
            bytes: 读取到的数据（长度不足时可返回空字节串或补零）
        """

    @abstractmethod
    def write_read(self, address: int, write_data: bytes, read_length: int) -> bytes:
        """
        先写后读（常见于寄存器读操作）

        Args:
            address: I2C 从机地址
            write_data: 先写入的数据（通常是寄存器地址）
            read_length: 读取长度

        Returns:
            bytes: 读取到的数据
        """

    @abstractmethod
    def scan(self) -> List[int]:
        """
        扫描总线上的从机地址

        Returns:
            List[int]: 探测到的地址列表
        """

    @abstractmethod
    def close(self) -> None:
        """关闭 I2C 接口，释放资源"""


class BitBangI2C(I2CInterfaceBase):
    """
    基于 GPIO 的软 I2C 占位实现

    说明：
    - 当前实现仅在方法调用时输出日志，并返回占位数据。
    - 在 RK3588 实机上，可将此类改为使用具体 GPIO 库（如 libgpiod、厂商 SDK 等）。
    """

    def __init__(
        self,
        sda_pin: int,
        scl_pin: int,
        freq_hz: int = 100_000,
        name: str = "bitbang-i2c",
    ) -> None:
        self.sda_pin = sda_pin
        self.scl_pin = scl_pin
        self.freq_hz = freq_hz
        self.name = name
        self._initialized = False

        logger.warning(
            "BitBangI2C(%s) 初始化：当前为占位实现，仅输出日志，不进行真实 GPIO 操作。"
            " 请在 RK3588 上替换为实际 GPIO I2C 实现。",
            self.name,
        )
        self._initialized = True

    def write(self, address: int, data: bytes) -> bool:
        logger.info(
            "[%s] I2C WRITE addr=0x%02X len=%d data=%s",
            self.name,
            address,
            len(data),
            data.hex(" "),
        )
        # TODO: 在 RK3588 实机上实现真实的 GPIO I2C 写入
        return True

    def read(self, address: int, length: int) -> bytes:
        logger.info(
            "[%s] I2C READ addr=0x%02X len=%d (占位返回全 0)",
            self.name,
            address,
            length,
        )
        # TODO: 在 RK3588 实机上实现真实的 GPIO I2C 读取
        return bytes([0x00] * max(0, length))

    def write_read(self, address: int, write_data: bytes, read_length: int) -> bytes:
        logger.info(
            "[%s] I2C WRITE_READ addr=0x%02X wlen=%d rlen=%d wdata=%s",
            self.name,
            address,
            len(write_data),
            read_length,
            write_data.hex(" "),
        )
        # TODO: 在 RK3588 实机上实现真实的 GPIO I2C 写读
        return bytes([0x00] * max(0, read_length))

    def scan(self) -> List[int]:
        logger.info("[%s] I2C SCAN 总线（占位实现，返回空列表）", self.name)
        # TODO: 在 RK3588 实机上实现地址扫描
        return []

    def close(self) -> None:
        if self._initialized:
            logger.info("[%s] 关闭 BitBangI2C（占位实现）", self.name)
        self._initialized = False

