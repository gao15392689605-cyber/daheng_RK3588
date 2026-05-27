"""大恒工业相机管理 — 安全打开/关闭, 帧采集"""
from __future__ import annotations

from typing import Any

import numpy as np

from utils.common import get_logger

log = get_logger("camera")


class CameraManager:
    """单实例相机管理. 设备一次打开多次启停流, 不要每次 stop 都 close_device

    状态机:
        _device_open=False / _streaming=False  → 完全关闭
        _device_open=True  / _streaming=False  → 设备就绪但流停止 (停止检测时)
        _device_open=True  / _streaming=True   → 检测中
    """

    def __init__(self) -> None:
        self._device: Any = None
        self._device_manager: Any = None
        self._device_open: bool = False
        self._streaming: bool = False
        self._info: str = "未连接"

    @property
    def _opened(self) -> bool:
        """兼容旧接口: 设备打开且在流式中视为 'opened'"""
        return self._device_open and self._streaming

    def open(self) -> bool:
        """打开设备 + 启动流. 设备已开就只启流, 不重新创建设备 (避免 SDK 报已打开)"""
        try:
            import gxipy as gx
        except ImportError:
            log.warning("gxipy 大恒相机 SDK 未安装, 相机模式不可用")
            return False

        try:
            if not self._device_open:
                self._device_manager = gx.DeviceManager()
                dev_num, dev_info_list = self._device_manager.update_device_list()
                if dev_num == 0:
                    log.warning("未检测到大恒相机")
                    return False
                self._device = self._device_manager.open_device_by_index(1)
                try:
                    self._device.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
                except Exception:
                    pass
                try:
                    self._device.ExposureTime.set(10000.0)
                    self._device.Gain.set(2.0)
                except Exception:
                    pass
                info0 = dev_info_list[0] if dev_info_list else {}
                self._info = f"{info0.get('model_name', '未知')} SN:{info0.get('sn', '?')}"
                self._device_open = True
                log.info("相机设备打开: %s", self._info)

            if not self._streaming:
                self._device.stream_on()
                self._streaming = True
                log.info("相机流启动")
            return True
        except Exception as e:
            log.error("打开相机失败: %s", e)
            return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        """带 500ms 超时, 避免 worker 卡在阻塞读上停不下来"""
        if not self._opened or self._device is None:
            return False, None
        try:
            # 500ms 超时 — 没帧就返回 None, worker 自己 continue 重试
            raw = self._device.data_stream[0].get_image(500)
            if raw is None:
                return False, None
            arr = raw.get_numpy_array()
            if arr is None:
                return False, None
            if arr.ndim == 2:
                rgb = np.stack([arr, arr, arr], axis=2)
            else:
                rgb = arr
            return True, rgb
        except Exception as e:
            log.error("读取相机帧失败: %s", e)
            return False, None

    def close(self) -> None:
        """停止流但保留设备打开 (允许下次 open 快速 stream_on, 不报"设备已打开"错)"""
        if not self._streaming:
            return
        try:
            self._device.stream_off()
            log.info("相机流停止")
        except Exception as e:
            log.error("停止相机流失败: %s", e)
        finally:
            self._streaming = False

    def shutdown(self) -> None:
        """彻底关闭设备 — app 退出时调"""
        try:
            if self._streaming and self._device is not None:
                self._device.stream_off()
                self._streaming = False
            if self._device_open and self._device is not None:
                self._device.close_device()
                log.info("相机设备已关闭")
        except Exception as e:
            log.error("关闭相机设备失败: %s", e)
        finally:
            self._device = None
            self._device_open = False
            self._streaming = False
            self._info = "未连接"

    def is_opened(self) -> bool:
        return self._device_open

    def info(self) -> str:
        return self._info

    # ── 实时调参 ──────────────────────────────────────
    def set_exposure(self, us: float) -> bool:
        """曝光时间 (微秒). 大恒 ME2P 一般 100-1000000us, 越大越亮"""
        if not self._opened or self._device is None:
            return False
        try:
            self._device.ExposureTime.set(float(us))
            log.info("曝光时间设为 %.0f us", us)
            return True
        except Exception as e:
            log.error("曝光设置失败: %s", e)
            return False

    def set_gain(self, db: float) -> bool:
        """模拟增益 (dB). 一般 0-20dB, 越大越亮但噪声越多"""
        if not self._opened or self._device is None:
            return False
        try:
            self._device.Gain.set(float(db))
            log.info("增益设为 %.2f dB", db)
            return True
        except Exception as e:
            log.error("增益设置失败: %s", e)
            return False

    def get_exposure(self) -> float:
        if not self._opened or self._device is None:
            return 0.0
        try:
            return float(self._device.ExposureTime.get())
        except Exception:
            return 0.0

    def get_gain(self) -> float:
        if not self._opened or self._device is None:
            return 0.0
        try:
            return float(self._device.Gain.get())
        except Exception:
            return 0.0
