"""下游分流闸门触发 — 严重异物检出时把那一份分流到隔离区(主线不停)。

RK3588 用 GPIO/继电器驱动闸门。当前为**安全占位**:无 GPIO 环境只记录不报错,
板端接好硬件后在 _drive_gpio() 里实现即可。
"""
from __future__ import annotations

from utils.common import get_logger

log = get_logger("divert")

# 板端接好后置 True,并实现 _drive_gpio
_GPIO_READY = False


def _drive_gpio(delay_ms: int) -> None:
    """实际驱动 GPIO 继电器(板端实现)。例:
        from periphery import GPIO
        gate = GPIO("/dev/gpiochip0", PIN, "out")
        # 按带速延时 delay_ms 后脉冲一下
    """
    raise NotImplementedError


def trigger_divert(delay_ms: int = 0) -> bool:
    """触发下游分流。返回是否成功(占位时返回 True 表示"已记录意图")。"""
    if not _GPIO_READY:
        log.info("[占位] 触发下游分流闸门 (delay=%dms) — 待接 RK3588 GPIO", delay_ms)
        return True
    try:
        _drive_gpio(delay_ms)
        return True
    except Exception as e:
        log.error("分流触发失败: %s", e)
        return False
