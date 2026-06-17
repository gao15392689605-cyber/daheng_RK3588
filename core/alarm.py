"""报警逻辑 — 严重度判定 + 同类冷却(同一异物移动多帧只报一次)+ 频次升级。

纯逻辑, 不依赖 UI/DB, 便于单测。UI 层(工人页)调 process() 拿到该触发的报警。
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable

from config import (
    ALARM_COOLDOWN_SEC, ESCALATE_COUNT, ESCALATE_WINDOW_SEC, SEVERE_CLASSES,
)


class AlarmManager:
    def __init__(self, now_fn: Callable[[], float] = time.monotonic) -> None:
        self._now = now_fn
        self._last_seen: dict[str, float] = {}      # class_eng -> 上次"出现"时刻(上升沿去重)
        self._severe_times: deque[float] = deque()  # 严重报警时刻(频次升级窗口)

    @staticmethod
    def is_severe(class_eng: str) -> bool:
        return class_eng in SEVERE_CLASSES

    def process(self, classes_eng: list[str], dedup: bool = True) -> dict:
        """传入本帧/本图检出的类别(英文)列表 → 返回应触发的报警。

        返回:
          {"severe": [新触发的严重类...], "escalate": bool(窗口内严重数达阈值)}

        dedup=True(相机/视频, 连续流): 上升沿去重 —— 某严重类"出现"才报一次, 持续在画面里
          (哪怕移动/抖动)不重复报, 消失超过 ALARM_COOLDOWN_SEC 秒再出现才算新一波。
          → 避免同一异物在视频里待久了被每隔几秒重复报。
        dedup=False(文件夹, 每张图=独立一片流动烟草): 不跨图去重 —— 本图出现的每个严重类都报,
          两张图都有石头 = 两次报警(它们是两片不同的烟草)。
        """
        now = self._now()
        present = {c for c in classes_eng if self.is_severe(c)}
        fired: list[str] = []
        for c in present:
            # dedup: 距该类"上次出现"超冷却秒数才算新一波; 非 dedup(文件夹): 每张图都报
            if (not dedup) or now - self._last_seen.get(c, -1e9) >= ALARM_COOLDOWN_SEC:
                fired.append(c)
                self._severe_times.append(now)
            self._last_seen[c] = now   # 每帧出现都刷新, 持续可见就不再报
        # 清理窗口外的记录
        while self._severe_times and now - self._severe_times[0] > ESCALATE_WINDOW_SEC:
            self._severe_times.popleft()
        return {"severe": fired, "escalate": len(self._severe_times) >= ESCALATE_COUNT}
