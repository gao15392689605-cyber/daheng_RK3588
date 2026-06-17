"""产线生产配置 — 技术员调好后"保存生效"的那套参数, 工人据此跑。

只有技术员能改并发布; 工人登录时加载只读使用。落盘为 db/line_config.json,
避免技术员每次改完都丢失。字段: 模型 + conf/iou + 每类阈值开关(best) + 禁检类别。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_RKNN, LINE_CONFIG_PATH
from utils.common import get_logger

log = get_logger("line_config")


@dataclass
class LineConfig:
    model_path: str = str(DEFAULT_RKNN)
    model_name: str = "best.rknn"
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    best_mode: bool = False                 # SEG「最佳」每类阈值
    disabled_classes: list[str] = field(default_factory=list)  # 禁检的中文类名
    exposure: float = 0.0                   # 相机曝光 (μs); 0 = 不设, 用相机默认
    gain: float = 0.0                       # 相机增益 (dB); 0 = 不设

    # ── 持久化 ──
    @classmethod
    def load(cls, path: Path = LINE_CONFIG_PATH) -> "LineConfig":
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(
                    model_path=data.get("model_path", str(DEFAULT_RKNN)),
                    model_name=data.get("model_name", "best.rknn"),
                    conf=float(data.get("conf", DEFAULT_CONF)),
                    iou=float(data.get("iou", DEFAULT_IOU)),
                    best_mode=bool(data.get("best_mode", False)),
                    disabled_classes=list(data.get("disabled_classes", [])),
                    exposure=float(data.get("exposure", 0.0)),
                    gain=float(data.get("gain", 0.0)),
                )
        except Exception as e:
            log.error("读取产线配置失败, 用默认: %s", e)
        return cls()

    def save(self, path: Path = LINE_CONFIG_PATH) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                            encoding="utf-8")
            return True
        except Exception as e:
            log.error("保存产线配置失败: %s", e)
            return False

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)
