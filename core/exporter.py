"""检测结果导出 — 当前帧 CSV + 带框图片 + 会话总计汇总 CSV, 格式写死"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.common import ensure_dir, get_logger

log = get_logger("exporter")


class Exporter:
    """同步导出工具(纯静态方法). 每次导出新建独立子文件夹 detection_YYYYMMDD_HHMMSS/."""

    @staticmethod
    def _fmt_points(points: Any) -> str:
        """OBB 四角点 → 'x0,y0 x1,y1 x2,y2 x3,y3' 字符串; 无坐标返回 '—'"""
        if not points:
            return "—"
        try:
            return " ".join(f"{float(x):.0f},{float(y):.0f}" for x, y in points)
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def write_detail_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
        """检测明细 — 每个目标一行: 序号/路径/帧序号/类别/置信度/坐标(OBB四角点).

        rows: [{"path": str, "frame_no": int|str, "det": detection_dict}, ...]
        照片单帧/文件夹逐图/视频相机逐帧都汇成这一种平铺格式.
        """
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["序号", "路径", "帧序号", "类别", "置信度", "坐标"])
            for i, r in enumerate(rows, 1):
                d = r.get("det", {})
                w.writerow([
                    i, r.get("path", "") or "—", r.get("frame_no", ""),
                    d.get("class_name", ""),
                    f"{d.get('confidence', 0.0):.4f}",
                    Exporter._fmt_points(d.get("points")),
                ])

    @staticmethod
    def _write_image(img_path: Path, annotated_rgb: np.ndarray | None) -> None:
        if annotated_rgb is not None and isinstance(annotated_rgb, np.ndarray):
            bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), bgr)

    @staticmethod
    def write_summary_csv(csv_path: Path, peak: dict[str, int]) -> None:
        """会话总计汇总 — 各类"单帧最多同时出现"峰值 + 总计.

        右侧导出与总计弹窗共用此函数, 保证两处保存的格式完全一致(联动).
        """
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["类别", "最多同时数量"])
            for cls, n in sorted(peak.items(), key=lambda kv: -kv[1]):
                w.writerow([cls, n])
            w.writerow(["总计", sum(peak.values())])

    @staticmethod
    def write_app_detail_csv(csv_path: Path, runs: list[dict[str, Any]]) -> None:
        """全程明细 — 每次推理一行汇总 (时间/模式/文件/异物分布/本次总计)"""
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["序号", "时间", "模式", "文件", "异物分布", "本次总计"])
            for i, r in enumerate(runs, 1):
                peak = r.get("peak", {})
                dist = " ".join(f"{k}:{v}" for k, v in sorted(peak.items(), key=lambda kv: -kv[1]))
                w.writerow([
                    i, r.get("time", ""), r.get("mode", ""),
                    r.get("file", "") or "—", dist or "—", r.get("total", 0),
                ])

    @staticmethod
    def make_session_dir(out_dir: str | Path) -> Path:
        """每次导出建一个独立子文件夹"""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = ensure_dir(Path(out_dir) / f"detection_{stamp}")
        return session
