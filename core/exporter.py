"""检测结果导出 — 当前帧 CSV + 带框图片 + 会话总计汇总 CSV, 格式写死"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from utils.common import ensure_dir, get_logger

log = get_logger("exporter")


class Exporter(QObject):
    """每次导出新建独立子文件夹: <out_dir>/detection_YYYYMMDD_HHMMSS/{result.csv, annotated.jpg}"""

    @staticmethod
    def _write_csv(csv_path: Path, results: list[dict[str, Any]]) -> None:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["序号", "类别", "置信度", "档位", "合并数", "面积占比"])
            for i, d in enumerate(results, 1):
                w.writerow([
                    i, d.get("class_name", ""),
                    f"{d.get('confidence', 0.0):.4f}",
                    d.get("tier", ""), d.get("merged", 1),
                    f"{d.get('area_ratio', 0.0):.4f}",
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


class ExportWorker(QThread):
    """后台导出线程 — 大批量场景用"""

    progress = Signal(int, int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        out_dir: str,
        results: list[dict[str, Any]],
        annotated_rgb: np.ndarray | None,
        summary: dict[str, int] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.out_dir = out_dir
        self.results = results
        self.annotated_rgb = annotated_rgb
        self.summary = summary  # 会话总计峰值, 非空则额外写 summary.csv

    def run(self) -> None:
        try:
            session = Exporter.make_session_dir(self.out_dir)
            steps = 3 if self.summary else 2
            self.progress.emit(0, steps)
            Exporter._write_csv(session / "result.csv", self.results)
            self.progress.emit(1, steps)
            Exporter._write_image(session / "annotated.jpg", self.annotated_rgb)
            self.progress.emit(2, steps)
            if self.summary:
                Exporter.write_summary_csv(session / "summary.csv", self.summary)
                self.progress.emit(3, steps)
            log.info("导出成功: %s", session)
            self.finished_ok.emit(str(session))
        except Exception as e:
            log.error("导出失败: %s", e)
            self.failed.emit(str(e))
