"""检测结果导出 — CSV + JSON + 带框图片, 格式写死"""
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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.out_dir = out_dir
        self.results = results
        self.annotated_rgb = annotated_rgb

    def run(self) -> None:
        try:
            session = Exporter.make_session_dir(self.out_dir)
            self.progress.emit(0, 2)
            Exporter._write_csv(session / "result.csv", self.results)
            self.progress.emit(1, 2)
            Exporter._write_image(session / "annotated.jpg", self.annotated_rgb)
            self.progress.emit(2, 2)
            log.info("导出成功: %s", session)
            self.finished_ok.emit(str(session))
        except Exception as e:
            log.error("导出失败: %s", e)
            self.failed.emit(str(e))
