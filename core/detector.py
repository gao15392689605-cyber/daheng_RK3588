"""推理工作线程 — DetectionWorker (单源) + FolderBatchWorker (批量)

线程通信仅靠 Qt 信号槽, 不直接跨线程读写变量.
停止用 _running 标志位 + QMutex, 不暴力 terminate.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from core.app_state import state
from core.source import BaseSource, FolderSource
from inference import CLASS_COLORS_BGR, CN_NAMES, run_inference_runtime
from utils.common import get_logger

log = get_logger("detector")


def _detections_to_unified(det_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """run_inference 输出 -> 统一字典格式 (type=obb)

    注: inference.run_inference 没有把 OBB points 输出, 只输出标签 + 标注图.
    points 字段保留空列表; 后续如需 canvas overlay 选中, 由 inference 内部
    或者额外 wrapper 提供.
    """
    out: list[dict[str, Any]] = []
    for d in det_list:
        eng = d.get("class_eng", "")
        color_bgr = CLASS_COLORS_BGR.get(eng, (0, 255, 0))
        out.append({
            "type": "obb",
            "class_name": d.get("class", eng),
            "class_eng": eng,
            "confidence": float(d.get("conf", 0.0)),
            "tier": d.get("tier", ""),
            "merged": int(d.get("merged", 1)),
            "area_ratio": float(d.get("area_ratio", 0.0)),
            "points": d.get("points", []),  # 由 inference wrapper 的 monkey-patch 注入
            "bbox": [],
            "mask": None,
            "track_id": -1,
            "color_rgb": (color_bgr[2], color_bgr[1], color_bgr[0]),
        })
    return out


class DetectionWorker(QThread):
    """单源(单图/视频/相机)推理线程"""

    frame_ready = Signal(object, list, float)  # annotated_rgb, detections_unified, infer_time
    progress = Signal(int, int)                # cur, total
    finished_clean = Signal(str)               # 完成原因
    error = Signal(str)

    def __init__(self, source: BaseSource, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self._running = True
        self._paused = False
        self._mutex = QMutex()
        self._last_raw: np.ndarray | None = None   # 最近一帧"干净原图"(无框), 供拍摄取用
        # 推理参数 (运行时可改)
        self.conf = state.conf_threshold
        self.iou = state.iou_threshold
        self.per_class = state.best_mode      # SEG「最佳」模式
        # 视频跳帧: 每 (skip+1) 帧推理 1 次, 0 = 不跳
        self.frame_skip = 0

    def latest_raw(self) -> "np.ndarray | None":
        """最近一帧干净原图(无检测框)的副本; 拍摄推理用, 避免拿到画了框的图。"""
        with QMutexLocker(self._mutex):
            return None if self._last_raw is None else self._last_raw.copy()

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False

    def set_paused(self, paused: bool) -> None:
        with QMutexLocker(self._mutex):
            self._paused = paused

    def update_thresholds(self, conf: float, iou: float, per_class: bool | None = None) -> None:
        with QMutexLocker(self._mutex):
            self.conf = conf
            self.iou = iou
            if per_class is not None:
                self.per_class = per_class

    def _is_running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def _is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    def run(self) -> None:
        try:
            if not self.source.open():
                self.error.emit("数据源打开失败")
                return

            idx = 0
            # 相机模式: read 超时是正常的(暂时没帧), 不能当作"读取结束"
            is_stream = self.source.total == -1
            was_paused = False
            while self._is_running():
                if self._is_paused():
                    was_paused = True
                    self.msleep(50)
                    continue

                # 刚从暂停恢复: 摄像头丢弃缓冲区积压的旧帧 —— 暂停期间相机流未停,
                # 缓冲里是调曝光前的旧亮度帧, 不丢的话继续后第一帧仍是旧曝光.
                # 丢几帧后取到的才是调曝光后产生的新帧, 让继续即时反映新亮度.
                if was_paused:
                    was_paused = False
                    if is_stream:
                        for _ in range(3):
                            if not self._is_running() or self._is_paused():
                                break
                            self.source.read()  # 取出即丢弃

                ok, frame_rgb = self.source.read()
                # stop 信号可能在 read 阻塞期间被设
                if not self._is_running():
                    break
                if ok and frame_rgb is not None:
                    with QMutexLocker(self._mutex):
                        self._last_raw = frame_rgb   # 存干净原帧, 供拍摄取用
                if not ok or frame_rgb is None:
                    if is_stream:
                        self.msleep(10)  # 相机超时, 重试
                        continue
                    self.finished_clean.emit("数据读取结束")
                    return

                idx += 1
                if self.frame_skip > 0 and self.source.total != 1 \
                        and (idx % (self.frame_skip + 1)) != 1:
                    continue

                # paused 标志可能在 read 期间被设, 再检查一次
                if self._is_paused():
                    continue

                try:
                    with QMutexLocker(self._mutex):
                        conf, iou, per_class = self.conf, self.iou, self.per_class
                    # 禁检类别(技术员检测项开关)在推理层就滤掉, seg mask 也不会画出
                    result = run_inference_runtime(frame_rgb, conf, iou, per_class=per_class,
                                                   drop_classes=set(state.disabled_classes))
                except Exception as e:
                    log.error("推理失败: %s", e)
                    self.error.emit(f"推理失败: {e}")
                    continue

                annotated = result.get("annotated_rgb", frame_rgb)
                detections = _detections_to_unified(result.get("detections", []))
                infer_t = float(result.get("inference_time", 0.0))

                self.frame_ready.emit(annotated, detections, infer_t)
                if self.source.total > 0:
                    self.progress.emit(idx, self.source.total)

                # 单图任务: 推理一次自动结束
                if self.source.total == 1 and idx >= 1:
                    self.finished_clean.emit("单图检测完成")
                    return

        except Exception as e:
            log.exception("DetectionWorker 异常: %s", e)
            self.error.emit(str(e))
        finally:
            try:
                self.source.close()
            except Exception:
                pass


class _LatestFrame:
    """共享槽: 预览线程写入最新帧, 推理线程取最新帧(自带序号, 避免重复推同一帧)。"""

    def __init__(self) -> None:
        self._m = QMutex()
        self._frame: np.ndarray | None = None
        self._seq = 0

    def put(self, frame: np.ndarray) -> None:
        with QMutexLocker(self._m):
            self._frame = frame
            self._seq += 1

    def get(self) -> tuple[np.ndarray | None, int]:
        with QMutexLocker(self._m):
            return self._frame, self._seq


class CameraPreviewWorker(QThread):
    """相机预览线程 — 唯一持有相机, 持续读流 + 高帧率显示, 并把最新帧塞进共享槽。

    不做推理(那是 CameraInferenceWorker 的事), 所以画面能按相机原速流畅刷新。
    """

    preview_ready = Signal(object)   # 原始 RGB 帧, 仅供显示
    error = Signal(str)
    finished_clean = Signal(str)

    def __init__(self, source: BaseSource, slot: _LatestFrame, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self.slot = slot
        self._running = True
        self._paused = False
        self._mutex = QMutex()

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False

    def set_paused(self, paused: bool) -> None:
        with QMutexLocker(self._mutex):
            self._paused = paused

    def _is_running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def _is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    def run(self) -> None:
        try:
            if not self.source.open():
                self.error.emit("相机打开失败")
                return
            while self._is_running():
                if self._is_paused():
                    self.msleep(30)
                    continue
                ok, frame_rgb = self.source.read()
                if not self._is_running():
                    break
                if not ok or frame_rgb is None:
                    self.msleep(5)   # 相机超时, 重试
                    continue
                self.slot.put(frame_rgb)
                self.preview_ready.emit(frame_rgb)
                self.msleep(10)      # 限制最高 ~60fps, 降低 UI 压力
        except Exception as e:
            log.exception("CameraPreviewWorker 异常: %s", e)
            self.error.emit(str(e))
        finally:
            try:
                self.source.close()
            except Exception:
                pass


class CameraInferenceWorker(QThread):
    """相机推理线程 — 只从共享槽取最新帧推理(不碰相机), emit 检测结果。

    推理仍走同一个 run_inference_runtime(合并/NMS/解码逻辑一字未改),
    只是与显示解耦: 它尽力快地推, 推到哪帧就更新哪帧的框, 不拖慢画面。
    """

    result_ready = Signal(list, float)   # detections_unified, infer_time
    error = Signal(str)
    finished_clean = Signal(str)

    def __init__(self, slot: _LatestFrame, parent=None) -> None:
        super().__init__(parent)
        self.slot = slot
        self._running = True
        self._paused = False
        self._mutex = QMutex()
        self.conf = state.conf_threshold
        self.iou = state.iou_threshold
        self.per_class = state.best_mode
        self._last_seq = -1

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False

    def set_paused(self, paused: bool) -> None:
        with QMutexLocker(self._mutex):
            self._paused = paused

    def update_thresholds(self, conf: float, iou: float, per_class: bool | None = None) -> None:
        with QMutexLocker(self._mutex):
            self.conf = conf
            self.iou = iou
            if per_class is not None:
                self.per_class = per_class

    def _is_running(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def _is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    def run(self) -> None:
        try:
            while self._is_running():
                if self._is_paused():
                    self.msleep(30)
                    continue
                frame_rgb, seq = self.slot.get()
                if frame_rgb is None or seq == self._last_seq:
                    self.msleep(15)   # 没新帧, 等一下
                    continue
                self._last_seq = seq
                try:
                    with QMutexLocker(self._mutex):
                        conf, iou, per_class = self.conf, self.iou, self.per_class
                    result = run_inference_runtime(frame_rgb, conf, iou, per_class=per_class,
                                                   drop_classes=set(state.disabled_classes))
                    detections = _detections_to_unified(result.get("detections", []))
                    self.result_ready.emit(detections, float(result.get("inference_time", 0.0)))
                except Exception as e:
                    log.error("相机推理失败: %s", e)
                    self.msleep(30)
        except Exception as e:
            log.exception("CameraInferenceWorker 异常: %s", e)
            self.error.emit(str(e))


class FolderBatchWorker(DetectionWorker):
    """文件夹批量检测 — 支持暂停/继续/中断, 边检测边记录"""

    file_finished = Signal(str, int)  # file_path, det_count

    def __init__(self, folder_source: FolderSource, parent=None) -> None:
        super().__init__(folder_source, parent)
        self.folder_source = folder_source
        self.per_file_summary: Counter = Counter()

    def run(self) -> None:
        try:
            if not self.source.open():
                self.error.emit("文件夹打开失败或无可检测文件")
                return

            for i in range(self.folder_source.total):
                if not self._is_running():
                    self.finished_clean.emit("用户中断")
                    return
                while self._is_paused() and self._is_running():
                    self.msleep(100)

                ok, frame_rgb = self.source.read()
                cur_file = self.folder_source.current_file()
                if not ok or frame_rgb is None:
                    log.warning("跳过无法读取: %s", cur_file)
                    self.progress.emit(i + 1, self.folder_source.total)
                    continue

                try:
                    with QMutexLocker(self._mutex):
                        conf, iou, per_class = self.conf, self.iou, self.per_class
                    # 禁检类别(技术员检测项开关)在推理层就滤掉, seg mask 也不会画出
                    result = run_inference_runtime(frame_rgb, conf, iou, per_class=per_class,
                                                   drop_classes=set(state.disabled_classes))
                except Exception as e:
                    log.error("批量推理失败 %s: %s", cur_file, e)
                    self.progress.emit(i + 1, self.folder_source.total)
                    continue

                annotated = result.get("annotated_rgb", frame_rgb)
                detections = _detections_to_unified(result.get("detections", []))
                infer_t = float(result.get("inference_time", 0.0))

                # 累加统计
                for d in detections:
                    self.per_file_summary[d["class_name"]] += 1

                self.frame_ready.emit(annotated, detections, infer_t)
                self.file_finished.emit(cur_file, len(detections))
                self.progress.emit(i + 1, self.folder_source.total)

            self.finished_clean.emit("批量检测完成")
        except Exception as e:
            log.exception("FolderBatchWorker 异常: %s", e)
            self.error.emit(str(e))
        finally:
            try:
                self.source.close()
            except Exception:
                pass
