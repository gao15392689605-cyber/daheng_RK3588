"""主窗口 — 1600×900 四栏布局 + 模式切换"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from config import (
    CLASS_LIST_CN, COLOR_BG_MAIN, COLOR_HIGHLIGHT, COLOR_TEXT,
    MODE_CAMERA, MODE_FOLDER, MODE_PHOTO, MODE_VIDEO,
)
from core.app_state import state
from core.camera_manager import CameraManager
from core.detector import DetectionWorker, FolderBatchWorker
from core.exporter import ExportWorker
from core.source import CameraSource, FolderSource, PhotoSource, VideoSource
from db.db_helper import DbHelper
from ui._main_layout import LayoutMixin
from ui.login_window import EditProfileDialog, RegisterDialog
from ui.panels import CameraSettingsDialog, DetectionSummaryDialog, HistoryQueryWidget
from ui.widgets import ndarray_to_pixmap
from utils.common import get_logger, list_media_files

log = get_logger("main_window")


class MainWindow(LayoutMixin, QMainWindow):

    logout_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("烟草异物智能检测系统")
        self.resize(1600, 900)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")

        self.worker: DetectionWorker | None = None
        self.camera_mgr = CameraManager()
        # 当前帧未过滤的完整 detections (过滤变化时直接重应用, 不等下一帧)
        self._last_full_detections: list[dict[str, Any]] = []
        self._current_file_path: str = ""
        self._current_log_id: int = -1
        self._export_worker: ExportWorker | None = None
        # 本次会话各类"单帧最多同时出现"峰值 (口径见 DetectionSummaryDialog)
        self._session_peak: Counter = Counter()
        self._summary_shown: bool = False
        # 文件夹批量结果缓存 (路径 + 检测), 供跑完后左右翻阅; 不存图, 翻阅时重读省内存
        self._folder_results: list[dict[str, Any]] = []
        self._folder_idx: int = 0

        self._build()
        self._wire_signals()
        self._switch_mode(MODE_PHOTO)

    def _wire_signals(self) -> None:
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.iou_slider.valueChanged.connect(self._on_iou_changed)
        self.start_btn.clicked.connect(self._on_start)
        # 中央 splash 的大按钮转发到顶部 start/stop
        self.center_splash.start_clicked.connect(self._on_start)
        self.center_splash.stop_clicked.connect(self._on_stop)
        self.pause_btn.toggled.connect(self._on_pause_toggled)
        self.stop_btn.clicked.connect(self._on_stop)
        self.capture_btn.clicked.connect(self._on_capture)
        self.table.row_selected.connect(self._on_row_selected)
        self.right_panel.filter_changed.connect(self._on_filter_changed)
        self.right_panel.export_request.connect(self._on_export)
        self.folder_prev_btn.clicked.connect(lambda: self._show_folder_index(self._folder_idx - 1))
        self.folder_next_btn.clicked.connect(lambda: self._show_folder_index(self._folder_idx + 1))

        for key, btn in self._nav_buttons.items():
            btn.clicked.connect(lambda _=False, k=key: self._on_nav_clicked(k))

        self.profile_panel.history_clicked.connect(self._open_history)
        self.profile_panel.edit_clicked.connect(self._open_edit)
        self.profile_panel.register_clicked.connect(self._open_register)
        self.profile_panel.logout_clicked.connect(self._on_logout)

    def init_for_user(self, username: str) -> None:
        self.user_footer.setText(f"当前用户: {username}")
        self.profile_panel.refresh()
        self._nav_buttons[MODE_PHOTO].setChecked(True)

    def _switch_mode(self, mode: str) -> None:
        state.detect_mode = mode
        self.center_status.setText(f"当前模式: {mode}")
        self.right_stack.setCurrentWidget(self.right_panel)
        # 拍摄按钮只在相机模式才显示
        is_cam = (mode == MODE_CAMERA)
        self.capture_btn.setVisible(is_cam)
        self.capture_btn.setEnabled(is_cam)
        self._set_folder_nav_visible(False)  # 切模式收起文件夹翻阅控件

    def _on_nav_clicked(self, key: str) -> None:
        if key in (MODE_PHOTO, MODE_VIDEO, MODE_FOLDER):
            # 即点即弹: 切模式后立即弹文件/文件夹框, 选完自动开始 (相机除外)
            self._switch_mode(key)
            self._on_start()
        elif key == MODE_CAMERA:
            self._switch_mode(key)  # 相机无文件框, 靠 ▶ 启流
        elif key == "profile":
            state.detect_mode = "profile"
            self.profile_panel.refresh()
            self.right_stack.setCurrentWidget(self.profile_panel)
            self.center_status.setText("个人中心")
        elif key == "model":
            self._choose_model()
        elif key == "export_dir":
            self._choose_export_dir()
        elif key == "cam_settings":
            self._open_camera_settings()

    def _choose_model(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择模型", str(Path(state.model_path).parent),
            "模型文件 (*.rknn *.pt *.onnx)",
        )
        if fp:
            state.model_path = fp
            state.model_name = Path(fp).name
            QMessageBox.information(self, "提示", f"模型已切换: {state.model_name}\n重启后生效")

    def _choose_export_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "选择默认导出目录", state.export_dir or str(Path.home()),
        )
        if d:
            state.export_dir = d
            QMessageBox.information(self, "提示", f"默认导出目录已设置:\n{d}")

    def _on_capture(self) -> None:
        """相机模式: 抓取当前帧, 弹窗选 [推理]/[保存图片]/[取消]"""
        # 优先用最近一帧 (worker 流式中), 否则现取
        frame = state.last_image_rgb
        if frame is None:
            if not self.camera_mgr.is_opened():
                if not self.camera_mgr.open():
                    QMessageBox.warning(self, "相机不可用", "相机未连接或 SDK 异常")
                    return
            ok, frame = self.camera_mgr.read()
            if not ok or frame is None:
                QMessageBox.warning(self, "拍摄失败", "未取到相机帧, 请重试")
                return
        frame = frame.copy()  # 拷贝避免后续被流式覆盖

        box = QMessageBox(self)
        box.setWindowTitle("拍摄完成")
        box.setText("已抓取当前帧, 用作?")
        btn_infer = box.addButton("🔍 推理这张", QMessageBox.AcceptRole)
        btn_save = box.addButton("💾 保存图片", QMessageBox.ActionRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec()

        clicked = box.clickedButton()
        if clicked == btn_infer:
            # 暂停流式 worker, 让画面定格在这张推理结果上
            was_running = self.worker is not None and self.worker.isRunning()
            if was_running:
                self.pause_btn.setChecked(True)  # 触发 _on_pause_toggled → worker.set_paused(True)
            self._infer_single_frame(frame)
            if was_running:
                self.center_status.setText("已冻结当前帧 — 点 ▶ 继续 恢复视频流, 或再次 📸 拍摄")
        elif clicked == btn_save:
            import time as _t
            default = str(Path.home() / f"capture_{int(_t.time())}.jpg")
            fp, _ = QFileDialog.getSaveFileName(
                self, "保存原图", default, "图片 (*.jpg *.png *.bmp)",
            )
            if fp:
                import cv2
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fp, bgr)
                QMessageBox.information(self, "已保存", f"原图已保存\n{fp}")

    def _infer_single_frame(self, frame_rgb: np.ndarray) -> None:
        """主线程同步推理一帧 (绕过 worker), 复用 _on_frame_ready 渲染"""
        try:
            from inference import run_inference_runtime
            from core.detector import _detections_to_unified
            result = run_inference_runtime(frame_rgb, state.conf_threshold, state.iou_threshold)
            annotated = result.get("annotated_rgb", frame_rgb)
            detections = _detections_to_unified(result.get("detections", []))
            self._on_frame_ready(annotated, detections, float(result.get("inference_time", 0.0)))
        except Exception as e:
            log.error("拍摄推理失败: %s", e)
            QMessageBox.warning(self, "推理失败", str(e))

    def _open_camera_settings(self) -> None:
        """打开相机亮度调节弹窗 — 没开相机时尝试自动打开"""
        if not self.camera_mgr.is_opened():
            ok = self.camera_mgr.open()
            if not ok:
                QMessageBox.warning(self, "相机不可用", "未检测到大恒相机或 SDK 异常")
                return
        CameraSettingsDialog(self.camera_mgr, self).exec()

    def _on_conf_changed(self, v: int) -> None:
        f = v / 100.0
        state.conf_threshold = f
        self.conf_label.setText(f"{f:.2f}")
        if self.worker is not None:
            self.worker.update_thresholds(state.conf_threshold, state.iou_threshold)

    def _on_iou_changed(self, v: int) -> None:
        f = v / 100.0
        state.iou_threshold = f
        self.iou_label.setText(f"{f:.2f}")
        if self.worker is not None:
            self.worker.update_thresholds(state.conf_threshold, state.iou_threshold)

    def _pick_source(self, mode: str) -> tuple[Any, str, bool]:
        """返回 (source, file_path, use_batch_worker); source 为 None 表示取消"""
        if mode == MODE_PHOTO:
            fp, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.tiff)")
            return (PhotoSource(fp), fp, False) if fp else (None, "", False)
        if mode == MODE_VIDEO:
            fp, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
            return (VideoSource(fp), fp, False) if fp else (None, "", False)
        if mode == MODE_FOLDER:
            d = QFileDialog.getExistingDirectory(self, "选择文件夹")
            if not d:
                return None, "", False
            files = list_media_files(d, recursive=False)
            if not files:
                QMessageBox.warning(self, "提示", "目录内无可检测的图片/视频")
                return None, "", False
            return FolderSource(d, files), d, True
        if mode == MODE_CAMERA:
            return CameraSource(self.camera_mgr), "", False
        return None, "", False

    def _on_start(self) -> None:
        # 运行中重新选源/重选视频: 先静默停掉旧任务 (不弹总计)
        if self.worker is not None and self.worker.isRunning():
            self._stop_worker_silent()
        mode = state.detect_mode
        source, file_path, use_batch = self._pick_source(mode)
        if source is None:
            if mode not in (MODE_PHOTO, MODE_VIDEO, MODE_FOLDER, MODE_CAMERA):
                QMessageBox.warning(self, "提示", "请先在左侧选择检测模式")
            return
        self.worker = FolderBatchWorker(source) if use_batch else DetectionWorker(source)
        # 视频跳帧: 每 3 帧推理 1 次, 减轻流式负载 (检测框由 UI ObbOverlay 渲染)
        if mode == MODE_VIDEO:
            self.worker.frame_skip = 2
        self._current_file_path = file_path
        # 会话级状态清零 — 重选视频/重开摄像头即重新计数
        self._session_peak = Counter()
        self._summary_shown = False
        self._folder_results = []
        self._folder_idx = 0
        self._set_folder_nav_visible(False)
        self.table.clear_all()
        self.overlay.clear_all()

        # 日志: 开始
        db = DbHelper.instance()
        self._current_log_id = db.insert_log(
            username=state.username,
            detect_mode=mode,
            model_name=state.model_name,
            conf_threshold=state.conf_threshold,
            iou_threshold=state.iou_threshold,
            file_path=file_path,
        )

        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished_clean.connect(self._on_worker_finished)
        self.worker.progress.connect(self._on_progress)
        if use_batch:
            self.worker.file_finished.connect(self._on_folder_file_done)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # 暂停只给摄像头 (视频/文件夹跑完看翻阅, 不做暂停)
        self.pause_btn.setEnabled(mode == MODE_CAMERA)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("⏸ 暂停")
        self.center_status.setText(f"检测中 — {file_path or '工业相机'}")
        self.worker.start()
        log.info("启动检测: mode=%s, src=%s", mode, file_path)

    def _stop_worker_silent(self) -> None:
        """停掉当前 worker 并结清审计日志, 但不弹总计 (用于切源/重选)"""
        if self.worker is None:
            return
        self.worker.stop()
        if not self.worker.wait(1500):
            self.worker.terminate()
            self.worker.wait(500)
        self._finalize_run(status="success")

    def _on_pause_toggled(self, paused: bool) -> None:
        if self.worker is None:
            return
        self.worker.set_paused(paused)
        self.pause_btn.setText("▶ 继续" if paused else "⏸ 暂停")
        self.center_status.setText("已暂停" if paused else "检测中")
        if paused:
            # 暂停即出累计总计 (峰值不清零, 继续可叠加)
            self._show_summary_popup()

    def _on_stop(self) -> None:
        if self.worker is None:
            return
        mode = state.detect_mode
        self.worker.stop()
        self.center_status.setText("正在停止...")
        # 兜底: 主动等 worker 最多 1.5s 退出 (read 500ms timeout + 推理时间)
        # 不阻塞太久 UI, 这里同步等是因为 UI 体验"卡一下"比"再开始没反应"好
        if not self.worker.wait(1500):
            log.warning("worker 1.5s 未退出, 强制 terminate")
            self.worker.terminate()
            self.worker.wait(500)
        self._finalize_run(status="success")
        self._maybe_show_summary(mode)

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.warning(self, "检测异常", msg)
        self._finalize_run(status="failed")

    def _on_worker_finished(self, reason: str) -> None:
        mode = state.detect_mode
        self.center_status.setText(f"完成: {reason}")
        self._finalize_run(status="success")
        self._maybe_show_summary(mode)
        # 文件夹批量完成 → 启用左右翻阅, 默认停在最后一张
        if mode == MODE_FOLDER and self._folder_results:
            self._set_folder_nav_visible(True)
            self._show_folder_index(len(self._folder_results) - 1)

    def _maybe_show_summary(self, mode: str) -> None:
        """视频/文件夹/相机 结束或停止时弹一次总计 (照片单图不弹)"""
        if mode != MODE_PHOTO and not self._summary_shown:
            self._summary_shown = True
            self._show_summary_popup()

    def _show_summary_popup(self) -> None:
        DetectionSummaryDialog(state.detect_mode, dict(self._session_peak), self).exec()

    def _finalize_run(self, status: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("⏸ 暂停")
        if self._current_log_id > 0:
            # 审计日志用会话峰值口径 (与总计弹窗一致, 不再逐帧累加虚高)
            DbHelper.instance().finalize_log(
                self._current_log_id,
                total_targets=sum(self._session_peak.values()),
                result_summary=dict(self._session_peak),
                status=status,
            )
            self._current_log_id = -1
        self.worker = None

    def _on_progress(self, cur: int, total: int) -> None:
        if total > 0:
            self.center_status.setText(f"进度: {cur}/{total}")

    def _on_frame_ready(
        self,
        annotated_rgb: np.ndarray,
        detections: list[dict[str, Any]],
        infer_time: float,
    ) -> None:
        self._render_frame(annotated_rgb, detections, self._current_file_path, infer_time)
        # 更新会话峰值: 本帧每类数量与历史峰值取大 (单帧最多同时出现 = 总计口径)
        frame_counts = Counter(d.get("class_name", "") for d in self._last_full_detections)
        for cls, n in frame_counts.items():
            if cls:
                self._session_peak[cls] = max(self._session_peak[cls], n)
        log.info("frame_ready: 收到 %d 个检测目标, 耗时 %.0fms", len(detections), infer_time * 1000)

    def _render_frame(
        self,
        image_rgb: np.ndarray,
        full_detections: list[dict[str, Any]],
        file_path: str,
        infer_time: float | None = None,
    ) -> None:
        """把一帧(图 + 检测)渲染到画布/表格/右侧 — 实时帧与文件夹翻阅共用.

        infer_time=None 表示翻阅缓存结果(不刷耗时标签、不计入峰值).
        """
        # skip_draw 模式下 image_rgb 就是原图; 缓存它给重画 overlay 和导出用
        state.last_image_rgb = image_rgb
        state.last_annotated_rgb = image_rgb
        if infer_time is not None:
            state.last_infer_time = infer_time
            self.infer_time_label.setText(f"🕐 检测耗时  {infer_time*1000:.0f}ms")
        self._current_file_path = file_path
        # 首帧到达 — splash 让位给 view
        if self.center_stack.currentIndex() != 1:
            self.center_stack.setCurrentIndex(1)
        pix = ndarray_to_pixmap(image_rgb)
        if self.pix_item is None:
            self.pix_item = self.scene.addPixmap(pix)
        else:
            self.pix_item.setPixmap(pix)
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self._fit_view()
        self._last_full_detections = list(full_detections)
        self._apply_filter_now()  # 表格/统计/目标数 + 重画 overlay
        self.right_panel.show_detail(state.current_results[0] if state.current_results else {})

    def _on_folder_file_done(self, path: str, count: int) -> None:
        """文件夹批量: 缓存每张图的 路径 + 检测结果, 供跑完后左右翻阅(不存图省内存)"""
        self._folder_results.append({
            "path": path,
            "detections": list(self._last_full_detections),
        })

    def _set_folder_nav_visible(self, visible: bool) -> None:
        for w in (self.folder_prev_btn, self.folder_next_btn, self.folder_page_label):
            w.setVisible(visible)

    def _show_folder_index(self, i: int) -> None:
        """翻阅文件夹批量结果第 i 张 — 从磁盘重读该图 + 用缓存检测重画框, 不重推理"""
        n = len(self._folder_results)
        if n == 0:
            return
        i = max(0, min(i, n - 1))
        self._folder_idx = i
        r = self._folder_results[i]
        bgr = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if bgr is None:
            self.center_status.setText(f"无法读取: {r['path']}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.view.user_zoomed = False  # 换图重新自适应
        self._render_frame(rgb, r["detections"], r["path"])
        self.folder_page_label.setText(f"第 {i+1}/{n} 张 — {Path(r['path']).name}")
        self.folder_prev_btn.setEnabled(i > 0)
        self.folder_next_btn.setEnabled(i < n - 1)
        self.center_status.setText("翻阅模式 — ◀ ▶ 或 ← → 切换")

    def _redraw_overlay(self) -> None:
        """根据 state.current_results 重画画布 OBB 框 (UI 自己画, 替代 inference 画框)"""
        self.overlay.clear_all()
        if self.pix_item is None:
            return
        rect = self.pix_item.boundingRect()
        short_edge = int(min(rect.width(), rect.height()))
        for d in state.current_results:
            pts = d.get("points") or []
            color = d.get("color_rgb", (0, 255, 0))
            label = f"{d.get('class_name', '')} {d.get('confidence', 0.0):.2f}"
            if pts:
                self.overlay.obb.add(pts, color, label, scene_short_edge=short_edge)

    def _apply_filter_now(self) -> None:
        """state.filter_classes 立即套用 — 刷表格/统计/目标数 + 重画 overlay"""
        filt = state.filter_classes
        full = self._last_full_detections
        filtered = [d for d in full if not filt or d.get("class_name") in filt]
        state.current_results = filtered
        self.target_count_label.setText(f"⊙ 检测目标:  {len(filtered)}个")
        self.table.load_results(self._current_file_path, filtered)
        self.right_panel.update_stats(filtered)
        # 同步重画画布上的 OBB (过滤后只画选中类别)
        self._redraw_overlay()

    def _on_row_selected(self, row: int) -> None:
        state.selected_row = row
        if not (0 <= row < len(state.current_results)):
            return
        det = state.current_results[row]
        self.right_panel.show_detail(det)
        # 选中行 → 画布上对应 OBB 高亮 (黄色加粗), 其他还原原色
        self.overlay.highlight(row)

    def _on_filter_changed(self, enabled: set[str]) -> None:
        # 空集 = 全部
        state.filter_classes = enabled if len(enabled) < len(CLASS_LIST_CN) else set()
        # 过滤即时套用到当前帧 — UI 端 ObbOverlay 重画, 不等下一帧
        self._apply_filter_now()

    def _on_export(self) -> None:
        if not state.current_results:
            QMessageBox.information(self, "提示", "当前无可导出结果")
            return
        # 优先用预设导出目录, 否则弹窗
        d = state.export_dir or QFileDialog.getExistingDirectory(
            self, "选择导出目录", str(Path.home()),
        )
        if not d:
            return
        # 导出带框图: skip_draw 模式下 last_annotated_rgb 是原图, 这里调 paint_annotations 重新画
        from inference import paint_annotations
        export_img = state.last_annotated_rgb
        if export_img is not None and state.current_results:
            try:
                export_img = paint_annotations(export_img, state.current_results)
            except Exception as e:
                log.error("导出画框失败, 使用原图: %s", e)
        self._export_worker = ExportWorker(d, state.current_results, export_img)
        ew = self._export_worker
        ew.progress.connect(lambda c, t: (
            self.export_progress.setRange(0, max(1, t)),
            self.export_progress.setValue(c),
        ))
        ew.finished_ok.connect(lambda p: (
            self.export_progress.setVisible(False),
            QMessageBox.information(self, "导出成功", f"已保存到\n{p}"),
        ))
        ew.failed.connect(lambda m: (
            self.export_progress.setVisible(False),
            QMessageBox.warning(self, "导出失败", m),
        ))
        ew.finished.connect(self._cleanup_export_worker)
        self.export_progress.setRange(0, 1); self.export_progress.setValue(0)
        self.export_progress.setVisible(True)
        self.export_progress.setFormat("导出中... %p%")
        ew.start()

    def _cleanup_export_worker(self) -> None:
        self._export_worker = None

    def _open_history(self) -> None:
        HistoryQueryWidget(state.username, self).exec()

    def _open_edit(self) -> None:
        dlg = EditProfileDialog(state.username, self)
        dlg.saved.connect(self.profile_panel.refresh)
        dlg.exec()

    def _open_register(self) -> None:
        RegisterDialog(self).exec()

    def _on_logout(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        self.logout_requested.emit()

    def _fit_view(self) -> None:
        """自适应当前 pixmap, 保持纵横比. 用户主动滚轮缩放过就不覆盖."""
        if self.pix_item is None or self.pix_item.boundingRect().isEmpty():
            return
        if getattr(self.view, "user_zoomed", False):
            return
        self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_view()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._fit_view()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # 文件夹翻阅态下 ←/→ 切换上一张/下一张
        if self.folder_prev_btn.isVisible() and self._folder_results:
            if event.key() == Qt.Key_Left:
                self._show_folder_index(self._folder_idx - 1)
                return
            if event.key() == Qt.Key_Right:
                self._show_folder_index(self._folder_idx + 1)
                return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        """资源清理. 是否退出 app 由 QApplication.quitOnLastWindowClosed 决定 —
        logout 流程会先 show 登录窗再 close 主窗, 此时不是最后一个窗口, 不退出.
        """
        try:
            if self.worker is not None:
                self.worker.stop()
                self.worker.wait(2000)
            # 彻底关闭相机设备 (close_device) 而不仅仅停流
            self.camera_mgr.shutdown()
            self.scene.clear()
            self.pix_item = None
        except Exception as e:
            log.error("关闭时清理异常: %s", e)
        event.accept()
