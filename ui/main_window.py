"""主窗口 — 1600×900 四栏布局 + 模式切换"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QVBoxLayout,
)

from config import (
    CAPTURE_DIR, CAPTURE_DIR_FALLBACK, CLASS_LIST_CN, COLOR_BG_MAIN, COLOR_HIGHLIGHT,
    COLOR_TEXT, DECOUPLED_CAMERA, DIVERT_DELAY_MS, MODE_CAMERA, MODE_FOLDER,
    MODE_PHOTO, MODE_VIDEO, SEVERE_CLASSES, VIDEO_DIR, VIDEO_DIR_FALLBACK, VIDEO_FPS,
)
from core.alarm import AlarmManager
from core.app_state import state
from core.camera_manager import CameraManager
from core.detector import (
    CameraInferenceWorker, CameraPreviewWorker, DetectionWorker, FolderBatchWorker,
    _LatestFrame,
)
from core.divert import trigger_divert
from core.exporter import Exporter
from core.line_config import LineConfig
from core.source import CameraSource, FolderSource, PhotoSource, VideoSource
from db.db_helper import DbHelper
from inference import CN_NAMES
from ui._main_layout import LayoutMixin
from ui.login_window import EditProfileDialog, RegisterDialog
from ui.panels import (
    CameraSettingsDialog, DetectionSummaryDialog, HistoryQueryWidget, ask_save_kind,
)

# 严重异物的中文名集合(看板"严重类别明细"用)
_SEVERE_CN = {CN_NAMES.get(e, e) for e in SEVERE_CLASSES}


class _BatchStartDialog(QDialog):
    """工人开始批次 — 录批次号 + 品种。品种支持历史下拉补全。"""

    def __init__(self, varieties: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("开始批次")
        self.setMinimumWidth(360)
        from config import (COLOR_BG_MAIN as _BG, COLOR_TEXT as _FG,
                            COLOR_BG_SIDE as _SD, COLOR_BORDER as _BD, COLOR_BTN_PRIMARY as _PR)
        self.setStyleSheet(f"background:{_BG}; color:{_FG};")
        # 输入框显式上色, 否则深色对话框里浅字+白底看不清(品种"不见了"的原因)
        _inp = (f"QLineEdit,QComboBox{{background:{_SD}; color:{_FG};"
                f"border:1px solid {_BD}; border-radius:4px; padding:6px; min-height:24px;}}"
                f"QLineEdit:focus,QComboBox:focus{{border:1px solid {_PR};}}"
                f"QComboBox QAbstractItemView{{background:{_SD}; color:{_FG};"
                f"selection-background-color:{_PR};}}")
        form = QFormLayout(self)
        form.setContentsMargins(20, 18, 20, 18); form.setSpacing(12)
        self.batch_id = QLineEdit()
        self.batch_id.setPlaceholderText("必填, 如 20260604-A")
        self.variety = QComboBox(); self.variety.setEditable(True)
        self.variety.addItem("")
        for v in varieties:
            self.variety.addItem(v)
        self.variety.setCurrentText("")
        self.variety.lineEdit().setPlaceholderText("如 云烟87 (可不填, 下拉选历史)")
        for wdg in (self.batch_id, self.variety):
            wdg.setStyleSheet(_inp)
        form.addRow("批次号/工单号", self.batch_id)
        form.addRow("品种/牌号", self.variety)
        row = QHBoxLayout()
        from config import btn_qss as _qss, COLOR_BTN_PRIMARY as _PRI, COLOR_BG_SIDE as _SIDE
        ok = QPushButton("开始"); cancel = QPushButton("取消")
        ok.setStyleSheet(_qss(_PRI)); cancel.setStyleSheet(_qss(_SIDE))
        ok.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        row.addStretch(); row.addWidget(cancel); row.addWidget(ok)
        form.addRow(row)

    def values(self) -> dict[str, str]:
        return {
            "batch_id": self.batch_id.text().strip(),
            "variety": self.variety.currentText().strip(),
        }
from ui.widgets import ndarray_to_pixmap
from utils.common import get_logger, list_media_files

log = get_logger("main_window")

_MODE_CN = {MODE_PHOTO: "照片", MODE_VIDEO: "视频", MODE_FOLDER: "文件夹", MODE_CAMERA: "工业相机"}

# 拒绝在限时内退出的 worker 暂存于此(模块级, 不随窗口销毁). 绝不 terminate/析构正在跑的线程
# (两者都会 "FATAL: exception not rethrown / Aborted"); 由其自行结束或进程退出时回收.
_ZOMBIE_WORKERS: list = []


class MainWindow(LayoutMixin, QMainWindow):

    logout_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("烟草异物智能检测系统")
        self.setMinimumSize(1100, 680)   # 可自由缩放, 设个合理下限
        self.resize(1600, 900)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};")

        self.worker: DetectionWorker | None = None
        # 相机解耦: 预览线程(读流+显示) + 推理线程(self.worker), 共享最新帧槽
        self.preview_worker: CameraPreviewWorker | None = None
        self._frame_slot: _LatestFrame | None = None
        self._decoupled: bool = False
        # 工人「文件夹推理 = 模拟视频批次」: 预选目录 + 标记跑完自动结束批次
        self._pending_folder: tuple[str, list[str]] | None = None
        self._folder_batch_active: bool = False
        self.camera_mgr = CameraManager()
        # 当前帧未过滤的完整 detections (过滤变化时直接重应用, 不等下一帧)
        self._last_full_detections: list[dict[str, Any]] = []
        self._current_file_path: str = ""
        self._current_log_id: int = -1
        # 本次会话各类"单帧最多同时出现"峰值 (口径见 DetectionSummaryDialog)
        self._session_peak: Counter = Counter()   # 本批次各类累计检出数(累计口径, 非峰值)
        self._cur_present: set = set()             # 相机/视频上升沿累计用: 上一帧在场的类
        self._summary_shown: bool = False
        # 文件夹批量结果缓存 (路径 + 检测), 供跑完后左右翻阅; 不存图, 翻阅时重读省内存
        self._folder_results: list[dict[str, Any]] = []
        self._folder_idx: int = 0
        # 视频/摄像头逐帧明细累积 — 每个目标一条 {frame_no, det}, 供保存"所有推理帧"明细
        self._run_detail_rows: list[dict[str, Any]] = []
        self._run_frame_idx: int = 0
        # 本次使用(从打开软件到现在)的全程检测记录: 每次推理一行汇总
        self._app_runs: list[dict[str, Any]] = []
        self._current_mode: str = ""  # 当前/最近一次推理的模式 (供全程记录用)
        # 报警(严重异物): 管理器(冷却/升级) + 当前未确认报警的 DB id
        self._alarm_mgr = AlarmManager()
        # 待处置报警队列: 每条 (报警id, 异物中文名)。逐个处置 —— 点一次"确认处置"只清最早一条。
        self._cur_alarm_ids: list[tuple[int, str]] = []
        # 批次计时(工人看板"已运行")
        self._batch_start_ts: float = 0.0
        # 工人录制视频
        self._video_writer = None
        self._recording: bool = False
        self._video_path: str = ""
        # 技术员暂存: 进入时的产线配置快照 + 脏标志(退出时对比, 决定是否发布生效)
        self._committed_cfg: dict[str, Any] | None = None
        self._cfg_dirty: bool = False

        self._build()
        self._wire_signals()
        self._switch_mode(MODE_PHOTO)
        self._rebuild_preset_buttons()   # 按当前模型 task 建滑块右侧预设按钮

        # 工人看板定时刷新(已运行时长/计数), 仅工人启用
        self._dash_timer = QTimer(self)
        self._dash_timer.setInterval(1000)
        self._dash_timer.timeout.connect(self._refresh_worker_dashboard)

        # 报警横幅闪烁: 有未确认严重报警时一直闪, 确认后才停
        self._alarm_flash_timer = QTimer(self)
        self._alarm_flash_timer.setInterval(550)
        self._alarm_flash_timer.timeout.connect(self._flash_alarm)
        self._alarm_flash_on = False

    def _wire_signals(self) -> None:
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.iou_slider.valueChanged.connect(self._on_iou_changed)
        # 用户手动拖滑块 → 取消预设选中/退出最佳 (sliderMoved 只在拖动时触发, 程序 setValue 不触发)
        self.conf_slider.sliderMoved.connect(self._on_slider_user_moved)
        self.iou_slider.sliderMoved.connect(self._on_slider_user_moved)
        # 拖完(释放)记一条审计 — 参数变更留痕
        self.conf_slider.sliderReleased.connect(self._audit_param)
        self.iou_slider.sliderReleased.connect(self._audit_param)
        # 报警确认(闭环)
        self.alarm_confirm_btn.clicked.connect(self._on_alarm_confirm)
        # 批次追溯
        self.batch_start_btn.clicked.connect(self._on_batch_start)
        self.batch_end_btn.clicked.connect(self._on_batch_end)
        # 工人录制视频
        self.record_start_btn.clicked.connect(self._on_record_start)
        self.record_stop_btn.clicked.connect(self._on_record_stop)
        self.start_btn.clicked.connect(self._on_start)
        # 中央 splash 的大按钮: 工人联动批次(开始检测=开批次), 技术员=普通开始/停止
        self.center_splash.start_clicked.connect(self._on_center_start)
        self.center_splash.stop_clicked.connect(self._on_center_stop)
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
        # 顶栏明面退出(工人/技术员)
        self.logout_top_btn.clicked.connect(self._on_logout)
        # 技术员「检测项开关」: 勾选变化 → 更新禁检集合 + 标记待发布
        for cn, cb in self._detect_checks.items():
            cb.toggled.connect(lambda _on, name=cn: self._on_detect_toggle(name))

    def init_for_user(self, username: str) -> None:
        role_cn = {"worker": "工人", "technician": "技术人员", "admin": "管理员"}.get(state.role, state.role)
        self.user_footer.setText(f"当前用户: {username}({role_cn})")
        self.profile_panel.refresh()
        self._load_line_config()          # 工人/技术员都从产线配置起步
        self._nav_buttons[MODE_PHOTO].setChecked(True)
        self._apply_role_permissions()
        self._refresh_batch_ui()

    def _load_line_config(self) -> None:
        """加载产线生产配置(技术员发布的那套) → 套到 state + UI。
        工人: 据此跑(只读); 技术员: 作为可编辑起点, 并记下快照用于退出对比。"""
        cfg = LineConfig.load()
        state.conf_threshold = cfg.conf
        state.iou_threshold = cfg.iou
        state.best_mode = cfg.best_mode
        state.disabled_classes = set(cfg.disabled_classes)
        state.cam_exposure = cfg.exposure
        state.cam_gain = cfg.gain
        # 同步滑块显示
        self.conf_slider.blockSignals(True); self.conf_slider.setValue(int(round(cfg.conf * 100)))
        self.conf_label.setText(f"{cfg.conf:.2f}"); self.conf_slider.blockSignals(False)
        self.iou_slider.blockSignals(True); self.iou_slider.setValue(int(round(cfg.iou * 100)))
        self.iou_label.setText(f"{cfg.iou:.2f}"); self.iou_slider.blockSignals(False)
        # 同步检测项勾选
        for cn, cb in self._detect_checks.items():
            cb.blockSignals(True); cb.setChecked(cn not in state.disabled_classes); cb.blockSignals(False)
        # 若发布的模型与当前已加载的不同, 切过去(工人也要用发布的模型)
        if cfg.model_path and cfg.model_path != state.model_path:
            try:
                from inference import load_model, get_task
                load_model(cfg.model_path, force=True)
                state.model_path = cfg.model_path
                state.model_name = cfg.model_name or Path(cfg.model_path).name
                state.model_task = get_task() or state.model_task
            except Exception as e:
                log.error("加载产线配置模型失败, 保持当前模型: %s", e)
        self._committed_cfg = cfg.snapshot()
        self._cfg_dirty = False

    def _current_cfg_snapshot(self) -> dict[str, Any]:
        return LineConfig(
            model_path=state.model_path, model_name=state.model_name,
            conf=state.conf_threshold, iou=state.iou_threshold,
            best_mode=state.best_mode,
            disabled_classes=sorted(state.disabled_classes),
            exposure=state.cam_exposure, gain=state.cam_gain,
        ).snapshot()

    def _apply_role_permissions(self) -> None:
        """按角色定制界面(职责不交叉)。
        工人   = 看护页:无参数区/结果表, 只有「开始批次/结束批次」+「拍摄」+「退出」, 右侧大字看板。
        技术员 = 调优页:参数区/预设/检测项开关/开始停止暂停 + 结果表 + 右侧详情, 无批次。
        """
        is_worker = (state.role == "worker")
        is_tech = (state.role == "technician")

        # 导航: 工人只留 个人中心 + 摄像头; 技术员全开
        worker_keep = {"profile", MODE_CAMERA, MODE_FOLDER}
        for key, btn in self._nav_buttons.items():
            btn.setVisible((key in worker_keep) if is_worker else True)

        # 参数区/预设/计数标签/结果表/检测项开关 — 技术员可见, 工人隐藏
        self._param_box.setVisible(is_tech)
        self._bottom_table.setVisible(is_tech)
        self.detect_toggle_box.setVisible(is_tech)
        self.infer_time_label.setVisible(is_tech)
        self.target_count_label.setVisible(is_tech)
        # 顶栏已包进横向滚动区(放不下出横向滚动条), 故窗口不需要为顶栏强制大下限;
        # 给个小下限即可, 实际大小由 main.py 锁成屏幕大小。
        self.setMinimumSize(900, 600)

        # 开始/暂停/停止 — 技术员(自由跑测试); 工人靠「开始批次」驱动
        for b in (self.start_btn, self.pause_btn, self.stop_btn):
            b.setVisible(is_tech)
        # 批次控件 — 仅工人
        for b in (self.batch_label, self.batch_start_btn, self.batch_end_btn):
            b.setVisible(is_worker)
        # 录制视频 — 工人 + 技术员都可(技术员也常需录一段留存/复查)
        for b in (self.record_start_btn, self.record_stop_btn):
            b.setVisible(is_worker or is_tech)
        self.record_stop_btn.setEnabled(self._recording)
        self.record_start_btn.setEnabled(not self._recording)

        if is_worker:
            # 右侧切看板, 锁定相机看护
            self.right_stack.setCurrentWidget(self.worker_dashboard)
            self._nav_buttons[MODE_CAMERA].setChecked(True)
            self._switch_mode(MODE_CAMERA)
            self.worker_dashboard.set_batch(state.batch_id, running=False)
            self._dash_timer.start()
        else:
            self.right_stack.setCurrentWidget(self.right_panel)
            # 技术员: 右侧显示过滤段无意义(检测项开关已在左侧), 隐藏之
            self.right_panel.set_filter_visible(False)

    def _switch_mode(self, mode: str) -> None:
        state.detect_mode = mode
        self.center_status.setText(f"当前模式: {mode}")
        # 工人检测态右侧用大字看板, 其余角色用检测详情面板
        if state.role == "worker":
            self.right_stack.setCurrentWidget(self.worker_dashboard)
        else:
            self.right_stack.setCurrentWidget(self.right_panel)
        # 拍摄按钮只在相机模式才显示
        is_cam = (mode == MODE_CAMERA)
        self.capture_btn.setVisible(is_cam)
        self.capture_btn.setEnabled(is_cam)
        self._set_folder_nav_visible(False)  # 切模式收起文件夹翻阅控件

    def _on_nav_clicked(self, key: str) -> None:
        if key == MODE_FOLDER and state.role == "worker":
            # 工人: 文件夹推理 = 模拟视频批次(先录批次, 跑完自动结束)
            self._start_folder_batch()
        elif key in (MODE_PHOTO, MODE_VIDEO, MODE_FOLDER):
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
            self._save_app_results()  # 左侧「结果保存」= 存全程检测
        elif key == "cam_settings":
            self._open_camera_settings()

    def _choose_model(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择模型", str(Path(state.model_path).parent),
            "模型文件 (*.rknn *.pt *.onnx)",
        )
        if not fp:
            return
        # 热切换: 正在检测则先安全停掉, 避免推理中途换模型
        if self.worker is not None and self.worker.isRunning():
            self._stop_worker_silent()
        from PySide6.QtWidgets import QApplication
        from inference import load_model, get_task
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            load_model(fp, force=True)      # 释放旧模型 + 加载新模型 + 自动判 task
            task = get_task()
        except Exception as e:
            log.error("模型切换失败: %s", e)
            QMessageBox.warning(self, "切换失败", f"加载模型失败:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        state.model_path = fp
        state.model_name = Path(fp).name
        state.model_task = task
        state.best_mode = False          # 切模型重置最佳模式
        self._mark_dirty()
        DbHelper.instance().add_audit(state.username, state.role, "model_switch",
                                      f"{state.model_name} ({task})")
        self._rebuild_preset_buttons()   # 按新模型 task 重建预设按钮 (OBB 3 / SEG 1)
        task_cn = {"seg": "实例分割 (mask)", "obb": "旋转框检测 (OBB)"}.get(task, task)
        QMessageBox.information(
            self, "提示",
            f"模型已切换: {state.model_name}\n类型: {task_cn}\n后处理已自动切换, 无需重启。",
        )

    def _save_app_results(self) -> None:
        """左侧「结果保存」— 保存本次使用(从打开软件到现在)全程检测.

        可选 明细 / 汇总 / 全部 (无照片).
        汇总 = 全程各类总数(各次峰值相加); 明细 = 每次推理一行.
        """
        if not self._app_runs:
            QMessageBox.information(self, "提示", "本次使用还没有任何检测可保存")
            return
        kinds = ask_save_kind(
            self, f"将保存本次使用全程 {len(self._app_runs)} 次推理的检测结果。\n保存哪些内容?",
            allow_image=False,
        )
        if kinds is None:
            return
        d = QFileDialog.getExistingDirectory(
            self, "选择保存目录", state.export_dir or str(Path.home()),
        )
        if not d:
            return
        try:
            session = Exporter.make_session_dir(d)
            if "summary" in kinds:
                agg: Counter = Counter()
                for r in self._app_runs:
                    agg.update(r["peak"])
                Exporter.write_summary_csv(session / "全程汇总.csv", dict(agg))
            if "detail" in kinds:
                Exporter.write_app_detail_csv(session / "全程明细.csv", self._app_runs)
            QMessageBox.information(self, "已保存", f"全程检测结果已保存到\n{session}")
        except Exception as e:
            log.error("全程结果保存失败: %s", e)
            QMessageBox.warning(self, "保存失败", str(e))

    def _on_capture(self) -> None:
        """相机模式: 抓取当前帧。
        工人: 自动存图到固定目录(不弹窗, 看到离谱东西随手拍);
        技术员: 弹窗选 [推理]/[保存图片]/[取消]。"""
        # 取"干净原帧": 优先推理线程最近的原图(无框), 否则现从相机读一帧。
        # 不能用 state.last_image_rgb —— 那是画了框的标注图, 拿来再推会叠框/出旧结果。
        frame = None
        if self.worker is not None and self.worker.isRunning() and hasattr(self.worker, "latest_raw"):
            frame = self.worker.latest_raw()
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

        if state.role == "worker":
            self._worker_capture_save(frame)
            return

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

    def _worker_capture_save(self, frame_rgb: np.ndarray) -> None:
        """工人拍摄: 直接落盘到固定目录(CAPTURE_DIR, 不可写则回退), 不弹窗, 状态栏提示。"""
        import time as _t
        target = CAPTURE_DIR
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            target = CAPTURE_DIR_FALLBACK
            target.mkdir(parents=True, exist_ok=True)
        bid = state.batch_id or "无批次"
        fp = target / f"{_t.strftime('%Y%m%d_%H%M%S')}_{bid}.jpg"
        try:
            cv2.imwrite(str(fp), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            self.center_status.setText(f"📸 已存图 → {fp}")
            DbHelper.instance().add_audit(state.username, state.role, "capture", str(fp))
            QMessageBox.information(self, "拍摄成功", f"✅ 已保存照片:\n{fp}")
        except Exception as e:
            log.error("工人拍摄存图失败: %s", e)
            self.center_status.setText(f"拍摄存图失败: {e}")
            QMessageBox.warning(self, "拍摄失败", f"存图失败:\n{e}")

    # ── 工人录制视频(录制/结束 → 固定目录) ──
    def _on_record_start(self) -> None:
        if self._recording:
            return
        # 需要相机在出帧(工人通常已开批次). 提示但不强制
        self._video_writer = None    # 拿到首帧再按尺寸初始化
        self._recording = True
        self._video_path = ""
        self.record_start_btn.setEnabled(False)
        self.record_stop_btn.setEnabled(True)
        self.center_status.setText("⏺ 正在录制视频…")
        if not (self.worker is not None and self.worker.isRunning()):
            QMessageBox.information(self, "提示", "已开始录制。请确保摄像头已打开并在检测, 否则录不到画面。")

    def _write_video_frame(self, frame_rgb: np.ndarray) -> None:
        """录制中: 把当前帧写入视频(首帧时按尺寸+目录初始化 writer)。"""
        if not self._recording or frame_rgb is None:
            return
        try:
            if self._video_writer is None:
                import time as _t
                target = VIDEO_DIR
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except Exception:
                    target = VIDEO_DIR_FALLBACK
                    target.mkdir(parents=True, exist_ok=True)
                bid = state.batch_id or "无批次"
                self._video_path = str(target / f"{_t.strftime('%Y%m%d_%H%M%S')}_{bid}.avi")
                h, w = frame_rgb.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")   # MJPG/.avi 兼容性最好
                self._video_writer = cv2.VideoWriter(self._video_path, fourcc, VIDEO_FPS, (w, h))
            self._video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            log.error("写视频帧失败: %s", e)

    def _on_record_stop(self, silent: bool = False) -> None:
        if not self._recording:
            return
        self._recording = False
        path = self._video_path
        try:
            if self._video_writer is not None:
                self._video_writer.release()
        except Exception as e:
            log.error("关闭视频失败: %s", e)
        self._video_writer = None
        self.record_start_btn.setEnabled(True)
        self.record_stop_btn.setEnabled(False)
        if silent:
            return
        if path:
            DbHelper.instance().add_audit(state.username, state.role, "video_record", path)
            self.center_status.setText(f"⏹ 录制结束 → {path}")
            QMessageBox.information(self, "录制完成", f"✅ 视频已保存:\n{path}")
        else:
            QMessageBox.information(self, "提示", "本次未录到画面(摄像头没出帧), 未生成文件。")

    def _infer_single_frame(self, frame_rgb: np.ndarray) -> None:
        """主线程同步推理一帧 (绕过 worker), 复用 _on_frame_ready 渲染"""
        try:
            from inference import run_inference_runtime
            from core.detector import _detections_to_unified
            result = run_inference_runtime(frame_rgb, state.conf_threshold, state.iou_threshold,
                                           per_class=state.best_mode,
                                           drop_classes=set(state.disabled_classes))
            annotated = result.get("annotated_rgb", frame_rgb)
            detections = _detections_to_unified(result.get("detections", []))
            # _on_frame_ready 内部已套 _apply_detection_filter, 此处直接转交
            self._on_frame_ready(annotated, detections, float(result.get("inference_time", 0.0)))
        except Exception as e:
            log.error("拍摄推理失败: %s", e)
            QMessageBox.warning(self, "推理失败", str(e))

    def _open_camera_settings(self) -> None:
        """打开相机亮度调节弹窗 — 没开相机时尝试自动打开。
        技术员关闭弹窗后, 把最终曝光/增益纳入暂存(退出时汇总是否发布到产线)。"""
        if not self.camera_mgr.is_opened():
            ok = self.camera_mgr.open()
            if not ok:
                QMessageBox.warning(self, "相机不可用", "未检测到大恒相机或 SDK 异常")
                return
        before = (self.camera_mgr.get_exposure(), self.camera_mgr.get_gain())
        CameraSettingsDialog(self.camera_mgr, self).exec()
        after_exp, after_gain = self.camera_mgr.get_exposure(), self.camera_mgr.get_gain()
        # 写回 state(供发布), 变了才标脏
        if after_exp:
            state.cam_exposure = float(after_exp)
        if after_gain is not None:
            state.cam_gain = float(after_gain)
        if (after_exp, after_gain) != before:
            self._mark_dirty()

    def _mark_dirty(self) -> None:
        """技术员改了检测配置 → 标记待发布(退出时汇总询问是否生效)。"""
        if state.role == "technician":
            self._cfg_dirty = True

    def _on_detect_toggle(self, cn: str) -> None:
        """技术员勾选/取消「检测项」: 同步禁检集合, 标记待发布。下一帧即反映。"""
        cb = self._detect_checks.get(cn)
        if cb is None:
            return
        if cb.isChecked():
            state.disabled_classes.discard(cn)
        else:
            state.disabled_classes.add(cn)
        self._mark_dirty()
        # 静态图(已暂停/单图)即时重套用, 视频/相机靠下一帧
        if self._last_full_detections is not None:
            self._last_full_detections = [
                d for d in self._last_full_detections if d.get("class_name") not in state.disabled_classes]
            self._apply_filter_now()

    def _on_conf_changed(self, v: int) -> None:
        f = v / 100.0
        state.conf_threshold = f
        self.conf_label.setText(f"{f:.2f}")
        self._mark_dirty()
        if self.worker is not None:
            self.worker.update_thresholds(state.conf_threshold, state.iou_threshold)

    def _on_iou_changed(self, v: int) -> None:
        f = v / 100.0
        state.iou_threshold = f
        self.iou_label.setText(f"{f:.2f}")
        self._mark_dirty()
        if self.worker is not None:
            self.worker.update_thresholds(state.conf_threshold, state.iou_threshold)

    def _audit_param(self) -> None:
        """滑块释放时记一条参数变更审计。"""
        DbHelper.instance().add_audit(
            state.username, state.role, "param_change",
            f"conf={state.conf_threshold:.2f} iou={state.iou_threshold:.2f}")

    # ── 预设按钮 (随模型 task: OBB 3 档 / SEG 最佳; 选中高亮, 拖滑块取消选中+退最佳) ──
    _PRESET_QSS = ("QPushButton{{background:{bg}; color:white; border:2px solid transparent;"
                   " border-radius:4px;}}"
                   " QPushButton:checked{{background:{bg2}; border:2px solid #FFD23F; font-weight:bold;}}")
    _OBB_PRESETS = (("标准", 0.5, 0.5), ("高召回", 0.4, 0.5), ("高准确率", 0.6, 0.5))

    def _rebuild_preset_buttons(self) -> None:
        task = state.model_task
        try:
            from inference import get_task
            task = get_task() or task
        except Exception:
            pass
        state.model_task = task or "obb"

        while self.preset_bar.count():
            it = self.preset_bar.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._preset_buttons = []

        if state.model_task == "seg":
            btn = QPushButton("最佳")
            btn.setCheckable(True)
            btn.setFixedHeight(36); btn.setMinimumWidth(70)
            btn.setStyleSheet(self._PRESET_QSS.format(bg="#9B59B6", bg2="#C27BE0"))
            btn.setToolTip("用每类最佳 conf 阈值(iou=0.7), 忽略左侧 conf 滑块; 拖动滑块即退出")
            btn.clicked.connect(lambda _=False, b=btn: self._select_best(b))
            self.preset_bar.addWidget(btn)
            self._preset_buttons.append(btn)
            btn.setChecked(state.best_mode)
        else:
            state.best_mode = False
            for name, conf, iou in self._OBB_PRESETS:
                b = QPushButton(name)
                b.setCheckable(True)
                # 固定宽度, 容得下最长的"高准确率"4字, 不被布局压缩截断
                b.setFixedHeight(36); b.setFixedWidth(92)
                b.setStyleSheet(self._PRESET_QSS.format(bg="#2D8CFE", bg2="#1F6FE0"))
                b.clicked.connect(lambda _=False, btn=b, c=conf, i=iou: self._select_obb_preset(btn, c, i))
                self.preset_bar.addWidget(b)
                self._preset_buttons.append(b)
                # 默认高亮与当前滑块值匹配的档 (默认 0.5/0.5 = 标准)
                if abs(state.conf_threshold - conf) < 1e-3 and abs(state.iou_threshold - iou) < 1e-3:
                    b.setChecked(True)

    def _clear_preset_selection(self) -> None:
        for b in getattr(self, "_preset_buttons", []):
            b.setChecked(False)

    def _select_obb_preset(self, btn, conf: float, iou: float) -> None:
        for b in self._preset_buttons:
            b.setChecked(b is btn)
        state.best_mode = False
        self.conf_slider.setValue(int(round(conf * 100)))   # valueChanged → 更新 state+worker
        self.iou_slider.setValue(int(round(iou * 100)))
        if self.worker is not None:
            self.worker.update_thresholds(conf, iou, per_class=False)

    def _select_best(self, btn) -> None:
        on = btn.isChecked()
        for b in self._preset_buttons:
            b.setChecked(b is btn and on)
        state.best_mode = on
        self._mark_dirty()
        if on:
            self.iou_slider.setValue(70)   # seg 最佳 NMS iou=0.7
        if self.worker is not None:
            self.worker.update_thresholds(state.conf_threshold, state.iou_threshold, per_class=on)

    def _on_slider_user_moved(self, *_a) -> None:
        """用户手动拖滑块 → 取消所有预设选中 + 退出最佳 (回手动模式)."""
        if not (state.best_mode or any(b.isChecked() for b in getattr(self, "_preset_buttons", []))):
            return
        state.best_mode = False
        self._clear_preset_selection()
        if self.worker is not None:
            self.worker.update_thresholds(state.conf_threshold, state.iou_threshold, per_class=False)

    def _pick_source(self, mode: str) -> tuple[Any, str, bool]:
        """返回 (source, file_path, use_batch_worker); source 为 None 表示取消"""
        if mode == MODE_PHOTO:
            fp, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.tiff)")
            return (PhotoSource(fp), fp, False) if fp else (None, "", False)
        if mode == MODE_VIDEO:
            # 默认打开工人录制的视频目录, 技术员可直接复查工人录的严重异物片段
            start = str(VIDEO_DIR) if VIDEO_DIR.exists() else ""
            fp, _ = QFileDialog.getOpenFileName(self, "选择视频(默认: 工人录制目录)", start,
                                                "视频 (*.mp4 *.avi *.mov *.mkv)")
            return (VideoSource(fp), fp, False) if fp else (None, "", False)
        if mode == MODE_FOLDER:
            # 工人批次流程已预选目录 → 直接用, 不再二次弹框
            if self._pending_folder is not None:
                d, files = self._pending_folder
                self._pending_folder = None
                return FolderSource(d, files), d, True
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
        # 相机解耦: 预览线程 + 推理线程(= self.worker); 其余模式走原逐帧 worker
        # 例外: seg 模型走逐帧(不解耦)—— seg 的 mask 是逐像素位图, 解耦下慢半拍会飘/错位,
        # 逐帧虽然只有 ~2.6fps 但 mask 跟当前帧严丝合缝; OBB 只有框, 继续解耦让画面顺。
        is_seg = (state.model_task == "seg")
        self._decoupled = (mode == MODE_CAMERA and DECOUPLED_CAMERA and not is_seg)
        if self._decoupled:
            self._frame_slot = _LatestFrame()
            self.preview_worker = CameraPreviewWorker(source, self._frame_slot)
            self.worker = CameraInferenceWorker(self._frame_slot)
        else:
            self.worker = FolderBatchWorker(source) if use_batch else DetectionWorker(source)
            # 视频跳帧: 每 3 帧推理 1 次, 减轻流式负载 (检测框由 UI ObbOverlay 渲染)
            if mode == MODE_VIDEO:
                self.worker.frame_skip = 2
        self._current_file_path = file_path
        self._current_mode = mode  # 记下本次推理模式 (供全程记录, 切源前 finalize 用的是上一次的值)
        # 会话级状态清零 — 重选视频/重开摄像头即重新计数
        self._session_peak = Counter()
        self._cur_present = set()
        self._summary_shown = False
        self._folder_results = []
        self._folder_idx = 0
        self._run_detail_rows = []
        self._run_frame_idx = 0
        self._set_folder_nav_visible(False)
        self.table.clear_all()
        self.overlay.clear_all()

        # 日志: 工人(生产)+ 技术员(个人调试记录)都记一条; 管理员不检测。
        # 注: 报表/报警统计 worker_only(role='worker' 子查询), 技术员记录只进自己的个人中心, 不污染生产报表。
        db = DbHelper.instance()
        if state.role in ("worker", "technician"):
            self._current_log_id = db.insert_log(
                username=state.username,
                detect_mode=mode,
                model_name=state.model_name,
                conf_threshold=state.conf_threshold,
                iou_threshold=state.iou_threshold,
                file_path=file_path,
            )
        else:
            self._current_log_id = -1

        if self._decoupled:
            self.preview_worker.preview_ready.connect(self._on_preview_frame)
            self.preview_worker.error.connect(self._on_worker_error)
            self.worker.result_ready.connect(self._on_infer_result)
            self.worker.error.connect(self._on_worker_error)
        else:
            self.worker.frame_ready.connect(self._on_frame_ready)
            self.worker.error.connect(self._on_worker_error)
            self.worker.finished_clean.connect(self._on_worker_finished)
            self.worker.progress.connect(self._on_progress)
            if use_batch:
                self.worker.file_finished.connect(self._on_folder_file_done)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # 暂停开放给 视频/文件夹/摄像头 (照片单图瞬间完成, 无需暂停)
        self.pause_btn.setEnabled(mode != MODE_PHOTO)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("⏸ 暂停")
        self.center_status.setText(f"检测中 — {file_path or '工业相机'}")
        # 相机模式: 应用产线发布的曝光/增益(工人据此跑, 技术员预览也用)
        if mode == MODE_CAMERA and (state.cam_exposure or state.cam_gain):
            self._apply_cam_settings()
        self.worker.start()
        if self._decoupled and self.preview_worker is not None:
            self.preview_worker.start()
        log.info("启动检测: mode=%s, src=%s, 解耦=%s", mode, file_path, self._decoupled)

    def _apply_cam_settings(self) -> None:
        """把产线配置里的曝光/增益应用到相机(需相机已开/可开)。"""
        try:
            if not self.camera_mgr.is_opened():
                self.camera_mgr.open()
            if self.camera_mgr.is_opened():
                if state.cam_exposure:
                    self.camera_mgr.set_exposure(float(state.cam_exposure))
                if state.cam_gain:
                    self.camera_mgr.set_gain(float(state.cam_gain))
                log.info("应用产线相机参数: 曝光=%s 增益=%s", state.cam_exposure, state.cam_gain)
        except Exception as e:
            log.error("应用相机参数失败: %s", e)

    def _stop_worker_silent(self) -> None:
        """停掉当前 worker 并结清审计日志, 但不弹总计 (用于切源/重选). 停+等在 _finalize_run 内安全完成."""
        self._finalize_run(status="success")

    def _on_pause_toggled(self, paused: bool) -> None:
        if self.worker is None:
            return
        self.worker.set_paused(paused)
        if self.preview_worker is not None:   # 解耦相机: 预览线程也一起暂停/继续(冻结画面)
            self.preview_worker.set_paused(paused)
        self.pause_btn.setText("▶ 继续" if paused else "⏸ 暂停")
        self.center_status.setText("已暂停" if paused else "检测中")
        # 仅摄像头: 暂停即出累计总计 (峰值不清零, 继续可叠加).
        # 视频/文件夹暂停只冻结画面, 不弹窗打断 (方便暂停看某一帧).
        if paused and state.detect_mode == MODE_CAMERA:
            self._show_summary_popup()

    def _on_stop(self) -> None:
        if self.worker is None:
            return
        # 停检测前若在录制 → 先收尾保存视频(否则摄像头停了, 视频文件会悬空损坏)
        if self._recording:
            self._on_record_stop()
        self.center_status.setText("正在停止...")
        self._finalize_run(status="success")  # 内部安全 stop + wait, 再弹总计
        self._maybe_show_summary()

    def _on_worker_error(self, msg: str) -> None:
        if self.worker is None:
            return  # 已被处理 (避免排队的多个 error 信号重复弹窗)
        self._finalize_run(status="failed")  # 先安全停掉 worker, 再提示
        QMessageBox.warning(self, "检测异常", msg)

    def _on_worker_finished(self, reason: str) -> None:
        mode = state.detect_mode
        self.center_status.setText(f"完成: {reason}")
        self._finalize_run(status="success")
        # 文件夹批量完成 → 启用左右翻阅, 默认停在最后一张
        if mode == MODE_FOLDER and self._folder_results:
            self._set_folder_nav_visible(True)
            self._show_folder_index(len(self._folder_results) - 1)
        # 工人「文件夹=模拟视频批次」: 跑完自动结束批次(汇总走批次结束弹窗, 不再单独弹文件夹总计)
        if self._folder_batch_active and state.role == "worker" and state.batch_id:
            self._folder_batch_active = False
            self._on_batch_end()
            return
        self._maybe_show_summary()

    def _maybe_show_summary(self) -> None:
        """视频/文件夹/相机 结束或停止时弹一次总计 + 保存入口.

        照片单图不自动弹窗 — 用右侧「结果导出」保存(点击时同样弹 仅汇总/仅明细/两者 选择).
        """
        if state.detect_mode != MODE_PHOTO and not self._summary_shown:
            self._summary_shown = True
            self._show_summary_popup()

    def _show_summary_popup(self) -> None:
        DetectionSummaryDialog(
            state.detect_mode, dict(self._session_peak),
            on_save=self._save_current, parent=self,
        ).exec()

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
            # 全程记录: 每次推理一行汇总 (log_id>0 保证一次会话只记一次, 不重复)
            if self._session_peak:
                self._app_runs.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": _MODE_CN.get(self._current_mode, self._current_mode),
                    "file": self._current_file_path,
                    "peak": dict(self._session_peak),
                    "total": sum(self._session_peak.values()),
                })
        self._dispose_worker()

    def _dispose_worker(self) -> None:
        """安全停止并释放当前 worker.

        关键: 绝不调用 QThread.terminate() — 强杀正在跑 Python / 持锁的线程会
        "FATAL: exception not rethrown" 直接 Aborted. 改为协作式停止(stop 标志)+ wait;
        万一线程没在限时内退出, 转入模块级僵尸列表保活(既不强杀也不让其被 GC 析构,
        两者都会 Aborted), 等其自行结束或进程退出时回收. 用局部引用防 wait 期间被 GC.
        """
        # 先回收相机预览线程(它持有相机, 先停它解开取流)
        pw = self.preview_worker
        self.preview_worker = None
        self._frame_slot = None
        self._decoupled = False
        if pw is not None:
            try:
                pw.stop()
                if not pw.wait(5000):
                    log.warning("预览线程未在 5s 内退出, 转入僵尸列表")
                    _ZOMBIE_WORKERS[:] = [z for z in _ZOMBIE_WORKERS if z.isRunning()]
                    _ZOMBIE_WORKERS.append(pw)
            except Exception as e:
                log.error("回收预览线程异常: %s", e)

        w = self.worker
        self.worker = None
        if w is None:
            return
        try:
            w.stop()
            if w.wait(5000):
                return  # 正常退出
            log.warning("worker 未在 5s 内退出, 转入僵尸列表(不 terminate/不析构), 待其自行结束")
            _ZOMBIE_WORKERS[:] = [z for z in _ZOMBIE_WORKERS if z.isRunning()]
            _ZOMBIE_WORKERS.append(w)
        except Exception as e:
            log.error("回收 worker 异常: %s", e)

    def _on_progress(self, cur: int, total: int) -> None:
        if total > 0:
            self.center_status.setText(f"进度: {cur}/{total}")

    def _apply_detection_filter(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """技术员「检测项开关」: 丢掉禁检类别 → 不计数/不报警/不画框。"""
        if not state.disabled_classes:
            return detections
        return [d for d in detections if d.get("class_name") not in state.disabled_classes]

    # ── 相机解耦: 显示(预览线程) 与 推理(推理线程) 分开 ──
    def _show_pixmap(self, image_rgb: np.ndarray) -> None:
        """只把图显示到画布(不动检测框/表格), 供预览高帧率刷新。"""
        if self.center_stack.currentIndex() != 1:
            self.center_stack.setCurrentIndex(1)
        pix = ndarray_to_pixmap(image_rgb)
        if self.pix_item is None:
            self.pix_item = self.scene.addPixmap(pix)
        else:
            self.pix_item.setPixmap(pix)
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self._fit_view()

    def _on_preview_frame(self, frame_rgb: np.ndarray) -> None:
        """预览线程: 原始帧高帧率显示 + 录制; 框沿用最近一次推理结果(由 _on_infer_result 更新)。"""
        state.last_image_rgb = frame_rgb
        state.last_annotated_rgb = frame_rgb
        self._current_file_path = ""
        self._show_pixmap(frame_rgb)
        if self._recording:
            self._write_video_frame(frame_rgb)

    def _on_infer_result(self, detections: list[dict[str, Any]], infer_time: float) -> None:
        """推理线程: 回传检测结果 → 更新框/表/计数/报警(画面由预览帧负责, 这里不动画布图)。
        注: 推理走同一个 run_inference_runtime, 合并/NMS 结果与逐帧模式完全一致。"""
        detections = self._apply_detection_filter(detections)
        state.last_infer_time = infer_time
        self.infer_time_label.setText(f"🕐 检测耗时  {infer_time*1000:.0f}ms")
        self._last_full_detections = list(detections)
        self._apply_filter_now()   # 重画 overlay + 表格 + 目标数
        self.right_panel.show_detail(state.current_results[0] if state.current_results else {})
        self._check_alarms(detections, state.last_image_rgb)
        self._accumulate_counts(self._last_full_detections)
        self._run_frame_idx += 1
        for d in self._last_full_detections:
            self._run_detail_rows.append({"frame_no": self._run_frame_idx, "det": d})

    def _accumulate_counts(self, detections: list[dict[str, Any]]) -> None:
        """累计本批次各类检出数(累计口径, 不是峰值)。
        文件夹: 每张图独立 → 当帧各类数量直接相加(Σ 全部图片)。
        相机/视频: 连续帧 → 上升沿累加(某类从"无"变"有"才加当帧数量), 同一物体持续在画面里不重复计,
                   与报警去重同口径, 故累计检出数与严重报警次数能对得上。"""
        frame_counts = Counter(d.get("class_name", "") for d in detections if d.get("class_name"))
        if state.detect_mode == MODE_FOLDER:
            for cls, n in frame_counts.items():
                self._session_peak[cls] += n
        else:
            present = set(frame_counts)
            for cls, n in frame_counts.items():
                if cls not in self._cur_present:      # 该类本帧新出现 → 计入
                    self._session_peak[cls] += n
            self._cur_present = present

    def _on_frame_ready(
        self,
        annotated_rgb: np.ndarray,
        detections: list[dict[str, Any]],
        infer_time: float,
    ) -> None:
        detections = self._apply_detection_filter(detections)
        self._render_frame(annotated_rgb, detections, self._current_file_path, infer_time)
        self._check_alarms(detections, annotated_rgb)
        if self._recording:
            self._write_video_frame(annotated_rgb)
        # 累计本批次检出数(文件夹=逐图相加 / 相机视频=上升沿累加)
        self._accumulate_counts(self._last_full_detections)
        # 视频/摄像头: 逐帧累积全部(未过滤)明细, 供保存"所有推理帧"
        if state.detect_mode in (MODE_VIDEO, MODE_CAMERA):
            self._run_frame_idx += 1
            for d in self._last_full_detections:
                self._run_detail_rows.append({"frame_no": self._run_frame_idx, "det": d})
        log.info("frame_ready: 收到 %d 个检测目标, 耗时 %.0fms", len(detections), infer_time * 1000)

    def _check_alarms(self, detections: list[dict[str, Any]], frame_rgb: np.ndarray | None = None) -> None:
        """严重异物 → 视觉强报警 + 下游分流 + 存证据帧 + 入库(报警闭环)。
        冷却/频次升级由 AlarmManager 处理(同类 3s 内只报一次, 存图天然随报警去重)。
        仅工人触发: 技术员是调试, 不报警不入库不弹高危横幅。"""
        if state.role != "worker":
            return
        # 文件夹: 每张图=独立一片流动烟草 → 不跨图去重(两张图都有石头算两次);
        # 相机/视频: 连续流 → 上升沿去重(同一物体持续在画面不重复报)。
        dedup = (state.detect_mode != MODE_FOLDER)
        res = self._alarm_mgr.process([d.get("class_eng", "") for d in detections], dedup=dedup)
        if res["severe"]:
            db = DbHelper.instance()
            img_path = self._save_alarm_frame(frame_rgb, res["severe"][0])  # 只严重项存图, 高频类不存
            cn_names = []
            for c in res["severe"]:
                cn = next((d.get("class_name", c) for d in detections if d.get("class_eng") == c), c)
                cn_names.append(cn)
                trigger_divert(DIVERT_DELAY_MS)              # 下游分流(占位, 待接 GPIO)
                aid = db.add_alarm(state.username, c, cn, severity="severe",
                                   diverted=True, image_path=img_path, batch_id=state.batch_id)
                self._cur_alarm_ids.append((aid, cn))        # 每条报警单独入队, 逐个处置
            # 横幅显示「待处置 N 个」, 闪烁直到逐条全部确认
            self._refresh_alarm_banner("⚠ 严重异物频发!建议立即停线检修" if res["escalate"] else "")

    def _flash_alarm(self) -> None:
        """红横幅明暗交替闪烁 — 只要有未确认报警就一直闪。"""
        if not self._cur_alarm_ids:
            self._alarm_flash_timer.stop()
            return
        self._alarm_flash_on = not self._alarm_flash_on
        bg = "#d9534f" if self._alarm_flash_on else "#7a2a27"
        self.alarm_box.setStyleSheet(f"background:{bg}; border-radius:4px;")

    def _refresh_alarm_banner(self, extra: str = "") -> None:
        """按待处置队列刷新横幅: 显示「待处置 N 个」; 队列空则停闪收起。
        逐个处置 —— 每点一次「确认处置」清最早一条, N 递减, 数字与报警数一一对应。"""
        n = len(self._cur_alarm_ids)
        if n == 0:
            self._alarm_flash_timer.stop()
            self.alarm_banner.setText("")
            self.alarm_box.setStyleSheet("background:#d9534f; border-radius:4px;")
            self.alarm_box.setVisible(False)
            return
        names = "、".join(dict.fromkeys(cn for _aid, cn in self._cur_alarm_ids))
        text = (f"🔴 严重异物待处置 {n} 个({names}),已自动分流隔离 "
                f"— 点「确认处置」逐个排查上游")
        if extra:
            text += "   " + extra
        self.alarm_banner.setText(text)
        self.alarm_box.setVisible(True)
        if not self._alarm_flash_timer.isActive():
            self._alarm_flash_on = True
            self._alarm_flash_timer.start()

    def _save_alarm_frame(self, frame_rgb: np.ndarray | None, cls_eng: str) -> str:
        """存严重项证据帧到 logs/alarms/。只在报警(已去重)时调, 故天然不重复存。返回路径或 ''。"""
        if frame_rgb is None:
            return ""
        try:
            import time as _t
            from config import LOG_DIR
            d = LOG_DIR / "alarms"; d.mkdir(parents=True, exist_ok=True)
            fp = d / f"alarm_{_t.strftime('%Y%m%d_%H%M%S')}_{cls_eng}.jpg"
            cv2.imwrite(str(fp), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            return str(fp)
        except Exception as e:
            log.error("存报警证据帧失败: %s", e)
            return ""

    def _on_alarm_confirm(self) -> None:
        """工人确认处置 → 逐个闭环: 每点一次只处理最早的一条报警, 横幅更新剩余数。
        全部处理完才停闪收起 —— 报警数与处置次数一一对应。"""
        if not self._cur_alarm_ids:
            return
        db = DbHelper.instance()
        aid, _cn = self._cur_alarm_ids.pop(0)        # 取最早一条
        db.handle_alarm(aid, "工人已确认(查上游)")
        # 不再把"处置报警"写进审计日志(审计只留对系统的操作)
        self._refresh_alarm_banner()                  # 还有剩余→更新数字; 清零→自动收起

    def _on_center_start(self) -> None:
        """中央大按钮「开始检测」: 工人 → 走开批次流程(与顶栏一致); 其余 → 普通开始。"""
        if state.role == "worker":
            self._on_batch_start()
        else:
            self._on_start()

    def _on_center_stop(self) -> None:
        """中央大按钮「停止」: 工人 → 结束批次; 其余 → 普通停止。"""
        if state.role == "worker":
            if state.batch_id:
                self._on_batch_end()
            else:
                self._on_stop()
        else:
            self._on_stop()

    # ── 批次追溯(工人: 开批次=录信息+开摄像头开跑; 结束=停+汇总) ──
    def _on_batch_start(self) -> None:
        import time as _t
        db = DbHelper.instance()
        dlg = _BatchStartDialog(db.list_varieties(), self)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        if not v["batch_id"]:
            QMessageBox.warning(self, "提示", "批次号必填"); return
        db.start_batch(v["batch_id"], v["variety"], state.username,
                       state.model_name, state.conf_threshold, state.iou_threshold)
        state.batch_id = v["batch_id"]
        self._batch_start_ts = _t.time()
        db.add_audit(state.username, state.role, "batch_start",
                     f"{v['batch_id']} 品种={v['variety'] or '—'}")
        self._refresh_batch_ui()
        # 自动开摄像头并开始检测
        self._switch_mode(MODE_CAMERA)
        self._on_start()
        self.worker_dashboard.set_batch(state.batch_id, running=True)

    def _start_folder_batch(self) -> None:
        """工人「文件夹推理」= 模拟视频批次: 选目录 → 录批次 → 跑 → 跑完自动结束批次。
        检出/报警都挂该批次, 进批次追溯与报表(与相机批次同等待遇)。"""
        import time as _t
        if state.batch_id:
            QMessageBox.information(self, "提示", "当前已有进行中的批次, 请先结束再开新批次"); return
        d = QFileDialog.getExistingDirectory(self, "选择异物图片文件夹(模拟视频批次)")
        if not d:
            return
        files = list_media_files(d, recursive=False)
        if not files:
            QMessageBox.warning(self, "提示", "目录内无可检测的图片/视频"); return
        db = DbHelper.instance()
        dlg = _BatchStartDialog(db.list_varieties(), self)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        if not v["batch_id"]:
            QMessageBox.warning(self, "提示", "批次号必填"); return
        db.start_batch(v["batch_id"], v["variety"], state.username,
                       state.model_name, state.conf_threshold, state.iou_threshold)
        state.batch_id = v["batch_id"]
        self._batch_start_ts = _t.time()
        db.add_audit(state.username, state.role, "batch_start",
                     f"{v['batch_id']} 品种={v['variety'] or '—'} 源=文件夹")
        self._refresh_batch_ui()
        self.worker_dashboard.set_batch(state.batch_id, running=True)
        # 预选目录交给 _pick_source 直接取用, 跑完 _on_worker_finished 自动结束批次
        self._pending_folder = (d, files)
        self._folder_batch_active = True
        self._switch_mode(MODE_FOLDER)
        self._on_start()

    def _on_batch_end(self) -> None:
        db = DbHelper.instance()
        bid = state.batch_id
        if not bid:
            QMessageBox.information(self, "提示", "当前没有进行中的批次"); return
        # 录制中则先收尾保存视频(结束批次=摄像头要停, 不收尾视频会损坏)
        if self._recording:
            self._on_record_stop()
        # 先停检测(安全停 worker + 结清日志), 再汇总批次
        if self.worker is not None and self.worker.isRunning():
            self._finalize_run(status="success")
        total = sum(self._session_peak.values())
        n_alarm = db.count_alarms(batch_id=bid)
        if db.end_batch():
            db.add_audit(state.username, state.role, "batch_end", bid)
        state.batch_id = ""
        self._refresh_batch_ui()
        self._session_peak = Counter()
        self.worker_dashboard.set_batch("", running=False)
        self.worker_dashboard.update_counts(0, 0, {}, 0.0)
        QMessageBox.information(
            self, "批次已结束",
            f"批次【{bid}】已结束并汇总:\n\n"
            f"  · 累计检出异物: {total} 个\n"
            f"  · 严重报警: {n_alarm} 次\n\n"
            f"明细可在管理员「批次追溯」中查看。")

    def _refresh_batch_ui(self) -> None:
        """据 DB 是否有 running 批次, 同步标签/按钮 + state.batch_id(登录后调用可恢复)。"""
        import time as _t
        running = DbHelper.instance().get_running_batch()
        if running:
            state.batch_id = running["batch_id"]
            self.batch_label.setText(f"批次: {running['batch_id']}")
            self.batch_start_btn.setEnabled(False)
            self.batch_end_btn.setEnabled(True)
            if self._batch_start_ts <= 0:
                self._batch_start_ts = _t.time()  # 跨登录恢复, 计时从现在起(近似)
        else:
            state.batch_id = ""
            self.batch_label.setText("批次: —")
            self.batch_start_btn.setEnabled(True)
            self.batch_end_btn.setEnabled(False)

    def _refresh_worker_dashboard(self) -> None:
        """工人看板每秒刷新: 累计检出/严重报警次数/各严重类/已运行时长。"""
        if state.role != "worker":
            return
        import time as _t
        running = bool(state.batch_id)
        total = sum(self._session_peak.values())
        per_severe = {k: v for k, v in self._session_peak.items() if k in _SEVERE_CN}
        severe_cnt = DbHelper.instance().count_alarms(batch_id=state.batch_id) if running else 0
        elapsed = (_t.time() - self._batch_start_ts) if (running and self._batch_start_ts > 0) else 0.0
        self.worker_dashboard.update_counts(total, severe_cnt, per_severe, elapsed)

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
        """文件夹批量: 缓存每张图的 路径 + 检测结果 + 推理时图像尺寸 (不存图省内存)"""
        img = state.last_image_rgb
        shape = tuple(img.shape[:2]) if img is not None else None  # (h, w) 推理所用尺寸
        self._folder_results.append({
            "path": path,
            "detections": list(self._last_full_detections),
            "shape": shape,
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
        # 检测框坐标基于"推理时的图像尺寸"; 推理对大图(>4096)会缩放, 重读原图尺寸不同会导致框偏移.
        # 这里把重读图缩放回推理尺寸, 保证框与目标对齐.
        shape = r.get("shape")
        if shape is not None and tuple(rgb.shape[:2]) != tuple(shape):
            rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
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
        self._save_current()

    def _save_current(self) -> None:
        """本次推理结果保存 — 右侧「结果导出」与总计弹窗「保存结果」共用.

        按当前模式给选项: 照片/文件夹可存图(照片), 视频/摄像头只明细+汇总.
        明细=未过滤完整检测(照片单帧/文件夹逐图/视频相机逐帧); 汇总=本次各类峰值.
        同步写入(数据小), 不开线程 — 避免 QThread 回收崩溃.
        """
        mode = state.detect_mode
        has_data = bool(self._last_full_detections) or bool(self._folder_results) \
            or bool(self._run_detail_rows)
        if not has_data and not self._session_peak:
            QMessageBox.information(self, "提示", "本次无可保存结果")
            return
        allow_image = mode in (MODE_PHOTO, MODE_FOLDER)
        kinds = ask_save_kind(self, "保存本次检测结果，存哪些?", allow_image=allow_image)
        if kinds is None:
            return
        d = state.export_dir or QFileDialog.getExistingDirectory(
            self, "选择保存目录", str(Path.home()),
        )
        if not d:
            return
        try:
            session = Exporter.make_session_dir(d)
            if "summary" in kinds:
                Exporter.write_summary_csv(session / "汇总.csv", dict(self._session_peak))
            if "detail" in kinds:
                Exporter.write_detail_csv(session / "明细.csv", self._build_detail_rows(mode))
            if "image" in kinds:
                self._save_images(session, mode)
            QMessageBox.information(self, "保存成功", f"已保存到\n{session}")
        except Exception as e:
            log.error("保存失败: %s", e)
            QMessageBox.warning(self, "保存失败", str(e))

    def _build_detail_rows(self, mode: str) -> list[dict[str, Any]]:
        """把当前模式的检测汇成 write_detail_csv 的平铺 rows (未过滤完整检测)"""
        rows: list[dict[str, Any]] = []
        if mode == MODE_FOLDER:
            for i, r in enumerate(self._folder_results, 1):
                for det in r.get("detections", []):
                    rows.append({"path": r.get("path", ""), "frame_no": i, "det": det})
        elif mode in (MODE_VIDEO, MODE_CAMERA):
            src = self._current_file_path or ("工业相机" if mode == MODE_CAMERA else "")
            for r in self._run_detail_rows:
                rows.append({"path": src, "frame_no": r["frame_no"], "det": r["det"]})
        else:  # photo — 当前单帧
            for det in self._last_full_detections:
                rows.append({"path": self._current_file_path, "frame_no": 1, "det": det})
        return rows

    def _save_images(self, session: Path, mode: str) -> None:
        """存带框图 — 仅照片/文件夹. 照片存当前图; 文件夹逐图重读+画框(全部框, 不过滤)."""
        from inference import paint_annotations
        if mode == MODE_FOLDER:
            for i, r in enumerate(self._folder_results, 1):
                bgr = cv2.imread(r["path"], cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                shape = r.get("shape")
                if shape is not None and tuple(rgb.shape[:2]) != tuple(shape):
                    rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
                dets = r.get("detections", [])
                try:
                    img = paint_annotations(rgb, dets) if dets else rgb
                except Exception as e:
                    log.error("文件夹画框失败, 用原图: %s", e)
                    img = rgb
                Exporter._write_image(session / f"{i:03d}_{Path(r['path']).stem}.jpg", img)
        else:  # photo
            img = state.last_annotated_rgb
            if img is not None and self._last_full_detections:
                try:
                    img = paint_annotations(img, self._last_full_detections)
                except Exception as e:
                    log.error("导出画框失败, 使用原图: %s", e)
            Exporter._write_image(session / "annotated.jpg", img)

    def _open_history(self) -> None:
        HistoryQueryWidget(state.username, self).exec()

    def _open_edit(self) -> None:
        dlg = EditProfileDialog(state.username, self)
        dlg.saved.connect(self.profile_panel.refresh)
        dlg.exec()

    def _open_register(self) -> None:
        RegisterDialog(self).exec()

    def _config_changes(self) -> list[str]:
        """技术员当前配置 vs 进入时快照的差异(人话列表), 供退出时汇总。"""
        old = self._committed_cfg or {}
        new = self._current_cfg_snapshot()
        out: list[str] = []
        if old.get("model_name") != new["model_name"]:
            out.append(f"模型: {old.get('model_name','?')} → {new['model_name']}")
        if abs(float(old.get("conf", -1)) - new["conf"]) > 1e-6:
            out.append(f"置信度 conf: {old.get('conf','?')} → {new['conf']:.2f}")
        if abs(float(old.get("iou", -1)) - new["iou"]) > 1e-6:
            out.append(f"交并比 iou: {old.get('iou','?')} → {new['iou']:.2f}")
        if bool(old.get("best_mode", False)) != new["best_mode"]:
            out.append(f"每类最佳阈值: {'开' if new['best_mode'] else '关'}")
        if abs(float(old.get("exposure", 0)) - new["exposure"]) > 1e-6:
            out.append(f"相机曝光: {old.get('exposure', 0):.0f} → {new['exposure']:.0f} μs")
        if abs(float(old.get("gain", 0)) - new["gain"]) > 1e-6:
            out.append(f"相机增益: {old.get('gain', 0):.1f} → {new['gain']:.1f} dB")
        old_dis, new_dis = set(old.get("disabled_classes", [])), set(new["disabled_classes"])
        if old_dis != new_dis:
            added = new_dis - old_dis   # 新关掉的(不再检测)
            back = old_dis - new_dis    # 重新打开的
            if added:
                out.append(f"停止检测: {', '.join(sorted(added))}")
            if back:
                out.append(f"恢复检测: {', '.join(sorted(back))}")
        return out

    def _maybe_publish_config(self) -> bool:
        """技术员退出前: 若有未发布改动, 汇总询问是否生效到产线。
        返回 True=可继续退出, False=取消退出(留在页面)。"""
        if state.role != "technician" or not self._cfg_dirty:
            return True
        changes = self._config_changes()
        if not changes:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("发布生产配置")
        box.setText("你本次修改了以下检测配置:\n\n• " + "\n• ".join(changes) +
                    "\n\n是否保存并对工人端生效?")
        save = box.addButton("保存生效", QMessageBox.AcceptRole)
        discard = box.addButton("不保存", QMessageBox.DestructiveRole)
        cancel = box.addButton("取消(留在页面)", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == cancel:
            return False
        if clicked == save:
            cfg = LineConfig(
                model_path=state.model_path, model_name=state.model_name,
                conf=state.conf_threshold, iou=state.iou_threshold,
                best_mode=state.best_mode,
                disabled_classes=sorted(state.disabled_classes),
                exposure=state.cam_exposure, gain=state.cam_gain)
            cfg.save()
            DbHelper.instance().add_audit(state.username, state.role, "config_publish",
                                          "; ".join(changes))
            self._committed_cfg = cfg.snapshot()
            self._cfg_dirty = False
            QMessageBox.information(self, "已发布", "新配置已保存, 工人端下次登录即按此运行。")
        else:
            # 不保存: 还原到上次发布的产线配置 + 清空本次暂存
            # (否则本次改动残留, 下次退出会和新改动累加显示)
            self._load_line_config()
        return True

    def _on_logout(self) -> None:
        if not self._maybe_publish_config():
            return  # 用户选择取消, 不退出
        if self._recording:
            self._on_record_stop(silent=True)
        self._dash_timer.stop()
        self._dispose_worker()  # 安全停掉再注销, 避免线程残留崩溃
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

        检测进行中点 X 直接拦下来, 要求先停止再退出 —— 从源头堵死"边跑边关"这条
        曾导致 RK3588 反复 Aborted 的路径(必须先走正常 ■停止 让 worker 干净退出).
        """
        running = (self.worker is not None and self.worker.isRunning()) or \
                  (self.preview_worker is not None and self.preview_worker.isRunning())
        if running:
            QMessageBox.information(
                self, "正在检测中",
                "检测正在进行, 请先点击「■ 停止」结束本次检测, 然后再退出。")
            event.ignore()
            return
        if not self._maybe_publish_config():  # 技术员未发布改动 → 询问
            event.ignore()
            return
        try:
            if self._recording:
                self._on_record_stop(silent=True)
            self._dash_timer.stop()
            if self.worker is not None:
                self.worker.stop()  # 先置停止标志
            # 先关相机(close_device): 解开可能阻塞在取流 get_image 的 worker, 让它能尽快退出
            self.camera_mgr.shutdown()
            self._dispose_worker()  # 再协作式等待回收(不 terminate)
            self.scene.clear()
            self.pix_item = None
        except Exception as e:
            log.error("关闭时清理异常: %s", e)
        event.accept()
