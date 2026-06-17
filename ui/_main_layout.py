"""MainWindow 布局构建 mixin — 把 UI 构建代码抽出来, 保持 main_window.py < 500 行"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QFrame, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

from config import (
    CLASS_LIST_CN, COLOR_BG_MAIN, COLOR_BG_SIDE, COLOR_BORDER, COLOR_BTN_PRIMARY,
    COLOR_TEXT, COLOR_TEXT_DIM, MODE_CAMERA, MODE_FOLDER, MODE_PHOTO, MODE_VIDEO,
    btn_qss,
)
from core.app_state import state
from PySide6.QtGui import QIcon

from ui.center_splash import CenterSplashWidget
from ui.panels import ProfilePanel, WorkerDashboard
from ui.widgets import (
    NavButton, ResultTable, RightPanel, ZoomableGraphicsView, make_color_icon,
)
from utils.overlay import OverlayDispatcher


class LayoutMixin:
    """MainWindow 用 mixin — 仅持有 UI 构建职责, 业务逻辑在 main_window.py"""

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        # 顶栏包进横向滚动区: 屏幕够宽就铺满, 不够宽出横向滚动条, 绝不把窗口撑出屏幕外
        _top = self._build_top_bar()
        _top_scroll = QScrollArea()
        _top_scroll.setWidget(_top)
        _top_scroll.setWidgetResizable(True)
        _top_scroll.setFixedHeight(78)
        _top_scroll.setFrameShape(QFrame.NoFrame)
        _top_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _top_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _top_scroll.setStyleSheet(
            f"QScrollArea{{background:{COLOR_BG_SIDE}; border:none;}}"
            f"QScrollBar:horizontal{{height:10px; background:{COLOR_BG_MAIN}; margin:0;}}"
            f"QScrollBar::handle:horizontal{{background:{COLOR_BORDER}; border-radius:5px; min-width:40px;}}"
            f"QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{{width:0;}}")
        root.addWidget(_top_scroll)
        mid = QHBoxLayout(); mid.setContentsMargins(0, 0, 0, 0); mid.setSpacing(0)
        mid.addWidget(self._build_left_nav())
        mid.addWidget(self._build_center(), 1)
        mid.addWidget(self._build_right_stack())
        w = QWidget(); w.setLayout(mid)
        root.addWidget(w, 1)
        self._bottom_table = self._build_bottom_table()
        root.addWidget(self._bottom_table)

    def _make_slider(self, init: float) -> tuple[QSlider, QLabel]:
        sl = QSlider(Qt.Horizontal)
        sl.setRange(10, 95); sl.setValue(int(init * 100)); sl.setFixedWidth(100)
        lbl = QLabel(f"{init:.2f}"); lbl.setFixedWidth(40)
        return sl, lbl

    def _build_top_bar(self) -> QWidget:
        bar = QFrame(); bar.setFixedHeight(60)
        bar.setStyleSheet(f"background:{COLOR_BG_SIDE}; border-bottom:1px solid {COLOR_BORDER};")
        lay = QHBoxLayout(bar); lay.setContentsMargins(14, 8, 14, 8); lay.setSpacing(10)

        # 参数区(conf/iou/预设)整体包进 self._param_box, 工人模式下一次性隐藏
        self._param_box = QWidget()
        pbox = QHBoxLayout(self._param_box); pbox.setContentsMargins(0, 0, 0, 0); pbox.setSpacing(10)
        self.conf_slider, self.conf_label = self._make_slider(state.conf_threshold)
        conf_lab = QLabel("置信度 (Conf)")
        conf_lab.setStyleSheet(f"color:{COLOR_TEXT}; font-size:13px;")
        pbox.addWidget(conf_lab); pbox.addWidget(self.conf_slider); pbox.addWidget(self.conf_label)

        self.iou_slider, self.iou_label = self._make_slider(state.iou_threshold)
        iou_lab = QLabel("交并比 (IoU)")
        iou_lab.setStyleSheet(f"color:{COLOR_TEXT}; font-size:13px;")
        pbox.addWidget(iou_lab); pbox.addWidget(self.iou_slider); pbox.addWidget(self.iou_label)

        # 预设按钮区 — 滑块右边. 随模型 task 动态填充(OBB 3 个 / SEG 1 个), 由 main_window 重建.
        self.preset_bar = QHBoxLayout(); self.preset_bar.setContentsMargins(0, 0, 0, 0); self.preset_bar.setSpacing(6)
        _preset_host = QWidget(); _preset_host.setLayout(self.preset_bar)
        pbox.addSpacing(12); pbox.addWidget(_preset_host)
        lay.addWidget(self._param_box)
        lay.addStretch()

        self.infer_time_label = QLabel("🕐 检测耗时  0.00s")
        self.target_count_label = QLabel("⊙ 检测目标:  0个")
        for w in (self.infer_time_label, self.target_count_label):
            w.setStyleSheet(f"color:{COLOR_TEXT}; font-size:13px;")
        lay.addWidget(self.infer_time_label); lay.addSpacing(12); lay.addWidget(self.target_count_label)

        # 批次控件(追溯): 当前批次 + 开始/结束(工人/技术可用)
        self.batch_label = QLabel("批次: —")
        self.batch_label.setStyleSheet("color:#FFD23F; font-size:13px;")
        self.batch_start_btn = QPushButton("开始批次")
        self.batch_end_btn = QPushButton("结束批次")
        for b in (self.batch_start_btn, self.batch_end_btn):
            b.setFixedHeight(36); b.setMinimumWidth(76)
            b.setStyleSheet("background:#6c5ce7; color:white; border-radius:4px;")
        self.batch_end_btn.setEnabled(False)
        lay.addSpacing(16); lay.addWidget(self.batch_label)
        lay.addWidget(self.batch_start_btn); lay.addWidget(self.batch_end_btn)

        # 工人录制视频(录制/结束), 存固定目录
        self.record_start_btn = QPushButton("● 录制视频")
        self.record_stop_btn = QPushButton("■ 结束视频")
        for b in (self.record_start_btn, self.record_stop_btn):
            b.setFixedHeight(36); b.setMinimumWidth(86)
        self.record_start_btn.setStyleSheet(btn_qss("#c0392b", hover="#d65448"))
        self.record_stop_btn.setStyleSheet(btn_qss("#7f8c8d", hover="#95a5a6"))
        self.record_stop_btn.setEnabled(False)
        lay.addSpacing(8); lay.addWidget(self.record_start_btn); lay.addWidget(self.record_stop_btn)

        self.start_btn = QPushButton("▶ 开始")
        self.pause_btn = QPushButton("⏸ 暂停")
        self.stop_btn = QPushButton("■ 停止")
        self.capture_btn = QPushButton("📸 拍摄")
        self.start_btn.setStyleSheet(
            f"background:{COLOR_BTN_PRIMARY}; color:white; border-radius:4px; font-weight:bold;")
        self.pause_btn.setStyleSheet("background:#f0ad4e; color:white; border-radius:4px; font-weight:bold;")
        self.stop_btn.setStyleSheet("background:#d9534f; color:white; border-radius:4px; font-weight:bold;")
        self.capture_btn.setStyleSheet("background:#5BC0AA; color:white; border-radius:4px; font-weight:bold;")
        for b in (self.start_btn, self.pause_btn, self.stop_btn, self.capture_btn):
            b.setFixedHeight(36); b.setMinimumWidth(78)
        self.pause_btn.setCheckable(True); self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setVisible(False)  # 默认隐藏, 切到相机模式才显示
        self.capture_btn.setEnabled(False)
        lay.addWidget(self.start_btn); lay.addWidget(self.pause_btn)
        lay.addWidget(self.stop_btn); lay.addWidget(self.capture_btn)

        # 退出登录 — 顶栏明面(工人/技术员都有, 与管理员一致); 个人中心里不再放退出
        self.logout_top_btn = QPushButton("退出登录")
        self.logout_top_btn.setFixedHeight(36); self.logout_top_btn.setMinimumWidth(86)
        self.logout_top_btn.setStyleSheet(btn_qss("#d9534f", hover="#e66662"))
        lay.addSpacing(10); lay.addWidget(self.logout_top_btn)
        return bar

    def _build_left_nav(self) -> QWidget:
        nav = QFrame(); nav.setFixedWidth(220)
        nav.setStyleSheet(f"background:{COLOR_BG_SIDE}; border-right:1px solid {COLOR_BORDER};")
        lay = QVBoxLayout(nav); lay.setContentsMargins(0, 20, 0, 0); lay.setSpacing(2)

        def ic(hex_color: str, ch: str) -> QIcon:
            return QIcon(make_color_icon(hex_color, ch, 26))

        items = [
            ("profile", "个人中心", ic("#3E8EFE", "人")),
            (MODE_PHOTO, "照片选择", ic("#2DB95B", "图")),
            (MODE_VIDEO, "视频选择", ic("#2D8CFE", "视")),
            (MODE_FOLDER, "文件夹选择", ic("#F0AD4E", "夹")),
            (MODE_CAMERA, "摄像头", ic("#56C2D6", "相")),
            ("cam_settings", "相机亮度", ic("#FFC850", "亮")),
            ("model", "模型选择", ic("#B05AC9", "模")),
            # 「结果保存」(左侧全程保存)已去掉 — 只保留右侧「结果导出」(按本次推理保存)
        ]
        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True)
        self._nav_buttons: dict[str, NavButton] = {}
        for key, text, icon in items:
            btn = NavButton(text, icon)
            self.nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            lay.addWidget(btn)

        # 检测项开关(技术员专用, 默认隐藏): 关掉某类 → 该类不检测/不计数/不报警
        self.detect_toggle_box = self._build_detect_toggles()
        lay.addWidget(self.detect_toggle_box)
        self.detect_toggle_box.setVisible(False)

        lay.addStretch()

        self.user_footer = QLabel("当前用户: —")
        self.user_footer.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; padding:10px; border-top:1px solid {COLOR_BORDER};")
        lay.addWidget(self.user_footer)
        return nav

    def _build_detect_toggles(self) -> QWidget:
        """左侧「检测项」复选组(技术员): 勾=检测, 取消勾=不检测该异物。"""
        box = QFrame()
        box.setStyleSheet(f"background:{COLOR_BG_SIDE}; border-top:1px solid {COLOR_BORDER};")
        v = QVBoxLayout(box); v.setContentsMargins(12, 8, 12, 8); v.setSpacing(4)
        title = QLabel("检测项(取消勾选即不检测)")
        title.setWordWrap(True)
        title.setStyleSheet(f"color:{COLOR_TEXT}; font-size:12px; font-weight:bold;")
        v.addWidget(title)
        area = QScrollArea(); area.setWidgetResizable(True); area.setMinimumHeight(252)
        area.setStyleSheet(f"QScrollArea{{background:{COLOR_BG_MAIN}; border:1px solid {COLOR_BORDER}; border-radius:4px;}}")
        inner = QWidget(); inner.setStyleSheet(f"background:{COLOR_BG_MAIN};")
        il = QVBoxLayout(inner); il.setContentsMargins(8, 6, 8, 6); il.setSpacing(2)
        self._detect_checks: dict[str, QCheckBox] = {}
        for cn in CLASS_LIST_CN:
            cb = QCheckBox(cn); cb.setChecked(True)
            cb.setStyleSheet(f"color:{COLOR_TEXT};")
            self._detect_checks[cn] = cb
            il.addWidget(cb)
        il.addStretch()
        area.setWidget(inner)
        v.addWidget(area)
        return box

    def _build_center(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap); lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        # 报警横幅(严重异物) — 默认隐藏, 由 main_window 控制显隐 + 闪烁
        self.alarm_box = QFrame()
        self.alarm_box.setStyleSheet("background:#d9534f; border-radius:4px;")
        _abox = QHBoxLayout(self.alarm_box); _abox.setContentsMargins(14, 8, 14, 8)
        self.alarm_banner = QLabel("")
        self.alarm_banner.setStyleSheet("color:white; font-size:16px; font-weight:bold;")
        self.alarm_confirm_btn = QPushButton("确认处置1个")
        self.alarm_confirm_btn.setStyleSheet(
            "background:white; color:#d9534f; border-radius:4px; padding:6px 16px; font-weight:bold;")
        _abox.addWidget(self.alarm_banner, 1); _abox.addWidget(self.alarm_confirm_btn)
        self.alarm_box.setVisible(False)
        lay.addWidget(self.alarm_box)

        self.scene = QGraphicsScene(); self.scene.setBackgroundBrush(QColor(COLOR_BG_MAIN))
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setStyleSheet(f"border:1px solid {COLOR_BORDER}; background:{COLOR_BG_MAIN};")
        self.pix_item: QGraphicsPixmapItem | None = None
        self.overlay = OverlayDispatcher(self.scene)

        # 中央用 QStackedWidget: 0=splash 占位, 1=view 推理画面
        self.center_stack = QStackedWidget()
        self.center_splash = CenterSplashWidget()
        self.center_stack.addWidget(self.center_splash)
        self.center_stack.addWidget(self.view)
        self.center_stack.setCurrentIndex(0)  # 默认显示 splash
        lay.addWidget(self.center_stack, 1)

        status_row = QHBoxLayout(); status_row.setContentsMargins(0, 0, 0, 0)
        self.center_status = QLabel("就绪")
        self.center_status.setStyleSheet(f"color:{COLOR_TEXT_DIM}; padding:4px;")

        # 文件夹结果翻阅控件 (默认隐藏, 文件夹批量完成后才显示)
        self.folder_prev_btn = QPushButton("◀ 上一张")
        self.folder_next_btn = QPushButton("下一张 ▶")
        self.folder_page_label = QLabel("")
        _nav_qss = (
            f"QPushButton {{ background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:4px 12px; }}"
            f"QPushButton:hover {{ background:{COLOR_BTN_PRIMARY}; color:white; }}"
            f"QPushButton:disabled {{ color:{COLOR_TEXT_DIM}; }}"
        )
        for b in (self.folder_prev_btn, self.folder_next_btn):
            b.setStyleSheet(_nav_qss); b.setVisible(False)
        self.folder_page_label.setStyleSheet(f"color:{COLOR_TEXT}; padding:0 8px;")
        self.folder_page_label.setVisible(False)

        status_row.addWidget(self.center_status, 1)
        status_row.addWidget(self.folder_prev_btn)
        status_row.addWidget(self.folder_page_label)
        status_row.addWidget(self.folder_next_btn)
        lay.addLayout(status_row)
        return wrap

    def _build_right_stack(self) -> QStackedWidget:
        self.right_stack = QStackedWidget(); self.right_stack.setFixedWidth(300)
        self.right_panel = RightPanel(); self.profile_panel = ProfilePanel()
        self.worker_dashboard = WorkerDashboard()   # 工人看护看板
        self.right_stack.addWidget(self.right_panel)
        self.right_stack.addWidget(self.profile_panel)
        self.right_stack.addWidget(self.worker_dashboard)
        return self.right_stack

    def _build_bottom_table(self) -> QWidget:
        wrap = QFrame(); wrap.setFixedHeight(220)
        wrap.setStyleSheet(f"background:{COLOR_BG_SIDE}; border-top:1px solid {COLOR_BORDER};")
        lay = QVBoxLayout(wrap); lay.setContentsMargins(10, 6, 10, 10); lay.setSpacing(4)

        head = QLabel("检测结果列表 — 点击行查看详情")
        head.setStyleSheet(f"color:{COLOR_TEXT}; font-weight:bold; font-size:12px;")
        lay.addWidget(head)

        self.table = ResultTable()
        self.table.setStyleSheet(
            # 深色表格 — 显式给 item 和交替行设色, 否则交替行用 palette 默认色看不见数据
            f"QTableWidget {{ background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
            f"gridline-color:{COLOR_BORDER}; alternate-background-color:#252830; }}"
            f"QTableWidget::item {{ color:{COLOR_TEXT}; padding:4px; }}"
            f"QTableWidget::item:selected {{ background:{COLOR_BTN_PRIMARY}; color:white; }}"
            f"QHeaderView::section {{ background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};"
            f"padding:4px; border:none; border-right:1px solid {COLOR_BORDER}; }}")
        lay.addWidget(self.table, 1)
        return wrap
