"""MainWindow 布局构建 mixin — 把 UI 构建代码抽出来, 保持 main_window.py < 500 行"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider, QStackedWidget,
    QStyle, QVBoxLayout, QWidget,
)

from config import (
    COLOR_BG_MAIN, COLOR_BG_SIDE, COLOR_BORDER, COLOR_BTN_PRIMARY, COLOR_TEXT,
    COLOR_TEXT_DIM, MODE_CAMERA, MODE_FOLDER, MODE_PHOTO, MODE_VIDEO,
)
from core.app_state import state
from PySide6.QtGui import QIcon

from ui.center_splash import CenterSplashWidget
from ui.panels import ProfilePanel
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
        root.addWidget(self._build_top_bar())
        mid = QHBoxLayout(); mid.setContentsMargins(0, 0, 0, 0); mid.setSpacing(0)
        mid.addWidget(self._build_left_nav())
        mid.addWidget(self._build_center(), 1)
        mid.addWidget(self._build_right_stack())
        w = QWidget(); w.setLayout(mid)
        root.addWidget(w, 1)
        root.addWidget(self._build_bottom_table())

    def _make_slider(self, init: float) -> tuple[QSlider, QLabel]:
        sl = QSlider(Qt.Horizontal)
        sl.setRange(10, 95); sl.setValue(int(init * 100)); sl.setFixedWidth(140)
        lbl = QLabel(f"{init:.2f}"); lbl.setFixedWidth(40)
        return sl, lbl

    def _build_top_bar(self) -> QWidget:
        bar = QFrame(); bar.setFixedHeight(60)
        bar.setStyleSheet(f"background:{COLOR_BG_SIDE}; border-bottom:1px solid {COLOR_BORDER};")
        lay = QHBoxLayout(bar); lay.setContentsMargins(20, 8, 20, 8); lay.setSpacing(20)

        # 顶部左侧不放标题, 直接放参数(参考图布局: 滑块占大头, 右侧统计)
        self.conf_slider, self.conf_label = self._make_slider(state.conf_threshold)
        conf_lab = QLabel("置信度 (Conf)")
        conf_lab.setStyleSheet(f"color:{COLOR_TEXT}; font-size:13px;")
        lay.addWidget(conf_lab); lay.addWidget(self.conf_slider); lay.addWidget(self.conf_label)

        self.iou_slider, self.iou_label = self._make_slider(state.iou_threshold)
        iou_lab = QLabel("交并比 (IoU)")
        iou_lab.setStyleSheet(f"color:{COLOR_TEXT}; font-size:13px;")
        lay.addWidget(iou_lab); lay.addWidget(self.iou_slider); lay.addWidget(self.iou_label)
        lay.addStretch()

        self.infer_time_label = QLabel("🕐 检测耗时  0.00s")
        self.target_count_label = QLabel("⊙ 检测目标:  0个")
        for w in (self.infer_time_label, self.target_count_label):
            w.setStyleSheet(f"color:{COLOR_TEXT}; font-size:13px;")
        lay.addWidget(self.infer_time_label); lay.addSpacing(24); lay.addWidget(self.target_count_label)

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
            ("export_dir", "结果保存", ic("#D9534F", "存")),
        ]
        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True)
        self._nav_buttons: dict[str, NavButton] = {}
        for key, text, icon in items:
            btn = NavButton(text, icon)
            self.nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            lay.addWidget(btn)
        lay.addStretch()

        self.user_footer = QLabel("当前用户: —")
        self.user_footer.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; padding:10px; border-top:1px solid {COLOR_BORDER};")
        lay.addWidget(self.user_footer)
        return nav

    def _build_center(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap); lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

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
        self.export_progress = QProgressBar(); self.export_progress.setFixedWidth(200)
        self.export_progress.setVisible(False)
        self.export_progress.setStyleSheet(
            f"QProgressBar {{ background:{COLOR_BG_SIDE}; border:1px solid {COLOR_BORDER};"
            f"border-radius:3px; color:{COLOR_TEXT}; text-align:center; }}"
            f"QProgressBar::chunk {{ background:{COLOR_BTN_PRIMARY}; }}")
        status_row.addWidget(self.center_status, 1)
        status_row.addWidget(self.export_progress)
        lay.addLayout(status_row)
        return wrap

    def _build_right_stack(self) -> QStackedWidget:
        self.right_stack = QStackedWidget(); self.right_stack.setFixedWidth(300)
        self.right_panel = RightPanel(); self.profile_panel = ProfilePanel()
        self.right_stack.addWidget(self.right_panel)
        self.right_stack.addWidget(self.profile_panel)
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
