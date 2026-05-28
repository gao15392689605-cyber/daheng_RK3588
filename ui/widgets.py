"""通用控件: ZoomableGraphicsView / NavButton / CaptchaWidget / ResultTable / RightPanel

ProfilePanel 和 HistoryQueryWidget 在 ui/panels.py
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFrame, QGraphicsPixmapItem, QGraphicsView,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from config import (
    CLASS_LIST_CN, COLOR_BG_MAIN, COLOR_BG_SIDE, COLOR_BORDER,
    COLOR_BTN_PRIMARY, COLOR_TEXT, COLOR_TEXT_DIM,
)
from utils.captcha import generate_captcha_text, render_captcha_image


class ZoomableGraphicsView(QGraphicsView):
    """画布: 滚轮缩放(以鼠标为中心), 双击重置自适应, 拖拽平移"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        # 用户主动缩放后为 True, _fit_view 就不再覆盖
        self.user_zoomed: bool = False

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.user_zoomed = True

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        # 双击 = 重置自适应
        self.user_zoomed = False
        for it in self.scene().items() if self.scene() else []:
            if isinstance(it, QGraphicsPixmapItem):
                self.resetTransform()
                self.fitInView(it, Qt.KeepAspectRatio)
                break
        super().mouseDoubleClickEvent(event)

    def reset_fit(self) -> None:
        self.user_zoomed = False
        for it in self.scene().items() if self.scene() else []:
            if isinstance(it, QGraphicsPixmapItem):
                self.resetTransform()
                self.fitInView(it, Qt.KeepAspectRatio)
                break


def make_color_icon(color_hex: str, glyph: str, size: int = 26) -> QPixmap:
    """生成圆角彩色方块 + 中央 unicode 字符的小图标"""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color_hex))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(0, 0, size, size, 6, 6)
    painter.setPen(QColor("white"))
    f = QFont("Microsoft YaHei", int(size * 0.55), QFont.Bold)
    painter.setFont(f)
    painter.drawText(pix.rect(), Qt.AlignCenter, glyph)
    painter.end()
    return pix


def make_round_pixmap(src: QPixmap, size: int) -> QPixmap:
    """把任意 QPixmap 裁成圆形 — QSS border-radius 只裁背景, 不裁 pixmap"""
    if src.isNull():
        return src
    src = src.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    out = QPixmap(size, size)
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    painter.setClipPath(path)
    # 居中
    dx = (size - src.width()) // 2
    dy = (size - src.height()) // 2
    painter.drawPixmap(dx, dy, src)
    painter.end()
    return out


def ndarray_to_pixmap(arr_rgb: "np.ndarray") -> QPixmap:
    """numpy RGB 数组转 QPixmap, 安全 copy"""
    h, w = arr_rgb.shape[:2]
    if arr_rgb.ndim == 2:
        img = QImage(arr_rgb.data, w, h, w, QImage.Format_Grayscale8)
    else:
        bpl = arr_rgb.strides[0]
        img = QImage(arr_rgb.data, w, h, bpl, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class NavButton(QPushButton):
    """侧栏导航按钮 — 可选中, 蓝色高亮"""

    def __init__(self, text: str, icon=None) -> None:
        super().__init__(text)
        if icon is not None:
            self.setIcon(icon)
        self.setCheckable(True)
        self.setMinimumHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{COLOR_TEXT};"
            f"border:none; text-align:left; padding-left:18px; font-size:13px; }}"
            f"QPushButton:hover {{ background:{COLOR_BTN_PRIMARY}; color:white; }}"
            f"QPushButton:checked {{ background:{COLOR_BTN_PRIMARY}; color:white; "
            f"border-left:3px solid #5BB0FF; }}"
        )

# ════════════════════════════════════════════════════════════
# CaptchaWidget — 验证码控件
# ════════════════════════════════════════════════════════════
class CaptchaWidget(QWidget):
    """4 位字母数字验证码 + 刷新按钮"""

    refreshed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.code: str = ""
        self._img_label = QLabel()
        self._img_label.setFixedSize(130, 44)
        self._img_label.setStyleSheet(f"border:1px solid {COLOR_BORDER};")
        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setFixedSize(36, 44)
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.clicked.connect(self.refresh)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self._img_label)
        lay.addWidget(self._refresh_btn)
        self.refresh()

    def refresh(self) -> None:
        self.code = generate_captcha_text(4)
        png_bytes = render_captcha_image(self.code)
        img = QImage.fromData(png_bytes, "PNG")
        self._img_label.setPixmap(QPixmap.fromImage(img))
        self.refreshed.emit()

    def verify(self, input_text: str) -> bool:
        return input_text.strip().upper() == self.code.upper()


# ════════════════════════════════════════════════════════════
# ResultTable — 检测结果表格
# ════════════════════════════════════════════════════════════
class ResultTable(QTableWidget):
    """5 列: 序号 / 路径 / 类别 / 置信度 / 坐标"""

    row_selected = Signal(int)

    HEADERS = ["序号", "路径", "类别", "置信度", "坐标"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 关掉 alternating row, 避免 stylesheet 不生效时出现白底白字
        self.setAlternatingRowColors(False)
        # QPalette 强制深色
        from PySide6.QtGui import QPalette
        pal = self.palette()
        pal.setColor(QPalette.Base, QColor(COLOR_BG_MAIN))
        pal.setColor(QPalette.AlternateBase, QColor(COLOR_BG_MAIN))
        pal.setColor(QPalette.Text, QColor(COLOR_TEXT))
        pal.setColor(QPalette.Highlight, QColor(COLOR_BTN_PRIMARY))
        pal.setColor(QPalette.HighlightedText, QColor("white"))
        self.setPalette(pal)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        # 路径列拉伸占剩余空间, 其余按内容/固定宽度
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 序号
        header.setSectionResizeMode(1, QHeaderView.Interactive)       # 路径(可拖, 默认窄)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 类别
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 置信度
        header.setSectionResizeMode(4, QHeaderView.Stretch)            # 坐标(拉伸)
        # 路径列默认 180px 宽 (只显示文件名, 够用), 用户可手动拖
        self.setColumnWidth(1, 180)
        self.itemSelectionChanged.connect(self._on_select)

    def load_results(self, file_path: str, results: list[dict[str, Any]]) -> None:
        self.setRowCount(0)
        from pathlib import Path
        short_name = Path(file_path).name if file_path else "—"
        for i, d in enumerate(results, 1):
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QTableWidgetItem(str(i)))
            self.setItem(row, 1, QTableWidgetItem(short_name))
            self.setItem(row, 2, QTableWidgetItem(d.get("class_name", "")))
            self.setItem(row, 3, QTableWidgetItem(f"{d.get('confidence', 0.0):.3f}"))
            pts = d.get("points") or []
            if pts:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                coord = f"({min(xs):.0f},{min(ys):.0f})-({max(xs):.0f},{max(ys):.0f})"
            else:
                coord = "—"
            self.setItem(row, 4, QTableWidgetItem(coord))

    def clear_all(self) -> None:
        self.setRowCount(0)

    def _on_select(self) -> None:
        rows = self.selectionModel().selectedRows()
        if rows:
            self.row_selected.emit(rows[0].row())


# ════════════════════════════════════════════════════════════
# RightPanel — 类别统计 / 过滤 / 详情 / 导出
# ════════════════════════════════════════════════════════════
class RightPanel(QWidget):

    filter_changed = Signal(set)
    export_request = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet(f"background:{COLOR_BG_SIDE}; color:{COLOR_TEXT};")
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # 顶部 YOLO 标志条
        head = QFrame()
        head.setStyleSheet(f"background:transparent;")
        head_lay = QHBoxLayout(head)
        head_lay.setContentsMargins(0, 0, 0, 0)
        head_lay.setSpacing(8)
        logo = QLabel()
        logo.setPixmap(make_color_icon("#2DB95B", "Y", 28))
        head_title = QLabel("YOLO检测系统")
        head_title.setStyleSheet(f"color:{COLOR_TEXT}; font-size:16px; font-weight:bold;")
        head_lay.addWidget(logo); head_lay.addWidget(head_title); head_lay.addStretch()
        lay.addWidget(head)

        # 类别统计
        lay.addWidget(self._section_title("📊  类别统计"))
        self.stat_label = QLabel("（无检测目标）")
        self.stat_label.setWordWrap(True)
        self.stat_label.setFont(QFont("Consolas", 11))
        self.stat_label.setStyleSheet(
            f"color:{COLOR_TEXT}; background:{COLOR_BG_MAIN};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:8px; min-height:80px;"
        )
        lay.addWidget(self.stat_label)

        # 类别过滤
        lay.addWidget(self._section_title("🎨  类别过滤"))
        self._filter_checks: dict[str, QCheckBox] = {}
        filter_area = QScrollArea()
        filter_area.setWidgetResizable(True)
        filter_area.setFixedHeight(120)
        filter_area.setStyleSheet(
            f"QScrollArea {{ background:{COLOR_BG_MAIN}; border:1px solid {COLOR_BORDER}; border-radius:4px; }}"
        )
        inner = QWidget()
        inner.setStyleSheet(f"background:{COLOR_BG_MAIN};")
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(8, 6, 8, 6); inner_lay.setSpacing(2)
        for cn in CLASS_LIST_CN:
            cb = QCheckBox(cn)
            cb.setChecked(True)
            cb.setStyleSheet(f"color:{COLOR_TEXT};")
            cb.stateChanged.connect(self._on_filter_changed)
            self._filter_checks[cn] = cb
            inner_lay.addWidget(cb)
        inner_lay.addStretch()
        filter_area.setWidget(inner)
        lay.addWidget(filter_area)

        # 检测类别 + 置信度
        lay.addWidget(self._section_title("🎯  检测类别"))
        self.class_field = QLineEdit("未选择")
        self.class_field.setReadOnly(True); self.class_field.setMinimumHeight(34)
        self.class_field.setStyleSheet(self._field_qss())
        lay.addWidget(self.class_field)

        lay.addWidget(self._section_title("📊  置信度"))
        self.conf_field = QLineEdit("0.00")
        self.conf_field.setReadOnly(True); self.conf_field.setMinimumHeight(34)
        self.conf_field.setStyleSheet(self._field_qss())
        lay.addWidget(self.conf_field)

        # 坐标位置 (x0/y0/x1/y1 — OBB 用最小外接 bbox 表达)
        lay.addWidget(self._section_title("📍  坐标位置"))
        coord_grid = QHBoxLayout()
        coord_grid.setSpacing(6)
        self.coord_fields: dict[str, QLineEdit] = {}
        for tag in ("x0", "y0", "x1", "y1"):
            col = QVBoxLayout()
            col.setSpacing(2)
            lab = QLabel(tag); lab.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:11px;")
            lab.setAlignment(Qt.AlignCenter)
            fld = QLineEdit(""); fld.setReadOnly(True); fld.setAlignment(Qt.AlignCenter)
            fld.setMinimumHeight(28)
            fld.setStyleSheet(self._field_qss())
            self.coord_fields[tag] = fld
            col.addWidget(lab); col.addWidget(fld)
            coord_grid.addLayout(col)
        lay.addLayout(coord_grid)

        lay.addStretch()

        # 橙色导出按钮
        self.export_btn = QPushButton("📂  结果导出")
        self.export_btn.setMinimumHeight(44)
        self.export_btn.setStyleSheet(
            "QPushButton { background:#F18B3A; color:white; font-weight:bold;"
            "font-size:14px; border-radius:4px; }"
            "QPushButton:hover { background:#FFA055; }"
        )
        self.export_btn.clicked.connect(self.export_request)
        lay.addWidget(self.export_btn)

    def _field_qss(self) -> str:
        return (
            f"QLineEdit {{ background:{COLOR_BG_MAIN}; color:{COLOR_TEXT};"
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:4px 8px; }}"
        )

    def _section_title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{COLOR_TEXT}; font-weight:bold; font-size:13px;"
            f"border-bottom:1px solid {COLOR_BORDER}; padding-bottom:4px;"
        )
        return lbl

    def _on_filter_changed(self) -> None:
        enabled = {cn for cn, cb in self._filter_checks.items() if cb.isChecked()}
        self.filter_changed.emit(enabled)

    def update_stats(self, results: list[dict[str, Any]]) -> None:
        if not results:
            self.stat_label.setText("（无检测目标）")
            return
        c = Counter(d.get("class_name", "") for d in results)
        lines = [f"{name}: {count}" for name, count in c.most_common()]
        lines.append(f"\n总计: {sum(c.values())}")
        self.stat_label.setText("\n".join(lines))

    def show_detail(self, d: dict[str, Any]) -> None:
        if not d:
            self.class_field.setText("未选择")
            self.conf_field.setText("0.00")
            for f in self.coord_fields.values():
                f.setText("")
            return
        self.class_field.setText(d.get("class_name", ""))
        self.conf_field.setText(f"{d.get('confidence', 0.0):.2f}")
        # OBB 点位 → 最小外接矩形 (x0,y0,x1,y1)
        pts = d.get("points") or []
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
            self.coord_fields["x0"].setText(f"{x0:.0f}")
            self.coord_fields["y0"].setText(f"{y0:.0f}")
            self.coord_fields["x1"].setText(f"{x1:.0f}")
            self.coord_fields["y1"].setText(f"{y1:.0f}")
        else:
            for f in self.coord_fields.values():
                f.setText("—")


# ════════════════════════════════════════════════════════════
# HistoryQueryWidget — 历史检测记录查询弹窗
# ════════════════════════════════════════════════════════════
