"""轻量柱状图/饼状图控件 — QPainter 手绘, 不引第三方图表库(板端无需额外装包)。

用于管理面板报表: 选「本批次/今天/本周/半月」→ 画该范围各类异物检出数量。
带入场生长动画 + 鼠标悬停高亮(提亮 + 弹出 + 数值), 不死板。
"""
from __future__ import annotations

import math

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from config import COLOR_BG_MAIN, COLOR_BORDER, COLOR_TEXT, COLOR_TEXT_DIM

_BAR_COLORS = [
    "#2D8CFE", "#2DB95B", "#F0AD4E", "#D9534F", "#9B59B6",
    "#56C2D6", "#E67E22", "#1ABC9C", "#E84393", "#FFD23F",
]


class BarChart(QWidget):
    """横轴=类别, 纵轴=数量。set_data({类别: 数量}) 刷新。

    入场: 柱子从底部生长(~600ms)。悬停: 该柱提亮 + 顶部弹出数值。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: list[tuple[str, int]] = []
        self._title = ""
        self._progress = 1.0           # 入场动画进度 0→1
        self._hover = -1               # 当前悬停柱索引
        self._geom: tuple | None = None  # 最近一次绘制的布局(供命中测试)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; border:1px solid {COLOR_BORDER}; border-radius:4px;")
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(600)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)

    def _on_anim(self, v) -> None:
        self._progress = float(v)
        self.update()

    def set_data(self, data: dict[str, int], title: str = "") -> None:
        self._data = sorted(data.items(), key=lambda kv: -kv[1])
        self._title = title
        self._hover = -1
        self._anim.stop()
        self._anim.start()   # 每次刷新都重新生长

    def leaveEvent(self, _e) -> None:  # noqa: N802
        if self._hover != -1:
            self._hover = -1
            self.update()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        idx = -1
        if self._geom:
            margin_l, margin_t, plot_w, plot_h, n, slot = self._geom
            x = e.position().x()
            if margin_l <= x <= margin_l + plot_w and n:
                idx = min(n - 1, int((x - margin_l) / slot))
        if idx != self._hover:
            self._hover = idx
            self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(COLOR_BG_MAIN))

        margin_l, margin_r, margin_t, margin_b = 44, 16, 34, 48
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b

        p.setPen(QColor(COLOR_TEXT))
        p.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        p.drawText(QRectF(0, 6, w, 22), Qt.AlignCenter, self._title or "异物检出统计")

        if not self._data or plot_w <= 0 or plot_h <= 0:
            self._geom = None
            p.setPen(QColor(COLOR_TEXT_DIM))
            p.setFont(QFont("Microsoft YaHei", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "该范围暂无检测数据")
            return

        max_v = max(v for _, v in self._data) or 1
        p.setPen(QPen(QColor(COLOR_BORDER), 1))
        p.drawLine(margin_l, margin_t, margin_l, margin_t + plot_h)
        p.drawLine(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h)

        p.setFont(QFont("Microsoft YaHei", 8))
        for frac in (0.0, 0.5, 1.0):
            val = int(round(max_v * frac))
            y = margin_t + plot_h - plot_h * frac
            p.setPen(QColor(COLOR_TEXT_DIM))
            p.drawText(QRectF(0, y - 8, margin_l - 6, 16), Qt.AlignRight | Qt.AlignVCenter, str(val))
            if frac > 0:
                p.setPen(QPen(QColor(COLOR_BORDER), 1, Qt.DotLine))
                p.drawLine(margin_l, int(y), margin_l + plot_w, int(y))

        n = len(self._data)
        slot = plot_w / n
        bar_w = min(48.0, slot * 0.6)
        self._geom = (margin_l, margin_t, plot_w, plot_h, n, slot)
        for i, (name, val) in enumerate(self._data):
            cx = margin_l + slot * (i + 0.5)
            bh = plot_h * (val / max_v) * self._progress
            hovered = (i == self._hover)
            bw = bar_w * (1.08 if hovered else 1.0)   # 悬停略微变宽=弹出感
            x = cx - bw / 2
            y = margin_t + plot_h - bh
            base = QColor(_BAR_COLORS[i % len(_BAR_COLORS)])
            col = base.lighter(135) if hovered else base
            if hovered:
                # 柱顶发光描边
                p.setPen(QPen(base.lighter(160), 2))
            else:
                p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(QRectF(x, y, bw, bh), 3, 3)
            # 数值(悬停时加大加亮)
            p.setPen(QColor(COLOR_TEXT) if not hovered else base.lighter(160))
            p.setFont(QFont("Microsoft YaHei", 11 if hovered else 9, QFont.Bold))
            p.drawText(QRectF(cx - slot / 2, y - 20, slot, 18), Qt.AlignCenter, str(val))
            # 类别名
            p.setPen(QColor(COLOR_TEXT) if hovered else QColor(COLOR_TEXT_DIM))
            p.setFont(QFont("Microsoft YaHei", 8))
            label = name if len(name) <= 5 else name[:4] + "…"
            p.drawText(QRectF(cx - slot / 2, margin_t + plot_h + 4, slot, margin_b - 6),
                       Qt.AlignHCenter | Qt.AlignTop, label)


class PieChart(QWidget):
    """饼状图 — 占比。set_data({标签: 数量}) 刷新; 用于"严重异物 vs 其他异物"占比。

    入场: 扇区从顶部顺时针扫开(~600ms)。悬停: 该扇区向外弹出 + 提亮 + 图例高亮。
    """

    _PIE_COLORS = ["#D9534F", "#2D8CFE", "#2DB95B", "#F0AD4E", "#9B59B6"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: list[tuple[str, int]] = []
        self._title = ""
        self._progress = 1.0
        self._hover = -1
        self._geom: tuple | None = None   # (cx, cy, radius)
        self.setMinimumHeight(260)
        self.setMinimumWidth(220)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{COLOR_BG_MAIN}; border:1px solid {COLOR_BORDER}; border-radius:4px;")
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(650)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)

    def _on_anim(self, v) -> None:
        self._progress = float(v)
        self.update()

    def set_data(self, data: dict[str, int], title: str = "") -> None:
        self._data = [(k, int(v)) for k, v in data.items() if int(v) > 0]
        self._title = title
        self._hover = -1
        self._anim.stop()
        self._anim.start()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        if self._hover != -1:
            self._hover = -1
            self.update()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        idx = -1
        total = sum(v for _, v in self._data)
        if self._geom and total > 0:
            cx, cy, radius = self._geom
            dx = e.position().x() - cx
            dy = e.position().y() - cy
            if math.hypot(dx, dy) <= radius:
                # Qt 角度: 0°=3点钟, 逆时针为正; 我们从 90°(12点)起顺时针画
                ang = math.degrees(math.atan2(-dy, dx)) % 360.0
                start = 90.0
                for i, (_n, val) in enumerate(self._data):
                    span = 360.0 * val / total
                    lo = (start - span) % 360.0
                    hi = start % 360.0
                    inside = (lo < hi and lo <= ang < hi) or (lo > hi and (ang >= lo or ang < hi))
                    if inside:
                        idx = i; break
                    start -= span
        if idx != self._hover:
            self._hover = idx
            self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(COLOR_BG_MAIN))

        p.setPen(QColor(COLOR_TEXT))
        p.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        p.drawText(QRectF(0, 6, w, 22), Qt.AlignCenter, self._title or "占比")

        total = sum(v for _, v in self._data)
        if total <= 0:
            self._geom = None
            p.setPen(QColor(COLOR_TEXT_DIM))
            p.setFont(QFont("Microsoft YaHei", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "该范围暂无检测数据")
            return

        d = max(60, min(w - 40, h - 110))
        cx, cy = w / 2, 34 + d / 2
        radius = d / 2
        self._geom = (cx, cy, radius)
        rect = QRectF(cx - radius, cy - radius, d, d)

        start = 90 * 16  # 从顶部开始, 顺时针
        for i, (name, val) in enumerate(self._data):
            span = -int(round(360 * 16 * val / total * self._progress))
            hovered = (i == self._hover)
            base = QColor(self._PIE_COLORS[i % len(self._PIE_COLORS)])
            if hovered:
                # 弹出: 扇区沿其角平分线向外平移一点
                mid_deg = (start + span / 2) / 16.0
                off = 8.0
                dx = off * math.cos(math.radians(mid_deg))
                dy = -off * math.sin(math.radians(mid_deg))
                er = QRectF(rect.translated(dx, dy))
                p.setBrush(base.lighter(130))
                p.setPen(QPen(base.lighter(165), 2))
                p.drawPie(er, start, span)
            else:
                p.setBrush(base)
                p.setPen(Qt.NoPen)
                p.drawPie(rect, start, span)
            start += span

        # 图例: 色块 + 标签 + 百分比(悬停项高亮)
        ly = int(cy + radius + 16)
        p.setFont(QFont("Microsoft YaHei", 9))
        for i, (name, val) in enumerate(self._data):
            pct = val / total * 100
            base = QColor(self._PIE_COLORS[i % len(self._PIE_COLORS)])
            hovered = (i == self._hover)
            p.setPen(Qt.NoPen)
            p.setBrush(base.lighter(140) if hovered else base)
            p.drawRoundedRect(QRectF(16, ly + i * 20, 12, 12), 2, 2)
            p.setPen(base.lighter(160) if hovered else QColor(COLOR_TEXT))
            p.setFont(QFont("Microsoft YaHei", 10 if hovered else 9, QFont.Bold if hovered else QFont.Normal))
            p.drawText(QRectF(34, ly + i * 20 - 2, w - 44, 16),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{name}  {val} ({pct:.0f}%)")
