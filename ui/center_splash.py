"""中央空状态占位 — 炫酷科技风, 替代黑屏

- 径向渐变背景 (深蓝中心 → 主背景外缘)
- 多层雷达光圈 + 8 端点 + 中央对勾
- YOLO 大字 + AI 角标
- 烟草异物智能检测系统 + 三个标签
- ▶ 开始检测 / ■ 停止检测 中央大按钮
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from config import COLOR_BG_MAIN, COLOR_BTN_PRIMARY, COLOR_TEXT, COLOR_TEXT_DIM


class _Radar(QWidget):
    """蓝色雷达 — 三层圆环 + 8 端点 + 中央对勾 + 扫描扇形 + 发光"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setMaximumSize(320, 320)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - 8

        # 发光晕 (多层半透明圆)
        for k, alpha in [(1.0, 18), (0.85, 24), (0.7, 36), (0.55, 56)]:
            grad = QRadialGradient(cx, cy, r * k)
            grad.setColorAt(0, QColor(70, 170, 255, alpha))
            grad.setColorAt(1, QColor(70, 170, 255, 0))
            p.setBrush(grad); p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), r * k, r * k)

        # 外圈虚线
        p.setPen(QPen(QColor(80, 160, 230, 140), 1, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # 中圈
        p.setPen(QPen(QColor(70, 180, 255, 220), 1.4))
        p.drawEllipse(QPointF(cx, cy), r * 0.72, r * 0.72)
        # 内圈
        p.drawEllipse(QPointF(cx, cy), r * 0.44, r * 0.44)

        # 扫描扇形 (静态, 增加科技感)
        p.save()
        p.translate(cx, cy)
        sweep_grad = QLinearGradient(0, 0, r, 0)
        sweep_grad.setColorAt(0, QColor(80, 200, 255, 100))
        sweep_grad.setColorAt(1, QColor(80, 200, 255, 0))
        p.setBrush(sweep_grad); p.setPen(Qt.NoPen)
        p.rotate(-35)
        p.drawPie(QRectF(-r, -r, r * 2, r * 2), 0, 50 * 16)
        p.restore()

        # 8 端点 (圆点)
        p.setBrush(QColor(90, 200, 255)); p.setPen(Qt.NoPen)
        for i in range(8):
            ang = i * math.pi / 4
            x = cx + r * math.cos(ang); y = cy + r * math.sin(ang)
            p.drawEllipse(QPointF(x, y), 4.5, 4.5)

        # 中央对勾在小圆内
        inner_r = r * 0.18
        # 小圆背景
        sm = QRadialGradient(cx, cy, inner_r)
        sm.setColorAt(0, QColor(40, 60, 90))
        sm.setColorAt(1, QColor(30, 50, 80))
        p.setBrush(sm); p.setPen(QPen(QColor(70, 200, 180, 200), 1.2))
        p.drawEllipse(QPointF(cx, cy), inner_r, inner_r)
        # 对勾
        pen_chk = QPen(QColor(90, 230, 180), 3); pen_chk.setCapStyle(Qt.RoundCap)
        p.setPen(pen_chk)
        s = inner_r * 0.55
        p.drawLine(QPointF(cx - s, cy + s * 0.1), QPointF(cx - s * 0.2, cy + s * 0.8))
        p.drawLine(QPointF(cx - s * 0.2, cy + s * 0.8), QPointF(cx + s * 1.0, cy - s * 0.7))


class CenterSplashWidget(QWidget):
    """中央空状态 — 含开始/停止按钮, 信号转发给主窗口"""

    start_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # 不在 stylesheet 里设固定背景, 用 paintEvent 画径向渐变
        self.setAutoFillBackground(False)
        self._build()

    def paintEvent(self, _e) -> None:  # noqa: N802
        """径向渐变背景: 中心略亮的深蓝 → 主背景色"""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        grad = QRadialGradient(w / 2, h / 2, max(w, h) * 0.7)
        grad.setColorAt(0, QColor(30, 50, 78))
        grad.setColorAt(1, QColor(COLOR_BG_MAIN))
        p.fillRect(self.rect(), QBrush(grad))

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 30, 20, 30); lay.setSpacing(14)
        lay.addStretch()

        # 雷达 — 直接放 VBoxLayout 用 AlignHCenter 居中, 不依赖 HBoxLayout+stretch
        lay.addWidget(_Radar(), 0, Qt.AlignHCenter)

        # YOLO 大字 — 删除 AI 角标, 独立居中
        yolo = QLabel("YOLO")
        yolo.setAlignment(Qt.AlignCenter)
        yolo.setStyleSheet(
            f"color:{COLOR_BTN_PRIMARY}; font-size:54px; font-weight:bold;"
        )
        lay.addWidget(yolo)

        sub = QLabel("烟  草  异  物  智  能  检  测  系  统")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color:{COLOR_TEXT}; font-size:22px; font-weight:bold;")
        lay.addWidget(sub)

        tags_row = QHBoxLayout()
        tags_row.setSpacing(32); tags_row.addStretch()
        for icon, text, color in (
            ("⚡", "快速检测", "#FFC850"),
            ("◎", "精准识别", "#2DB95B"),
            ("✦", "深度学习", "#B05AC9"),
        ):
            t = QLabel(f"<span style='color:{color}; font-size:14px;'>{icon}</span> "
                       f"<span style='color:{COLOR_TEXT_DIM}; font-size:13px;'>{text}</span>")
            tags_row.addWidget(t)
        tags_row.addStretch()
        lay.addLayout(tags_row)

        lay.addSpacing(20)

        # 开始/停止 大按钮
        btn_row = QHBoxLayout(); btn_row.setSpacing(20); btn_row.addStretch()
        self.start_big = QPushButton("▶  开始检测")
        self.start_big.setMinimumSize(180, 56)
        self.start_big.setCursor(Qt.PointingHandCursor)
        self.start_big.setStyleSheet(
            f"QPushButton {{ background:{COLOR_BTN_PRIMARY}; color:white;"
            f"border:none; border-radius:8px; font-size:17px; font-weight:bold; }}"
            f"QPushButton:hover {{ background:#4FA0FF; }}"
        )
        self.start_big.clicked.connect(self.start_clicked)

        self.stop_big = QPushButton("■  停止检测")
        self.stop_big.setMinimumSize(180, 56)
        self.stop_big.setCursor(Qt.PointingHandCursor)
        self.stop_big.setStyleSheet(
            "QPushButton { background:#2DB95B; color:white;"
            "border:none; border-radius:8px; font-size:17px; font-weight:bold; }"
            "QPushButton:hover { background:#4ACE74; }"
        )
        self.stop_big.clicked.connect(self.stop_clicked)
        btn_row.addWidget(self.start_big); btn_row.addWidget(self.stop_big); btn_row.addStretch()
        lay.addLayout(btn_row)

        lay.addStretch()
