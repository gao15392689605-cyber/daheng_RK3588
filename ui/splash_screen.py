"""启动闪屏 — 展示初始化进度, 完成后跳登录页"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QLabel, QProgressBar, QSplashScreen, QVBoxLayout, QWidget

from config import COLOR_BG_MAIN, COLOR_BTN_PRIMARY, COLOR_TEXT, COLOR_TEXT_DIM


class SplashScreen(QSplashScreen):
    """500x320 启动闪屏 — logo + 标题 + 进度条 + 提示文字"""

    def __init__(self) -> None:
        pix = self._build_background(500, 320)
        super().__init__(pix, Qt.WindowStaysOnTopHint)
        self.setEnabled(False)

        self._tip_text = "正在初始化推理引擎..."

    def _build_background(self, w: int, h: int) -> QPixmap:
        pix = QPixmap(w, h)
        pix.fill(QColor(COLOR_BG_MAIN))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        # 主标题
        painter.setPen(QColor(COLOR_TEXT))
        painter.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        painter.drawText(
            pix.rect().adjusted(0, 60, 0, 0),
            Qt.AlignHCenter | Qt.AlignTop,
            "烟草异物智能检测系统",
        )
        # 副标题
        painter.setPen(QColor(COLOR_TEXT_DIM))
        painter.setFont(QFont("Microsoft YaHei", 11))
        painter.drawText(
            pix.rect().adjusted(0, 110, 0, 0),
            Qt.AlignHCenter | Qt.AlignTop,
            "RK3588 NPU 加速 · 10 类异物检测",
        )
        # 装饰条
        painter.fillRect(170, 180, 160, 3, QColor(COLOR_BTN_PRIMARY))

        painter.end()
        return pix

    def set_tip(self, text: str) -> None:
        self._tip_text = text
        self.showMessage(
            text,
            Qt.AlignHCenter | Qt.AlignBottom,
            QColor(COLOR_TEXT),
        )
        # 强制刷新
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def drawContents(self, painter: QPainter) -> None:  # noqa: N802
        painter.setPen(QColor(COLOR_TEXT))
        painter.setFont(QFont("Microsoft YaHei", 11))
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -30),
            Qt.AlignHCenter | Qt.AlignBottom,
            self._tip_text,
        )
        painter.setPen(QColor(COLOR_TEXT_DIM))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -10),
            Qt.AlignHCenter | Qt.AlignBottom,
            "© 2026 烟草异物检测系统",
        )
