"""渲染分发器 — obb / det / seg 三类型路由

业务仅启用 obb, det/seg 留作扩展骨架.
当前实现: 直接显示 inference.draw_annotations 渲染好的图像,
高亮选中靠在 QGraphicsScene 额外叠加 QGraphicsRectItem.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsPolygonItem, QGraphicsScene, QGraphicsSimpleTextItem,
)

from config import COLOR_HIGHLIGHT


class ObbOverlay:
    """OBB 多边形 + 中文标签渲染 + 高亮"""

    def __init__(self, scene: QGraphicsScene) -> None:
        self.scene = scene
        self.items: list[QGraphicsPolygonItem] = []
        self.labels: list[QGraphicsSimpleTextItem] = []
        # 每个 item 缓存原始颜色 (高亮时改色, 取消时还原)
        self._orig_colors: list[tuple[int, int, int]] = []

    def clear(self) -> None:
        for it in self.items:
            self.scene.removeItem(it)
        for lab in self.labels:
            self.scene.removeItem(lab)
        self.items.clear(); self.labels.clear(); self._orig_colors.clear()

    def add(
        self,
        points: list[tuple[float, float]],
        color_rgb: tuple[int, int, int],
        label_text: str = "",
        scene_short_edge: int = 640,
    ) -> QGraphicsPolygonItem:
        poly = QPolygonF([QPointF(x, y) for x, y in points])
        item = QGraphicsPolygonItem(poly)
        pen = QPen(QColor(*color_rgb), max(2, scene_short_edge // 400))
        pen.setCosmetic(False)  # 跟随缩放, 视觉比例稳定
        item.setPen(pen)
        item.setBrush(QBrush(Qt.transparent))
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.scene.addItem(item)
        self.items.append(item)
        self._orig_colors.append(color_rgb)

        # 标签 — 放在 OBB 第一个点(左上角通常)
        if label_text:
            lab = QGraphicsSimpleTextItem(label_text)
            font_size = max(12, scene_short_edge // 60)
            lab.setFont(QFont("Microsoft YaHei", font_size, QFont.Bold))
            lab.setBrush(QBrush(QColor(*color_rgb)))
            lab.setPen(QPen(QColor(20, 20, 20), 1))  # 描边增强可读性
            # 位置: 第一个点上方
            x, y = points[0]
            lab.setPos(QPointF(x, max(0, y - font_size * 1.5)))
            self.scene.addItem(lab)
            self.labels.append(lab)
        else:
            self.labels.append(None)  # 占位对齐
        return item

    def highlight(self, index: int) -> None:
        for i, it in enumerate(self.items):
            pen = it.pen()
            if i == index:
                pen.setColor(QColor(COLOR_HIGHLIGHT))
                pen.setWidth(pen.width() + 2)
            else:
                pen.setColor(QColor(*self._orig_colors[i]))
            it.setPen(pen)


class DetOverlay:
    """常规检测框 — 预留骨架"""

    def __init__(self, scene: QGraphicsScene) -> None:
        self.scene = scene

    def clear(self) -> None:
        pass

    def add(self, *_: Any, **__: Any) -> None:
        pass


class SegOverlay:
    """分割掩膜 — 预留骨架"""

    def __init__(self, scene: QGraphicsScene) -> None:
        self.scene = scene

    def clear(self) -> None:
        pass

    def add(self, *_: Any, **__: Any) -> None:
        pass


class OverlayDispatcher:
    """根据 detection.type 路由到对应 overlay"""

    def __init__(self, scene: QGraphicsScene) -> None:
        self.obb = ObbOverlay(scene)
        self.det = DetOverlay(scene)
        self.seg = SegOverlay(scene)

    def clear_all(self) -> None:
        self.obb.clear()
        self.det.clear()
        self.seg.clear()

    def dispatch(self, detection: dict[str, Any]) -> None:
        t = detection.get("type", "obb")
        if t == "obb":
            pts = detection.get("points") or []
            color = detection.get("color_rgb", (0, 255, 0))
            if pts:
                self.obb.add(pts, color)
        elif t == "det":
            self.det.add(detection)
        elif t == "seg":
            self.seg.add(detection)

    def highlight(self, index: int) -> None:
        self.obb.highlight(index)
