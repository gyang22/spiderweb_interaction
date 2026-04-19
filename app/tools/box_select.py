"""Rubber-band rectangle selection tool (PCD or skeleton nodes)."""

import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor
from app.tools.base_tool import AbstractTool


class BoxSelectTool(AbstractTool):
    def __init__(self):
        self._dragging = False
        self._start: QPoint | None = None
        self._end: QPoint | None = None

    def mouse_press(self, event, viewport) -> None:
        self._dragging = True
        pos = event.position().toPoint()
        self._start = pos
        self._end = pos

    def mouse_move(self, event, viewport) -> None:
        if self._dragging:
            self._end = event.position().toPoint()
            viewport.update()

    def mouse_release(self, event, viewport) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._end = event.position().toPoint()

        if not viewport.has_selectable():
            self._start = self._end = None
            return

        x1, y1 = self._start.x(), self._start.y()
        x2, y2 = self._end.x(), self._end.y()

        if abs(x2 - x1) < 3 and abs(y2 - y1) < 3:
            self._start = self._end = None
            viewport.update()
            return

        screen_xy, all_idx = viewport.project_selectable()
        rx1, rx2 = sorted([x1, x2])
        ry1, ry2 = sorted([y1, y2])
        mask = (
            (screen_xy[:, 0] >= rx1) & (screen_xy[:, 0] <= rx2) &
            (screen_xy[:, 1] >= ry1) & (screen_xy[:, 1] <= ry2)
        )
        indices = all_idx[mask]

        add = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        viewport.apply_region_selection(indices, add)

        self._start = self._end = None
        viewport.update()

    def draw_overlay(self, painter: QPainter) -> None:
        if not self._dragging or self._start is None or self._end is None:
            return
        pen = QPen(QColor(255, 200, 50), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        rect = QRect(self._start, self._end).normalized()
        painter.drawRect(rect)
