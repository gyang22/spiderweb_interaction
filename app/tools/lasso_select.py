"""Freehand polygon (lasso) selection tool."""

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygon
from app.tools.base_tool import AbstractTool


def _points_in_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Ray-casting polygon test (vectorised).
    points_xy: (M, 2) float — screen XY coordinates to test
    polygon:   (K, 2) float — lasso vertex screen coordinates
    Returns boolean mask of length M.
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points_xy), dtype=bool)

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i, 0], polygon[i, 1]
        xj, yj = polygon[j, 0], polygon[j, 1]
        denom = yj - yi
        if abs(denom) < 1e-9:
            j = i
            continue
        cond1 = (yi > y) != (yj > y)
        intersect_x = (xj - xi) * (y - yi) / denom + xi
        cond2 = x < intersect_x
        inside ^= cond1 & cond2
        j = i

    return inside


class LassoSelectTool(AbstractTool):
    def __init__(self):
        self._drawing = False
        self._polygon: list[QPoint] = []

    def mouse_press(self, event, viewport) -> None:
        self._drawing = True
        self._polygon = [event.position().toPoint()]

    def mouse_move(self, event, viewport) -> None:
        if self._drawing:
            pt = event.position().toPoint()
            # Add vertex only if it moved enough (avoids huge polygon arrays)
            if not self._polygon or (
                abs(pt.x() - self._polygon[-1].x()) > 2 or
                abs(pt.y() - self._polygon[-1].y()) > 2
            ):
                self._polygon.append(pt)
                viewport.update()

    def mouse_release(self, event, viewport) -> None:
        if not self._drawing or len(self._polygon) < 3:
            self._drawing = False
            self._polygon = []
            viewport.update()
            return

        self._drawing = False

        if not viewport.has_point_cloud():
            self._polygon = []
            viewport.update()
            return

        # CPU-side projection: select ALL points inside polygon regardless of depth
        poly = np.array([[p.x(), p.y()] for p in self._polygon], dtype=np.float32)
        screen_xy, alive_idx = viewport.screen_project_alive()
        inside = _points_in_polygon(screen_xy, poly)
        indices = alive_idx[inside]

        pc = viewport.point_cloud
        add = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        pc.select_indices(indices, add=add)

        self._polygon = []
        viewport.on_selection_changed()
        viewport.update()

    def draw_overlay(self, painter: QPainter) -> None:
        if len(self._polygon) < 2:
            return
        pen = QPen(QColor(100, 200, 255), 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        for i in range(len(self._polygon) - 1):
            painter.drawLine(self._polygon[i], self._polygon[i + 1])
        # Close the lasso visually
        if len(self._polygon) >= 3:
            painter.drawLine(self._polygon[-1], self._polygon[0])
