"""Status bar with live point count, selection count, and FPS display."""

from PyQt6.QtWidgets import QStatusBar, QLabel


class StatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lbl_mode     = QLabel("")
        self._lbl_points   = QLabel("Points: —")
        self._lbl_selected = QLabel("Selected: —")
        self._lbl_fps      = QLabel("FPS: —")

        self.addWidget(self._lbl_mode)
        self.addWidget(self._lbl_points)
        self.addWidget(self._lbl_selected)
        self.addPermanentWidget(self._lbl_fps)

        for lbl in (self._lbl_mode, self._lbl_points, self._lbl_selected, self._lbl_fps):
            lbl.setStyleSheet("padding: 0 8px;")

    def update_point_cloud(self, pc) -> None:
        self._lbl_points.setText(f"Points: {pc.alive_count:,} / {pc.total_count:,}")
        self._lbl_selected.setText(f"Selected: {pc.selected_count:,}")

    def update_selection(self, pc) -> None:
        self._lbl_points.setText(f"Points: {pc.alive_count:,} / {pc.total_count:,}")
        self._lbl_selected.setText(f"Selected: {pc.selected_count:,}")

    def update_fps(self, fps: float) -> None:
        self._lbl_fps.setText(f"FPS: {fps:.0f}")

    def set_fps_mode(self, active: bool) -> None:
        if active:
            self._lbl_mode.setText("🔒 FPS MODE  (Tab to release mouse)")
            self._lbl_mode.setStyleSheet("padding: 0 8px; color: #ffaa00; font-weight: bold;")
        else:
            self._lbl_mode.setText("")
            self._lbl_mode.setStyleSheet("padding: 0 8px;")

    def clear(self) -> None:
        self._lbl_points.setText("Points: —")
        self._lbl_selected.setText("Selected: —")
        self._lbl_fps.setText("FPS: —")
