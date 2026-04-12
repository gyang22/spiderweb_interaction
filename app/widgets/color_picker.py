"""A button that opens a QColorDialog and stores the chosen color."""

from PyQt6.QtWidgets import QPushButton, QColorDialog
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal


class ColorPickerButton(QPushButton):
    color_changed = pyqtSignal(QColor)

    def __init__(self, initial_color: QColor | None = None, parent=None):
        super().__init__(parent)
        self._color = initial_color or QColor(255, 165, 0)  # orange default
        self._update_swatch()
        self.clicked.connect(self._open_dialog)
        self.setToolTip("Click to choose highlight color")
        self.setFixedWidth(36)

    def _update_swatch(self) -> None:
        c = self._color
        self.setStyleSheet(
            f"background-color: rgb({c.red()},{c.green()},{c.blue()}); border: 1px solid #888;"
        )

    def _open_dialog(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Choose highlight color")
        if color.isValid():
            self._color = color
            self._update_swatch()
            self.color_changed.emit(color)

    @property
    def current_color(self) -> QColor:
        return self._color

    def rgba_float(self) -> tuple[float, float, float, float]:
        c = self._color
        return (c.redF(), c.greenF(), c.blueF(), c.alphaF())
