"""Left-docked toolbar with tool buttons, point size control, and color actions."""

from PyQt6.QtWidgets import (
    QToolBar, QLabel, QSlider, QSpinBox, QPushButton,
    QWidget, QHBoxLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from app.widgets.color_picker import ColorPickerButton
from app import settings


class ToolBar(QToolBar):
    tool_selected       = pyqtSignal(str)   # 'click', 'box', 'lasso'
    point_size_changed  = pyqtSignal(int)
    apply_color_clicked = pyqtSignal()
    delete_clicked      = pyqtSignal()
    reset_camera_clicked = pyqtSignal()
    open_pcd_clicked    = pyqtSignal()
    save_pcd_clicked    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Tools", parent)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setMovable(False)
        self.setMinimumWidth(160)

        # ── Navigation ───────────────────────────────────────────────────────
        self._btn_reset = QPushButton("⌂  Reset Camera")
        self._btn_reset.setToolTip("Return camera to the starting position (Home)")
        self._btn_reset.setFixedHeight(36)
        self._btn_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_reset.clicked.connect(self.reset_camera_clicked)
        self.addWidget(self._btn_reset)

        self._btn_open = QPushButton("📂  Open PCD…")
        self._btn_open.setToolTip("Browse and load a new point cloud")
        self._btn_open.setFixedHeight(36)
        self._btn_open.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_open.clicked.connect(self.open_pcd_clicked)
        self.addWidget(self._btn_open)

        self._btn_save = QPushButton("💾  Save PCD…")
        self._btn_save.setToolTip("Save current point cloud (Ctrl+S)")
        self._btn_save.setFixedHeight(36)
        self._btn_save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_save.clicked.connect(self.save_pcd_clicked)
        self.addWidget(self._btn_save)

        self.addSeparator()

        # ── Selection tools ───────────────────────────────────────────────────
        self.addWidget(_label("Selection Tool"))
        self._btn_click = self._add_tool_button("✦  Click",  'click', checkable=True)
        self._btn_box   = self._add_tool_button("▭  Box",    'box',   checkable=True)
        self._btn_lasso = self._add_tool_button("⌒  Lasso",  'lasso', checkable=True)
        self._tool_btns = [self._btn_click, self._btn_box, self._btn_lasso]
        self._btn_lasso.setChecked(True)

        self.addSeparator()

        # ── Point size ────────────────────────────────────────────────────────
        self.addWidget(_label("Point Size"))

        # +/- quick buttons on one row
        ps_row = QWidget()
        ps_layout = QHBoxLayout(ps_row)
        ps_layout.setContentsMargins(4, 0, 4, 0)
        ps_layout.setSpacing(4)

        self._btn_ps_minus = QPushButton("−")
        self._btn_ps_minus.setFixedWidth(28)
        self._btn_ps_minus.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_ps_minus.clicked.connect(self._decrease_point_size)

        self._spinbox = QSpinBox()
        self._spinbox.setRange(settings.MIN_POINT_SIZE, settings.MAX_POINT_SIZE)
        self._spinbox.setValue(int(settings.DEFAULT_POINT_SIZE))
        self._spinbox.valueChanged.connect(self.point_size_changed)

        self._btn_ps_plus = QPushButton("+")
        self._btn_ps_plus.setFixedWidth(28)
        self._btn_ps_plus.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_ps_plus.clicked.connect(self._increase_point_size)

        ps_layout.addWidget(self._btn_ps_minus)
        ps_layout.addWidget(self._spinbox, stretch=1)
        ps_layout.addWidget(self._btn_ps_plus)
        self.addWidget(ps_row)

        # Slider below
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(settings.MIN_POINT_SIZE, settings.MAX_POINT_SIZE)
        self._slider.setValue(int(settings.DEFAULT_POINT_SIZE))
        self._slider.setContentsMargins(6, 0, 6, 0)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.valueChanged.connect(self._spinbox.setValue)
        self._spinbox.valueChanged.connect(self._slider.setValue)
        self.addWidget(self._slider)

        self.addSeparator()

        # ── Color picker + apply ──────────────────────────────────────────────
        self.addWidget(_label("Highlight Color"))
        color_row = QWidget()
        cr_layout = QHBoxLayout(color_row)
        cr_layout.setContentsMargins(4, 0, 4, 0)
        cr_layout.setSpacing(4)

        self.color_picker = ColorPickerButton()
        self._btn_apply_color = QPushButton("Apply")
        self._btn_apply_color.setToolTip("Apply color to selected points")
        self._btn_apply_color.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_apply_color.clicked.connect(self.apply_color_clicked)

        cr_layout.addWidget(self.color_picker)
        cr_layout.addWidget(self._btn_apply_color, stretch=1)
        self.addWidget(color_row)

        self.addSeparator()

        # ── Delete ────────────────────────────────────────────────────────────
        self._btn_delete = QPushButton("🗑  Delete Selected")
        self._btn_delete.setToolTip("Delete selected points (Del)")
        self._btn_delete.setFixedHeight(36)
        self._btn_delete.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_delete.clicked.connect(self.delete_clicked)
        self._btn_delete.setStyleSheet("color: #ff6666; font-weight: bold;")
        self.addWidget(self._btn_delete)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _add_tool_button(self, label: str, name: str, checkable: bool = False) -> QPushButton:
        btn = QPushButton(label)
        btn.setCheckable(checkable)
        btn.setFixedHeight(34)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if checkable:
            btn.clicked.connect(lambda checked, n=name: self._on_tool_clicked(n))
        self.addWidget(btn)
        return btn

    def _on_tool_clicked(self, name: str) -> None:
        for btn in self._tool_btns:
            btn.setChecked(False)
        mapping = {'click': self._btn_click, 'box': self._btn_box, 'lasso': self._btn_lasso}
        mapping[name].setChecked(True)
        self.tool_selected.emit(name)

    def _increase_point_size(self) -> None:
        self._spinbox.setValue(min(self._spinbox.value() + 1, settings.MAX_POINT_SIZE))

    def _decrease_point_size(self) -> None:
        self._spinbox.setValue(max(self._spinbox.value() - 1, settings.MIN_POINT_SIZE))

    def set_active_tool(self, name: str) -> None:
        mapping = {'click': self._btn_click, 'box': self._btn_box, 'lasso': self._btn_lasso}
        for n, btn in mapping.items():
            btn.setChecked(n == name)


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #aaa; font-size: 10px; padding: 2px 6px 0 6px;")
    return lbl
