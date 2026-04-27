"""Skeleton Node Editor — dock panel for manual skeleton editing."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QGroupBox, QCheckBox,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal


class SkeletonEditorPanel(QDockWidget):
    edit_mode_changed      = pyqtSignal(bool)
    select_all_clicked     = pyqtSignal()
    deselect_all_clicked   = pyqtSignal()
    select_by_degree_clicked = pyqtSignal(int)
    reextract_clicked      = pyqtSignal()
    delete_nodes_clicked   = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__("Skeleton Node Editor", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.setMinimumWidth(220)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Mode toggle ───────────────────────────────────────────────────────
        self._chk_edit = QCheckBox("Edit skeleton nodes")
        self._chk_edit.setToolTip(
            "When enabled, selection tools act on skeleton nodes instead of the\n"
            "point cloud. Use click / box / lasso to select nodes."
        )
        self._chk_edit.toggled.connect(self.edit_mode_changed)
        layout.addWidget(self._chk_edit)

        self._lbl_stats = QLabel("No skeleton loaded")
        self._lbl_stats.setStyleSheet("color: #aaa; font-size: 11px; padding: 2px 0;")
        self._lbl_stats.setWordWrap(True)
        layout.addWidget(self._lbl_stats)

        # ── Selection controls ────────────────────────────────────────────────
        sel_group = QGroupBox("Selection")
        sel_layout = QHBoxLayout(sel_group)
        sel_layout.setSpacing(6)

        self._btn_sel_all = QPushButton("Select All")
        self._btn_sel_all.setFixedHeight(30)
        self._btn_sel_all.clicked.connect(self.select_all_clicked)
        sel_layout.addWidget(self._btn_sel_all)

        self._btn_desel = QPushButton("Deselect All")
        self._btn_desel.setFixedHeight(30)
        self._btn_desel.clicked.connect(self.deselect_all_clicked)
        sel_layout.addWidget(self._btn_desel)

        layout.addWidget(sel_group)

        # ── Select by degree ──────────────────────────────────────────────────
        deg_group = QGroupBox("Select by degree")
        deg_layout = QHBoxLayout(deg_group)
        deg_layout.setSpacing(6)

        deg_layout.addWidget(QLabel("Degree ="))
        self._spin_degree = QSpinBox()
        self._spin_degree.setRange(0, 999)
        self._spin_degree.setValue(1)
        self._spin_degree.setToolTip(
            "Select all skeleton nodes with exactly this many connected edges.\n"
            "Degree 1 = leaf/endpoint nodes, degree 0 = isolated nodes."
        )
        deg_layout.addWidget(self._spin_degree)

        self._btn_sel_degree = QPushButton("Select")
        self._btn_sel_degree.setFixedHeight(28)
        self._btn_sel_degree.setToolTip("Select all nodes whose edge count equals the degree value.")
        self._btn_sel_degree.clicked.connect(
            lambda: self.select_by_degree_clicked.emit(self._spin_degree.value())
        )
        deg_layout.addWidget(self._btn_sel_degree)

        layout.addWidget(deg_group)

        # ── Re-extract connectivity ───────────────────────────────────────────
        reex_group = QGroupBox("Re-extract connectivity")
        reex_layout = QVBoxLayout(reex_group)
        reex_layout.setSpacing(6)

        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("k neighbors:"))
        self._spin_k = QSpinBox()
        self._spin_k.setRange(2, 20)
        self._spin_k.setValue(4)
        self._spin_k.setToolTip(
            "Nearest-neighbor connections per node used to build the graph\n"
            "before MST extraction."
        )
        k_row.addWidget(self._spin_k)
        reex_layout.addLayout(k_row)

        self._btn_reextract = QPushButton("Re-extract Selected")
        self._btn_reextract.setFixedHeight(34)
        self._btn_reextract.setToolTip(
            "Run k-NN + MST on the selected skeleton nodes to rebuild their\n"
            "connectivity. Edges to non-selected nodes are preserved."
        )
        self._btn_reextract.clicked.connect(self.reextract_clicked)
        reex_layout.addWidget(self._btn_reextract)

        layout.addWidget(reex_group)

        # ── Delete ────────────────────────────────────────────────────────────
        self._btn_delete = QPushButton("Delete Selected Nodes")
        self._btn_delete.setFixedHeight(32)
        self._btn_delete.setStyleSheet("color: #ff9966;")
        self._btn_delete.setToolTip(
            "Remove the selected skeleton nodes and all edges connected to them."
        )
        self._btn_delete.clicked.connect(self.delete_nodes_clicked)
        layout.addWidget(self._btn_delete)

        # ── Degree distribution ───────────────────────────────────────────────
        stats_group = QGroupBox("Degree distribution")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(2)

        self._lbl_deg_stats = QLabel("—")
        self._lbl_deg_stats.setStyleSheet(
            "color: #aaa; font-size: 11px; font-family: monospace; padding: 2px 0;"
        )
        self._lbl_deg_stats.setWordWrap(True)
        stats_layout.addWidget(self._lbl_deg_stats)

        layout.addWidget(stats_group)

        layout.addStretch()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setWidget(scroll)

        self._set_controls_enabled(False)

    # ── public API ────────────────────────────────────────────────────────────

    def set_node_stats(self, selected: int, total: int) -> None:
        if total == 0:
            self._lbl_stats.setText("No skeleton loaded")
        else:
            self._lbl_stats.setText(f"Selected: {selected} / {total} nodes")

    def set_degree_stats(self, deg_counts: dict[int, int]) -> None:
        """Display degree → node count mapping."""
        if not deg_counts:
            self._lbl_deg_stats.setText("—")
            return
        lines = [f"deg {d}: {c} node{'s' if c != 1 else ''}"
                 for d, c in sorted(deg_counts.items())]
        self._lbl_deg_stats.setText("\n".join(lines))

    def get_k_neighbors(self) -> int:
        return self._spin_k.value()

    def set_edit_mode(self, active: bool) -> None:
        """Sync the checkbox without emitting the signal (for external resets)."""
        self._chk_edit.blockSignals(True)
        self._chk_edit.setChecked(active)
        self._chk_edit.blockSignals(False)
        self._set_controls_enabled(active)

    # ── private ───────────────────────────────────────────────────────────────

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (self._btn_sel_all, self._btn_desel, self._btn_sel_degree,
                  self._btn_reextract, self._btn_delete):
            w.setEnabled(enabled)
