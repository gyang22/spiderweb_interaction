"""Right-docked panel: point cloud downsampling + strand skeleton extraction."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QGroupBox, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class GraphPanel(QDockWidget):
    # Downsampling
    downsample_clicked = pyqtSignal()

    # Skeleton
    extract_clicked = pyqtSignal()
    export_clicked  = pyqtSignal()
    clear_clicked   = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__("Tools", parent)
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

        # ══ Downsample Cloud ══════════════════════════════════════════════════
        ds_group = QGroupBox("Downsample Cloud")
        ds_layout = QVBoxLayout(ds_group)
        ds_layout.setSpacing(6)

        self._chk_ds_auto = QCheckBox("Auto voxel size")
        self._chk_ds_auto.setChecked(True)
        self._chk_ds_auto.toggled.connect(self._on_ds_auto_toggled)
        ds_layout.addWidget(self._chk_ds_auto)

        ds_vox_row = QHBoxLayout()
        ds_vox_row.addWidget(QLabel("Voxel size:"))
        self._spin_ds_voxel = QDoubleSpinBox()
        self._spin_ds_voxel.setRange(1e-6, 9999.0)
        self._spin_ds_voxel.setDecimals(4)
        self._spin_ds_voxel.setValue(0.01)
        self._spin_ds_voxel.setEnabled(False)
        self._spin_ds_voxel.setToolTip(
            "Voxel edge length used for downsampling.\n"
            "All points inside a cell are collapsed to their centroid.\n"
            "Auto = extent / 100  (keeps ~100 voxels along the longest axis)."
        )
        ds_vox_row.addWidget(self._spin_ds_voxel)
        ds_layout.addLayout(ds_vox_row)

        self._btn_downsample = QPushButton("Downsample")
        self._btn_downsample.setFixedHeight(34)
        self._btn_downsample.setToolTip(
            "Replace the current point cloud with a voxel-downsampled version.\n"
            "This cannot be undone — save a copy first if needed."
        )
        self._btn_downsample.clicked.connect(self.downsample_clicked)
        ds_layout.addWidget(self._btn_downsample)

        self._lbl_ds_stats = QLabel("")
        self._lbl_ds_stats.setStyleSheet("color: #aaa; font-size: 11px; padding: 2px 0;")
        self._lbl_ds_stats.setWordWrap(True)
        ds_layout.addWidget(self._lbl_ds_stats)

        layout.addWidget(ds_group)

        # ══ Strand Skeleton ═══════════════════════════════════════════════════
        skel_group = QGroupBox("Strand Skeleton")
        skel_layout = QVBoxLayout(skel_group)
        skel_layout.setSpacing(6)

        self._chk_auto_voxel = QCheckBox("Auto voxel size")
        self._chk_auto_voxel.setChecked(True)
        self._chk_auto_voxel.toggled.connect(self._on_skel_auto_toggled)
        skel_layout.addWidget(self._chk_auto_voxel)

        vox_row = QHBoxLayout()
        vox_row.addWidget(QLabel("Voxel size:"))
        self._spin_voxel = QDoubleSpinBox()
        self._spin_voxel.setRange(1e-4, 9999.0)
        self._spin_voxel.setDecimals(4)
        self._spin_voxel.setValue(0.01)
        self._spin_voxel.setEnabled(False)
        self._spin_voxel.setToolTip(
            "Grid cell size for skeleton downsampling.\n"
            "Smaller = denser nodes; larger = sparser skeleton.\n"
            "Auto = selection extent / 20."
        )
        vox_row.addWidget(self._spin_voxel)
        skel_layout.addLayout(vox_row)

        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("k neighbors:"))
        self._spin_k = QSpinBox()
        self._spin_k.setRange(2, 20)
        self._spin_k.setValue(4)
        self._spin_k.setToolTip(
            "Nearest-neighbor connections per node before MST extraction."
        )
        k_row.addWidget(self._spin_k)
        skel_layout.addLayout(k_row)

        prune_row = QHBoxLayout()
        prune_row.addWidget(QLabel("Prune factor:"))
        self._spin_prune = QDoubleSpinBox()
        self._spin_prune.setRange(0.0, 10.0)
        self._spin_prune.setDecimals(2)
        self._spin_prune.setValue(0.5)
        self._spin_prune.setToolTip(
            "Remove leaf edges shorter than factor × median edge length.\n"
            "0.5 = only prune clear noise stubs (safe default).\n"
            "0 = no pruning."
        )
        prune_row.addWidget(self._spin_prune)
        skel_layout.addLayout(prune_row)

        self._btn_extract = QPushButton("Extract Skeleton")
        self._btn_extract.setFixedHeight(34)
        self._btn_extract.setToolTip(
            "Compute backbone graph from the current point selection.\n"
            "Results accumulate — use Clear to start over."
        )
        self._btn_extract.clicked.connect(self.extract_clicked)
        skel_layout.addWidget(self._btn_extract)

        self._lbl_stats = QLabel("No skeleton extracted")
        self._lbl_stats.setStyleSheet("color: #aaa; font-size: 11px; padding: 2px 0;")
        self._lbl_stats.setWordWrap(True)
        skel_layout.addWidget(self._lbl_stats)

        self._btn_export = QPushButton("Export JSON…")
        self._btn_export.setFixedHeight(32)
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip("Save accumulated skeleton graph as JSON.")
        self._btn_export.clicked.connect(self.export_clicked)
        skel_layout.addWidget(self._btn_export)

        self._btn_clear = QPushButton("Clear Skeleton")
        self._btn_clear.setFixedHeight(32)
        self._btn_clear.setEnabled(False)
        self._btn_clear.setStyleSheet("color: #ff9966;")
        self._btn_clear.setToolTip("Remove the skeleton overlay and reset accumulated graph.")
        self._btn_clear.clicked.connect(self.clear_clicked)
        skel_layout.addWidget(self._btn_clear)

        layout.addWidget(skel_group)
        layout.addStretch()
        self.setWidget(root)

    # ── Downsample public API ─────────────────────────────────────────────────

    def get_ds_voxel_size(self) -> float | None:
        """None = auto (extent/100), otherwise the spinbox value."""
        return None if self._chk_ds_auto.isChecked() else self._spin_ds_voxel.value()

    def set_ds_stats(self, n_before: int, n_after: int) -> None:
        self._lbl_ds_stats.setText(f"{n_before:,} → {n_after:,} points")

    def clear_ds_stats(self) -> None:
        self._lbl_ds_stats.setText("")

    # ── Skeleton public API ───────────────────────────────────────────────────

    def set_stats(self, n_nodes: int, n_edges: int) -> None:
        self._lbl_stats.setText(f"Nodes: {n_nodes}   Edges: {n_edges}")
        self._btn_export.setEnabled(True)
        self._btn_clear.setEnabled(True)

    def clear_stats(self) -> None:
        self._lbl_stats.setText("No skeleton extracted")
        self._btn_export.setEnabled(False)
        self._btn_clear.setEnabled(False)

    def get_voxel_size(self) -> float | None:
        return None if self._chk_auto_voxel.isChecked() else self._spin_voxel.value()

    def get_k_neighbors(self) -> int:
        return self._spin_k.value()

    def get_prune_factor(self) -> float:
        return self._spin_prune.value()

    # ── private ───────────────────────────────────────────────────────────────

    def _on_ds_auto_toggled(self, checked: bool) -> None:
        self._spin_ds_voxel.setEnabled(not checked)

    def _on_skel_auto_toggled(self, checked: bool) -> None:
        self._spin_voxel.setEnabled(not checked)
