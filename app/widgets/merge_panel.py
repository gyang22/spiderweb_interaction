"""
Right-docked panel: secondary PCD loading, alignment, and merge controls.

Workflow
────────
1. Load Secondary PCD   → blue overlay appears in viewport
2. (Optional) adjust 6-DOF spinboxes for coarse manual alignment
3. Run ICP              → rigid alignment; spinboxes reset, transform applied
4. Run CPD              → non-rigid warp; replaces secondary cloud geometry
5. Merge                → concatenates (transformed) secondary into primary
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class MergePanel(QDockWidget):
    # ── signals ───────────────────────────────────────────────────────────────
    load_secondary_clicked  = pyqtSignal()
    clear_secondary_clicked = pyqtSignal()
    run_icp_clicked         = pyqtSignal()
    run_cpd_clicked         = pyqtSignal()
    merge_clicked           = pyqtSignal()

    # Fired whenever any manual-transform spinbox changes.
    # Args: tx, ty, tz (translation), yaw, pitch, roll (degrees)
    transform_changed = pyqtSignal(float, float, float, float, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__("Merge / Align", parent)
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

        # ══ Secondary PCD ═════════════════════════════════════════════════════
        sec_group = QGroupBox("Secondary PCD")
        sec_layout = QVBoxLayout(sec_group)
        sec_layout.setSpacing(6)

        self._btn_load = QPushButton("Load Secondary PCD…")
        self._btn_load.setFixedHeight(34)
        self._btn_load.clicked.connect(self.load_secondary_clicked)
        sec_layout.addWidget(self._btn_load)

        self._lbl_sec_status = QLabel("No secondary cloud loaded")
        self._lbl_sec_status.setWordWrap(True)
        self._lbl_sec_status.setStyleSheet("color: #aaa; font-size: 11px;")
        sec_layout.addWidget(self._lbl_sec_status)

        self._btn_clear_sec = QPushButton("Clear Secondary")
        self._btn_clear_sec.setFixedHeight(28)
        self._btn_clear_sec.setEnabled(False)
        self._btn_clear_sec.setStyleSheet("color: #ff9966;")
        self._btn_clear_sec.clicked.connect(self.clear_secondary_clicked)
        sec_layout.addWidget(self._btn_clear_sec)

        layout.addWidget(sec_group)

        # ══ Manual Transform ══════════════════════════════════════════════════
        xform_group = QGroupBox("Manual Adjustment")
        xform_layout = QVBoxLayout(xform_group)
        xform_layout.setSpacing(4)

        self._spins: dict[str, QDoubleSpinBox] = {}
        for label, key, lo, hi, step, dec in [
            ("TX:",    "tx",    -9999.0, 9999.0, 0.001, 4),
            ("TY:",    "ty",    -9999.0, 9999.0, 0.001, 4),
            ("TZ:",    "tz",    -9999.0, 9999.0, 0.001, 4),
            ("Yaw °:", "yaw",   -180.0,  180.0,  1.0,   2),
            ("Pitch°:","pitch", -90.0,   90.0,   1.0,   2),
            ("Roll °:","roll",  -180.0,  180.0,  1.0,   2),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            sp.setValue(0.0)
            sp.setEnabled(False)
            sp.valueChanged.connect(self._on_transform_spinbox_changed)
            row.addWidget(sp)
            self._spins[key] = sp
            xform_layout.addLayout(row)

        self._btn_reset_xform = QPushButton("Reset Transform")
        self._btn_reset_xform.setFixedHeight(28)
        self._btn_reset_xform.setEnabled(False)
        self._btn_reset_xform.clicked.connect(self._reset_transform)
        xform_layout.addWidget(self._btn_reset_xform)

        layout.addWidget(xform_group)

        # ══ ICP (Rigid) ════════════════════════════════════════════════════════
        icp_group = QGroupBox("ICP  (Rigid)")
        icp_layout = QVBoxLayout(icp_group)
        icp_layout.setSpacing(4)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max iter:"))
        self._spin_icp_iter = QSpinBox()
        self._spin_icp_iter.setRange(1, 500)
        self._spin_icp_iter.setValue(50)
        self._spin_icp_iter.setEnabled(False)
        iter_row.addWidget(self._spin_icp_iter)
        icp_layout.addLayout(iter_row)

        self._btn_icp = QPushButton("Run ICP")
        self._btn_icp.setFixedHeight(32)
        self._btn_icp.setEnabled(False)
        self._btn_icp.setToolTip(
            "Iterative Closest Point — rigid alignment.\n"
            "Good for initial coarse registration."
        )
        self._btn_icp.clicked.connect(self.run_icp_clicked)
        icp_layout.addWidget(self._btn_icp)

        self._lbl_icp_result = QLabel("")
        self._lbl_icp_result.setStyleSheet("color: #aaa; font-size: 11px;")
        self._lbl_icp_result.setWordWrap(True)
        icp_layout.addWidget(self._lbl_icp_result)

        layout.addWidget(icp_group)

        # ══ CPD (Non-rigid) ═══════════════════════════════════════════════════
        cpd_group = QGroupBox("CPD  (Non-rigid)")
        cpd_layout = QVBoxLayout(cpd_group)
        cpd_layout.setSpacing(4)

        beta_row = QHBoxLayout()
        beta_row.addWidget(QLabel("β schedule:"))
        beta_row.addWidget(QLabel("60→15→5→2"), stretch=1)
        cpd_layout.addLayout(beta_row)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Alpha:"))
        self._spin_cpd_alpha = QDoubleSpinBox()
        self._spin_cpd_alpha.setRange(0.001, 10.0)
        self._spin_cpd_alpha.setSingleStep(0.01)
        self._spin_cpd_alpha.setDecimals(3)
        self._spin_cpd_alpha.setValue(0.1)
        self._spin_cpd_alpha.setEnabled(False)
        self._spin_cpd_alpha.setToolTip(
            "CPD regularisation weight.\n"
            "Higher = smoother / more constrained warp."
        )
        alpha_row.addWidget(self._spin_cpd_alpha)
        cpd_layout.addLayout(alpha_row)

        self._btn_cpd = QPushButton("Run CPD  (Non-rigid)")
        self._btn_cpd.setFixedHeight(32)
        self._btn_cpd.setEnabled(False)
        self._btn_cpd.setToolTip(
            "Coherent Point Drift — deformable registration.\n"
            "Run after ICP for a fine non-rigid warp.\n"
            "Replaces the secondary cloud with the warped result."
        )
        self._btn_cpd.clicked.connect(self.run_cpd_clicked)
        cpd_layout.addWidget(self._btn_cpd)

        self._lbl_cpd_status = QLabel("")
        self._lbl_cpd_status.setStyleSheet("color: #aaa; font-size: 11px;")
        self._lbl_cpd_status.setWordWrap(True)
        cpd_layout.addWidget(self._lbl_cpd_status)

        layout.addWidget(cpd_group)

        # ══ Merge ══════════════════════════════════════════════════════════════
        self._btn_merge = QPushButton("Merge into Primary")
        self._btn_merge.setFixedHeight(36)
        self._btn_merge.setEnabled(False)
        self._btn_merge.setToolTip(
            "Apply the current transform to the secondary cloud\n"
            "and concatenate its points into the primary."
        )
        self._btn_merge.clicked.connect(self.merge_clicked)
        layout.addWidget(self._btn_merge)

        layout.addStretch()
        self.setWidget(root)

    # ── public API ────────────────────────────────────────────────────────────

    def set_secondary_loaded(self, name: str, n_points: int) -> None:
        """Call after a secondary cloud is successfully loaded."""
        self._lbl_sec_status.setText(f"{name}\n{n_points:,} points")
        self._lbl_sec_status.setStyleSheet("color: #88cc88; font-size: 11px;")
        self._set_secondary_controls_enabled(True)

    def clear_secondary_status(self) -> None:
        self._lbl_sec_status.setText("No secondary cloud loaded")
        self._lbl_sec_status.setStyleSheet("color: #aaa; font-size: 11px;")
        self._lbl_icp_result.setText("")
        self._lbl_cpd_status.setText("")
        self._set_secondary_controls_enabled(False)
        self._reset_transform_silent()

    def set_icp_result(self, rmse: float, n_inliers: int) -> None:
        self._lbl_icp_result.setText(f"RMSE: {rmse:.5f}   Inliers: {n_inliers:,}")

    def set_cpd_status(self, msg: str) -> None:
        self._lbl_cpd_status.setText(msg)

    def reset_transform_spinboxes(self) -> None:
        """Reset all transform spinboxes to zero without emitting transform_changed."""
        self._reset_transform_silent()

    def get_icp_max_iter(self) -> int:
        return self._spin_icp_iter.value()

    def get_cpd_alpha(self) -> float:
        return self._spin_cpd_alpha.value()

    # ── private ───────────────────────────────────────────────────────────────

    def _set_secondary_controls_enabled(self, enabled: bool) -> None:
        self._btn_clear_sec.setEnabled(enabled)
        self._btn_icp.setEnabled(enabled)
        self._btn_cpd.setEnabled(enabled)
        self._btn_merge.setEnabled(enabled)
        self._btn_reset_xform.setEnabled(enabled)
        self._spin_icp_iter.setEnabled(enabled)
        self._spin_cpd_alpha.setEnabled(enabled)
        for sp in self._spins.values():
            sp.setEnabled(enabled)

    def _reset_transform(self) -> None:
        self._reset_transform_silent()
        self._emit_transform()

    def _reset_transform_silent(self) -> None:
        """Zero all spinboxes without firing transform_changed."""
        for sp in self._spins.values():
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)

    def _on_transform_spinbox_changed(self) -> None:
        self._emit_transform()

    def _emit_transform(self) -> None:
        self.transform_changed.emit(
            self._spins["tx"].value(),
            self._spins["ty"].value(),
            self._spins["tz"].value(),
            self._spins["yaw"].value(),
            self._spins["pitch"].value(),
            self._spins["roll"].value(),
        )
