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
    QGroupBox, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal


class MergePanel(QDockWidget):
    # ── signals ───────────────────────────────────────────────────────────────
    load_secondary_clicked  = pyqtSignal()
    clear_secondary_clicked = pyqtSignal()
    switch_active_clicked   = pyqtSignal()   # toggle which cloud is being edited
    run_icp_clicked         = pyqtSignal()
    run_cpd_clicked         = pyqtSignal()
    run_webmerge_clicked    = pyqtSignal()
    anchor_mode_toggled     = pyqtSignal(bool)
    apply_warp_clicked      = pyqtSignal()
    auto_match_clicked      = pyqtSignal()
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

        # Active-cloud indicator + switch button
        self._lbl_editing = QLabel("")
        self._lbl_editing.setStyleSheet("font-size: 11px; font-weight: bold;")
        sec_layout.addWidget(self._lbl_editing)

        self._btn_switch = QPushButton("")
        self._btn_switch.setFixedHeight(28)
        self._btn_switch.setEnabled(False)
        self._btn_switch.setToolTip(
            "Switch which cloud is active for editing.\n"
            "The inactive cloud is shown as a dim reference overlay."
        )
        self._btn_switch.clicked.connect(self.switch_active_clicked)
        sec_layout.addWidget(self._btn_switch)

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

        # ══ WebMerge Pipeline ══════════════════════════════════════════════════
        webmerge_group = QGroupBox("WebMerge Pipeline")
        webmerge_layout = QVBoxLayout(webmerge_group)
        webmerge_layout.setSpacing(4)
        
        self._spin_wm_radius = QDoubleSpinBox()
        self._spin_wm_radius.setRange(0.1, 1000.0)
        self._spin_wm_radius.setValue(20.0)
        self._spin_wm_radius.setSingleStep(1.0)
        self._spin_wm_radius.setDecimals(1)
        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("Search Rad:"))
        r_layout.addWidget(self._spin_wm_radius)
        webmerge_layout.addLayout(r_layout)
        
        self._spin_wm_vote = QSpinBox()
        self._spin_wm_vote.setRange(1, 50)
        self._spin_wm_vote.setValue(5)
        v_layout = QHBoxLayout()
        v_layout.addWidget(QLabel("Vote Steps:"))
        v_layout.addWidget(self._spin_wm_vote)
        webmerge_layout.addLayout(v_layout)
        
        self._spin_wm_step = QDoubleSpinBox()
        self._spin_wm_step.setRange(0.1, 100.0)
        self._spin_wm_step.setValue(2.5)
        self._spin_wm_step.setSingleStep(0.5)
        self._spin_wm_step.setDecimals(1)
        s_layout = QHBoxLayout()
        s_layout.addWidget(QLabel("Step Size:"))
        s_layout.addWidget(self._spin_wm_step)
        webmerge_layout.addLayout(s_layout)
        
        self._spin_wm_lam = QDoubleSpinBox()
        self._spin_wm_lam.setRange(0.01, 1.0)
        self._spin_wm_lam.setValue(0.4)
        self._spin_wm_lam.setSingleStep(0.1)
        l_layout = QHBoxLayout()
        l_layout.addWidget(QLabel("Lap. Lam:"))
        l_layout.addWidget(self._spin_wm_lam)
        webmerge_layout.addLayout(l_layout)
        
        self._spin_wm_iter = QSpinBox()
        self._spin_wm_iter.setRange(1, 200)
        self._spin_wm_iter.setValue(30)
        i_layout = QHBoxLayout()
        i_layout.addWidget(QLabel("Lap. Iter:"))
        i_layout.addWidget(self._spin_wm_iter)
        webmerge_layout.addLayout(i_layout)
        
        self._btn_webmerge = QPushButton("Run WebMerge Pipeline")
        self._btn_webmerge.setFixedHeight(32)
        self._btn_webmerge.setEnabled(False)
        self._btn_webmerge.setToolTip("Extract skeletons from both clouds and align them.")
        self._btn_webmerge.clicked.connect(self.run_webmerge_clicked)
        webmerge_layout.addWidget(self._btn_webmerge)
        
        self._lbl_webmerge_status = QLabel("")
        self._lbl_webmerge_status.setStyleSheet("color: #aaa; font-size: 11px;")
        self._lbl_webmerge_status.setWordWrap(True)
        webmerge_layout.addWidget(self._lbl_webmerge_status)
        
        layout.addWidget(webmerge_group)

        # ══ Manual Anchor Warp ════════════════════════════════════════════════
        manual_group = QGroupBox("Manual Anchor Warp")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.setSpacing(4)
        
        self._btn_anchor_mode = QPushButton("Enter Anchor Mode")
        self._btn_anchor_mode.setCheckable(True)
        self._btn_anchor_mode.setFixedHeight(32)
        self._btn_anchor_mode.setEnabled(False)
        self._btn_anchor_mode.setToolTip("Extract key anchors and enter manual pairing mode.")
        self._btn_anchor_mode.toggled.connect(self.anchor_mode_toggled)
        manual_layout.addWidget(self._btn_anchor_mode)
        
        self._lbl_anchor_status = QLabel("0 anchors paired")
        self._lbl_anchor_status.setStyleSheet("color: #aaa; font-size: 11px;")
        manual_layout.addWidget(self._lbl_anchor_status)
        
        self._btn_auto_match = QPushButton("Auto-Match Selected Regions")
        self._btn_auto_match.setFixedHeight(32)
        self._btn_auto_match.setEnabled(False)
        self._btn_auto_match.setToolTip("Finds the best FPFH match between your selections on the Primary and Secondary clouds.")
        self._btn_auto_match.clicked.connect(self.auto_match_clicked)
        manual_layout.addWidget(self._btn_auto_match)
        
        self._btn_apply_warp = QPushButton("Apply TPS Warp")
        self._btn_apply_warp.setFixedHeight(32)
        self._btn_apply_warp.setEnabled(False)
        self._btn_apply_warp.setToolTip("Non-rigidly stretch the secondary cloud to match the paired anchors.")
        self._btn_apply_warp.clicked.connect(self.apply_warp_clicked)
        manual_layout.addWidget(self._btn_apply_warp)
        
        layout.addWidget(manual_group)

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
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setWidget(scroll)

    # ── public API ────────────────────────────────────────────────────────────

    def set_secondary_loaded(self, name: str, n_points: int) -> None:
        """Call after a secondary cloud is successfully loaded."""
        self._lbl_sec_status.setText(f"{name}\n{n_points:,} points")
        self._lbl_sec_status.setStyleSheet("color: #88cc88; font-size: 11px;")
        self._set_secondary_controls_enabled(True)

    def set_editing_state(self, editing_secondary: bool) -> None:
        """Update the indicator showing which cloud is currently active."""
        if editing_secondary:
            self._lbl_editing.setText("Editing: Secondary")
            self._lbl_editing.setStyleSheet(
                "font-size: 11px; font-weight: bold; color: #88ccff;")
            self._btn_switch.setText("Switch to Primary")
        else:
            self._lbl_editing.setText("Editing: Primary")
            self._lbl_editing.setStyleSheet(
                "font-size: 11px; font-weight: bold; color: #ffcc88;")
            self._btn_switch.setText("Switch to Secondary")
        self._btn_switch.setEnabled(True)

    def clear_secondary_status(self) -> None:
        self._lbl_sec_status.setText("No secondary cloud loaded")
        self._lbl_sec_status.setStyleSheet("color: #aaa; font-size: 11px;")
        self._lbl_editing.setText("")
        self._lbl_icp_result.setText("")
        self._lbl_cpd_status.setText("")
        self._lbl_webmerge_status.setText("")
        self._lbl_anchor_status.setText("0 anchors paired")
        if self._btn_anchor_mode.isChecked():
            self._btn_anchor_mode.setChecked(False)
        self._set_secondary_controls_enabled(False)
        self._reset_transform_silent()

    def set_icp_result(self, rmse: float, n_inliers: int) -> None:
        self._lbl_icp_result.setText(f"RMSE: {rmse:.5f}   Inliers: {n_inliers:,}")

    def set_cpd_status(self, msg: str) -> None:
        self._lbl_cpd_status.setText(msg)

    def set_webmerge_status(self, msg: str) -> None:
        self._lbl_webmerge_status.setText(msg)

    def set_anchor_status(self, pairs: int) -> None:
        self._lbl_anchor_status.setText(f"{pairs} anchors paired")
        self._btn_apply_warp.setEnabled(pairs > 0)

    def reset_transform_spinboxes(self) -> None:
        """Reset all transform spinboxes to zero without emitting transform_changed."""
        self._reset_transform_silent()

    def get_icp_max_iter(self) -> int:
        return self._spin_icp_iter.value()

    def get_cpd_alpha(self) -> float:
        return self._spin_cpd_alpha.value()

    def get_webmerge_params(self) -> dict:
        return {
            'search_radius': self._spin_wm_radius.value(),
            'vote_steps': self._spin_wm_vote.value(),
            'step_size': self._spin_wm_step.value(),
            'lam': self._spin_wm_lam.value(),
            'iterations': self._spin_wm_iter.value(),
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _set_secondary_controls_enabled(self, enabled: bool) -> None:
        self._btn_clear_sec.setEnabled(enabled)
        self._btn_switch.setEnabled(enabled)
        self._btn_icp.setEnabled(enabled)
        self._btn_cpd.setEnabled(enabled)
        self._btn_webmerge.setEnabled(enabled)
        self._btn_anchor_mode.setEnabled(enabled)
        self._btn_auto_match.setEnabled(enabled)
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
