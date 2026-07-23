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
    remove_edge_clicked    = pyqtSignal()
    delete_nodes_clicked   = pyqtSignal()
    smooth_clicked         = pyqtSignal(float)
    simplify_chains_clicked    = pyqtSignal()
    prune_leaves_clicked       = pyqtSignal()
    collapse_triangles_clicked = pyqtSignal(float)
    grow_rays_clicked          = pyqtSignal(float)
    beam_latch_clicked         = pyqtSignal(float)
    run_pipeline_clicked       = pyqtSignal(float, float, float)  # tol, beam_r, tri

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

        self._btn_remove_edge = QPushButton("Remove Edge(s) Between Selected")
        self._btn_remove_edge.setFixedHeight(30)
        self._btn_remove_edge.setToolTip(
            "Delete edges whose BOTH endpoints are selected. Select two nodes\n"
            "to cut the edge between them (inverse of Re-extract); select a\n"
            "cluster to remove all edges inside it. Nodes are kept."
        )
        self._btn_remove_edge.clicked.connect(self.remove_edge_clicked)
        reex_layout.addWidget(self._btn_remove_edge)

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

        # ── Smooth ────────────────────────────────────────────────────────────
        smooth_group = QGroupBox("Smooth")
        smooth_layout = QHBoxLayout(smooth_group)
        smooth_layout.setSpacing(6)

        smooth_layout.addWidget(QLabel("Max Dev:"))
        self._spin_dev = QDoubleSpinBox()
        self._spin_dev.setRange(0.0, 10000.0)
        self._spin_dev.setDecimals(2)
        self._spin_dev.setSingleStep(1.0)
        self._spin_dev.setValue(5.0)
        self._spin_dev.setToolTip(
            "Maximum allowed distance a selected node can deviate from the straight line\n"
            "between its neighbors to be smoothed out."
        )
        smooth_layout.addWidget(self._spin_dev)

        self._btn_smooth = QPushButton("Smooth Selected")
        self._btn_smooth.setFixedHeight(28)
        self._btn_smooth.setToolTip("Smooth out selected chains of nodes if they form straight lines.")
        self._btn_smooth.clicked.connect(
            lambda: self.smooth_clicked.emit(self._spin_dev.value())
        )
        smooth_layout.addWidget(self._btn_smooth)

        layout.addWidget(smooth_group)

        # ── Topology refinement ───────────────────────────────────────────────
        # Global operations (act on the whole skeleton, ignoring the selection).
        refine_group = QGroupBox("Topology Refinement")
        refine_layout = QVBoxLayout(refine_group)
        refine_layout.setSpacing(4)

        self._btn_simplify = QPushButton("Simplify Chains")
        self._btn_simplify.setFixedHeight(28)
        self._btn_simplify.setToolTip(
            "Collapse chains of degree-2 nodes into single direct edges between\n"
            "junctions/endpoints. Acts on the whole skeleton.")
        self._btn_simplify.clicked.connect(self.simplify_chains_clicked)
        refine_layout.addWidget(self._btn_simplify)

        self._btn_prune = QPushButton("Prune Leaves (deg-1)")
        self._btn_prune.setFixedHeight(28)
        self._btn_prune.setToolTip(
            "Remove all degree-1 'hair' edges in a single pass and drop the\n"
            "orphaned nodes. Acts on the whole skeleton.")
        self._btn_prune.clicked.connect(self.prune_leaves_clicked)
        refine_layout.addWidget(self._btn_prune)

        tri_row = QHBoxLayout()
        tri_row.addWidget(QLabel("Tri size:"))
        self._spin_tri = QDoubleSpinBox()
        self._spin_tri.setRange(0.0, 10000.0)
        self._spin_tri.setDecimals(2)
        self._spin_tri.setSingleStep(1.0)
        self._spin_tri.setValue(5.0)
        self._spin_tri.setToolTip(
            "Max edge length (point-coordinate units) for a 3-clique to be\n"
            "collapsed into its centroid.")
        tri_row.addWidget(self._spin_tri)
        self._btn_collapse = QPushButton("Collapse Triangles")
        self._btn_collapse.setFixedHeight(28)
        self._btn_collapse.setToolTip(
            "Merge small triangles (all edges <= 'Tri size') into a single node\n"
            "at their centroid to remove jittery noise.")
        self._btn_collapse.clicked.connect(
            lambda: self.collapse_triangles_clicked.emit(self._spin_tri.value()))
        tri_row.addWidget(self._btn_collapse)
        refine_layout.addLayout(tri_row)

        ray_row = QHBoxLayout()
        ray_row.addWidget(QLabel("Ray tol:"))
        self._spin_ray = QDoubleSpinBox()
        self._spin_ray.setRange(0.0, 10000.0)
        self._spin_ray.setDecimals(2)
        self._spin_ray.setSingleStep(0.5)
        self._spin_ray.setValue(1.0)
        self._spin_ray.setToolTip(
            "Max distance (point-coordinate units) between two endpoint rays for\n"
            "them to be snapped together with a new junction node.")
        ray_row.addWidget(self._spin_ray)
        self._btn_grow_rays = QPushButton("Grow Rays")
        self._btn_grow_rays.setFixedHeight(28)
        self._btn_grow_rays.setToolTip(
            "Shoot rays outward from dangling endpoints; where two rays nearly\n"
            "intersect, add a junction node connecting both.")
        self._btn_grow_rays.clicked.connect(
            lambda: self.grow_rays_clicked.emit(self._spin_ray.value()))
        ray_row.addWidget(self._btn_grow_rays)
        refine_layout.addLayout(ray_row)

        beam_row = QHBoxLayout()
        beam_row.addWidget(QLabel("Beam r:"))
        self._spin_beam = QDoubleSpinBox()
        self._spin_beam.setRange(0.0, 10000.0)
        self._spin_beam.setDecimals(2)
        self._spin_beam.setSingleStep(1.0)
        self._spin_beam.setValue(5.0)
        self._spin_beam.setToolTip(
            "Radius (point-coordinate units) of the fat beam shot from each\n"
            "endpoint when latching onto the nearest point.")
        beam_row.addWidget(self._spin_beam)
        self._btn_beam = QPushButton("Beam Latch")
        self._btn_beam.setFixedHeight(28)
        self._btn_beam.setToolTip(
            "From each degree-1 endpoint, shoot a fat beam along its outward\n"
            "direction and connect it to the closest point caught inside.")
        self._btn_beam.clicked.connect(
            lambda: self.beam_latch_clicked.emit(self._spin_beam.value()))
        beam_row.addWidget(self._btn_beam)
        refine_layout.addLayout(beam_row)

        self._btn_pipeline = QPushButton("Run Full Refinement Pipeline")
        self._btn_pipeline.setFixedHeight(32)
        self._btn_pipeline.setToolTip(
            "Run all refinements in order: prune → grow rays → prune →\n"
            "beam latch → (simplify chains + collapse triangles, looped) using\n"
            "the Ray tol / Beam r / Tri size values above. One undo step.")
        self._btn_pipeline.clicked.connect(
            lambda: self.run_pipeline_clicked.emit(
                self._spin_ray.value(), self._spin_beam.value(),
                self._spin_tri.value()))
        refine_layout.addWidget(self._btn_pipeline)

        layout.addWidget(refine_group)

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
                  self._btn_reextract, self._btn_remove_edge,
                  self._btn_delete, self._btn_smooth,
                  self._btn_simplify, self._btn_prune, self._btn_collapse,
                  self._btn_grow_rays, self._btn_beam, self._btn_pipeline):
            w.setEnabled(enabled)
