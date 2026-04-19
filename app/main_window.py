"""Main application window — orchestrates all components."""

import numpy as np
from pathlib import Path

import open3d as o3d
try:
    from pcd_graph_recon import generate_graph
    _HAS_GRAPH_RECON = True
except ImportError:
    _HAS_GRAPH_RECON = False

from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QProgressDialog,
    QApplication, QAbstractSpinBox, QLineEdit, QTextEdit, QComboBox,
)
from PyQt6.QtCore import Qt, QThread, QEvent, pyqtSignal, QObject
from PyQt6.QtGui import QKeySequence, QAction

from app.gl.viewport import GLViewport
from app.widgets.toolbar import ToolBar
from app.widgets.status_bar import StatusBar
from app.widgets.graph_panel import GraphPanel
from app.widgets.merge_panel import MergePanel
from app.widgets.skeleton_editor_panel import SkeletonEditorPanel
from app.commands.undo_stack import UndoStack
from app.commands.delete_command import DeleteCommand
from app.commands.color_command import ColorCommand
from app.data.pcd_io import load_pcd, save_pcd
from app.data.ply_export import export_ply
from app.data.strand_graph import (
    extract_skeleton, merge_graphs, merge_graphs_with_bridges, StrandGraph,
    clean as clean_graph, degree_counts,
    _build_knn_edges, _kruskal_mst,
)
from app.data.graph_io import export_graph_json, import_graph_json
from app.data.downsample import voxel_downsample
from app.data.align import icp_align, cpd_align, euler_to_transform
from app.data.point_cloud import PointCloud
from app.commands.replace_cloud_command import ReplaceCloudCommand
from app.commands.edit_skeleton_command import EditSkeletonCommand
from app.widgets.pcd_selector import PcdSelectorDialog
from app import settings


# ── Background PCD loader (keeps UI responsive for large files) ───────────────

class _LoadWorker(QObject):
    finished = pyqtSignal(object)   # PointCloud
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self) -> None:
        try:
            pc = load_pcd(self._path)
            self.finished.emit(pc)
        except Exception as exc:
            self.error.emit(str(exc))


class _GraphWorker(QObject):
    finished  = pyqtSignal(object)  # StrandGraph
    error     = pyqtSignal(str)
    progress  = pyqtSignal(str)     # human-readable stage description
    cancelled = pyqtSignal()

    def __init__(self, pcd, tau_detour, keep_tau, voxel_size, persistence_threshold):
        super().__init__()
        self._pcd = pcd
        self._tau_detour = tau_detour
        self._keep_tau = keep_tau
        self._voxel_size = voxel_size
        self._persistence_threshold = persistence_threshold
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        import os, shutil, tempfile
        import dmpcd as dm
        from pcd_graph_recon.api import (
            filter_marker2_by_detour, edge_length_percentile_filter,
        )
        try:
            from MomentumConnect import MomentumConnect
        except ImportError:
            import sys as _sys
            _sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(
                __import__('pcd_graph_recon').__file__
            ))))
            from MomentumConnect import MomentumConnect

        tmpdir = tempfile.mkdtemp(prefix="spiderweb_skel_")
        try:
            # ── Stage 1: downsample ───────────────────────────────────────────
            self.progress.emit("1/5  Downsampling…")
            pcd_down = (self._pcd.voxel_down_sample(self._voxel_size)
                        if self._voxel_size else self._pcd)
            points = np.asarray(pcd_down.points)

            if self._cancel_requested:
                self.cancelled.emit()
                return

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            feature_filename = os.path.join(tmpdir, "features.txt")
            np.savetxt(feature_filename, points, fmt="%.6f")

            # ── Stage 2: build filtration (pure Python, can be slow) ──────────
            self.progress.emit("2/5  Building filtration…")
            dm.pcd.build_sparse_weighted_rips_filtration(
                feature_filename, output_dir, 15, "euclidean", 0.99
            )

            if self._cancel_requested:
                self.cancelled.emit()
                return

            # ── Stage 3: compute persistence (C++ subprocess) ─────────────────
            self.progress.emit("3/5  Computing persistence…")
            filtration_filename = os.path.join(output_dir,
                                               "sparse_weighted_rips_filtration.txt")
            dm.pcd.compute_persistence_swr(filtration_filename, output_dir)

            if self._cancel_requested:
                self.cancelled.emit()
                return

            # ── Stage 4: graph reconstruction (C++ subprocess) ────────────────
            self.progress.emit("4/5  Reconstructing graph…")
            weights_filename        = os.path.join(output_dir, "weights.txt")
            edge_filename           = os.path.join(output_dir, "edge_for_morse_only.txt")
            sorted_weights_filename = os.path.join(output_dir, "sorted-weights.txt")
            dm.pcd.reorder_weights(weights_filename, sorted_weights_filename)
            dm.pcd.compute_graph_reconstruction(
                sorted_weights_filename, edge_filename,
                self._persistence_threshold, output_dir,
            )

            if self._cancel_requested:
                self.cancelled.emit()
                return

            # ── Stage 5: post-processing ──────────────────────────────────────
            self.progress.emit("5/5  Filtering edges…")

            potential = ["edge.txt", "dimo_edge.txt"]
            edge_txt_path = next(
                (os.path.join(output_dir, f) for f in potential
                 if os.path.exists(os.path.join(output_dir, f))),
                None,
            )
            if edge_txt_path is None:
                raise RuntimeError(
                    "Graph reconstruction failed to produce an edge file."
                )
            final_edge_txt = os.path.join(output_dir, "edge.txt")
            if edge_txt_path != final_edge_txt:
                os.rename(edge_txt_path, final_edge_txt)
                edge_txt_path = final_edge_txt

            sorted_feature_filename = os.path.join(output_dir, "sorted-feature.txt")
            dm.pcd.reorder_verts_by_weight(
                weights_filename, feature_filename, sorted_feature_filename
            )

            points_sorted = np.loadtxt(sorted_feature_filename)
            if points_sorted.ndim == 1:
                points_sorted = points_sorted.reshape(-1, 3)
            if len(points_sorted) > 0 and points_sorted.shape[1] == 2:
                points_sorted = np.hstack(
                    [points_sorted, np.zeros((len(points_sorted), 1))]
                )

            edges_list = []
            with open(edge_txt_path) as fh:
                for line in fh:
                    s = line.strip().split()
                    if len(s) >= 3:
                        edges_list.append([int(s[0]), int(s[1]), int(s[2])])
            E = (np.array(edges_list, dtype=int) if edges_list
                 else np.empty((0, 3), dtype=int))

            good2, bad2 = filter_marker2_by_detour(
                points_sorted, E, tau_detour=self._tau_detour
            )
            base_mask     = np.isin(E[:, 2], (-1, 1))
            base_edges    = E[base_mask, :2]
            filtered_edges = (np.vstack([base_edges, good2])
                              if len(good2) > 0 else base_edges)
            added_back     = MomentumConnect(
                filtered_edges, bad2, points_sorted, self._keep_tau, 30
            )
            final_edges_indices = (np.vstack([filtered_edges, added_back])
                                   if len(added_back) > 0 else filtered_edges)

            base_edges_full = E[base_mask, :]
            good2_full = (np.hstack([good2, 2 * np.ones((len(good2), 1), dtype=int)])
                          if len(good2) > 0 else np.empty((0, 3), dtype=int))
            added_back_full = (np.hstack([added_back,
                                          2 * np.ones((len(added_back), 1), dtype=int)])
                               if len(added_back) > 0 else np.empty((0, 3), dtype=int))
            final_edges_full = np.vstack([base_edges_full, good2_full, added_back_full])

            if final_edges_full.shape[0] > 0:
                final_edges_full, _, _ = edge_length_percentile_filter(
                    points_sorted, final_edges_full, percentile=75.0
                )

            nodes = np.array(points_sorted, dtype=np.float32)
            edges = np.array(
                final_edges_full[:, :2] if len(final_edges_full) > 0
                else np.empty((0, 2), dtype=int),
                dtype=np.int32,
            )
            self.finished.emit(StrandGraph(nodes=nodes, edges=edges))

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── MainWindow ─────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spiderweb PCD Explorer")
        self.resize(1400, 900)

        # Ensure the saves directory exists (all writes go here by default)
        settings.SAVES_DIR.mkdir(parents=True, exist_ok=True)

        # Active cloud — all tools always operate on self._pc
        self._pc: PointCloud | None = None

        # Two-cloud state for the merge workflow
        self._pc_primary:   PointCloud | None = None   # first-loaded web
        self._pc_secondary: PointCloud | None = None   # second-loaded web
        self._editing_secondary: bool = False           # True = secondary is active

        # Alignment transform for the secondary cloud (updated by ICP / manual controls).
        # Always expressed relative to secondary's own coordinate system.
        self._secondary_alignment_T: np.ndarray = np.eye(4, dtype=np.float32)

        self._skeleton: StrandGraph | None = None
        self._graph_thread: QThread | None = None
        self._graph_worker: _GraphWorker | None = None
        self._undo_stack = UndoStack(self)
        self._undo_stack.changed.connect(self._on_command_executed)

        # Central viewport
        self._viewport = GLViewport(self)
        self.setCentralWidget(self._viewport)
        self._viewport.frame_rendered.connect(self._status_bar_update_fps)
        self._viewport.selection_changed.connect(self._on_selection_changed)

        # Toolbar (left-docked)
        self._toolbar = ToolBar(self)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self._toolbar)
        self._toolbar.tool_selected.connect(self._on_tool_selected)
        self._toolbar.point_size_changed.connect(self._viewport.set_point_size)
        self._toolbar.apply_color_clicked.connect(self._apply_color)
        self._toolbar.delete_clicked.connect(self._delete_selected)
        self._toolbar.reset_camera_clicked.connect(self._viewport.reset_camera)
        self._toolbar.open_pcd_clicked.connect(self._open_pcd)
        self._toolbar.save_pcd_clicked.connect(self._save_pcd)

        # Status bar — must be created before connecting fps_mode_changed
        self._status = StatusBar(self)
        self.setStatusBar(self._status)
        self._viewport.fps_mode_changed.connect(self._status.set_fps_mode)

        # Graph panel (right-docked)
        self._graph_panel = GraphPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._graph_panel)
        self._graph_panel.downsample_clicked.connect(self._downsample_cloud)
        self._graph_panel.extract_clicked.connect(self._extract_skeleton)
        self._graph_panel.extract_intelligent_clicked.connect(self._extract_intelligent_skeleton)
        self._graph_panel.cancel_intelligent_clicked.connect(self._cancel_intelligent_skeleton)
        self._graph_panel.import_clicked.connect(self._import_graph)
        self._graph_panel.export_clicked.connect(self._export_graph)
        self._graph_panel.clear_clicked.connect(self._clear_skeleton)

        # Merge panel (right-docked, tabbed with graph panel)
        self._merge_panel = MergePanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._merge_panel)
        self.tabifyDockWidget(self._graph_panel, self._merge_panel)
        self._merge_panel.load_secondary_clicked.connect(self._load_secondary_pcd)
        self._merge_panel.clear_secondary_clicked.connect(self._clear_secondary)
        self._merge_panel.switch_active_clicked.connect(self._switch_active_cloud)
        self._merge_panel.run_icp_clicked.connect(self._run_icp)
        self._merge_panel.run_cpd_clicked.connect(self._run_cpd)
        self._merge_panel.merge_clicked.connect(self._merge_clouds)
        self._merge_panel.transform_changed.connect(self._on_manual_transform_changed)

        # Skeleton editor panel (right-docked, tabbed)
        self._skel_editor = SkeletonEditorPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._skel_editor)
        self.tabifyDockWidget(self._merge_panel, self._skel_editor)
        self._skel_editor.edit_mode_changed.connect(self._on_skel_edit_mode_changed)
        self._skel_editor.select_all_clicked.connect(self._skel_select_all)
        self._skel_editor.deselect_all_clicked.connect(self._skel_deselect_all)
        self._skel_editor.select_by_degree_clicked.connect(self._skel_select_by_degree)
        self._skel_editor.reextract_clicked.connect(self._reextract_selected_skel_nodes)
        self._skel_editor.delete_nodes_clicked.connect(self._delete_selected_skel_nodes)
        self._viewport.skel_selection_changed.connect(self._on_skel_selection_changed)

        self._build_menu()

        # App-level event filter so camera keys work regardless of which
        # widget has focus (toolbar clicks steal focus from the viewport)
        QApplication.instance().installEventFilter(self)

    # ── menu bar ──────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")

        act_open = QAction("&Browse Spiderweb PCDs…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_pcd)
        file_menu.addAction(act_open)

        act_open_file = QAction("Open &File…", self)
        act_open_file.triggered.connect(self._open_file_dialog)
        file_menu.addAction(act_open_file)

        act_save = QAction("&Save PCD…", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_pcd)
        file_menu.addAction(act_save)

        act_export = QAction("Export &PLY…", self)
        act_export.triggered.connect(self._export_ply)
        file_menu.addAction(act_export)

        act_export_graph = QAction("Export Skeleton &JSON…", self)
        act_export_graph.triggered.connect(self._export_graph)
        file_menu.addAction(act_export_graph)

        act_import_graph = QAction("&Import Skeleton JSON…", self)
        act_import_graph.triggered.connect(self._import_graph)
        file_menu.addAction(act_import_graph)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Edit
        edit_menu = mb.addMenu("&Edit")

        self._act_undo = QAction("&Undo", self)
        self._act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._act_undo.triggered.connect(self._undo_stack.undo)
        self._act_undo.setEnabled(False)
        edit_menu.addAction(self._act_undo)

        self._act_redo = QAction("&Redo", self)
        self._act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self._act_redo.triggered.connect(self._undo_stack.redo)
        self._act_redo.setEnabled(False)
        edit_menu.addAction(self._act_redo)

        edit_menu.addSeparator()

        act_invert = QAction("&Invert Selection", self)
        act_invert.setShortcut(QKeySequence("Ctrl+I"))
        act_invert.triggered.connect(self._invert_selection)
        edit_menu.addAction(act_invert)

        act_clear = QAction("&Clear Selection", self)
        act_clear.setShortcut(QKeySequence("Escape"))
        act_clear.triggered.connect(self._clear_selection)
        edit_menu.addAction(act_clear)

    # ── File actions ──────────────────────────────────────────────────────────

    def show_selector_on_startup(self) -> None:
        """Called after the window is shown to present the PCD selector."""
        self._open_pcd()

    def _open_pcd(self) -> None:
        """Show the spiderweb PCD browser dialog."""
        dlg = PcdSelectorDialog(self)
        if dlg.exec() != PcdSelectorDialog.DialogCode.Accepted:
            return
        if dlg.import_json_requested:
            self._import_graph()
            return
        if dlg.selected_path is None:
            return
        self._load_path(str(dlg.selected_path))

    def _open_file_dialog(self) -> None:
        """Fallback: open any PCD via a standard file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PCD file", "", "Point Cloud (*.pcd);;All files (*)"
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        progress = QProgressDialog("Loading point cloud…", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)
        progress.show()
        QApplication.processEvents()

        self._load_thread = QThread()
        self._load_worker = _LoadWorker(path)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.finished.connect(progress.close)

        self._load_thread.start()

    def _on_load_finished(self, pc) -> None:
        self._pc = pc
        self._pc_primary = pc
        self._editing_secondary = False
        self._viewport.load_point_cloud(pc)
        self._undo_stack._undo.clear()
        self._undo_stack._redo.clear()
        self._update_undo_actions()
        self._status.update_point_cloud(pc)
        # Clear any skeleton left over from the previous cloud
        self._skeleton = None
        self._viewport.clear_skeleton()
        self._graph_panel.clear_stats()
        self._skel_editor.set_node_stats(0, 0)
        self._skel_editor.set_edit_mode(False)
        self._viewport.skeleton_edit_mode = False
        # Clear secondary cloud whenever a fresh primary is loaded
        self._pc_secondary = None
        self._secondary_alignment_T = np.eye(4, dtype=np.float32)
        self._viewport.clear_reference()
        self._merge_panel.clear_secondary_status()
        self._viewport.setFocus()   # give viewport focus so WASD works immediately

    def _on_load_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Load error", f"Failed to load PCD:\n{msg}")

    def _save_pcd(self) -> None:
        if self._pc is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PCD file", settings.default_dir(), "Point Cloud (*.pcd)"
        )
        if not path:
            return
        try:
            save_pcd(self._pc, path)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _export_ply(self) -> None:
        if self._pc is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PLY", settings.default_dir(), "PLY file (*.ply)"
        )
        if not path:
            return
        try:
            export_ply(self._pc, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    # ── Active-cloud helpers ──────────────────────────────────────────────────

    def _apply_active_cloud(self, pc: PointCloud) -> None:
        """
        Swap `pc` in as the active cloud (self._pc + the correct primary/secondary
        slot) and reload the viewport renderer.  Used by ReplaceCloudCommand.
        """
        self._pc = pc
        if self._editing_secondary:
            self._pc_secondary = pc
        else:
            self._pc_primary = pc
        self._viewport.reload_point_cloud(pc)
        self._status.update_point_cloud(pc)
        # Update reference overlay to reflect the now-active cloud's counterpart
        self._refresh_reference_overlay()

    def _refresh_reference_overlay(self) -> None:
        """Reload the reference overlay to match the current two-cloud state."""
        if self._editing_secondary and self._pc_primary is not None:
            self._viewport.load_reference(self._pc_primary, transform=None)
        elif not self._editing_secondary and self._pc_secondary is not None:
            self._viewport.load_reference(
                self._pc_secondary, transform=self._secondary_alignment_T
            )

    # ── Downsample action ─────────────────────────────────────────────────────

    def _downsample_cloud(self) -> None:
        if self._pc is None:
            return
        n_before = self._pc.alive_count

        voxel_size = self._graph_panel.get_ds_voxel_size()
        if voxel_size is None:
            positions = self._pc.positions[self._pc.alive_mask]
            extent = float(np.ptp(positions, axis=0).max())
            voxel_size = max(extent / 100.0, 1e-9)

        try:
            new_pc = voxel_downsample(self._pc, voxel_size)
        except Exception as exc:
            QMessageBox.critical(self, "Downsample error", str(exc))
            return

        n_after = new_pc.alive_count
        self._graph_panel.set_ds_stats(n_before, n_after)

        old_pc = self._pc
        cmd = ReplaceCloudCommand(old_pc, new_pc, self._apply_active_cloud, "downsample")
        self._undo_stack.push(cmd)
        self._graph_panel.set_ds_stats(n_before, n_after)

    # ── Skeleton actions ──────────────────────────────────────────────────────

    def _extract_skeleton(self) -> None:
        if self._pc is None:
            return
        indices = self._pc.selected_alive_indices()
        if len(indices) == 0:
            QMessageBox.warning(self, "No selection",
                                "Select a region of the point cloud first.")
            return
        if len(indices) < 2:
            QMessageBox.warning(self, "Too few points",
                                "Select at least 2 points to extract a skeleton.")
            return

        positions = self._pc.positions[indices]

        voxel_size = self._graph_panel.get_voxel_size()   # None = auto
        k          = self._graph_panel.get_k_neighbors()
        prune_fac  = self._graph_panel.get_prune_factor()

        if voxel_size is None:
            extent = float(np.ptp(positions, axis=0).max())
            voxel_size = max(extent / 20.0, 1e-9)

        try:
            graph = extract_skeleton(
                positions,
                voxel_size=voxel_size,
                k_neighbors=k,
                prune_factor=prune_fac,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Extraction failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Extraction error", str(exc))
            return

        if self._skeleton is not None:
            graph = merge_graphs_with_bridges(self._skeleton, graph)
        self._set_skeleton(graph)

    def _extract_intelligent_skeleton(self) -> None:
        if not _HAS_GRAPH_RECON:
            QMessageBox.critical(self, "Missing Dependency", "pcd_graph_recon is not installed.")
            return

        if self._pc is None:
            return
        indices = self._pc.selected_alive_indices()
        if len(indices) == 0:
            QMessageBox.warning(self, "No selection",
                                "Select a region of the point cloud first.")
            return
        if len(indices) < 2:
            QMessageBox.warning(self, "Too few points",
                                "Select at least 2 points to extract a skeleton.")
            return

        positions = self._pc.positions[indices]
        colors = self._pc.colors[indices] if self._pc.colors is not None else None

        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
        if colors is not None:
            # open3d expects (N, 3) for colors, but PointCloud stores (N, 4) RGBA
            o3d_pc.colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64))

        voxel_size = self._graph_panel.get_intel_voxel_size()   # None = auto
        tau_detour = self._graph_panel.get_tau_detour()
        keep_tau = self._graph_panel.get_keep_tau()
        persist_thresh = self._graph_panel.get_persistence_threshold()

        self._graph_panel.set_intel_running(True)
        self._graph_panel.set_intel_progress("Starting…")

        self._graph_thread = QThread()
        self._graph_worker = _GraphWorker(
            o3d_pc, tau_detour, keep_tau, voxel_size, persist_thresh
        )
        self._graph_worker.moveToThread(self._graph_thread)

        self._graph_thread.started.connect(self._graph_worker.run)
        self._graph_worker.progress.connect(self._graph_panel.set_intel_progress)
        self._graph_worker.finished.connect(self._on_graph_finished)
        self._graph_worker.cancelled.connect(self._on_graph_cancelled)
        self._graph_worker.error.connect(self._on_graph_error)
        self._graph_worker.finished.connect(self._graph_thread.quit)
        self._graph_worker.cancelled.connect(self._graph_thread.quit)
        self._graph_worker.error.connect(self._graph_thread.quit)
        self._graph_thread.finished.connect(lambda: self._graph_panel.set_intel_running(False))

        self._graph_thread.start()

    def _cancel_intelligent_skeleton(self) -> None:
        if self._graph_worker is not None:
            self._graph_panel.set_intel_progress("Cancelling… (waiting for current stage)")
            self._graph_worker.request_cancel()

    def _on_graph_finished(self, graph) -> None:
        if self._skeleton is not None:
            graph = merge_graphs(self._skeleton, graph)
        self._set_skeleton(graph)
        self._graph_worker = None

    def _on_graph_cancelled(self) -> None:
        self._graph_worker = None

    def _on_graph_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Extraction error", f"pcd_graph_recon failed:\n{msg}")
        self._graph_worker = None

    def _export_graph(self) -> None:
        if self._skeleton is None:
            QMessageBox.information(self, "No skeleton",
                                    "Extract a skeleton first before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Skeleton JSON", settings.default_dir(), "JSON file (*.json)"
        )
        if not path:
            return
        try:
            export_graph_json(self._skeleton, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _import_graph(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Skeleton JSON", settings.default_dir(), "JSON file (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            graph = import_graph_json(path)
            print(f"Imported graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            if len(graph.nodes) > 0:
                mins = graph.nodes.min(axis=0)
                maxs = graph.nodes.max(axis=0)
                print(f"Graph bounding box: min={mins}, max={maxs}")
            
            if self._skeleton is not None:
                graph = merge_graphs(self._skeleton, graph)
            self._set_skeleton(graph)
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            import traceback
            traceback.print_exc()

    def _set_skeleton(self, graph: StrandGraph) -> None:
        """Clean, store, render, and update all stats for a new skeleton."""
        graph = clean_graph(graph)
        self._skeleton = graph
        self._viewport.set_skeleton(graph)
        self._graph_panel.set_stats(len(graph.nodes), len(graph.edges))
        self._skel_editor.set_degree_stats(degree_counts(graph))
        # Refresh selection mask size if edit mode is active
        if self._viewport.skeleton_edit_mode:
            self._viewport.reset_skel_selection()
            self._skel_editor.set_node_stats(0, len(graph.nodes))

    def _clear_skeleton(self) -> None:
        self._skeleton = None
        self._viewport.clear_skeleton()
        self._graph_panel.clear_stats()
        self._skel_editor.set_node_stats(0, 0)
        # Exit skeleton edit mode so controls reset cleanly
        if self._viewport.skeleton_edit_mode:
            self._viewport.skeleton_edit_mode = False
            self._skel_editor.set_edit_mode(False)

    # ── Skeleton node editor ──────────────────────────────────────────────────

    def _on_skel_edit_mode_changed(self, active: bool) -> None:
        self._viewport.skeleton_edit_mode = active
        if active:
            if self._skeleton is None or len(self._skeleton.nodes) == 0:
                self._skel_editor.set_edit_mode(False)
                self._viewport.skeleton_edit_mode = False
                QMessageBox.information(self, "No skeleton",
                                        "Import or extract a skeleton first.")
                return
            self._viewport.reset_skel_selection()
            n = len(self._skeleton.nodes)
            self._skel_editor.set_node_stats(0, n)
            self._skel_editor._set_controls_enabled(True)
        else:
            self._viewport._skel_selection = None
            self._skel_editor.set_node_stats(0,
                len(self._skeleton.nodes) if self._skeleton else 0)
            self._skel_editor._set_controls_enabled(False)

    def _on_skel_selection_changed(self) -> None:
        if self._skeleton is None or self._viewport._skel_selection is None:
            return
        n_sel   = int(self._viewport._skel_selection.sum())
        n_total = len(self._skeleton.nodes)
        self._skel_editor.set_node_stats(n_sel, n_total)

    def _skel_select_by_degree(self, degree: int) -> None:
        if self._skeleton is None or self._viewport._skel_selection is None:
            return
        n = len(self._skeleton.nodes)
        counts = np.zeros(n, dtype=np.int32)
        # Deduplicate edges (normalise to min,max) so (u,v) and (v,u) aren't double-counted
        seen: set[tuple[int, int]] = set()
        for u, v in self._skeleton.edges:
            key = (min(int(u), int(v)), max(int(u), int(v)))
            if key not in seen:
                seen.add(key)
                counts[key[0]] += 1
                counts[key[1]] += 1
        self._viewport._skel_selection[:] = counts == degree
        self._viewport._upload_skel_selection()
        self._viewport.skel_selection_changed.emit()

    def _skel_select_all(self) -> None:
        if self._viewport._skel_selection is None:
            return
        self._viewport._skel_selection[:] = True
        self._viewport._upload_skel_selection()
        self._viewport.skel_selection_changed.emit()

    def _skel_deselect_all(self) -> None:
        if self._viewport._skel_selection is None:
            return
        self._viewport._skel_selection[:] = False
        self._viewport._upload_skel_selection()
        self._viewport.skel_selection_changed.emit()

    def _reextract_selected_skel_nodes(self) -> None:
        if self._skeleton is None or self._viewport._skel_selection is None:
            return
        mask         = self._viewport._skel_selection
        selected_idx = np.where(mask)[0]
        if len(selected_idx) < 2:
            QMessageBox.warning(self, "Too few nodes",
                                "Select at least 2 skeleton nodes.")
            return

        selected_pos = self._skeleton.nodes[selected_idx]
        k = self._skel_editor.get_k_neighbors()
        k_actual = min(k, len(selected_idx) - 1)

        # Build k-NN edges and MST on just the selected positions (no voxelization,
        # since the skeleton nodes are already sparse and we want to keep them as-is).
        edge_list = _build_knn_edges(selected_pos, k_actual)
        mst_local = _kruskal_mst(len(selected_pos), edge_list)

        # Map local MST edge indices back to global skeleton node indices
        new_edges = [(int(selected_idx[u]), int(selected_idx[v]))
                     for u, v in mst_local]

        # Keep only existing edges where NOT both endpoints are selected
        selected_set = set(selected_idx.tolist())
        kept_edges = [
            (int(u), int(v)) for u, v in self._skeleton.edges
            if not (u in selected_set and v in selected_set)
        ]

        all_edges = kept_edges + new_edges
        edges_arr = (np.array(all_edges, dtype=np.int32) if all_edges
                     else np.empty((0, 2), dtype=np.int32))

        old_skeleton = self._skeleton
        new_skeleton = StrandGraph(
            nodes=self._skeleton.nodes.copy(),
            edges=edges_arr,
        )
        cmd = EditSkeletonCommand(old_skeleton, new_skeleton, self._set_skeleton,
                                  "Re-extract skeleton connectivity")
        self._undo_stack.push(cmd)
        self._viewport.reset_skel_selection()

    def _delete_selected_skel_nodes(self) -> None:
        if self._skeleton is None or self._viewport._skel_selection is None:
            return
        mask         = self._viewport._skel_selection
        selected_set = set(np.where(mask)[0].tolist())
        kept_idx     = np.where(~mask)[0]

        if len(kept_idx) == 0:
            self._clear_skeleton()
            return

        remap     = {int(old): new for new, old in enumerate(kept_idx)}
        new_nodes = self._skeleton.nodes[kept_idx]

        new_edges_list = [
            (remap[int(u)], remap[int(v)])
            for u, v in self._skeleton.edges
            if int(u) not in selected_set and int(v) not in selected_set
        ]
        new_edges = (np.array(new_edges_list, dtype=np.int32) if new_edges_list
                     else np.empty((0, 2), dtype=np.int32))

        old_skeleton = self._skeleton
        new_skeleton = StrandGraph(nodes=new_nodes, edges=new_edges)
        cmd = EditSkeletonCommand(old_skeleton, new_skeleton, self._set_skeleton,
                                  "Delete skeleton nodes")
        self._undo_stack.push(cmd)
        self._viewport.reset_skel_selection()

    # ── Merge / Align actions ─────────────────────────────────────────────────

    def _load_secondary_pcd(self) -> None:
        """Open the PCD selector to pick a secondary cloud for alignment."""
        dlg = PcdSelectorDialog(self)
        if dlg.exec() != PcdSelectorDialog.DialogCode.Accepted:
            return
        if dlg.selected_path is None:
            return
        try:
            pc = load_pcd(str(dlg.selected_path))
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Failed to load secondary PCD:\n{exc}")
            return

        self._pc_secondary = pc
        self._secondary_alignment_T = np.eye(4, dtype=np.float32)

        # Make the secondary the active cloud so the user can edit it immediately
        self._editing_secondary = True
        self._pc = self._pc_secondary
        self._viewport.reload_point_cloud(self._pc_secondary)

        # Primary becomes the reference overlay
        if self._pc_primary is not None:
            self._viewport.load_reference(self._pc_primary, transform=None)

        self._undo_stack._undo.clear()
        self._undo_stack._redo.clear()
        self._update_undo_actions()
        self._status.update_point_cloud(self._pc)

        self._merge_panel.reset_transform_spinboxes()
        self._merge_panel.set_secondary_loaded(dlg.selected_path.name, pc.alive_count)
        self._merge_panel.set_editing_state(editing_secondary=True)
        self._viewport.setFocus()

    def _switch_active_cloud(self) -> None:
        """Toggle which cloud (primary / secondary) is currently being edited."""
        if self._pc_secondary is None:
            return

        self._editing_secondary = not self._editing_secondary

        if self._editing_secondary:
            # Secondary → active; primary → reference overlay (no transform)
            self._pc = self._pc_secondary
            self._viewport.reload_point_cloud(self._pc_secondary)
            if self._pc_primary is not None:
                self._viewport.load_reference(self._pc_primary, transform=None)
        else:
            # Primary → active; secondary → reference overlay (with alignment transform)
            self._pc = self._pc_primary
            self._viewport.reload_point_cloud(self._pc_primary)
            self._viewport.load_reference(
                self._pc_secondary, transform=self._secondary_alignment_T
            )

        # Clear undo history — history from the other cloud isn't applicable
        self._undo_stack._undo.clear()
        self._undo_stack._redo.clear()
        self._update_undo_actions()
        self._status.update_point_cloud(self._pc)
        self._merge_panel.set_editing_state(self._editing_secondary)
        self._viewport.setFocus()

    def _clear_secondary(self) -> None:
        self._pc_secondary = None
        self._secondary_alignment_T = np.eye(4, dtype=np.float32)

        # If we were editing the secondary, switch back to primary
        if self._editing_secondary and self._pc_primary is not None:
            self._editing_secondary = False
            self._pc = self._pc_primary
            self._viewport.reload_point_cloud(self._pc_primary)
            self._undo_stack._undo.clear()
            self._undo_stack._redo.clear()
            self._update_undo_actions()
            self._status.update_point_cloud(self._pc)

        self._viewport.clear_reference()
        self._merge_panel.clear_secondary_status()

    def _on_manual_transform_changed(
        self, tx: float, ty: float, tz: float,
        yaw: float, pitch: float, roll: float
    ) -> None:
        """
        Compose the manual spinbox delta on top of the ICP base transform.
        Total = _secondary_alignment_T @ manual_delta.
        """
        if self._pc_secondary is None:
            return
        alive_pos = self._pc_secondary.positions[self._pc_secondary.alive_mask]
        center = alive_pos.mean(axis=0)
        manual_delta = euler_to_transform(tx, ty, tz, yaw, pitch, roll, center=center)
        total = (self._secondary_alignment_T @ manual_delta).astype(np.float32)
        # Only affects the reference overlay (secondary is the reference when
        # editing primary, or when the transform controls are being used)
        self._viewport.update_reference_transform(total)

    def _run_icp(self) -> None:
        """Run rigid ICP to align secondary cloud to primary."""
        if self._pc_primary is None or self._pc_secondary is None:
            return

        primary_pos   = self._pc_primary.positions[self._pc_primary.alive_mask]
        secondary_pos = self._pc_secondary.positions[self._pc_secondary.alive_mask]

        max_iter = self._merge_panel.get_icp_max_iter()

        try:
            T, rmse, n_inliers = icp_align(
                source=secondary_pos,
                target=primary_pos,
                init_transform=self._secondary_alignment_T,
                max_iter=max_iter,
            )
        except Exception as exc:
            QMessageBox.critical(self, "ICP error", str(exc))
            return

        self._secondary_alignment_T = T
        self._merge_panel.reset_transform_spinboxes()
        self._merge_panel.set_icp_result(rmse, n_inliers)

        # Show aligned secondary as overlay; switch view to primary so the
        # alignment result is immediately visible
        if self._editing_secondary:
            self._switch_active_cloud()   # → primary active, secondary as aligned overlay
        else:
            self._viewport.update_reference_transform(T)

    def _run_cpd(self) -> None:
        """Run non-rigid CPD to warp the secondary cloud to the primary."""
        if self._pc_primary is None or self._pc_secondary is None:
            return

        primary_pos   = self._pc_primary.positions[self._pc_primary.alive_mask]
        secondary_pos = self._pc_secondary.positions[self._pc_secondary.alive_mask]

        # Apply current alignment transform to secondary before CPD
        T = self._secondary_alignment_T.astype(np.float64)
        ones  = np.ones((len(secondary_pos), 1), dtype=np.float64)
        sec_h = np.concatenate([secondary_pos.astype(np.float64), ones], axis=1)
        aligned_pos = (T @ sec_h.T).T[:, :3].astype(np.float32)

        alpha = self._merge_panel.get_cpd_alpha()
        self._merge_panel.set_cpd_status("Running CPD…")
        QApplication.processEvents()

        try:
            warped = cpd_align(source=aligned_pos, target=primary_pos, alpha=alpha)
        except ImportError as exc:
            QMessageBox.critical(self, "pycpd missing", str(exc))
            self._merge_panel.set_cpd_status("pycpd not installed")
            return
        except Exception as exc:
            QMessageBox.critical(self, "CPD error", str(exc))
            self._merge_panel.set_cpd_status("CPD failed")
            return

        # CPD bakes the warp into positions → reset alignment transform to identity
        new_pc = PointCloud(warped)
        self._pc_secondary = new_pc
        self._secondary_alignment_T = np.eye(4, dtype=np.float32)
        self._merge_panel.reset_transform_spinboxes()
        self._merge_panel.set_secondary_loaded("(warped)", len(warped))
        self._merge_panel.set_cpd_status(f"Done — {len(warped):,} warped points")

        # Switch to primary view so user can inspect the warped overlay
        if self._editing_secondary:
            self._pc = self._pc_primary
            self._editing_secondary = False
            self._viewport.reload_point_cloud(self._pc_primary)
            self._merge_panel.set_editing_state(editing_secondary=False)
        self._viewport.load_reference(new_pc, transform=None)
        self._status.update_point_cloud(self._pc)

    def _merge_clouds(self) -> None:
        """
        Apply the current alignment transform to the secondary cloud and
        concatenate it into the primary.  The result is undoable.
        """
        if self._pc_primary is None or self._pc_secondary is None:
            return

        # Apply alignment transform to secondary alive positions
        T = self._viewport._secondary_transform.astype(np.float64)
        sec_pos    = self._pc_secondary.positions[self._pc_secondary.alive_mask].astype(np.float64)
        ones       = np.ones((len(sec_pos), 1), dtype=np.float64)
        warped_pos = (T @ np.concatenate([sec_pos, ones], axis=1).T).T[:, :3].astype(np.float32)
        sec_colors = self._pc_secondary.colors[self._pc_secondary.alive_mask]

        pri_pos    = self._pc_primary.positions[self._pc_primary.alive_mask]
        pri_colors = self._pc_primary.colors[self._pc_primary.alive_mask]

        all_pos    = np.concatenate([pri_pos, warped_pos], axis=0)
        all_colors = np.concatenate([pri_colors, sec_colors], axis=0)
        merged_pc  = PointCloud(all_pos, all_colors)

        # Clear secondary state before the replacement so that _apply_active_cloud
        # (called by execute()) sees no secondary and updates _pc_primary correctly.
        old_primary = self._pc_primary
        self._pc_secondary = None
        self._secondary_alignment_T = np.eye(4, dtype=np.float32)
        self._editing_secondary = False
        self._viewport.clear_reference()
        self._merge_panel.clear_secondary_status()

        # Push: execute() calls _apply_active_cloud(merged_pc)
        cmd = ReplaceCloudCommand(old_primary, merged_pc, self._apply_active_cloud, "merge")
        self._undo_stack.push(cmd)

    def _apply_skeleton_from_command(self, skeleton) -> None:
        self._set_skeleton(skeleton)
        self._graph_panel.set_stats(len(skeleton.nodes), len(skeleton.edges))

    def _delete_selected(self) -> None:
        if self._pc is None:
            return
        indices = self._pc.selected_alive_indices()
        if len(indices) == 0:
            return

        old_skeleton = self._skeleton
        new_skeleton = self._skeleton
        
        if self._skeleton is not None and len(self._skeleton.nodes) > 0:
            try:
                from sklearn.neighbors import KDTree
                alive_idx = np.where(self._pc.alive_mask)[0]
                alive_pos = self._pc.positions[alive_idx]
                tree = KDTree(alive_pos)
                _, closest_local_idx = tree.query(self._skeleton.nodes, k=1)
                closest_global_idx = alive_idx[closest_local_idx.flatten()]
                
                deleted_set = set(indices.tolist())
                keep_node_mask = np.array([idx not in deleted_set for idx in closest_global_idx], dtype=bool)
                
                if not keep_node_mask.all():
                    kept_indices = np.where(keep_node_mask)[0]
                    if len(kept_indices) == 0:
                        new_skeleton = StrandGraph(np.empty((0,3), dtype=np.float32), np.empty((0,2), dtype=np.int32))
                    else:
                        remap = {old: new for new, old in enumerate(kept_indices)}
                        nodes_out = self._skeleton.nodes[kept_indices]
                        edges_out_list = []
                        for u, v in self._skeleton.edges:
                            if u in remap and v in remap:
                                edges_out_list.append((remap[u], remap[v]))
                        edges_out = np.array(edges_out_list, dtype=np.int32) if edges_out_list else np.empty((0,2), dtype=np.int32)
                        new_skeleton = StrandGraph(nodes=nodes_out, edges=edges_out)
            except Exception as exc:
                print("Could not prune skeleton", exc)

        cmd = DeleteCommand(
            self._pc, indices, 
            old_skeleton=old_skeleton, 
            new_skeleton=new_skeleton, 
            apply_skeleton_func=self._apply_skeleton_from_command
        )
        self._undo_stack.push(cmd)
        self._viewport.on_alive_changed()

    def _apply_color(self) -> None:
        if self._pc is None:
            return
        indices = self._pc.selected_alive_indices()
        if len(indices) == 0:
            return
        rgba = self._toolbar.color_picker.rgba_float()
        cmd = ColorCommand(self._pc, indices, rgba)
        self._undo_stack.push(cmd)
        self._viewport.on_colors_changed()

    def _invert_selection(self) -> None:
        if self._pc is None:
            return
        self._pc.invert_selection()
        self._viewport.on_selection_changed()

    def _clear_selection(self) -> None:
        if self._pc is None:
            return
        self._pc.deselect_all()
        self._viewport.on_selection_changed()

    # ── Tool selection ────────────────────────────────────────────────────────

    def _on_tool_selected(self, name: str) -> None:
        self._viewport.tool_manager.set_tool(name)

    # ── Undo/redo callbacks ───────────────────────────────────────────────────

    def _on_command_executed(self) -> None:
        """Called after every undo/redo push — refresh VBOs and UI."""
        # ReplaceCloudCommand already calls reload_point_cloud, which uploads
        # fresh GPU buffers.  For DeleteCommand / ColorCommand the cloud object
        # is the same but its arrays changed, so mark them dirty here.
        if self._viewport.renderer:
            self._viewport.renderer.mark_alive_dirty()
            self._viewport.renderer.mark_colors_dirty()
        self._viewport.update()
        if self._pc:
            self._status.update_point_cloud(self._pc)
        self._update_undo_actions()

    def _update_undo_actions(self) -> None:
        self._act_undo.setEnabled(self._undo_stack.can_undo)
        self._act_redo.setEnabled(self._undo_stack.can_redo)
        undo_desc = self._undo_stack.undo_description
        redo_desc = self._undo_stack.redo_description
        self._act_undo.setText(f"&Undo {undo_desc}" if undo_desc else "&Undo")
        self._act_redo.setText(f"&Redo {redo_desc}" if redo_desc else "&Redo")

    # ── Status bar helpers ────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        if self._pc:
            self._status.update_selection(self._pc)

    def _status_bar_update_fps(self, fps: float) -> None:
        self._status.update_fps(fps)

    # ── Global key routing (app-level event filter) ───────────────────────────
    # Keys go to whichever widget has focus.  After clicking a toolbar button
    # the viewport loses focus, so WASD would silently do nothing.  The filter
    # intercepts camera keys before they reach any widget and sends them
    # straight to the camera, unless a text-entry widget is active.

    # Plain ints — Qt.Key enums don't compare equal to ints in PyQt6 strict mode
    _CAM_KEYS = {
        int(Qt.Key.Key_W), int(Qt.Key.Key_A),
        int(Qt.Key.Key_S), int(Qt.Key.Key_D),
        int(Qt.Key.Key_Space), int(Qt.Key.Key_Shift),
    }

    def eventFilter(self, obj, event) -> bool:
        if not self.isActiveWindow():
            return False
        if QApplication.activeModalWidget() is not None:
            return False

        ev_type = event.type()
        if ev_type not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False

        # Don't steal keys from text-entry widgets
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QComboBox, QAbstractSpinBox)):
            return False

        key = int(event.key())   # normalise Qt.Key enum → plain int

        if ev_type == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if key == int(Qt.Key.Key_Tab):
                self._viewport.toggle_fps_mode()
                return True          # consume — prevent Qt focus-tab navigation
            if key in (int(Qt.Key.Key_Delete), int(Qt.Key.Key_Q)):
                if self._viewport.skeleton_edit_mode:
                    self._delete_selected_skel_nodes()
                else:
                    self._delete_selected()
                return True
            if key == int(Qt.Key.Key_G):
                if self._viewport.skeleton_edit_mode:
                    self._reextract_selected_skel_nodes()
                else:
                    self._extract_skeleton()
                return True
            if key == int(Qt.Key.Key_Home):
                self._viewport.reset_camera()
                return True
            if key in self._CAM_KEYS:
                self._viewport.camera.key_down(key)
                return True

        elif ev_type == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if key in self._CAM_KEYS:
                self._viewport.camera.key_up(key)
                return True

        return False
