"""Main application window — orchestrates all components."""

import numpy as np
from pathlib import Path

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
from app.commands.undo_stack import UndoStack
from app.commands.delete_command import DeleteCommand
from app.commands.color_command import ColorCommand
from app.data.pcd_io import load_pcd, save_pcd
from app.data.ply_export import export_ply
from app.data.strand_graph import extract_skeleton, merge_graphs, StrandGraph
from app.data.graph_io import export_graph_json, import_graph_json
from app.data.downsample import voxel_downsample
from app.data.align import icp_align, cpd_align, euler_to_transform
from app.data.point_cloud import PointCloud
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


# ── MainWindow ─────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spiderweb PCD Explorer")
        self.resize(1400, 900)

        # Ensure the saves directory exists (all writes go here by default)
        settings.SAVES_DIR.mkdir(parents=True, exist_ok=True)

        self._pc = None
        self._skeleton: StrandGraph | None = None
        self._secondary_pc: PointCloud | None = None
        # Base transform set by ICP; manual spinbox delta is composed on top of this
        self._secondary_base_transform: np.ndarray = np.eye(4, dtype=np.float32)
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
        self._graph_panel.export_clicked.connect(self._export_graph)
        self._graph_panel.clear_clicked.connect(self._clear_skeleton)

        # Merge panel (right-docked, tabbed with graph panel)
        self._merge_panel = MergePanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._merge_panel)
        self.tabifyDockWidget(self._graph_panel, self._merge_panel)
        self._merge_panel.load_secondary_clicked.connect(self._load_secondary_pcd)
        self._merge_panel.clear_secondary_clicked.connect(self._clear_secondary)
        self._merge_panel.run_icp_clicked.connect(self._run_icp)
        self._merge_panel.run_cpd_clicked.connect(self._run_cpd)
        self._merge_panel.merge_clicked.connect(self._merge_clouds)
        self._merge_panel.transform_changed.connect(self._on_manual_transform_changed)

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
        self._viewport.load_point_cloud(pc)
        self._undo_stack._undo.clear()
        self._undo_stack._redo.clear()
        self._update_undo_actions()
        self._status.update_point_cloud(pc)
        # Clear any skeleton left over from the previous cloud
        self._skeleton = None
        self._viewport.clear_skeleton()
        self._graph_panel.clear_stats()
        # Clear secondary cloud if a new primary is loaded
        self._secondary_pc = None
        self._secondary_base_transform = np.eye(4, dtype=np.float32)
        self._viewport.clear_secondary()
        self._merge_panel.clear_secondary_status()
        self._viewport.setFocus()   # give viewport focus so WASD works immediately

    def _on_load_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Load error", f"Failed to load PCD:\n{msg}")

    def _save_pcd(self) -> None:
        if self._pc is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PCD file", str(settings.SAVES_DIR), "Point Cloud (*.pcd)"
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
            self, "Export PLY", str(settings.SAVES_DIR), "PLY file (*.ply)"
        )
        if not path:
            return
        try:
            export_ply(self._pc, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

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
        self._on_load_finished(new_pc)

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

        # Accumulate: merge new graph onto whatever was already extracted
        if self._skeleton is not None:
            graph = merge_graphs(self._skeleton, graph)
        self._skeleton = graph
        self._viewport.set_skeleton(graph)
        self._graph_panel.set_stats(len(graph.nodes), len(graph.edges))

    def _export_graph(self) -> None:
        if self._skeleton is None:
            QMessageBox.information(self, "No skeleton",
                                    "Extract a skeleton first before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Skeleton JSON", str(settings.SAVES_DIR), "JSON file (*.json)"
        )
        if not path:
            return
        try:
            export_graph_json(self._skeleton, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _import_graph(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Skeleton JSON", str(settings.SAVES_DIR), "JSON file (*.json);;All files (*)"
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
            
            # Accumulate: merge new graph onto whatever was already extracted/imported
            if self._skeleton is not None:
                graph = merge_graphs(self._skeleton, graph)
            self._skeleton = graph
            self._viewport.set_skeleton(graph)
            self._graph_panel.set_stats(len(graph.nodes), len(graph.edges))
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            import traceback
            traceback.print_exc()

    def _clear_skeleton(self) -> None:
        self._skeleton = None
        self._viewport.clear_skeleton()
        self._graph_panel.clear_stats()

    # ── Merge / Align actions ─────────────────────────────────────────────────

    def _load_secondary_pcd(self) -> None:
        """Open the PCD selector to pick a secondary cloud for alignment."""
        from app.widgets.pcd_selector import PcdSelectorDialog
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

        self._secondary_pc = pc
        self._secondary_base_transform = np.eye(4, dtype=np.float32)
        self._merge_panel.reset_transform_spinboxes()
        self._merge_panel.set_secondary_loaded(dlg.selected_path.name, pc.alive_count)
        self._viewport.load_secondary(pc)

    def _clear_secondary(self) -> None:
        self._secondary_pc = None
        self._secondary_base_transform = np.eye(4, dtype=np.float32)
        self._merge_panel.clear_secondary_status()
        self._viewport.clear_secondary()

    def _on_manual_transform_changed(
        self, tx: float, ty: float, tz: float,
        yaw: float, pitch: float, roll: float
    ) -> None:
        """
        Compose the manual (spinbox) adjustment on top of the ICP base transform.

        Total transform = _secondary_base_transform @ manual_delta
        """
        if self._secondary_pc is None:
            return
        # Rotation center = secondary cloud centroid (alive points)
        alive_pos = self._secondary_pc.positions[self._secondary_pc.alive_mask]
        center = alive_pos.mean(axis=0)
        manual_delta = euler_to_transform(tx, ty, tz, yaw, pitch, roll, center=center)
        total = (self._secondary_base_transform @ manual_delta).astype(np.float32)
        self._viewport.update_secondary_transform(total)

    def _run_icp(self) -> None:
        """Run rigid ICP to align secondary cloud to primary."""
        if self._pc is None or self._secondary_pc is None:
            return

        primary_pos = self._pc.positions[self._pc.alive_mask]
        secondary_pos = self._secondary_pc.positions[self._secondary_pc.alive_mask]

        max_iter = self._merge_panel.get_icp_max_iter()

        try:
            T, rmse, n_inliers = icp_align(
                source=secondary_pos,
                target=primary_pos,
                init_transform=self._secondary_base_transform,
                max_iter=max_iter,
            )
        except Exception as exc:
            QMessageBox.critical(self, "ICP error", str(exc))
            return

        self._secondary_base_transform = T
        # Reset manual spinboxes — the ICP result IS the new base
        self._merge_panel.reset_transform_spinboxes()
        self._merge_panel.set_icp_result(rmse, n_inliers)
        self._viewport.update_secondary_transform(T)

    def _run_cpd(self) -> None:
        """Run non-rigid CPD to warp the secondary cloud to the primary."""
        if self._pc is None or self._secondary_pc is None:
            return

        primary_pos = self._pc.positions[self._pc.alive_mask]
        secondary_pos = self._secondary_pc.positions[self._secondary_pc.alive_mask]

        # Apply current base transform to secondary positions first
        base_T = self._secondary_base_transform.astype(np.float64)
        ones = np.ones((len(secondary_pos), 1), dtype=np.float64)
        sec_h = np.concatenate([secondary_pos.astype(np.float64), ones], axis=1)
        aligned_pos = (base_T @ sec_h.T).T[:, :3].astype(np.float32)

        alpha = self._merge_panel.get_cpd_alpha()
        self._merge_panel.set_cpd_status("Running CPD…")
        QApplication.processEvents()

        try:
            warped = cpd_align(
                source=aligned_pos,
                target=primary_pos,
                alpha=alpha,
            )
        except ImportError as exc:
            QMessageBox.critical(self, "pycpd missing", str(exc))
            self._merge_panel.set_cpd_status("pycpd not installed")
            return
        except Exception as exc:
            QMessageBox.critical(self, "CPD error", str(exc))
            self._merge_panel.set_cpd_status("CPD failed")
            return

        # Replace secondary cloud with the warped geometry; reset transform to identity
        new_pc = PointCloud(warped)
        self._secondary_pc = new_pc
        self._secondary_base_transform = np.eye(4, dtype=np.float32)
        self._merge_panel.reset_transform_spinboxes()
        self._merge_panel.set_secondary_loaded("(warped)", len(warped))
        self._merge_panel.set_cpd_status(f"Done — {len(warped):,} warped points")
        self._viewport.load_secondary(new_pc)

    def _merge_clouds(self) -> None:
        """Apply the current transform to the secondary cloud and concatenate into primary."""
        if self._pc is None or self._secondary_pc is None:
            return

        # Build total transform (base @ current manual spinbox state —
        # the viewport already has the composed transform, so read it directly)
        T = self._viewport._secondary_transform.astype(np.float64)

        # Transform secondary alive positions
        sec_pos = self._secondary_pc.positions[self._secondary_pc.alive_mask].astype(np.float64)
        ones = np.ones((len(sec_pos), 1), dtype=np.float64)
        sec_h = np.concatenate([sec_pos, ones], axis=1)
        merged_pos = (T @ sec_h.T).T[:, :3].astype(np.float32)

        sec_colors = self._secondary_pc.colors[self._secondary_pc.alive_mask]

        # Concatenate with primary alive points
        pri_pos    = self._pc.positions[self._pc.alive_mask]
        pri_colors = self._pc.colors[self._pc.alive_mask]

        all_pos    = np.concatenate([pri_pos, merged_pos], axis=0)
        all_colors = np.concatenate([pri_colors, sec_colors], axis=0)

        merged_pc = PointCloud(all_pos, all_colors)
        self._clear_secondary()
        self._on_load_finished(merged_pc)

    # ── Edit actions ──────────────────────────────────────────────────────────

    def _delete_selected(self) -> None:
        if self._pc is None:
            return
        indices = self._pc.selected_alive_indices()
        if len(indices) == 0:
            return
        cmd = DeleteCommand(self._pc, indices)
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
        # The command may have touched alive or colors; mark both dirty to be safe
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
        int(Qt.Key.Key_Space), int(Qt.Key.Key_Control),
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
                self._delete_selected()
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
