"""
GLViewport — QOpenGLWidget that integrates rendering, picking, camera, and tools.

Interaction modes
─────────────────
  Selection mode (default)
    • Left-click/drag  → active selection tool (click / box / lasso)
    • Right-click drag → camera look (position-based, releases on button-up)
    • WASD/Space/Ctrl  → fly movement

  FPS mode  (toggle with Tab)
    • Cursor hidden and locked to widget center
    • Any mouse movement → camera look (delta-based, no button needed)
    • WASD/Space/Ctrl  → fly movement
    • Tab again         → exit FPS mode, cursor restored
"""

import time
import numpy as np

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, QPoint, QEvent, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QCursor, QSurfaceFormat

from OpenGL.GL import (
    glClearColor, glClear, glEnable, glViewport, glGetError,
    GL_DEPTH_TEST, GL_PROGRAM_POINT_SIZE, GL_MULTISAMPLE,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_NO_ERROR,
)

from app.gl.camera import FlyCamera
from app.gl.renderer import PointCloudRenderer
from app.gl.picking import PickingRenderer
from app.gl.skeleton_renderer import SkeletonRenderer
from app.tools.tool_manager import ToolManager
from app import settings


class GLViewport(QOpenGLWidget):
    frame_rendered    = pyqtSignal(float)   # FPS
    selection_changed = pyqtSignal()
    fps_mode_changed  = pyqtSignal(bool)    # True = entered FPS mode
    skel_selection_changed = pyqtSignal()   # fired when skeleton node selection changes

    def __init__(self, parent=None):
        super().__init__(parent)

        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)

        self.camera       = FlyCamera()
        self.renderer: PointCloudRenderer | None = None
        self.picking: PickingRenderer | None = None
        self.skeleton_renderer: SkeletonRenderer | None = None
        self._secondary_renderer: PointCloudRenderer | None = None
        self._secondary_transform: np.ndarray = np.eye(4, dtype=np.float32)
        self.tool_manager = ToolManager(self)
        self.point_cloud  = None

        # Skeleton edit mode state
        self.skeleton_edit_mode: bool = False
        self._skel_nodes: np.ndarray | None = None      # (M, 3) CPU copy of skeleton nodes
        self._skel_selection: np.ndarray | None = None  # (M,) bool mask
        self._home_position     = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self._home_yaw          = 0.0
        self._home_pitch        = 0.0
        self._home_speed        = settings.CAMERA_MOVE_SPEED
        self._home_acceleration = settings.CAMERA_ACCELERATION

        self._last_time   = time.perf_counter()
        self._fps         = 0.0
        self._frame_count = 0

        # FPS mode state
        self._fps_mode = False
        self._fps_warp_pending = False   # ignore the first mouse event after cursor warp

        # 60Hz render loop
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ── public API ────────────────────────────────────────────────────────────

    def has_point_cloud(self) -> bool:
        return self.point_cloud is not None and self.renderer is not None

    def load_point_cloud(self, pc) -> None:
        self.point_cloud = pc
        self.makeCurrent()
        self.renderer.load(pc)

        alive  = pc.positions[pc.alive_mask]
        center = alive.mean(axis=0)
        extent = float(np.ptp(alive, axis=0).max())

        # Clamp extent so degenerate clouds (all-same-Z etc.) still work
        extent = max(extent, 1e-3)

        # Position camera outside the cloud, looking straight at its center
        # offset along +Z in world space, yaw=0 → forward=(0,0,-1) → looks back at center
        self.camera.position   = (center + np.array([0.0, 0.0, extent * 2.0],
                                                     dtype=np.float32))
        self.camera.yaw        = 0.0    # forward = (0,0,-1) → looking toward origin
        self.camera.pitch      = 0.0
        self.camera.move_speed = extent * 0.5   # sensible speed for this scale
        # Steady-state velocity = acceleration / damping, so set acceleration
        # to move_speed * damping so the camera actually reaches move_speed.
        # Without this, the default acceleration=40 gives v_ss=3.3 regardless
        # of cloud size — imperceptible for large clouds.
        self.camera.acceleration = self.camera.move_speed * self.camera.damping

        # Near/far scaled to the cloud so tiny and huge clouds both look right
        self.camera.near = extent * 1e-3
        self.camera.far  = extent * 1000.0

        # Save home position for camera reset
        self._home_position     = self.camera.position.copy()
        self._home_yaw          = self.camera.yaw
        self._home_pitch        = self.camera.pitch
        self._home_speed        = self.camera.move_speed
        self._home_acceleration = self.camera.acceleration

        self.doneCurrent()
        self.tool_manager.enable()
        self._timer.start()

    def on_selection_changed(self) -> None:
        if self.renderer:
            self.renderer.mark_selection_dirty()
        self.selection_changed.emit()
        self.update()

    def on_alive_changed(self) -> None:
        if self.renderer:
            self.renderer.mark_alive_dirty()
        self.selection_changed.emit()
        self.update()

    def on_colors_changed(self) -> None:
        if self.renderer:
            self.renderer.mark_colors_dirty()
        self.update()

    def reset_camera(self) -> None:
        """Return camera to the position it was at when the PCD was loaded."""
        self.camera.position     = self._home_position.copy()
        self.camera.yaw          = self._home_yaw
        self.camera.pitch        = self._home_pitch
        self.camera.move_speed   = self._home_speed
        self.camera.acceleration = self._home_acceleration
        self.camera.clear_keys()

    def set_point_size(self, size: int) -> None:
        if self.renderer:
            self.renderer.point_size = float(size)

    def set_skeleton(self, graph) -> None:
        """Upload a StrandGraph to the skeleton renderer and repaint."""
        if self.skeleton_renderer is None:
            return
        self._skel_nodes = graph.nodes.copy() if len(graph.nodes) > 0 else None
        self.makeCurrent()
        self.skeleton_renderer.upload(graph)
        self.doneCurrent()
        self.update()

    def clear_skeleton(self) -> None:
        """Remove the skeleton overlay."""
        if self.skeleton_renderer:
            self.skeleton_renderer.clear()
        self._skel_nodes = None
        self._skel_selection = None
        self.update()

    # ── skeleton edit mode ────────────────────────────────────────────────────

    def has_selectable(self) -> bool:
        """True when there is something to select (PCD or skeleton nodes in edit mode)."""
        if self.skeleton_edit_mode:
            return self._skel_nodes is not None and len(self._skel_nodes) > 0
        return self.has_point_cloud()

    def reset_skel_selection(self) -> None:
        """Allocate/reset the skeleton selection mask to all-False."""
        n = len(self._skel_nodes) if self._skel_nodes is not None else 0
        self._skel_selection = np.zeros(n, dtype=bool) if n > 0 else None
        self._upload_skel_selection()
        self.skel_selection_changed.emit()

    def _upload_skel_selection(self) -> None:
        if self.skeleton_renderer is None or self._skel_selection is None:
            return
        self.makeCurrent()
        self.skeleton_renderer.upload_selection(self._skel_selection)
        self.doneCurrent()
        self.update()

    def screen_project_skeleton_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Project skeleton nodes to logical widget screen coords.

        Returns
        -------
        screen_xy    : (M, 2) float32  — logical pixel coords (x right, y down)
        node_indices : (M,) int32      — index into skeleton node array for each row
        """
        nodes = self._skel_nodes
        if nodes is None or len(nodes) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int32)

        mvp  = self.camera.get_mvp_matrix()
        ones = np.ones((len(nodes), 1), dtype=np.float32)
        pos_h = np.concatenate([nodes.astype(np.float32), ones], axis=1)
        clip  = pos_h @ mvp.T.astype(np.float32)

        w       = clip[:, 3]
        visible = w > 0.0
        clip    = clip[visible]
        idx     = np.where(visible)[0].astype(np.int32)

        if len(clip) == 0:
            return np.empty((0, 2), dtype=np.float32), idx

        w   = clip[:, 3:4]
        ndc = clip[:, :3] / w
        W, H = float(self.width()), float(self.height())
        sx = (ndc[:, 0] + 1.0) * 0.5 * W
        sy = (1.0 - ndc[:, 1]) * 0.5 * H
        return np.stack([sx, sy], axis=1).astype(np.float32), idx

    # ── unified selection dispatch ─────────────────────────────────────────────

    def project_selectable(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (screen_xy, indices) for the current selection target."""
        if self.skeleton_edit_mode:
            return self.screen_project_skeleton_nodes()
        return self.screen_project_alive()

    def apply_click_selection(self, x: int, y: int, add: bool) -> None:
        """Handle a click-select at logical coords (x, y)."""
        if self.skeleton_edit_mode:
            if self._skel_selection is None:
                return
            idx = self._pick_skeleton_node(x, y)
            if idx >= 0:
                if add:
                    self._skel_selection[idx] = not self._skel_selection[idx]
                else:
                    self._skel_selection[:] = False
                    self._skel_selection[idx] = True
            elif not add:
                self._skel_selection[:] = False
            self._upload_skel_selection()
            self.skel_selection_changed.emit()
        else:
            if not self.has_point_cloud():
                return
            index = self.do_picking(x, y)
            pc = self.point_cloud
            if index >= 0 and index < pc.total_count and pc.alive_mask[index]:
                if add:
                    pc.selection_mask[index] = not pc.selection_mask[index]
                else:
                    pc.deselect_all()
                    pc.selection_mask[index] = True
            else:
                if not add:
                    pc.deselect_all()
            self.on_selection_changed()

    def apply_region_selection(self, indices: np.ndarray, add: bool) -> None:
        """Apply a region selection result (box or lasso)."""
        if self.skeleton_edit_mode:
            if self._skel_selection is None:
                return
            if not add:
                self._skel_selection[:] = False
            self._skel_selection[indices] = True
            self._upload_skel_selection()
            self.skel_selection_changed.emit()
        else:
            if not self.has_point_cloud():
                return
            self.point_cloud.select_indices(indices, add=add)
            self.on_selection_changed()

    def _pick_skeleton_node(self, x: int, y: int, radius: float = 15.0) -> int:
        """Return index of nearest skeleton node within `radius` screen pixels, or -1."""
        screen_xy, node_idx = self.screen_project_skeleton_nodes()
        if len(screen_xy) == 0:
            return -1
        dx = screen_xy[:, 0] - x
        dy = screen_xy[:, 1] - y
        dist2 = dx * dx + dy * dy
        nearest = int(dist2.argmin())
        if dist2[nearest] <= radius * radius:
            return int(node_idx[nearest])
        return -1

    def reload_point_cloud(self, pc) -> None:
        """Re-upload a point cloud to GPU without resetting the camera or tools."""
        self.point_cloud = pc
        self.makeCurrent()
        self.renderer.load(pc)
        self.doneCurrent()
        self.update()

    def load_reference(self, pc, transform: np.ndarray | None = None) -> None:
        """
        Load a cloud into the reference overlay renderer (dim tint, non-interactive).

        `transform` is the 4×4 model transform to apply when drawing this overlay
        (default identity).  Pass the alignment transform here so the reference
        shows where the secondary *would* land relative to the primary.
        """
        if self._secondary_renderer is None:
            return
        from app.data.point_cloud import PointCloud

        alive_mask = pc.alive_mask
        positions  = pc.positions[alive_mask].copy()
        # Original colors, dimmed and slightly desaturated so it reads as "inactive"
        orig_colors = pc.colors[alive_mask].copy()
        # Blend toward a neutral blue-grey to signal "not the active cloud"
        tint   = np.array([0.35, 0.55, 0.75, 0.55], dtype=np.float32)
        colors = orig_colors * 0.45 + tint * 0.55
        colors[:, 3] = 0.55        # fixed alpha for clarity
        colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

        tinted = PointCloud(positions, colors)

        self.makeCurrent()
        self._secondary_renderer.load(tinted)
        self._secondary_transform = np.asarray(
            transform if transform is not None else np.eye(4), dtype=np.float32
        )
        self.doneCurrent()
        self.update()

    def update_reference_transform(self, T: np.ndarray) -> None:
        """Set the 4×4 model transform applied to the reference overlay cloud."""
        self._secondary_transform = np.asarray(T, dtype=np.float32)
        self.update()

    def clear_reference(self) -> None:
        """Remove the reference overlay."""
        if self._secondary_renderer:
            self._secondary_renderer.clear()
        self._secondary_transform = np.eye(4, dtype=np.float32)
        self.update()

    # Keep legacy names as thin aliases so nothing else breaks during transition
    def load_secondary(self, pc) -> None:
        self.load_reference(pc)

    def update_secondary_transform(self, T: np.ndarray) -> None:
        self.update_reference_transform(T)

    def clear_secondary(self) -> None:
        self.clear_reference()

    # ── FPS mode ──────────────────────────────────────────────────────────────

    def toggle_fps_mode(self) -> None:
        if self._fps_mode:
            self._exit_fps_mode()
        else:
            self._enter_fps_mode()

    def _enter_fps_mode(self) -> None:
        self._fps_mode = True
        self._fps_warp_pending = True    # discard the move event generated by setPos below
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._recenter_cursor()
        self.fps_mode_changed.emit(True)

    def _exit_fps_mode(self) -> None:
        self._fps_mode = False
        self.unsetCursor()
        self.camera.stop_look()
        self.fps_mode_changed.emit(False)

    def _recenter_cursor(self) -> None:
        center_local  = QPoint(self.width() // 2, self.height() // 2)
        center_global = self.mapToGlobal(center_local)
        QCursor.setPos(center_global)

    def _widget_center_global(self) -> QPoint:
        return self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2))

    # ── picking helpers ───────────────────────────────────────────────────────

    def _fbo_scale(self) -> tuple[float, float]:
        """
        Scale factors from logical widget coords (Qt mouse events) → physical FBO pixels.
        Derived from actual FBO dimensions rather than devicePixelRatio(), so it's
        correct regardless of whether Qt passes logical or physical dims to resizeGL.
        """
        fw = self.picking.fbo_width  if self.picking else self.width()
        fh = self.picking.fbo_height if self.picking else self.height()
        return fw / max(1, self.width()), fh / max(1, self.height())

    def do_picking(self, x: int, y: int) -> int:
        """Single-point pick. x/y are logical widget coords (from Qt mouse events)."""
        if not self.has_point_cloud():
            return -1
        sx, sy = self._fbo_scale()
        px = int(round(x * sx))
        py = int(round(y * sy))
        ph = self.picking.fbo_height
        self.makeCurrent()
        self.picking.render(self.renderer, self.camera.get_mvp_matrix())
        idx = self.picking.read_pixel(px, ph - py - 1)
        self.doneCurrent()
        return idx

    def do_region_picking(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Box pick. All coords are logical widget pixels."""
        if not self.has_point_cloud():
            return np.array([], dtype=np.int32)
        sx, sy = self._fbo_scale()
        ph = self.picking.fbo_height
        x1p = int(round(x1 * sx));  x2p = int(round(x2 * sx))
        y1p = int(round(y1 * sy));  y2p = int(round(y2 * sy))
        self.makeCurrent()
        self.picking.render(self.renderer, self.camera.get_mvp_matrix())
        rx1, rx2 = sorted([x1p, x2p])
        ry1 = ph - max(y1p, y2p) - 1
        ry2 = ph - min(y1p, y2p) - 1
        indices = self.picking.read_region(rx1, ry1, rx2 - rx1 + 1, ry2 - ry1 + 1)
        self.doneCurrent()
        return indices

    def do_full_picking(self) -> np.ndarray | None:
        """Full-FBO pick for lasso. Returns physical-pixel index map (row 0 = GL bottom)."""
        if not self.has_point_cloud():
            return None
        self.makeCurrent()
        self.picking.render(self.renderer, self.camera.get_mvp_matrix())
        result = self.picking.read_full()
        self.doneCurrent()
        return result

    def screen_project_alive(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Project all alive points to logical widget screen coordinates.

        Returns
        -------
        screen_xy : (M, 2) float32 — logical pixel coords (x right, y down)
        alive_indices : (M,) int32 — original point-cloud indices for each row
        """
        pc = self.point_cloud
        alive_idx = np.where(pc.alive_mask)[0].astype(np.int32)
        positions = pc.positions[alive_idx]   # (M, 3)

        mvp = self.camera.get_mvp_matrix()    # (4, 4) row-major

        # Homogeneous positions: (M, 4)
        ones = np.ones((len(positions), 1), dtype=np.float32)
        pos_h = np.concatenate([positions, ones], axis=1)

        # clip = pos_h @ mvp.T  →  (M, 4)
        clip = pos_h @ mvp.T.astype(np.float32)

        # Discard points behind camera (w <= 0)
        w = clip[:, 3]
        visible = w > 0.0
        clip = clip[visible]
        alive_idx = alive_idx[visible]

        w = clip[:, 3:4]
        ndc = clip[:, :3] / w          # (M', 3)  NDC in [-1, 1]

        W = float(self.width())
        H = float(self.height())
        sx = (ndc[:, 0] + 1.0) * 0.5 * W
        sy = (1.0 - ndc[:, 1]) * 0.5 * H

        screen_xy = np.stack([sx, sy], axis=1).astype(np.float32)
        return screen_xy, alive_idx

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def initializeGL(self) -> None:
        # Drain stale GL errors left by Qt's internal context setup —
        # PyOpenGL_accelerate reports them on the very first call otherwise
        while glGetError() != GL_NO_ERROR:
            pass

        glClearColor(*settings.BACKGROUND_COLOR)
        glEnable(GL_DEPTH_TEST)
        try:
            glEnable(GL_PROGRAM_POINT_SIZE)
        except Exception:
            pass
        try:
            glEnable(GL_MULTISAMPLE)
        except Exception:
            pass

        self.renderer = PointCloudRenderer()
        self.renderer.initialize()

        self.picking = PickingRenderer()
        self.picking.initialize()

        self.skeleton_renderer = SkeletonRenderer()
        self.skeleton_renderer.initialize()

        self._secondary_renderer = PointCloudRenderer()
        self._secondary_renderer.initialize()

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)
        self.camera.aspect = w / max(1, h)
        if self.picking and w > 0 and h > 0:
            self.makeCurrent()
            self.picking.resize(w, h)

    def paintGL(self) -> None:
        now = time.perf_counter()
        dt  = min(now - self._last_time, 0.1)
        self._last_time = now

        self.camera.tick(dt)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        mvp = self.camera.get_mvp_matrix()
        if self.renderer and self.renderer.has_data:
            self.renderer.draw(mvp)
        if self._secondary_renderer and self._secondary_renderer.has_data:
            sec_mvp = (mvp @ self._secondary_transform).astype(np.float32)
            self._secondary_renderer.draw(sec_mvp)
        if self.skeleton_renderer and self.skeleton_renderer.has_data:
            self.skeleton_renderer.draw(mvp)

        self._frame_count += 1
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        if self._frame_count % settings.FPS_DISPLAY_INTERVAL == 0:
            self.frame_rendered.emit(self._fps)

    # ── 2D overlay ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        self.tool_manager.active_tool.draw_overlay(painter)
        if self._fps_mode:
            self._draw_crosshair(painter)
        painter.end()
        self.makeCurrent()

    def _draw_crosshair(self, painter: QPainter) -> None:
        cx = self.width() // 2
        cy = self.height() // 2
        gap, arm = 4, 12
        pen = QPen(QColor(0, 0, 0, 160), 3)
        painter.setPen(pen)
        painter.drawLine(cx - arm, cy, cx - gap, cy)
        painter.drawLine(cx + gap, cy, cx + arm, cy)
        painter.drawLine(cx, cy - arm, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + arm)
        pen = QPen(QColor(255, 255, 255, 230), 1)
        painter.setPen(pen)
        painter.drawLine(cx - arm, cy, cx - gap, cy)
        painter.drawLine(cx + gap, cy, cx + arm, cy)
        painter.drawLine(cx, cy - arm, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + arm)

    # ── keyboard ──────────────────────────────────────────────────────────────

    def event(self, event) -> bool:
        # Override at event() level — keyPressEvent never sees Tab because Qt
        # intercepts it for focus-tab navigation before reaching keyPressEvent.
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab:
            self.toggle_fps_mode()
            return True
        return super().event(event)

    # Camera movement keys handled directly here (primary path).
    # The app-level event filter in MainWindow is a backup for cases where
    # a non-NoFocus widget (e.g. the point-size spinbox) steals focus.
    _CAM_KEYS = frozenset([
        int(Qt.Key.Key_W), int(Qt.Key.Key_A),
        int(Qt.Key.Key_S), int(Qt.Key.Key_D),
        int(Qt.Key.Key_Space), int(Qt.Key.Key_Shift),
    ])

    def keyPressEvent(self, event) -> None:
        key = int(event.key())
        if not event.isAutoRepeat() and key in self._CAM_KEYS:
            self.camera.key_down(key)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        key = int(event.key())
        if not event.isAutoRepeat() and key in self._CAM_KEYS:
            self.camera.key_up(key)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:
        """Exit FPS mode and clear ghost keys whenever the window loses focus."""
        if self._fps_mode:
            self._exit_fps_mode()
        self.camera.clear_keys()
        super().focusOutEvent(event)

    # ── mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if self._fps_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                # Temporarily unhide cursor so the user can drag a selection
                self.unsetCursor()
                self.tool_manager.active_tool.mouse_press(event, self)
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.camera.start_look(event.position().x(), event.position().y())
        else:
            self.tool_manager.active_tool.mouse_press(event, self)

    def mouseMoveEvent(self, event) -> None:
        if self._fps_mode:
            # Left button held → selection drag; skip FPS look while dragging
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.tool_manager.active_tool.mouse_move(event, self)
                return
            # Discard the synthetic move event produced by our own cursor warp
            if self._fps_warp_pending:
                self._fps_warp_pending = False
                return
            # Delta from the locked center → apply to camera → re-center cursor
            center = self._widget_center_global()
            pos    = event.globalPosition().toPoint()
            dx     = pos.x() - center.x()
            dy     = pos.y() - center.y()
            if dx != 0 or dy != 0:
                self.camera.apply_delta(dx, dy)
                self._fps_warp_pending = True   # next event will be our own re-center
                QCursor.setPos(center)
            return

        if event.buttons() & Qt.MouseButton.RightButton:
            self.camera.update_look(event.position().x(), event.position().y())
        else:
            self.tool_manager.active_tool.mouse_move(event, self)

    def mouseReleaseEvent(self, event) -> None:
        if self._fps_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self.tool_manager.active_tool.mouse_release(event, self)
                # Re-lock cursor and absorb the re-center warp
                self.setCursor(Qt.CursorShape.BlankCursor)
                self._fps_warp_pending = True
                self._recenter_cursor()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.camera.stop_look()
        else:
            self.tool_manager.active_tool.mouse_release(event, self)
