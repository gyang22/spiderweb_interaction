import numpy as np
from scipy.spatial import cKDTree
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from app.tools.base_tool import AbstractTool

# Screen-space radii (logical pixels)
CLICK_RADIUS2 = 15 * 15      # how close a click must be to pair an anchor
DELETE_RADIUS2 = 11 * 11     # how close a pick-click must be to delete an existing anchor
SNAP_RADIUS2 = 22 * 22       # how close the cursor must be to a cloud point to add an anchor
HOVER_SUBSAMPLE = 60000      # cap on cloud points projected each hover frame

PRIMARY_COLOR = QColor(255, 50, 50, 220)
SECONDARY_COLOR = QColor(0, 100, 255, 220)


class ManualAlignTool(AbstractTool):
    def __init__(self):
        super().__init__()
        self.primary_anchors = None
        self.secondary_anchors = None

        # Currently selected active anchor (if any)
        # Type: ('primary', idx) or ('secondary', idx)
        self.active_selection = None

        # Paired anchors: list of (primary_idx, secondary_idx)
        self.pairs = []

        # Screen projection cache for anchors
        self._proj_prim = np.empty((0, 2))
        self._proj_sec = np.empty((0, 2))
        self._viewport = None

        # ── Manual picking ────────────────────────────────────────────────────
        # 'pair' = click existing anchors to pair them (legacy behaviour)
        # 'pick' = click on either web to drop / remove anchors
        self.mode = 'pair'
        # Which web a click targets in pick mode: 'auto' | 'primary' | 'secondary'
        self.pick_target = 'auto'
        # Which cloud the viewport is currently editing (active renderer). Kept in
        # sync by the main window so projections use the right model transform.
        self._editing_secondary = False

        # Full-resolution cloud positions used for picking.
        self._prim_cloud = None          # (N, 3) primary points (world space)
        self._sec_cloud = None           # (M, 3) secondary points (secondary local space)
        # Down-sampled views used for smooth hover feedback on huge clouds.
        self._prim_hover = None
        self._prim_hover_src = None      # indices back into the full array
        self._sec_hover = None
        self._sec_hover_src = None

        # Live hover result, recomputed on mouse move while in pick mode.
        # ('add', kind, point3d, (sx, sy)) or ('remove', kind, anchor_idx, (sx, sy))
        self.hover = None
        # Last cursor position (logical px) so the overlay can re-snap the hover
        # marker every frame as the camera orbits, not just on mouse-move events.
        self._last_cursor = None

    # ── activation / data ─────────────────────────────────────────────────────

    def on_activate(self, viewport):
        self._viewport = viewport

    def on_deactivate(self, viewport):
        self.hover = None

    def set_anchors(self, primary_anchors: np.ndarray, secondary_anchors: np.ndarray):
        self.primary_anchors = primary_anchors
        self.secondary_anchors = secondary_anchors
        self.active_selection = None
        self.pairs = []

    def ensure_anchor_arrays(self):
        """Make sure the anchor arrays exist so picking can append to them."""
        if self.primary_anchors is None:
            self.primary_anchors = np.empty((0, 3), dtype=np.float32)
        if self.secondary_anchors is None:
            self.secondary_anchors = np.empty((0, 3), dtype=np.float32)

    def merge_anchors(self, primary_new: np.ndarray, secondary_new: np.ndarray):
        """Append new candidate anchors, skipping ones that duplicate an existing
        anchor. Existing anchors keep their indices so current pairs stay valid."""
        self.ensure_anchor_arrays()
        self.primary_anchors = self._append_unique(self.primary_anchors, primary_new)
        self.secondary_anchors = self._append_unique(self.secondary_anchors, secondary_new)

    @staticmethod
    def _append_unique(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
        if new is None or len(new) == 0:
            return existing
        new = np.asarray(new, dtype=np.float32)
        if existing is None or len(existing) == 0:
            return new.copy()
        # Tolerance scaled to the data so exact FPS re-adds are dropped while
        # genuinely distinct features (always far apart) are kept.
        extent = float(np.ptp(np.vstack((existing, new)), axis=0).max())
        tol = max(extent, 1e-6) * 1e-4
        tree = cKDTree(existing)
        dist, _ = tree.query(new)
        keep = new[dist > tol]
        if len(keep) == 0:
            return existing
        return np.vstack((existing, keep)).astype(np.float32)

    def set_clouds(self, primary_cloud: np.ndarray, secondary_cloud: np.ndarray):
        """Provide the raw point positions used to snap manual picks."""
        self._prim_cloud = None if primary_cloud is None else np.ascontiguousarray(
            primary_cloud, dtype=np.float32)
        self._sec_cloud = None if secondary_cloud is None else np.ascontiguousarray(
            secondary_cloud, dtype=np.float32)
        self._prim_hover, self._prim_hover_src = self._make_hover_view(self._prim_cloud)
        self._sec_hover, self._sec_hover_src = self._make_hover_view(self._sec_cloud)

    @staticmethod
    def _make_hover_view(cloud):
        if cloud is None or len(cloud) == 0:
            return None, None
        if len(cloud) <= HOVER_SUBSAMPLE:
            return cloud, np.arange(len(cloud))
        stride = int(np.ceil(len(cloud) / HOVER_SUBSAMPLE))
        src = np.arange(0, len(cloud), stride)
        return cloud[src], src

    def set_mode(self, mode: str):
        self.mode = mode
        self.active_selection = None
        self.hover = None

    def set_editing_secondary(self, editing_secondary: bool):
        self._editing_secondary = bool(editing_secondary)

    # Order the hotkey cycles through.
    _TARGET_CYCLE = ('auto', 'primary', 'secondary')

    def cycle_pick_target(self) -> str:
        """Advance the pick target to the next web and return the new value."""
        try:
            i = self._TARGET_CYCLE.index(self.pick_target)
        except ValueError:
            i = -1
        self.pick_target = self._TARGET_CYCLE[(i + 1) % len(self._TARGET_CYCLE)]
        self.hover = None
        return self.pick_target

    # ── projection helpers ─────────────────────────────────────────────────────

    def _mvp_for(self, viewport, secondary: bool):
        mvp = viewport.camera.get_mvp_matrix()
        # Each web is drawn either by the active renderer or the reference overlay,
        # depending on which cloud the user is currently editing. Project with the
        # SAME model transform the viewport uses to draw that web, so picking lines
        # up regardless of whether Primary or Secondary is the active cloud.
        cloud_is_active = (secondary == self._editing_secondary)
        T = viewport._active_transform if cloud_is_active else viewport._secondary_transform
        return (mvp @ T).astype(np.float32)

    def _project(self, points, mvp, viewport):
        """Project (N,3) points to (N,2) screen coords; points behind camera → -1000."""
        if points is None or len(points) == 0:
            return np.empty((0, 2))
        ones = np.ones((len(points), 1), dtype=np.float32)
        pos_h = np.concatenate([points.astype(np.float32), ones], axis=1)
        clip = pos_h @ mvp.T
        w = clip[:, 3]
        visible = w > 0.0
        screen = np.full((len(points), 2), -1000.0)
        if np.any(visible):
            clip_v = clip[visible]
            ndc = clip_v[:, :3] / clip_v[:, 3:4]
            W, H = float(viewport.width()), float(viewport.height())
            sx = (ndc[:, 0] + 1.0) * 0.5 * W
            sy = (1.0 - ndc[:, 1]) * 0.5 * H
            screen[visible] = np.stack([sx, sy], axis=1)
        return screen

    def project_anchors(self, viewport):
        """Project 3D anchors to 2D screen coordinates."""
        self._proj_prim = self._project(self.primary_anchors,
                                        self._mvp_for(viewport, False), viewport)
        self._proj_sec = self._project(self.secondary_anchors,
                                       self._mvp_for(viewport, True), viewport)

    @staticmethod
    def _closest(screen_pts, x, y):
        """Return (index, dist2) of the closest visible projected point to (x, y)."""
        if screen_pts is None or len(screen_pts) == 0:
            return -1, float('inf')
        visible = screen_pts[:, 0] > -500
        if not np.any(visible):
            return -1, float('inf')
        dx = screen_pts[:, 0] - x
        dy = screen_pts[:, 1] - y
        dist2 = dx * dx + dy * dy
        dist2[~visible] = np.inf
        idx = int(np.argmin(dist2))
        return idx, float(dist2[idx])

    # ── overlay ────────────────────────────────────────────────────────────────

    def draw_overlay(self, painter: QPainter) -> None:
        if self._viewport is not None:
            self.project_anchors(self._viewport)

        if self.primary_anchors is None or self.secondary_anchors is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw lines between paired anchors
        pen_line = QPen(QColor(255, 255, 0, 150), 2)
        painter.setPen(pen_line)
        for p_idx, s_idx in self.pairs:
            p_pt = self._proj_prim[p_idx]
            s_pt = self._proj_sec[s_idx]
            if p_pt[0] > -500 and s_pt[0] > -500:  # If both visible
                painter.drawLine(QPointF(p_pt[0], p_pt[1]), QPointF(s_pt[0], s_pt[1]))

        # Helper to draw circles
        def draw_circles(screen_pts, base_color, kind):
            for i, pt in enumerate(screen_pts):
                if pt[0] <= -500:
                    continue

                is_paired = any((kind == 'primary' and pair[0] == i) or
                                (kind == 'secondary' and pair[1] == i) for pair in self.pairs)
                is_active = self.active_selection == (kind, i)

                # Active = bright yellow. Paired = faded green. Base = solid red/blue
                if is_active:
                    color = QColor(255, 255, 0, 255)
                    radius = 8
                elif is_paired:
                    color = QColor(0, 255, 100, 200)
                    radius = 5
                else:
                    color = base_color
                    radius = 6

                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.drawEllipse(QPointF(pt[0], pt[1]), radius, radius)

        draw_circles(self._proj_sec, SECONDARY_COLOR, 'secondary')
        draw_circles(self._proj_prim, PRIMARY_COLOR, 'primary')

        if self.mode == 'pick':
            # Re-snap the hover marker every frame so it tracks the point under the
            # crosshair/cursor as the camera moves (fixes the marker drifting away
            # from the geometry in FPS mode and during orbit).
            if self._viewport is not None:
                cx, cy = self._cursor_xy(self._viewport)
                if cx is not None:
                    self._update_hover(cx, cy, self._viewport)
            self._draw_pick_overlay(painter)

    def _draw_pick_overlay(self, painter: QPainter) -> None:
        # Live cursor feedback: show what a click would do right now.
        if self.hover is not None:
            action, kind, _, (sx, sy) = self.hover
            web_color = PRIMARY_COLOR if kind == 'primary' else SECONDARY_COLOR
            if action == 'add':
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(web_color.red(), web_color.green(),
                                           web_color.blue(), 255), 2))
                painter.drawEllipse(QPointF(sx, sy), 10, 10)
                # small plus
                painter.drawLine(QPointF(sx - 5, sy), QPointF(sx + 5, sy))
                painter.drawLine(QPointF(sx, sy - 5), QPointF(sx, sy + 5))
            else:  # remove
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 60, 60, 255), 2))
                r = 9
                painter.drawEllipse(QPointF(sx, sy), r, r)
                painter.drawLine(QPointF(sx - 5, sy - 5), QPointF(sx + 5, sy + 5))
                painter.drawLine(QPointF(sx - 5, sy + 5), QPointF(sx + 5, sy - 5))

        # Persistent hint banner
        n_p = 0 if self.primary_anchors is None else len(self.primary_anchors)
        n_s = 0 if self.secondary_anchors is None else len(self.secondary_anchors)
        target_label = {'auto': 'Nearest', 'primary': 'Primary',
                        'secondary': 'Secondary'}.get(self.pick_target, self.pick_target)
        hint = (f"PICK MODE — click a web to add, click a dot to remove   "
                f"[P:{n_p}  S:{n_s}]   target: {target_label}  (E to switch)")
        painter.setFont(QFont("", 10))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(hint)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
        painter.drawRoundedRect(8, 8, tw + 16, 24, 5, 5)
        painter.setPen(QPen(QColor(255, 255, 255, 230)))
        painter.drawText(16, 25, hint)

    # ── mouse ──────────────────────────────────────────────────────────────────

    def mouse_press(self, event, viewport) -> None:
        if self.mode == 'pick':
            self._pick_press(event, viewport)
            return
        self._pair_press(event, viewport)

    def _pair_press(self, event, viewport) -> None:
        if self.primary_anchors is None or self.secondary_anchors is None:
            return

        x = event.position().x()
        y = event.position().y()

        p_idx, p_dist2 = self._closest(self._proj_prim, x, y)
        s_idx, s_dist2 = self._closest(self._proj_sec, x, y)

        clicked = None
        if p_dist2 < CLICK_RADIUS2 and p_dist2 <= s_dist2:
            clicked = ('primary', p_idx)
        elif s_dist2 < CLICK_RADIUS2:
            clicked = ('secondary', s_idx)

        if clicked:
            # If we already have one selected, and we click the OTHER kind, make a pair!
            if self.active_selection is not None and self.active_selection[0] != clicked[0]:
                prim_idx = clicked[1] if clicked[0] == 'primary' else self.active_selection[1]
                sec_idx = clicked[1] if clicked[0] == 'secondary' else self.active_selection[1]

                # Remove any existing pairs involving these nodes
                self.pairs = [pair for pair in self.pairs
                              if pair[0] != prim_idx and pair[1] != sec_idx]
                self.pairs.append((prim_idx, sec_idx))

                self.active_selection = None

                # Notify UI via manager -> viewport -> parent window
                if hasattr(viewport.window(), "on_manual_anchors_paired"):
                    viewport.window().on_manual_anchors_paired(len(self.pairs))
            else:
                self.active_selection = clicked
        else:
            self.active_selection = None

        viewport.update()

    def _pick_press(self, event, viewport) -> None:
        self.ensure_anchor_arrays()
        # In FPS mode the cursor is locked to the centre crosshair, so pick there.
        if getattr(viewport, '_fps_mode', False):
            x, y = viewport.width() / 2.0, viewport.height() / 2.0
        else:
            x = event.position().x()
            y = event.position().y()

        # Recompute hover for this exact position (mouse_move may lag a click).
        self._update_hover(x, y, viewport)

        if self.hover is None:
            viewport.update()
            return

        action, kind, payload, _ = self.hover
        if action == 'remove':
            self._remove_anchor(kind, payload)
        else:  # add
            if kind == 'primary':
                self.primary_anchors = np.vstack((self.primary_anchors,
                                                  payload.astype(np.float32)))
            else:
                self.secondary_anchors = np.vstack((self.secondary_anchors,
                                                    payload.astype(np.float32)))

        self._notify_counts(viewport)
        # Refresh hover after mutating anchors so the overlay stays accurate.
        self._update_hover(x, y, viewport)
        viewport.update()

    def mouse_move(self, event, viewport) -> None:
        if self.mode == 'pick':
            self._last_cursor = (event.position().x(), event.position().y())
            self._update_hover(*self._last_cursor, viewport)
        else:
            self.project_anchors(viewport)
        viewport.update()

    def mouse_release(self, event, viewport) -> None:
        pass

    # ── pick helpers ────────────────────────────────────────────────────────────

    def _cursor_xy(self, viewport):
        """Effective pick location: the centre crosshair in FPS mode, else the
        last known cursor position. Returns (x, y) or (None, None)."""
        if getattr(viewport, '_fps_mode', False):
            return viewport.width() / 2.0, viewport.height() / 2.0
        if self._last_cursor is not None:
            return self._last_cursor
        return None, None

    def _update_hover(self, x, y, viewport):
        """Decide what a click at (x, y) would do and cache it for overlay + press."""
        self.project_anchors(viewport)

        # 1) Over an existing anchor? → removal takes precedence.
        best_remove = None  # (dist2, kind, idx, sx, sy)
        for kind, proj in (('primary', self._proj_prim), ('secondary', self._proj_sec)):
            if self.pick_target != 'auto' and self.pick_target != kind:
                continue
            idx, d2 = self._closest(proj, x, y)
            if idx >= 0 and d2 < DELETE_RADIUS2:
                if best_remove is None or d2 < best_remove[0]:
                    best_remove = (d2, kind, idx, proj[idx][0], proj[idx][1])
        if best_remove is not None:
            _, kind, idx, sx, sy = best_remove
            self.hover = ('remove', kind, idx, (sx, sy))
            return

        # 2) Otherwise snap to the nearest cloud point on the target web(s).
        best_add = None  # (dist2, kind, point3d, sx, sy)
        candidates = []
        if self.pick_target in ('auto', 'primary'):
            candidates.append(('primary', self._prim_hover, self._prim_hover_src,
                               self._mvp_for(viewport, False)))
        if self.pick_target in ('auto', 'secondary'):
            candidates.append(('secondary', self._sec_hover, self._sec_hover_src,
                               self._mvp_for(viewport, True)))

        for kind, hover_pts, src, mvp in candidates:
            if hover_pts is None:
                continue
            proj = self._project(hover_pts, mvp, viewport)
            idx, d2 = self._closest(proj, x, y)
            if idx >= 0 and d2 < SNAP_RADIUS2:
                if best_add is None or d2 < best_add[0]:
                    full = self._prim_cloud if kind == 'primary' else self._sec_cloud
                    point3d = full[src[idx]]
                    best_add = (d2, kind, point3d, proj[idx][0], proj[idx][1])

        if best_add is not None:
            _, kind, point3d, sx, sy = best_add
            self.hover = ('add', kind, point3d, (sx, sy))
        else:
            self.hover = None

    def _remove_anchor(self, kind, idx):
        if kind == 'primary':
            self.primary_anchors = np.delete(self.primary_anchors, idx, axis=0)
            new_pairs = []
            for p, s in self.pairs:
                if p == idx:
                    continue
                new_pairs.append((p - 1 if p > idx else p, s))
            self.pairs = new_pairs
        else:
            self.secondary_anchors = np.delete(self.secondary_anchors, idx, axis=0)
            new_pairs = []
            for p, s in self.pairs:
                if s == idx:
                    continue
                new_pairs.append((p, s - 1 if s > idx else s))
            self.pairs = new_pairs
        self.active_selection = None

    def _notify_counts(self, viewport):
        win = viewport.window()
        if hasattr(win, "on_manual_anchors_changed"):
            win.on_manual_anchors_changed(
                0 if self.primary_anchors is None else len(self.primary_anchors),
                0 if self.secondary_anchors is None else len(self.secondary_anchors),
                len(self.pairs),
            )
