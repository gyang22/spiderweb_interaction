import numpy as np
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
from app.tools.base_tool import AbstractTool

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
        
        # Screen projection cache
        self._proj_prim = np.empty((0, 2))
        self._proj_sec = np.empty((0, 2))
        self._viewport = None

    def on_activate(self, viewport):
        self._viewport = viewport

    def set_anchors(self, primary_anchors: np.ndarray, secondary_anchors: np.ndarray):
        self.primary_anchors = primary_anchors
        self.secondary_anchors = secondary_anchors
        self.active_selection = None
        self.pairs = []
        
    def project_anchors(self, viewport):
        """Project 3D anchors to 2D screen coordinates."""
        def project(anchors, transform=None):
            if anchors is None or len(anchors) == 0:
                return np.empty((0, 2))
            mvp = viewport.camera.get_mvp_matrix()
            if transform is not None:
                mvp = (mvp @ transform).astype(np.float32)
            
            ones = np.ones((len(anchors), 1), dtype=np.float32)
            pos_h = np.concatenate([anchors.astype(np.float32), ones], axis=1)
            clip = pos_h @ mvp.T.astype(np.float32)
            
            w = clip[:, 3]
            visible = w > 0.0
            
            screen = np.full((len(anchors), 2), -1000.0) # Hidden by default
            if np.any(visible):
                clip_v = clip[visible]
                ndc = clip_v[:, :3] / clip_v[:, 3:4]
                W, H = float(viewport.width()), float(viewport.height())
                sx = (ndc[:, 0] + 1.0) * 0.5 * W
                sy = (1.0 - ndc[:, 1]) * 0.5 * H
                screen[visible] = np.stack([sx, sy], axis=1)
            return screen

        self._proj_prim = project(self.primary_anchors)
        self._proj_sec = project(self.secondary_anchors, viewport._secondary_transform)

    def draw_overlay(self, painter: QPainter) -> None:
        if self._viewport is not None:
            self.project_anchors(self._viewport)
            
        if self.primary_anchors is None or self.secondary_anchors is None:
            return
            
        # Draw lines between paired anchors
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen_line = QPen(QColor(255, 255, 0, 150), 2)
        painter.setPen(pen_line)
        for p_idx, s_idx in self.pairs:
            p_pt = self._proj_prim[p_idx]
            s_pt = self._proj_sec[s_idx]
            if p_pt[0] > -500 and s_pt[0] > -500: # If both visible
                painter.drawLine(QPointF(p_pt[0], p_pt[1]), QPointF(s_pt[0], s_pt[1]))
        
        # Helper to draw circles
        def draw_circles(screen_pts, base_color, kind):
            for i, pt in enumerate(screen_pts):
                if pt[0] <= -500: continue
                
                is_paired = any((kind == 'primary' and pair[0] == i) or (kind == 'secondary' and pair[1] == i) for pair in self.pairs)
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
                
        draw_circles(self._proj_sec, QColor(0, 100, 255, 220), 'secondary')
        draw_circles(self._proj_prim, QColor(255, 50, 50, 220), 'primary')

    def mouse_press(self, event, viewport) -> None:
        if self.primary_anchors is None or self.secondary_anchors is None:
            return
            
        x = event.position().x()
        y = event.position().y()
        
        # Find clicked anchor
        def find_closest(screen_pts):
            if len(screen_pts) == 0: return -1, float('inf')
            dx = screen_pts[:, 0] - x
            dy = screen_pts[:, 1] - y
            dist2 = dx*dx + dy*dy
            idx = np.argmin(dist2)
            return idx, dist2[idx]
            
        p_idx, p_dist2 = find_closest(self._proj_prim)
        s_idx, s_dist2 = find_closest(self._proj_sec)
        
        CLICK_RADIUS2 = 15 * 15
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
                self.pairs = [pair for pair in self.pairs if pair[0] != prim_idx and pair[1] != sec_idx]
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

    def mouse_move(self, event, viewport) -> None:
        self.project_anchors(viewport)
        viewport.update()

    def mouse_release(self, event, viewport) -> None:
        pass
