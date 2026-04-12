"""Velocity-based first-person fly camera."""

import numpy as np
from PyQt6.QtCore import Qt
from app import settings

# Pre-compute plain ints for all movement keys.
# In PyQt6, Qt.Key members are strict enums that do NOT compare equal to ints,
# so we must normalise to int before storing in / comparing against sets.
_K_W     = int(Qt.Key.Key_W)
_K_A     = int(Qt.Key.Key_A)
_K_S     = int(Qt.Key.Key_S)
_K_D     = int(Qt.Key.Key_D)
_K_SPACE = int(Qt.Key.Key_Space)
_K_SHIFT = int(Qt.Key.Key_Shift)


class FlyCamera:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.yaw   = 0.0   # degrees; 0 → looking toward -Z (OpenGL default)
        self.pitch = 0.0   # degrees; clamped to ±89

        self.aspect = 1.0

        # near/far set per-cloud in viewport.load_point_cloud so they scale
        # with the point cloud's extent and never clip geometry
        self.near = settings.CAMERA_NEAR
        self.far  = settings.CAMERA_FAR

        self.move_speed        = settings.CAMERA_MOVE_SPEED
        self.acceleration      = settings.CAMERA_ACCELERATION
        self.damping           = settings.CAMERA_DAMPING
        self.mouse_sensitivity = settings.CAMERA_MOUSE_SENSITIVITY

        self._velocity   = np.zeros(3, dtype=np.float32)
        self._held_keys: set[int] = set()   # always plain ints

        # Position-based look (right-click drag in selection mode)
        self._looking      = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

    # ── key input ─────────────────────────────────────────────────────────────

    def key_down(self, key) -> None:
        self._held_keys.add(int(key))   # int() normalises Qt.Key enum → int

    def key_up(self, key) -> None:
        self._held_keys.discard(int(key))

    def clear_keys(self) -> None:
        """Call when the viewport loses focus so held keys don't ghost."""
        self._held_keys.clear()
        self._velocity[:] = 0.0

    # ── mouse look — position-based (right-drag in selection mode) ────────────

    def start_look(self, x: float, y: float) -> None:
        self._looking = True
        self._last_mouse_x = x
        self._last_mouse_y = y

    def update_look(self, x: float, y: float) -> None:
        if not self._looking:
            return
        dx = x - self._last_mouse_x
        dy = y - self._last_mouse_y
        self._last_mouse_x = x
        self._last_mouse_y = y
        self.apply_delta(dx, dy)

    def stop_look(self) -> None:
        self._looking = False

    # ── mouse look — delta-based (FPS locked-cursor mode) ─────────────────────

    def apply_delta(self, dx: float, dy: float) -> None:
        """Apply a raw pixel delta directly (no position tracking needed)."""
        self.yaw   += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity   # invert Y: drag down = look down
        self.pitch  = float(np.clip(self.pitch, -89.0, 89.0))

    # ── per-frame update ──────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        forward = self._forward()
        right   = self._right()

        desired = np.zeros(3, dtype=np.float32)
        # All comparisons use module-level int constants — no enum/int mismatch
        if _K_W     in self._held_keys: desired += forward
        if _K_S     in self._held_keys: desired -= forward
        if _K_A     in self._held_keys: desired -= right
        if _K_D     in self._held_keys: desired += right
        if _K_SPACE in self._held_keys: desired[1] += 1.0
        if _K_SHIFT in self._held_keys: desired[1] -= 1.0

        norm = np.linalg.norm(desired)
        if norm > 1e-6:
            desired /= norm

        self._velocity += desired * self.acceleration * dt
        self._velocity *= max(0.0, 1.0 - self.damping * dt)

        speed = np.linalg.norm(self._velocity)
        if speed > self.move_speed:
            self._velocity *= self.move_speed / speed

        self.position += self._velocity * dt

    # ── direction vectors ─────────────────────────────────────────────────────

    def _forward(self) -> np.ndarray:
        yaw_r   = np.radians(self.yaw)
        pitch_r = np.radians(self.pitch)
        x =  float(np.cos(pitch_r) * np.sin(yaw_r))
        y =  float(np.sin(pitch_r))
        z = -float(np.cos(pitch_r) * np.cos(yaw_r))
        return np.array([x, y, z], dtype=np.float32)

    def _right(self) -> np.ndarray:
        fwd      = self._forward()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r        = np.cross(fwd, world_up).astype(np.float32)
        norm     = np.linalg.norm(r)
        return r / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # ── matrices ──────────────────────────────────────────────────────────────

    def get_view_matrix(self) -> np.ndarray:
        fwd   = self._forward()
        right = self._right()
        up    = np.cross(right, fwd).astype(np.float32)

        view = np.array([
            [ right[0],  right[1],  right[2], -np.dot(right, self.position)],
            [ up[0],     up[1],     up[2],    -np.dot(up,    self.position)],
            [-fwd[0],   -fwd[1],   -fwd[2],    np.dot(fwd,   self.position)],
            [ 0.0,       0.0,       0.0,        1.0],
        ], dtype=np.float32)
        return view

    def get_projection_matrix(self) -> np.ndarray:
        f  = 1.0 / np.tan(np.radians(settings.CAMERA_FOV) / 2.0)
        n, far = self.near, self.far

        proj = np.array([
            [f / self.aspect, 0.0,  0.0,                  0.0                ],
            [0.0,             f,    0.0,                  0.0                ],
            [0.0,             0.0, (far + n) / (n - far), (2*far*n) / (n-far)],
            [0.0,             0.0, -1.0,                  0.0                ],
        ], dtype=np.float32)
        return proj

    def get_mvp_matrix(self) -> np.ndarray:
        """Row-major MVP. Upload with transpose=True to OpenGL."""
        return self.get_projection_matrix() @ self.get_view_matrix()
