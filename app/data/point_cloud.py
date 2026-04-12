import numpy as np
from app import settings


class PointCloud:
    """Central data structure. Uses mask arrays so deletions never resize GPU buffers."""

    def __init__(self, positions: np.ndarray, colors: np.ndarray | None = None):
        """
        positions: (N, 3) float32 XYZ
        colors:    (N, 4) float32 RGBA [0, 1] — or None for default gray
        """
        n = len(positions)
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)

        if colors is not None:
            self.colors = np.ascontiguousarray(colors, dtype=np.float32)
            # Ensure 4 channels
            if self.colors.shape[1] == 3:
                alpha = np.ones((n, 1), dtype=np.float32)
                self.colors = np.concatenate([self.colors, alpha], axis=1)
        else:
            r, g, b, a = settings.DEFAULT_POINT_COLOR
            self.colors = np.full((n, 4), [r, g, b, a], dtype=np.float32)

        self.alive_mask = np.ones(n, dtype=bool)
        self.selection_mask = np.zeros(n, dtype=bool)

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def total_count(self) -> int:
        return len(self.positions)

    @property
    def alive_count(self) -> int:
        return int(self.alive_mask.sum())

    @property
    def selected_count(self) -> int:
        return int((self.selection_mask & self.alive_mask).sum())

    # ── selection helpers ─────────────────────────────────────────────────────

    def select_indices(self, indices: np.ndarray, add: bool = False) -> None:
        if not add:
            self.selection_mask[:] = False
        self.selection_mask[indices] = True

    def deselect_all(self) -> None:
        self.selection_mask[:] = False

    def invert_selection(self) -> None:
        self.selection_mask[self.alive_mask] = ~self.selection_mask[self.alive_mask]

    def selected_alive_indices(self) -> np.ndarray:
        """Return indices of points that are both selected and alive."""
        return np.where(self.selection_mask & self.alive_mask)[0]

    # ── alive float array (uploaded to GPU) ───────────────────────────────────

    def alive_float(self) -> np.ndarray:
        """(N,) float32 — 1.0 = alive, 0.0 = deleted — for GPU VBO."""
        return self.alive_mask.astype(np.float32)

    def selection_float(self) -> np.ndarray:
        """(N,) float32 — 1.0 = selected, 0.0 = not — for GPU VBO."""
        return self.selection_mask.astype(np.float32)
