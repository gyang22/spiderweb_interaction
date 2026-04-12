"""Recolor selected points, with undo by restoring snapshot of old colors."""

import numpy as np
from app.commands.command import AbstractCommand


class ColorCommand(AbstractCommand):
    def __init__(self, pc, indices: np.ndarray, new_color: tuple[float, float, float, float]):
        self._pc = pc
        self._indices = indices.copy()
        self._old_colors = pc.colors[indices].copy()  # snapshot only selected rows
        self._new_color = np.array(new_color, dtype=np.float32)

    @property
    def description(self) -> str:
        return f"Color {len(self._indices)} points"

    def execute(self) -> None:
        self._pc.colors[self._indices] = self._new_color

    def undo(self) -> None:
        self._pc.colors[self._indices] = self._old_colors
