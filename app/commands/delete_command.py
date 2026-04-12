"""Soft-delete selected points via alive_mask."""

import numpy as np
from app.commands.command import AbstractCommand


class DeleteCommand(AbstractCommand):
    def __init__(self, pc, indices: np.ndarray):
        self._pc = pc
        self._indices = indices.copy()

    @property
    def description(self) -> str:
        return f"Delete {len(self._indices)} points"

    def execute(self) -> None:
        self._pc.alive_mask[self._indices] = False
        # Clear selection for deleted points
        self._pc.selection_mask[self._indices] = False

    def undo(self) -> None:
        self._pc.alive_mask[self._indices] = True
