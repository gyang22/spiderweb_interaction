"""Soft-delete selected points via alive_mask."""

import numpy as np
from app.commands.command import AbstractCommand


class DeleteCommand(AbstractCommand):
    def __init__(self, pc, indices: np.ndarray, old_skeleton=None, new_skeleton=None, apply_skeleton_func=None):
        self._pc = pc
        self._indices = indices.copy()
        self._old_skeleton = old_skeleton
        self._new_skeleton = new_skeleton
        self._apply_skeleton = apply_skeleton_func

    @property
    def description(self) -> str:
        return f"Delete {len(self._indices)} points"

    def execute(self) -> None:
        self._pc.alive_mask[self._indices] = False
        # Clear selection for deleted points
        self._pc.selection_mask[self._indices] = False
        if self._apply_skeleton is not None and self._new_skeleton is not None:
            self._apply_skeleton(self._new_skeleton)

    def undo(self) -> None:
        self._pc.alive_mask[self._indices] = True
        if self._apply_skeleton is not None and self._old_skeleton is not None:
            self._apply_skeleton(self._old_skeleton)
