"""Command that replaces the active point cloud (downsample, merge, etc.)."""

from app.commands.command import AbstractCommand


class ReplaceCloudCommand(AbstractCommand):
    """
    Stores a before/after snapshot of the active PointCloud and calls a
    caller-supplied `apply_fn(pc)` to swap it in on execute/redo/undo.

    The caller performs the actual work BEFORE pushing the command, so
    `execute()` on first push just re-applies the new state (a no-op if
    the command is pushed via `push_done()`).  On undo/redo the full swap
    is applied.
    """

    def __init__(self, old_pc, new_pc, apply_fn, description: str = "replace"):
        self._old      = old_pc
        self._new      = new_pc
        self._apply    = apply_fn   # callable(PointCloud) — updates app + viewport
        self._desc     = description

    def execute(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)

    @property
    def description(self) -> str:
        return self._desc
