"""Undoable skeleton mutation (re-extract connectivity or delete nodes)."""

from app.commands.command import AbstractCommand
from app.data.strand_graph import StrandGraph


class EditSkeletonCommand(AbstractCommand):
    def __init__(
        self,
        old_skeleton: StrandGraph,
        new_skeleton: StrandGraph,
        apply_func,          # callable(StrandGraph) → None
        description: str = "Edit skeleton",
    ) -> None:
        self._old = old_skeleton
        self._new = new_skeleton
        self._apply = apply_func
        self._desc  = description

    @property
    def description(self) -> str:
        return self._desc

    def execute(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)
