"""Generic undoable state swap.

Captures an `old` and `new` snapshot of some state plus an `apply_fn` that
installs a snapshot. On execute/redo it applies `new`; on undo it applies `old`.
Used for lightweight, non-cloud state (alignment transform, manual anchors) where
a full command subclass would be overkill.
"""

from app.commands.command import AbstractCommand


class StateCommand(AbstractCommand):
    def __init__(self, apply_fn, old_state, new_state, description: str = "edit"):
        self._apply = apply_fn      # callable(state) → installs it
        self._old   = old_state
        self._new   = new_state
        self._desc  = description

    def execute(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)

    @property
    def description(self) -> str:
        return self._desc
