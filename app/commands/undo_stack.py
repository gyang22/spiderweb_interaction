"""Undo/redo stack with configurable depth cap."""

from collections import deque
from PyQt6.QtCore import QObject, pyqtSignal
from app.commands.command import AbstractCommand
from app import settings


class UndoStack(QObject):
    changed = pyqtSignal()  # emitted after every push/undo/redo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._undo: deque[AbstractCommand] = deque(maxlen=settings.UNDO_MAX_DEPTH)
        self._redo: deque[AbstractCommand] = deque(maxlen=settings.UNDO_MAX_DEPTH)

    def push(self, cmd: AbstractCommand) -> None:
        cmd.execute()
        self._undo.append(cmd)
        self._redo.clear()
        self.changed.emit()

    def undo(self) -> None:
        if not self._undo:
            return
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        self.changed.emit()

    def redo(self) -> None:
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.execute()
        self._undo.append(cmd)
        self.changed.emit()

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    @property
    def undo_description(self) -> str:
        return self._undo[-1].description if self._undo else ""

    @property
    def redo_description(self) -> str:
        return self._redo[-1].description if self._redo else ""
