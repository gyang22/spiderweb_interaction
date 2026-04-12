"""Abstract command base class for undo/redo."""

from abc import ABC, abstractmethod


class AbstractCommand(ABC):
    @abstractmethod
    def execute(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...

    @property
    def description(self) -> str:
        return self.__class__.__name__
