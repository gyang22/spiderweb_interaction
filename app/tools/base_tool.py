"""Abstract base class for all interaction tools."""

from abc import ABC, abstractmethod
from PyQt6.QtGui import QPainter


class AbstractTool(ABC):
    @abstractmethod
    def mouse_press(self, event, viewport) -> None: ...

    @abstractmethod
    def mouse_move(self, event, viewport) -> None: ...

    @abstractmethod
    def mouse_release(self, event, viewport) -> None: ...

    def draw_overlay(self, painter: QPainter) -> None:
        """Optional 2D overlay drawn on top of the GL viewport."""

    def on_activate(self, viewport) -> None:
        """Called when this tool becomes the active tool."""

    def on_deactivate(self, viewport) -> None:
        """Called when this tool is replaced by another tool."""
