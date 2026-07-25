"""Single-point selection via GPU picking (PCD) or nearest-node (skeleton edit)."""

from PyQt6.QtCore import Qt
from app.tools.base_tool import AbstractTool


class ClickSelectTool(AbstractTool):
    def mouse_press(self, event, viewport) -> None:
        if not viewport.has_selectable():
            return
        x   = int(event.position().x())
        y   = int(event.position().y())
        # Alt = add to selection (Shift is taken by the camera "fly down" key).
        add = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        viewport.apply_click_selection(x, y, add)

    def mouse_move(self, event, viewport) -> None:
        pass

    def mouse_release(self, event, viewport) -> None:
        pass
