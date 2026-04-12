"""Single-point selection via GPU picking."""

from PyQt6.QtCore import Qt
from app.tools.base_tool import AbstractTool


class ClickSelectTool(AbstractTool):
    def mouse_press(self, event, viewport) -> None:
        if not viewport.has_point_cloud():
            return
        x = int(event.position().x())
        y = int(event.position().y())
        index = viewport.do_picking(x, y)
        pc = viewport.point_cloud

        add = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if index >= 0 and index < pc.total_count and pc.alive_mask[index]:
            if add:
                # Toggle
                pc.selection_mask[index] = not pc.selection_mask[index]
            else:
                pc.deselect_all()
                pc.selection_mask[index] = True
        else:
            if not add:
                pc.deselect_all()

        viewport.on_selection_changed()

    def mouse_move(self, event, viewport) -> None:
        pass

    def mouse_release(self, event, viewport) -> None:
        pass
