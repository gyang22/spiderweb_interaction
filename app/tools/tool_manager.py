"""Manages the currently active tool and dispatches input events."""

from app.tools.base_tool import AbstractTool
from app.tools.click_select import ClickSelectTool
from app.tools.box_select import BoxSelectTool
from app.tools.lasso_select import LassoSelectTool
from app.tools.manual_align import ManualAlignTool


class NullTool(AbstractTool):
    """No-op tool used when no point cloud is loaded."""
    def mouse_press(self, event, viewport): pass
    def mouse_move(self, event, viewport):  pass
    def mouse_release(self, event, viewport): pass


class ToolManager:
    CLICK  = 'click'
    BOX    = 'box'
    LASSO  = 'lasso'

    def __init__(self, viewport):
        self._viewport = viewport
        self._tools = {
            self.CLICK: ClickSelectTool(),
            self.BOX:   BoxSelectTool(),
            self.LASSO: LassoSelectTool(),
            'manual_align': ManualAlignTool(),
            'null':     NullTool(),
        }
        self._active_name = 'null'

    @property
    def active_tool(self) -> AbstractTool:
        return self._tools[self._active_name]

    @property
    def active_name(self) -> str:
        return self._active_name

    def set_tool(self, name: str) -> None:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name!r}")
        if name == self._active_name:
            return
        self._tools[self._active_name].on_deactivate(self._viewport)
        self._active_name = name
        self._tools[name].on_activate(self._viewport)

    def enable(self) -> None:
        """Switch to lasso-select after a point cloud is loaded."""
        if self._active_name == 'null':
            self.set_tool(self.LASSO)
