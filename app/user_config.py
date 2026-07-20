"""
User-editable configuration: persistent overrides for app defaults, file paths,
and hotkeys.

Values live in a JSON file next to the project (``user_settings.json``) and are
loaded once at import time into the module-level ``config`` singleton. Any module
can read a value with ``config.get(key)``; the Preferences dialog writes values
back with ``config.set`` / ``config.save`` and emits :pyattr:`UserConfig.changed`
so live consumers can refresh.

The set of editable settings is declared once in :data:`SCHEMA`; both the dialog
UI and the defaults are derived from it, so adding a new setting is a one-line
change here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from app import settings


CONFIG_PATH: Path = Path(__file__).parent.parent / "user_settings.json"


@dataclass(frozen=True)
class Field:
    key: str
    group: str                     # tab name: "Defaults" | "Paths" | "Hotkeys"
    label: str
    kind: str                      # 'int' | 'float' | 'path' | 'keyseq'
    default: Any
    minimum: float = 0.0
    maximum: float = 0.0
    decimals: int = 2
    step: float = 1.0
    tooltip: str = ""


# ── Declarative schema ─────────────────────────────────────────────────────────
# Order here is the order shown in each tab.
SCHEMA: list[Field] = [
    # ---- Defaults (numbers) ----
    Field("point_size", "Defaults", "Default point size", "int",
          int(settings.DEFAULT_POINT_SIZE), settings.MIN_POINT_SIZE, settings.MAX_POINT_SIZE,
          tooltip="Point size applied when a cloud is first shown."),
    Field("camera_speed_mult", "Defaults", "Camera speed multiplier", "float",
          1.0, 0.05, 20.0, decimals=2, step=0.1,
          tooltip="Scales the fly-camera speed (which is auto-fit to cloud size)."),
    Field("camera_mouse_sensitivity", "Defaults", "Mouse sensitivity", "float",
          settings.CAMERA_MOUSE_SENSITIVITY, 0.01, 5.0, decimals=2, step=0.05,
          tooltip="Degrees of camera rotation per pixel of mouse movement."),
    Field("camera_fov", "Defaults", "Field of view (°)", "float",
          settings.CAMERA_FOV, 20.0, 120.0, decimals=1, step=1.0,
          tooltip="Vertical field of view in degrees."),
    Field("undo_max_depth", "Defaults", "Undo history depth", "int",
          settings.UNDO_MAX_DEPTH, 1, 1000,
          tooltip="How many operations can be undone."),
    Field("anchor_primary_count", "Defaults", "Anchor: primary FPS count", "int",
          100, 1, 2000,
          tooltip="How many candidate anchors are sampled on the primary web."),
    Field("anchor_secondary_extra", "Defaults", "Anchor: secondary extra FPS", "int",
          30, 0, 2000,
          tooltip="Extra independent candidate anchors sampled on the secondary web."),
    Field("icp_max_iter", "Defaults", "ICP max iterations", "int",
          50, 1, 500),
    Field("cpd_alpha", "Defaults", "CPD alpha", "float",
          0.1, 0.001, 10.0, decimals=3, step=0.01),
    Field("wm_search_radius", "Defaults", "WebMerge search radius", "float",
          20.0, 0.1, 1000.0, decimals=1, step=1.0),
    Field("wm_vote_steps", "Defaults", "WebMerge vote steps", "int",
          5, 1, 50),
    Field("wm_step_size", "Defaults", "WebMerge step size", "float",
          2.5, 0.1, 100.0, decimals=1, step=0.5),
    Field("wm_lam", "Defaults", "WebMerge lambda", "float",
          0.4, 0.01, 1.0, decimals=2, step=0.05),
    Field("wm_iterations", "Defaults", "WebMerge iterations", "int",
          30, 1, 200),

    # ---- Paths ----
    Field("default_dir", "Paths", "Default file dialog folder", "path",
          "", tooltip="Starting folder for open/save dialogs. Blank = the "
                      "project 'saves' folder (or your home folder)."),

    # ---- Hotkeys (action shortcuts only; movement keys are fixed) ----
    Field("hk_fps_toggle", "Hotkeys", "Toggle FPS fly mode", "keyseq", "Tab"),
    Field("hk_delete_selected", "Hotkeys", "Delete selected", "keyseq", "Q"),
    Field("hk_extract_skeleton", "Hotkeys", "Extract skeleton", "keyseq", "G"),
    Field("hk_reset_camera", "Hotkeys", "Reset camera", "keyseq", "Home"),
    Field("hk_cycle_pick_target", "Hotkeys", "Cycle pick target web", "keyseq", "E"),
    Field("hk_invert_selection", "Hotkeys", "Invert selection", "keyseq", "Ctrl+I"),
    Field("hk_clear_selection", "Hotkeys", "Clear selection", "keyseq", "Escape"),
]

_BY_KEY = {f.key: f for f in SCHEMA}

# Which hotkey actions are consumed by the global event filter as single keys,
# vs. those applied to QActions (which support full chords like Ctrl+I).
SINGLE_KEY_HOTKEYS = (
    "hk_fps_toggle", "hk_delete_selected", "hk_extract_skeleton",
    "hk_reset_camera", "hk_cycle_pick_target",
)
QACTION_HOTKEYS = ("hk_invert_selection", "hk_clear_selection")


class UserConfig(QObject):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._values: dict[str, Any] = {f.key: f.default for f in SCHEMA}
        self.load()

    # ── access ──────────────────────────────────────────────────────────────
    def get(self, key: str) -> Any:
        return self._values.get(key, _BY_KEY[key].default if key in _BY_KEY else None)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def reset_defaults(self) -> None:
        self._values = {f.key: f.default for f in SCHEMA}

    def all_values(self) -> dict[str, Any]:
        return dict(self._values)

    # ── convenience getters ─────────────────────────────────────────────────
    def default_dir(self) -> str:
        """Folder for file dialogs. Falls back to settings.default_dir()."""
        d = str(self.get("default_dir") or "").strip()
        if d and Path(d).is_dir():
            return d
        return settings.default_dir()

    def hotkey_key(self, action: str) -> int | None:
        """Primary Qt key int for a single-key hotkey action, or None."""
        from PyQt6.QtGui import QKeySequence
        seq = QKeySequence(str(self.get(action) or ""))
        if seq.count() == 0:
            return None
        # PyQt6: indexing returns a QKeyCombination.
        combo = seq[0]
        try:
            return int(combo.key().value)
        except AttributeError:
            return int(combo)

    def hotkey_sequence(self, action: str):
        from PyQt6.QtGui import QKeySequence
        return QKeySequence(str(self.get(action) or ""))

    # ── persistence ──────────────────────────────────────────────────────────
    def load(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if key in _BY_KEY:
                self._values[key] = value

    def save(self) -> None:
        try:
            CONFIG_PATH.write_text(json.dumps(self._values, indent=2))
        except OSError:
            pass
        self.changed.emit()


# Module-level singleton — import this everywhere.
config = UserConfig()
