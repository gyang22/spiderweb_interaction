"""
Preferences dialog — edits the user-editable settings declared in
``app.user_config.SCHEMA``. The UI is generated from the schema so new settings
appear automatically.

Open with Edit ▸ Preferences… (Ctrl+,). On OK the values are written back to the
``config`` singleton and persisted; ``config.changed`` then lets the app apply
them live.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QFormLayout, QVBoxLayout, QHBoxLayout,
    QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton, QKeySequenceEdit,
    QDialogButtonBox, QFileDialog, QLabel,
)
from PyQt6.QtGui import QKeySequence

from app.user_config import config, SCHEMA, Field


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(420)

        self._editors: dict[str, object] = {}

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)

        # Group fields by tab, preserving schema order.
        groups: dict[str, list[Field]] = {}
        for f in SCHEMA:
            groups.setdefault(f.group, []).append(f)

        for group_name, fields in groups.items():
            tabs.addTab(self._build_tab(fields), group_name)

        # Buttons: Restore Defaults | Cancel | OK
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._on_restore_defaults)
        root.addWidget(buttons)

    # ── build ────────────────────────────────────────────────────────────────
    def _build_tab(self, fields: list[Field]) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setSpacing(8)
        for f in fields:
            editor = self._make_editor(f)
            self._editors[f.key] = editor
            row = self._wrap_editor(f, editor)
            label = QLabel(f.label)
            if f.tooltip:
                label.setToolTip(f.tooltip)
            form.addRow(label, row)
        return page

    def _make_editor(self, f: Field):
        if f.kind == "int":
            w = QSpinBox()
            w.setRange(int(f.minimum), int(f.maximum))
            w.setSingleStep(int(f.step) or 1)
            w.setValue(int(config.get(f.key)))
            return w
        if f.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(float(f.minimum), float(f.maximum))
            w.setDecimals(f.decimals)
            w.setSingleStep(f.step)
            w.setValue(float(config.get(f.key)))
            return w
        if f.kind == "path":
            w = QLineEdit()
            w.setText(str(config.get(f.key) or ""))
            w.setPlaceholderText("(default: project 'saves' folder)")
            return w
        if f.kind == "keyseq":
            w = QKeySequenceEdit()
            w.setKeySequence(QKeySequence(str(config.get(f.key) or "")))
            return w
        raise ValueError(f"Unknown field kind: {f.kind}")

    def _wrap_editor(self, f: Field, editor) -> QWidget:
        """Paths get a Browse button; others are returned as-is (wrapped)."""
        if f.tooltip:
            editor.setToolTip(f.tooltip)
        if f.kind != "path":
            return editor
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(editor, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_path(editor))
        h.addWidget(browse)
        return container

    def _browse_path(self, line_edit: QLineEdit) -> None:
        start = line_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose default folder", start)
        if folder:
            line_edit.setText(folder)

    # ── actions ────────────────────────────────────────────────────────────────
    def _on_restore_defaults(self) -> None:
        for f in SCHEMA:
            editor = self._editors[f.key]
            if f.kind in ("int", "float"):
                editor.setValue(f.default)
            elif f.kind == "path":
                editor.setText(str(f.default or ""))
            elif f.kind == "keyseq":
                editor.setKeySequence(QKeySequence(str(f.default or "")))

    def _on_accept(self) -> None:
        for f in SCHEMA:
            editor = self._editors[f.key]
            if f.kind == "int":
                config.set(f.key, int(editor.value()))
            elif f.kind == "float":
                config.set(f.key, float(editor.value()))
            elif f.kind == "path":
                config.set(f.key, editor.text().strip())
            elif f.kind == "keyseq":
                config.set(f.key, editor.keySequence().toString())
        config.save()   # emits config.changed
        self.accept()
