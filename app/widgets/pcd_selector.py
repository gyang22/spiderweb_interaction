"""
PCD file selector dialog.

Supports two browsing modes depending on the folder's contents:

  Structured mode  (default — spiderweb folder)
    Files matching  "<series> <param> <date> <time> T<threshold>.pcd"
    are grouped into a series → session → threshold hierarchy.

  Flat mode  (any other folder)
    All *.pcd files in the folder are listed directly.

The folder root can be changed at any time via the path bar at the top.
Reads are allowed from any location; no files are written by this dialog.
"""

import re
from pathlib import Path
from collections import defaultdict

from PyQt6.QtWidgets import (
    QDialog, QSplitter, QTreeWidget, QTreeWidgetItem,
    QListWidget, QListWidgetItem, QDialogButtonBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QFileDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


_DEFAULT_ROOT = Path(
    "/Users/grantyang/Documents/python_projects/spiderweb/"
    "video_processing/point_clouds"
)

# Matches: "<series> <rest> T<threshold>.pcd"
_NAME_RE = re.compile(
    r'^(?P<series>tangle\w+)\s+'
    r'(?P<rest>.+?)\s+'
    r'T(?P<threshold>[0-9.]+)\.pcd$',
    re.IGNORECASE,
)

_FLAT_SERIES  = "__flat__"
_FLAT_SESSION = "All files"


def _scan_structured(root: Path) -> dict:
    """
    Returns { series: { session_label: [(threshold_str, Path), …] } }
    Only files matching _NAME_RE are included.
    """
    tree: dict = defaultdict(lambda: defaultdict(list))
    for p in sorted(root.glob("*.pcd")):
        m = _NAME_RE.match(p.name)
        if not m:
            continue
        series    = m.group("series")
        rest      = m.group("rest").strip()
        threshold = m.group("threshold")
        tree[series][rest].append((threshold, p))

    for series in tree:
        for session in tree[series]:
            tree[series][session].sort(key=lambda x: float(x[0]))

    return tree


def _scan_flat(root: Path) -> dict:
    """
    Returns a single-series dict with all *.pcd files as (name, Path) entries
    using the flat sentinel keys so the rest of the UI works unchanged.
    """
    files = sorted(root.glob("*.pcd"))
    if not files:
        return {}
    # Use filename stem as the "threshold" label so the right panel shows names
    entries = [(p.stem, p) for p in files]
    return {_FLAT_SERIES: {_FLAT_SESSION: entries}}


def _scan(root: Path) -> tuple[dict, bool]:
    """
    Scan root for .pcd files.

    Returns (data_dict, is_structured).
    Falls back to flat mode when no files match the naming pattern.
    """
    if not root.is_dir():
        return {}, False
    structured = _scan_structured(root)
    if structured:
        return structured, True
    flat = _scan_flat(root)
    return flat, False


class PcdSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Point Cloud")
        self.resize(820, 560)
        self.selected_path: Path | None = None
        self.import_json_requested: bool = False

        self._root = _DEFAULT_ROOT
        self._data, self._structured = _scan(self._root)
        self._build_ui()
        self._populate_tree()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(8)

        # Title
        title = QLabel("Open Point Cloud")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        title.setFont(font)
        root_layout.addWidget(title)

        # ── Folder path bar ───────────────────────────────────────────────────
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Folder:"))
        self._path_edit = QLineEdit(str(self._root))
        self._path_edit.setPlaceholderText("Enter a folder path…")
        self._path_edit.returnPressed.connect(self._on_path_edited)
        path_row.addWidget(self._path_edit, stretch=1)
        btn_browse_dir = QPushButton("Browse…")
        btn_browse_dir.setFixedWidth(80)
        btn_browse_dir.clicked.connect(self._on_browse_dir)
        path_row.addWidget(btn_browse_dir)
        root_layout.addLayout(path_row)

        # Search bar (only meaningful in structured mode)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter experiments…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_filter)
        root_layout.addWidget(self._search)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: experiment / folder tree ────────────────────────────────────
        left = QFrame()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._tree_header = QLabel("Experiments")
        left_layout.addWidget(self._tree_header)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(220)
        self._tree.currentItemChanged.connect(self._on_tree_selection)
        left_layout.addWidget(self._tree)
        splitter.addWidget(left)

        # ── Right: file list ──────────────────────────────────────────────────
        right = QFrame()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._frame_header = QLabel("Select a file")
        right_layout.addWidget(self._frame_header)

        self._frame_list = QListWidget()
        self._frame_list.setSpacing(2)
        self._frame_list.currentItemChanged.connect(self._on_frame_selection)
        self._frame_list.itemDoubleClicked.connect(self._accept_current)
        right_layout.addWidget(self._frame_list)

        self._path_label = QLabel("")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #888; font-size: 11px;")
        right_layout.addWidget(self._path_label)

        splitter.addWidget(right)
        splitter.setSizes([240, 560])
        root_layout.addWidget(splitter, stretch=1)

        # Button row
        btn_row = QHBoxLayout()

        btn_import_json = QPushButton("Import Skeleton JSON…")
        btn_import_json.setToolTip("Skip PCD selection and import a skeleton graph from JSON.")
        btn_import_json.clicked.connect(self._on_import_json)
        btn_row.addWidget(btn_import_json)

        btn_row.addStretch()

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open |
            QDialogButtonBox.StandardButton.Cancel
        )
        self._open_btn = btn_box.button(QDialogButtonBox.StandardButton.Open)
        self._open_btn.setEnabled(False)
        btn_box.accepted.connect(self._accept_current)
        btn_box.rejected.connect(self.reject)
        btn_row.addWidget(btn_box)

        root_layout.addLayout(btn_row)

    def _on_import_json(self) -> None:
        self.import_json_requested = True
        self.accept()

    # ── Folder navigation ─────────────────────────────────────────────────────

    def _on_browse_dir(self) -> None:
        start = str(self._root) if self._root.is_dir() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder containing .pcd files", start
        )
        if folder:
            self._set_root(Path(folder))

    def _on_path_edited(self) -> None:
        p = Path(self._path_edit.text().strip())
        if p.is_dir():
            self._set_root(p)
        else:
            self._path_edit.setStyleSheet("color: #ff6666;")

    def _set_root(self, new_root: Path) -> None:
        self._root = new_root
        self._path_edit.setText(str(new_root))
        self._path_edit.setStyleSheet("")
        self._data, self._structured = _scan(new_root)
        self._search.setText("")
        self._frame_list.clear()
        self._path_label.setText("")
        self._open_btn.setEnabled(False)
        self.selected_path = None
        self._populate_tree()

    # ── Tree population ───────────────────────────────────────────────────────

    def _populate_tree(self, filter_text: str = "") -> None:
        self._tree.clear()
        ft = filter_text.lower()

        if not self._data:
            self._tree_header.setText("No .pcd files found")
            self._frame_header.setText("(empty folder)")
            return

        if self._structured:
            self._tree_header.setText("Experiments")
            for series in sorted(self._data.keys()):
                if ft and ft not in series.lower():
                    continue
                series_item = QTreeWidgetItem([series])
                series_item.setData(0, Qt.ItemDataRole.UserRole, None)
                bold = QFont(); bold.setBold(True)
                series_item.setFont(0, bold)
                for session in sorted(self._data[series].keys()):
                    session_item = QTreeWidgetItem([session])
                    session_item.setData(
                        0, Qt.ItemDataRole.UserRole, (series, session)
                    )
                    series_item.addChild(session_item)
                if series_item.childCount() > 0:
                    self._tree.addTopLevelItem(series_item)
                    series_item.setExpanded(True)
        else:
            # Flat mode — single entry in the tree
            self._tree_header.setText("Files")
            all_item = QTreeWidgetItem([f"{self._root.name}  ({len(self._data[_FLAT_SERIES][_FLAT_SESSION])} files)"])
            all_item.setData(0, Qt.ItemDataRole.UserRole, (_FLAT_SERIES, _FLAT_SESSION))
            bold = QFont(); bold.setBold(True)
            all_item.setFont(0, bold)
            self._tree.addTopLevelItem(all_item)
            self._tree.setCurrentItem(all_item)   # auto-select so files show immediately

    def _on_filter(self, text: str) -> None:
        self._populate_tree(text)
        self._frame_list.clear()
        self._path_label.setText("")
        self._open_btn.setEnabled(False)

    # ── Selection handlers ────────────────────────────────────────────────────

    def _on_tree_selection(self, current: QTreeWidgetItem, _) -> None:
        self._frame_list.clear()
        self._path_label.setText("")
        self._open_btn.setEnabled(False)

        if current is None:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return  # series header — no files to show

        series, session = data
        frames = self._data[series][session]

        if self._structured:
            self._frame_header.setText(f"{series}  ·  {session}")
            for threshold, path in frames:
                item = QListWidgetItem(f"T = {threshold}")
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setToolTip(str(path))
                self._frame_list.addItem(item)
        else:
            self._frame_header.setText(f"{self._root.name}")
            for stem, path in frames:
                item = QListWidgetItem(path.name)
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setToolTip(str(path))
                self._frame_list.addItem(item)

        if self._frame_list.count() > 0:
            self._frame_list.setCurrentRow(0)

    def _on_frame_selection(self, current: QListWidgetItem, _) -> None:
        if current is None:
            self._open_btn.setEnabled(False)
            self._path_label.setText("")
            return
        path: Path = current.data(Qt.ItemDataRole.UserRole)
        self._path_label.setText(str(path))
        self._open_btn.setEnabled(True)
        self.selected_path = path

    # ── Accept ────────────────────────────────────────────────────────────────

    def _accept_current(self, *_) -> None:
        item = self._frame_list.currentItem()
        if item is None:
            return
        self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
