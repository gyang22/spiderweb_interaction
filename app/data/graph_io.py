"""Export StrandGraph to JSON."""

from __future__ import annotations
import json
from pathlib import Path

from app.data.strand_graph import StrandGraph


def export_graph_json(graph: StrandGraph, path: str | Path) -> None:
    """
    Write a StrandGraph to a JSON file.

    Output format::

        {
            "nodes": [[x, y, z], ...],
            "edges": [[i, j], ...]
        }

    Parameters
    ----------
    graph : StrandGraph
    path  : str or Path — destination file path (should end in .json)
    """
    data = {
        "nodes": graph.nodes.tolist(),
        "edges": graph.edges.tolist(),
    }
    with open(Path(path), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
