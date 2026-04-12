"""Export StrandGraph to JSON."""

from __future__ import annotations
import json
import numpy as np
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


def import_graph_json(path: str | Path) -> StrandGraph:
    """
    Read a StrandGraph from a JSON file.
    Robustly handles nested lists, dictionaries, 'links' vs 'edges', and 2D geometries.

    Parameters
    ----------
    path : str or Path — source file path

    Returns
    -------
    StrandGraph
    """
    with open(Path(path), 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes_out = []
    edges_out = []

    # Robust node parsing
    raw_nodes = data.get("nodes", [])
    for node in raw_nodes:
        if isinstance(node, dict):
            # E.g. networkx format {"id": 0, "pos": [x,y,z]}
            pos = node.get("pos", node.get("position", [0.0, 0.0, 0.0]))
            nodes_out.append(pos)
        else:
            nodes_out.append(node)

    # Robust edge parsing
    raw_edges = data.get("edges", data.get("links", []))
    for e in raw_edges:
        if isinstance(e, dict):
            # E.g. networkx style {"source": 0, "target": 1}
            edges_out.append([e.get("source", 0), e.get("target", 0)])
        else:
            edges_out.append(e)

    nodes = np.array(nodes_out, dtype=np.float32)
    edges = np.array(edges_out, dtype=np.int32)
    
    # Ensure nodes is N x 3 (handle empty arrays or 2D inputs)
    if nodes.ndim == 1 and len(nodes) > 0:
        nodes = nodes.reshape(-1, 3)
    elif nodes.ndim == 2 and nodes.shape[1] == 2:
        zeros = np.zeros((nodes.shape[0], 1), dtype=np.float32)
        nodes = np.hstack((nodes, zeros))
    elif len(nodes) == 0:
        nodes = np.empty((0, 3), dtype=np.float32)

    if len(edges) == 0:
        edges = np.empty((0, 2), dtype=np.int32)

    return StrandGraph(nodes=nodes, edges=edges)
