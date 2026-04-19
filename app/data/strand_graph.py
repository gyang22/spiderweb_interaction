"""
Strand skeleton extraction from a dense point cloud selection.

Algorithm: voxel downsampling → k-NN graph → Kruskal MST → leaf pruning.
Pure numpy, no scipy/sklearn required.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class StrandGraph:
    """
    Sparse backbone graph representing a strand skeleton.

    nodes : (M, 3) float32 — 3D positions of skeleton nodes
    edges : (E, 2) int32   — pairs of node indices (undirected)
    """
    nodes: np.ndarray   # (M, 3) float32
    edges: np.ndarray   # (E, 2) int32


def merge_graphs(g1: StrandGraph, g2: StrandGraph) -> StrandGraph:
    """
    Combine two StrandGraphs by concatenating their nodes and edges.

    The second graph's edge indices are offset by len(g1.nodes) so that all
    indices remain valid after the node arrays are stacked.
    """
    nodes = np.concatenate([g1.nodes, g2.nodes], axis=0)
    if len(g2.edges) == 0:
        edges = g1.edges.copy()
    elif len(g1.edges) == 0:
        edges = g2.edges + len(g1.nodes)
    else:
        edges = np.concatenate([g1.edges, g2.edges + len(g1.nodes)], axis=0)
    return StrandGraph(nodes=nodes.astype(np.float32), edges=edges.astype(np.int32))


def merge_graphs_with_bridges(
    existing: StrandGraph,
    new_sub: StrandGraph,
    bridge_factor: float = 2.0,
) -> StrandGraph:
    """
    Merge two StrandGraphs like merge_graphs, but also add bridge edges that
    connect nodes in new_sub to nearby nodes in existing.

    All existing edges are preserved unchanged.  Bridge edges are added for
    every new_sub node whose nearest existing node is within
    ``bridge_factor × median_existing_edge_length``.
    """
    N_e = len(existing.nodes)
    combined_nodes = np.concatenate([existing.nodes, new_sub.nodes], axis=0)

    if len(existing.edges) == 0:
        base_edges = (new_sub.edges + N_e).copy() if len(new_sub.edges) > 0 \
                     else np.empty((0, 2), dtype=np.int32)
    elif len(new_sub.edges) == 0:
        base_edges = existing.edges.copy()
    else:
        base_edges = np.concatenate(
            [existing.edges, new_sub.edges + N_e], axis=0
        )

    # Compute bridge distance threshold from existing skeleton edge lengths.
    if N_e > 0 and len(new_sub.nodes) > 0:
        if len(existing.edges) > 0:
            ex_lengths = np.linalg.norm(
                existing.nodes[existing.edges[:, 0]] -
                existing.nodes[existing.edges[:, 1]],
                axis=1,
            )
            threshold = float(np.median(ex_lengths)) * bridge_factor
        elif len(new_sub.edges) > 0:
            ns_lengths = np.linalg.norm(
                new_sub.nodes[new_sub.edges[:, 0]] -
                new_sub.nodes[new_sub.edges[:, 1]],
                axis=1,
            )
            threshold = float(np.median(ns_lengths)) * bridge_factor
        else:
            threshold = float("inf")

        # (M_new, N_existing) pairwise distances
        diff = new_sub.nodes[:, None, :] - existing.nodes[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        nearest_ex = dists.argmin(axis=1)
        nearest_d  = dists[np.arange(len(new_sub.nodes)), nearest_ex]

        seen: set[tuple[int, int]] = set()
        bridge_list: list[tuple[int, int]] = []
        for new_i, (ex_i, d) in enumerate(zip(nearest_ex.tolist(), nearest_d.tolist())):
            if d <= threshold:
                a, b = N_e + new_i, int(ex_i)
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    bridge_list.append(key)

        if bridge_list:
            bridges = np.array(bridge_list, dtype=np.int32)
            base_edges = np.concatenate([base_edges, bridges], axis=0)

    return StrandGraph(
        nodes=combined_nodes.astype(np.float32),
        edges=base_edges.astype(np.int32),
    )


# ── Public entry point ────────────────────────────────────────────────────────

def extract_skeleton(
    positions: np.ndarray,
    voxel_size: float | None = None,
    k_neighbors: int = 4,
    prune_factor: float = 0.5,
) -> StrandGraph:
    """
    Extract a strand skeleton from a dense point selection.

    Parameters
    ----------
    positions : (N, 3) float32
        3D positions of the selected points.
    voxel_size : float or None
        Grid cell size for downsampling. Auto-computed as
        ``np.ptp(positions, axis=0).max() / 40`` if None.
    k_neighbors : int
        Number of nearest neighbors per voxel node for graph construction.
    prune_factor : float
        Leaf edges shorter than ``prune_factor × median_edge_length`` are
        removed. A value of 0 disables pruning. A value of 0.5 removes
        leaf stubs that are less than half the typical edge length — only
        clear noise stubs, not real strand ends.

    Returns
    -------
    StrandGraph

    Raises
    ------
    ValueError
        If fewer than 2 points survive filtering / voxelization.
    """
    positions = np.asarray(positions, dtype=np.float32)

    # Filter non-finite
    valid = np.isfinite(positions).all(axis=1)
    positions = positions[valid]

    if len(positions) < 2:
        raise ValueError(
            f"Need at least 2 finite points to extract a skeleton "
            f"(got {len(positions)})."
        )

    # Auto voxel size
    extent = float(np.ptp(positions, axis=0).max())
    if voxel_size is None:
        voxel_size = max(extent / 40.0, 1e-9)

    # Step 1 — voxelize
    nodes = _voxelize(positions, voxel_size)

    if len(nodes) < 2:
        raise ValueError(
            f"Voxelization with voxel_size={voxel_size:.6g} produced only "
            f"{len(nodes)} node(s). Reduce voxel_size or select more points."
        )

    # Step 2 — k-NN edge list
    k = min(k_neighbors, len(nodes) - 1)
    edge_list = _build_knn_edges(nodes, k)

    # Step 3 — minimum spanning forest (Kruskal)
    mst_edges = _kruskal_mst(len(nodes), edge_list)

    # Step 4 — prune short leaf stubs (relative to median edge length)
    if prune_factor > 0.0 and mst_edges:
        edge_lengths = np.array(
            [float(np.linalg.norm(nodes[i] - nodes[j])) for i, j in mst_edges],
            dtype=np.float64,
        )
        median_len = float(np.median(edge_lengths))
        prune_threshold = prune_factor * median_len
        nodes_out, edges_out = _prune_leaves(nodes, mst_edges, prune_threshold)
    else:
        # No pruning — compact into arrays directly
        nodes_out = nodes
        edges_out = np.array(mst_edges, dtype=np.int32) if mst_edges \
                    else np.empty((0, 2), dtype=np.int32)

    return StrandGraph(
        nodes=nodes_out.astype(np.float32),
        edges=edges_out.astype(np.int32),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _voxelize(positions: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Voxel centroid downsampling.

    Returns (M, 3) float32 array of one centroid per occupied voxel, M << N.
    """
    shifted = positions - positions.min(axis=0)
    keys = np.floor(shifted / voxel_size).astype(np.int64)

    # Encode 3D voxel key as a single structured value for np.unique grouping
    key_struct = np.ascontiguousarray(keys).view(
        np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])
    ).ravel()

    _, first_idx, inverse = np.unique(
        key_struct, return_index=True, return_inverse=True
    )
    M = len(first_idx)

    # Accumulate centroid per voxel
    centroids = np.zeros((M, 3), dtype=np.float64)
    counts = np.zeros(M, dtype=np.int64)
    np.add.at(centroids, inverse, positions)
    np.add.at(counts, inverse, 1)
    centroids /= counts[:, None]

    return centroids.astype(np.float32)


def _build_knn_edges(
    nodes: np.ndarray, k: int
) -> list[tuple[int, int, float]]:
    """
    Build undirected k-NN edge list (i < j, deduplicated) with Euclidean distances.
    Uses O(M²) pairwise distances — acceptable since M is small after voxelization.
    """
    M = len(nodes)
    # (M, M) pairwise squared distances
    diff = nodes[:, None, :] - nodes[None, :, :]   # (M, M, 3)
    dist2 = (diff ** 2).sum(axis=-1)               # (M, M)
    np.fill_diagonal(dist2, np.inf)

    edge_set: dict[tuple[int, int], float] = {}
    for i in range(M):
        # Indices of k nearest neighbors
        nn = np.argpartition(dist2[i], k)[:k]
        for j in nn:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            d = float(dist2[i, j] ** 0.5)
            if (a, b) not in edge_set or edge_set[(a, b)] > d:
                edge_set[(a, b)] = d

    return [(a, b, d) for (a, b), d in edge_set.items()]


def _kruskal_mst(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
) -> list[tuple[int, int]]:
    """
    Kruskal's minimum spanning forest via union-find.

    Returns list of (i, j) edge pairs. Disconnected components each get their
    own spanning tree (no artificial bridge edges between them).
    """
    uf = _UnionFind(n_nodes)
    mst: list[tuple[int, int]] = []
    for i, j, _ in sorted(edges, key=lambda e: e[2]):
        if uf.union(i, j):
            mst.append((i, j))
    return mst


class _UnionFind:
    """Path-compressed, rank-based union-find."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Merge sets. Returns True if they were different (edge added)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return True


def _prune_leaves(
    nodes: np.ndarray,
    mst_edges: list[tuple[int, int]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively remove leaf nodes whose single edge is shorter than threshold.

    After removal, node indices are compacted (gaps removed) and edges re-mapped.
    Returns (nodes_out (M', 3), edges_out (E', 2)).
    """
    if not mst_edges:
        return nodes, np.empty((0, 2), dtype=np.int32)

    # Build adjacency as sets (mutable during pruning)
    adj: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    edge_len: dict[tuple[int, int], float] = {}

    for i, j in mst_edges:
        adj[i].add(j)
        adj[j].add(i)
        key = (min(i, j), max(i, j))
        d = float(np.linalg.norm(nodes[i] - nodes[j]))
        edge_len[key] = d

    # Iterative leaf pruning
    changed = True
    while changed:
        changed = False
        leaves = [n for n, nbrs in adj.items() if len(nbrs) == 1]
        for leaf in leaves:
            if leaf not in adj or len(adj[leaf]) != 1:
                continue  # already removed in this iteration
            nbr = next(iter(adj[leaf]))
            key = (min(leaf, nbr), max(leaf, nbr))
            if edge_len.get(key, 0.0) < threshold:
                # Guard: don't prune below 2 nodes
                remaining = sum(1 for n in adj if len(adj[n]) > 0 or n == leaf)
                if remaining <= 2:
                    break
                adj[leaf].discard(nbr)
                adj[nbr].discard(leaf)
                del adj[leaf]
                edge_len.pop(key, None)
                changed = True

    # Compact: keep only nodes that still exist in adj
    kept = sorted(adj.keys())
    if len(kept) < 2:
        # Over-pruned safety: return original
        kept = list(range(len(nodes)))
        mst_edge_set = set(mst_edges)
        edges_out = np.array(list(mst_edge_set), dtype=np.int32) if mst_edge_set \
                    else np.empty((0, 2), dtype=np.int32)
        return nodes, edges_out

    remap = {old: new for new, old in enumerate(kept)}
    nodes_out = nodes[kept]

    edges_out_list = []
    seen = set()
    for old_i in kept:
        for old_j in adj[old_i]:
            a, b = remap[old_i], remap[old_j]
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                edges_out_list.append(key)

    edges_out = np.array(edges_out_list, dtype=np.int32) if edges_out_list \
                else np.empty((0, 2), dtype=np.int32)

    return nodes_out, edges_out
