"""
Topology refinement operations for strand skeletons.

Ported (pure-numpy) from the standalone ``RayRecon/RayRecon_simplified.py`` script
in the sibling ``PCD-Graph-Recon-DM`` project, which was never importable (its only
entry point was a hardcoded ``__main__`` block). The five refinement operations are
exposed here as thin ``StrandGraph`` -> ``StrandGraph`` wrappers so the UI never
touches raw arrays:

    simplify_chains_graph      - collapse degree-2 chains into direct edges
    prune_leaves_graph         - remove degree-1 "hair" edges (one pass)
    collapse_triangles_graph   - merge small 3-cliques into their centroid
    grow_rays_graph            - shoot rays from endpoints to snap branches together
    beam_latch_graph           - fat-beam from endpoints to the nearest point

All operate on the whole graph (not a vertex subset). Node indices may change; the
caller re-renders via a full StrandGraph swap, so renumbering is safe.

Note: this deliberately duplicates the source functions (the upstream pipeline had
already forked its own copies); consolidating the copies is out of scope here.
"""

from __future__ import annotations
from collections import Counter, defaultdict

import numpy as np

from app.data.strand_graph import StrandGraph


# ── low-level edge helpers ────────────────────────────────────────────────────

def _dedup_undirected_edges_uv(edges_uv: np.ndarray) -> np.ndarray:
    edges_uv = np.asarray(edges_uv, dtype=int).reshape(-1, 2)
    if edges_uv.size == 0:
        return edges_uv
    uv = np.sort(edges_uv, axis=1)
    _, idx = np.unique(uv, axis=0, return_index=True)
    return edges_uv[np.sort(idx)]


def _max_edge_length(points: np.ndarray, edges_uv: np.ndarray) -> float:
    edges_uv = np.asarray(edges_uv, dtype=int).reshape(-1, 2)
    if edges_uv.size == 0:
        return 0.0
    d = points[edges_uv[:, 1]] - points[edges_uv[:, 0]]
    return float(np.max(np.linalg.norm(d, axis=1)))


def _degree1_vertices(edges_uv: np.ndarray):
    deg = Counter()
    for u, v in edges_uv:
        deg[int(u)] += 1
        deg[int(v)] += 1
    return [v for v, d in deg.items() if d == 1], deg


def _build_adj(edges_uv: np.ndarray):
    adj = defaultdict(list)
    for u, v in edges_uv:
        u = int(u); v = int(v)
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _closest_points_on_segments(p1, q1, p2, q2, eps=1e-12):
    """Closest approach between segments p1->q1 and p2->q2.

    Returns (c1, c2, dist, s, t) with s,t in [0,1] the segment parameters.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= eps and e <= eps:
        return p1, p2, float(np.linalg.norm(p1 - p2)), 0.0, 0.0

    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0) if e > eps else 0.0
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom != 0.0 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return c1, c2, float(np.linalg.norm(c1 - c2)), s, t


def _ray_beam_candidates(Pu, d_hat, P_all, beam_radius, max_length, exclude_idx=()):
    """Indices of points inside a fat beam (tube) in front of ``Pu``.

    Returns (cand_idx, t, perp): forward distances and perpendicular distances.
    """
    W = P_all - Pu
    t = W @ d_hat
    mask = (t > 0) & (t <= max_length)

    if np.any(mask):
        Wm = W[mask]
        tm = t[mask]
        perp = np.linalg.norm(Wm - tm[:, None] * d_hat[None, :], axis=1)
        mask2 = perp <= beam_radius
        idx_mask = np.flatnonzero(mask)
        cand_idx = idx_mask[mask2]
        t_keep = tm[mask2]
        perp_keep = perp[mask2]
    else:
        cand_idx = np.array([], dtype=int)
        t_keep = np.array([], dtype=float)
        perp_keep = np.array([], dtype=float)

    if exclude_idx:
        exclude_idx = set(int(x) for x in exclude_idx)
        if cand_idx.size > 0:
            keep = np.array([int(i) not in exclude_idx for i in cand_idx], dtype=bool)
            cand_idx = cand_idx[keep]
            t_keep = t_keep[keep]
            perp_keep = perp_keep[keep]

    return cand_idx, t_keep, perp_keep


# ── core refinement operations (raw arrays) ───────────────────────────────────

def prune_degree1_once(edges_uv):
    """Remove every edge touching a degree-1 vertex (single pass).

    Returns (pruned_edges_uv, removed_vertices).
    """
    edges_uv = np.asarray(edges_uv, dtype=int).reshape(-1, 2)
    if edges_uv.size == 0:
        return edges_uv, set()

    deg = Counter()
    for u, v in edges_uv:
        deg[int(u)] += 1
        deg[int(v)] += 1

    leaves = {v for v, d in deg.items() if d == 1}
    if not leaves:
        return edges_uv, set()

    mask = np.array([(u not in leaves) and (v not in leaves) for u, v in edges_uv],
                    dtype=bool)
    return edges_uv[mask], leaves


def grow_rays_and_connect(points, edges_uv, tol=1.0, connect_triangle=False):
    """Shoot rays outward from degree-1 endpoints; where two rays come within
    ``tol`` add a new junction vertex connecting both endpoints to it.

    Returns (new_points, new_edges_uv).
    """
    P = np.asarray(points, dtype=float)
    uv = _dedup_undirected_edges_uv(edges_uv)

    deg1, _ = _degree1_vertices(uv)
    adj = _build_adj(uv)
    if len(deg1) == 0:
        return P, uv

    L = _max_edge_length(P, uv)
    if L <= 0:
        return P, uv

    rays = []
    for u in deg1:
        nbrs = adj[int(u)]
        if len(nbrs) != 1:
            continue
        v = int(nbrs[0])
        dir_vec = P[u] - P[v]
        n = np.linalg.norm(dir_vec)
        if n == 0:
            continue
        dir_vec = dir_vec / n
        rays.append((P[u], P[u] + dir_vec * L, int(u)))

    new_pts = []
    new_edges_uv = []
    for i in range(len(rays)):
        p1, q1, u1 = rays[i]
        for j in range(i + 1, len(rays)):
            p2, q2, u2 = rays[j]
            c1, c2, dist, _, _ = _closest_points_on_segments(p1, q1, p2, q2)
            if dist <= tol:
                x = 0.5 * (c1 + c2)
                new_index = P.shape[0] + len(new_pts)
                new_pts.append(x)
                new_edges_uv.append([u1, new_index])
                new_edges_uv.append([u2, new_index])
                if connect_triangle:
                    new_edges_uv.append([u1, u2])

    if not new_pts:
        return P, uv

    P2 = np.vstack([P, np.asarray(new_pts, dtype=float)])
    uv2 = _dedup_undirected_edges_uv(np.vstack([uv, np.asarray(new_edges_uv, dtype=int)]))
    return P2, uv2


def beam_latch_from_degree1(points, edges_uv, beam_radius=5.0, max_length=None,
                            pick="forward", perp_tiebreak=True):
    """For each degree-1 vertex, shoot a fat beam along its outward direction and
    connect it to the closest point caught in the beam.

    Returns (points, new_edges_uv) - points are unchanged.
    """
    P = np.asarray(points, dtype=float)
    uv = _dedup_undirected_edges_uv(edges_uv)

    deg1, _ = _degree1_vertices(uv)
    if len(deg1) == 0:
        return P, uv

    adj = _build_adj(uv)
    L = _max_edge_length(P, uv) if max_length is None else float(max_length)
    if L <= 0:
        return P, uv

    existing = set(map(tuple, np.sort(uv, axis=1))) if uv.size else set()
    new_edges = []

    for u in deg1:
        u = int(u)
        nbrs = adj.get(u, [])
        if len(nbrs) != 1:
            continue
        v = int(nbrs[0])
        d = P[u] - P[v]
        n = np.linalg.norm(d)
        if n == 0:
            continue
        d_hat = d / n

        cand_idx, t, perp = _ray_beam_candidates(
            Pu=P[u], d_hat=d_hat, P_all=P,
            beam_radius=float(beam_radius), max_length=float(L),
            exclude_idx=(u, v))
        if cand_idx.size == 0:
            continue

        if pick == "forward":
            order = np.lexsort((perp, t)) if perp_tiebreak else np.argsort(t)
            w = int(cand_idx[order[0]])
        elif pick == "euclid":
            dists = np.linalg.norm(P[cand_idx] - P[u], axis=1)
            order = np.lexsort((perp, dists)) if perp_tiebreak else np.argsort(dists)
            w = int(cand_idx[order[0]])
        else:
            raise ValueError("pick must be 'forward' or 'euclid'")

        key = tuple(sorted((u, w)))
        if key in existing:
            continue
        new_edges.append([u, w])
        existing.add(key)

    if not new_edges:
        return P, uv

    uv2 = _dedup_undirected_edges_uv(np.vstack([uv, np.asarray(new_edges, dtype=int)]))
    return P, uv2


def simplify_chains(points, edges_uv):
    """Collapse maximal chains of degree-2 vertices into a single direct edge
    between the junction/endpoint nodes that bound them.

    Returns (points, new_edges_uv) - points are unchanged (interior chain nodes
    become isolated and are typically dropped afterwards).
    """
    P = np.asarray(points, dtype=float)
    uv = _dedup_undirected_edges_uv(edges_uv)

    adj = _build_adj(uv)
    deg = {u: len(nbrs) for u, nbrs in adj.items()}

    start_nodes = [u for u, d in deg.items() if d != 2]
    visited_edges = set()
    new_edges_uv = []
    established_connections = set()

    for u in start_nodes:
        for v in adj[u]:
            edge = tuple(sorted((u, v)))
            if edge in visited_edges:
                continue

            visited_edges.add(edge)
            curr, prev = v, u
            path = [u, v]

            while deg.get(curr, 0) == 2:
                nbs = adj[curr]
                nxt = nbs[0] if nbs[1] == prev else nbs[1]
                edge_next = tuple(sorted((curr, nxt)))
                if edge_next in visited_edges:
                    break
                visited_edges.add(edge_next)
                path.append(nxt)
                prev, curr = curr, nxt
                if curr == u:
                    break

            start_node = path[0]
            end_node = path[-1]

            if start_node == end_node:
                mid = path[len(path) // 2]
                new_edges_uv.extend([[start_node, mid], [mid, start_node]])
                continue

            connection_key = tuple(sorted((start_node, end_node)))
            if connection_key in established_connections:
                mid = path[len(path) // 2]
                new_edges_uv.append([start_node, mid])
                new_edges_uv.append([mid, end_node])
            else:
                new_edges_uv.append([start_node, end_node])
                established_connections.add(connection_key)

    uv2 = np.asarray(new_edges_uv, dtype=int)
    if uv2.size > 0:
        uv2 = _dedup_undirected_edges_uv(uv2)
    else:
        uv2 = np.empty((0, 2), dtype=int)
    return P, uv2


def collapse_small_triangles(points, edges_uv, threshold=5.0):
    """Merge every 3-clique whose edges are all <= ``threshold`` into a single
    vertex at the clique centroid (transitively, via union-find).

    Returns (new_points, new_edges_uv).
    """
    P = np.asarray(points, dtype=float)
    uv = _dedup_undirected_edges_uv(edges_uv)

    adj = _build_adj(uv)

    small_triangles = []
    nodes = sorted(adj.keys())
    for u in nodes:
        nbrs = adj[u]
        for v in nbrs:
            if v <= u:
                continue
            if np.linalg.norm(P[u] - P[v]) > threshold:
                continue
            for w in nbrs:
                if w <= v:
                    continue
                if w in adj[v]:
                    d_uw = np.linalg.norm(P[u] - P[w])
                    d_vw = np.linalg.norm(P[v] - P[w])
                    if d_uw <= threshold and d_vw <= threshold:
                        small_triangles.append((u, v, w))

    if not small_triangles:
        return P, uv

    parent = {n: n for n in nodes}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for u, v, w in small_triangles:
        union(u, v)
        union(v, w)

    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)

    P2 = P.copy()
    remap = {n: n for n in range(len(P))}
    for members in groups.values():
        if len(members) > 1:
            rep = members[0]
            P2[rep] = np.mean(P[members], axis=0)
            for m in members:
                remap[m] = rep

    new_edges_uv = []
    for u, v in uv:
        u2, v2 = remap[int(u)], remap[int(v)]
        if u2 != v2:
            new_edges_uv.append(sorted((u2, v2)))

    uv_final = (np.unique(new_edges_uv, axis=0) if new_edges_uv
                else np.empty((0, 2), dtype=int))
    return P2, uv_final


def remove_isolated_points(points, edges_uv):
    """Drop vertices with no incident edge and renumber the rest.

    Returns (new_points, new_edges_uv).
    """
    P = np.asarray(points, dtype=float)
    uv = np.asarray(edges_uv, dtype=int).reshape(-1, 2)
    if uv.size == 0:
        return P[:0], uv

    used = np.zeros(P.shape[0], dtype=bool)
    used[uv[:, 0]] = True
    used[uv[:, 1]] = True

    old_idx = np.flatnonzero(used)
    old_to_new = -np.ones(P.shape[0], dtype=int)
    old_to_new[old_idx] = np.arange(old_idx.size)
    return P[old_idx], old_to_new[uv]


# ── StrandGraph wrappers ──────────────────────────────────────────────────────

def _graph(points: np.ndarray, edges_uv: np.ndarray) -> StrandGraph:
    nodes = np.ascontiguousarray(points, dtype=np.float32)
    edges = (np.asarray(edges_uv, dtype=np.int32).reshape(-1, 2)
             if np.asarray(edges_uv).size else np.empty((0, 2), dtype=np.int32))
    return StrandGraph(nodes=nodes, edges=edges)


def simplify_chains_graph(g: StrandGraph) -> StrandGraph:
    P, E = simplify_chains(g.nodes, g.edges)
    P, E = remove_isolated_points(P, E)
    return _graph(P, E)


def prune_leaves_graph(g: StrandGraph) -> StrandGraph:
    E, _ = prune_degree1_once(g.edges)
    P, E = remove_isolated_points(g.nodes, E)
    return _graph(P, E)


def collapse_triangles_graph(g: StrandGraph, threshold: float) -> StrandGraph:
    P, E = collapse_small_triangles(g.nodes, g.edges, threshold=threshold)
    P, E = remove_isolated_points(P, E)
    return _graph(P, E)


def grow_rays_graph(g: StrandGraph, tol: float) -> StrandGraph:
    P, E = grow_rays_and_connect(g.nodes, g.edges, tol=tol)
    return _graph(P, E)


def beam_latch_graph(g: StrandGraph, beam_radius: float) -> StrandGraph:
    P, E = beam_latch_from_degree1(g.nodes, g.edges, beam_radius=beam_radius)
    return _graph(P, E)


def run_pipeline_graph(g: StrandGraph, tol: float, beam_radius: float,
                       tri_threshold: float, max_simplify_iters: int = 50
                       ) -> StrandGraph:
    """Run the full RayRecon refinement pipeline in the canonical order used by
    the standalone script's ``__main__``:

        prune -> grow rays -> prune -> beam latch -> clean
        -> (simplify chains + collapse triangles, looped until stable) -> clean

    ``connect_interior_leaves`` from the original script is intentionally omitted
    (it was not ported). All stages use the given parameters.
    """
    P, E = np.asarray(g.nodes, dtype=float), g.edges

    E, _ = prune_degree1_once(E)
    P, E = grow_rays_and_connect(P, E, tol=tol)
    E, _ = prune_degree1_once(E)
    P, E = beam_latch_from_degree1(P, E, beam_radius=beam_radius)
    P, E = remove_isolated_points(P, E)

    # Iterate chain-collapse until the edge count stops changing (or a cap hit).
    prev_edges = -1
    for _ in range(max_simplify_iters):
        if len(E) == prev_edges:
            break
        prev_edges = len(E)
        P, E = simplify_chains(P, E)
        P, E = collapse_small_triangles(P, E, threshold=tri_threshold)

    P, E = remove_isolated_points(P, E)
    return _graph(P, E)
