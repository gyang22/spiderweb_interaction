import numpy as np
from dataclasses import dataclass

@dataclass
class StrandGraph:
    nodes: np.ndarray
    edges: np.ndarray

def point_line_distance(point, start, end):
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point - start)
    line_dir = line_vec / line_len
    point_vec = point - start
    proj = np.dot(point_vec, line_dir)
    proj_clamped = max(0.0, min(line_len, proj))
    closest = start + proj_clamped * line_dir
    return np.linalg.norm(point - closest)

def rdp(points, indices, epsilon):
    if len(points) <= 2:
        return indices
        
    start, end = points[0], points[-1]
    
    dists = []
    for i in range(1, len(points) - 1):
        dists.append(point_line_distance(points[i], start, end))
        
    max_idx = np.argmax(dists) + 1
    max_dist = dists[max_idx - 1]
    
    if max_dist > epsilon:
        left = rdp(points[:max_idx+1], indices[:max_idx+1], epsilon)
        right = rdp(points[max_idx:], indices[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [indices[0], indices[-1]]

def get_chains(n, edges, smoothable):
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
        
    visited_smoothable = set()
    chains = []
    
    for i in smoothable:
        if i in visited_smoothable:
            continue
            
        start_node = i
        curr = i
        prev = -1
        while True:
            neighbors = adj[curr]
            nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if nxt not in smoothable:
                start_node = curr
                break
            if nxt == i:
                break
            prev = curr
            curr = nxt
            
        chain = [start_node]
        visited_smoothable.add(start_node)
        
        neighbors = adj[start_node]
        nxt0, nxt1 = neighbors[0], neighbors[1]
        
        if nxt0 in smoothable and nxt0 not in visited_smoothable:
            trace_dir = nxt0
            end_back = nxt1
        elif nxt1 in smoothable and nxt1 not in visited_smoothable:
            trace_dir = nxt1
            end_back = nxt0
        else:
            chains.append([nxt0, start_node, nxt1])
            continue
            
        curr = start_node
        nxt = trace_dir
        while nxt in smoothable and nxt not in visited_smoothable:
            chain.append(nxt)
            visited_smoothable.add(nxt)
            nxt_neighbors = adj[nxt]
            nxt_nxt = nxt_neighbors[0] if nxt_neighbors[0] != curr else nxt_neighbors[1]
            curr = nxt
            nxt = nxt_nxt
            
        end_forward = nxt
        chains.append([end_back] + chain + [end_forward])
        
    return chains

def smooth_graph(graph: StrandGraph, selected_mask: np.ndarray, max_deviation: float) -> StrandGraph:
    n = len(graph.nodes)
    if n == 0 or len(graph.edges) == 0:
        return graph

    adj = {i: [] for i in range(n)}
    for u, v in graph.edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
        
    smoothable = set()
    for i in range(n):
        if selected_mask[i] and len(adj[i]) == 2:
            smoothable.add(i)
            
    if not smoothable:
        return graph
        
    chains = get_chains(n, graph.edges, smoothable)
    
    new_edges = []
    for u, v in graph.edges:
        if int(u) not in smoothable and int(v) not in smoothable:
            new_edges.append((int(u), int(v)))
            
    kept_nodes = set()
    for i in range(n):
        if i not in smoothable:
            kept_nodes.add(i)

    for chain in chains:
        points = graph.nodes[chain]
        simplified_chain = rdp(points, chain, max_deviation)
        
        for i in range(len(simplified_chain) - 1):
            u, v = simplified_chain[i], simplified_chain[i+1]
            new_edges.append((u, v))
            kept_nodes.add(u)
            kept_nodes.add(v)
            
    kept_list = sorted(list(kept_nodes))
    remap = {old: new for new, old in enumerate(kept_list)}

    nodes_out = graph.nodes[kept_list]
    edges_out = np.array([(remap[u], remap[v]) for u, v in new_edges], dtype=np.int32)
    if len(edges_out) == 0:
        edges_out = np.empty((0, 2), dtype=np.int32)

    return StrandGraph(nodes=nodes_out, edges=edges_out)

nodes = np.array([
    [0, 0, 0], # 0 (end)
    [1, 0, 0], # 1 (smoothable)
    [2, 0.05, 0], # 2 (smoothable)
    [3, 0, 0], # 3 (smoothable)
    [4, 0, 0]  # 4 (end)
], dtype=np.float32)

edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 4]
])
graph = StrandGraph(nodes, edges)
mask = np.array([False, True, True, True, False])
smoothed = smooth_graph(graph, mask, 0.1)

print("Original nodes:", len(graph.nodes))
print("Smoothed nodes:", len(smoothed.nodes))
print("Edges:", smoothed.edges)

mask = np.array([False, True, True, True, False])
smoothed = smooth_graph(graph, mask, 0.01)
print("Smoothed nodes tight:", len(smoothed.nodes))

