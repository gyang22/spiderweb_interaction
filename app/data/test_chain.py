import numpy as np

def _get_chains(n: int, edges: np.ndarray, smoothable: set[int]) -> list[list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
        
    visited_smoothable: set[int] = set()
    chains: list[list[int]] = []
    
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

edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
n = 5
smoothable = {1, 2, 3}
print("Chain 1:", _get_chains(n, edges, smoothable))

edges = np.array([[0, 1], [1, 2], [2, 3]])
n = 4
smoothable = {1, 2}
print("Chain 2:", _get_chains(n, edges, smoothable))
