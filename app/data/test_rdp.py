import numpy as np

def point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return float(np.linalg.norm(point - start))
    line_dir = line_vec / line_len
    point_vec = point - start
    proj = np.dot(point_vec, line_dir)
    proj_clamped = max(0.0, min(float(line_len), float(proj)))
    closest = start + proj_clamped * line_dir
    return float(np.linalg.norm(point - closest))

def rdp(points: np.ndarray, indices: list[int], epsilon: float) -> list[int]:
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

points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [4.0, 0.0, 0.0]
], dtype=np.float32)
indices = [0, 1, 2, 3, 4]
print(rdp(points, indices, 0.1))

points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.1, 0.0],
    [2.0, 0.0, 0.0],
    [3.0, -0.1, 0.0],
    [4.0, 0.0, 0.0]
], dtype=np.float32)
print(rdp(points, indices, 0.15))
print(rdp(points, indices, 0.05))

