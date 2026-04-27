import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    N = points.shape[0]
    if n_samples >= N:
        return np.arange(N)
    selected_indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)
    for _ in range(1, n_samples):
        last_point = points[selected_indices[-1]]
        dists = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dists)
        next_index = int(np.argmax(distances))
        selected_indices.append(next_index)
    return np.array(selected_indices)

def laplacian_contraction(points: np.ndarray, knn_idx: np.ndarray, lam: float = 0.4, iterations: int = 30) -> np.ndarray:
    pts = points.copy()
    N = pts.shape[0]
    for _ in range(iterations):
        new_pts = pts.copy()
        for i in range(N):
            neigh_idx = knn_idx[i, 1:]
            neigh = pts[neigh_idx]
            if len(neigh) == 0:
                continue
            centroid = neigh.mean(axis=0)
            lam_eff = lam / (1.0 + len(neigh) * 0.1)
            new_pts[i] = pts[i] + lam_eff * (centroid - pts[i])
        pts = new_pts
    return pts

def tensor_vote_extrapolate(points: np.ndarray, search_radius: float = 20.0, vote_steps: int = 5, step_size: float = 2.5) -> np.ndarray:
    points = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
    pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30))
    covariances = np.asarray(pcd.covariances)
    
    new_points = []
    nbrs = NearestNeighbors(n_neighbors=10).fit(points)
    
    for i in range(len(points)):
        cov = covariances[i]
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = eigvals.argsort()[::-1]
        v1 = eigvecs[:, idx[0]]
        lambda1, lambda2 = eigvals[idx[0]], eigvals[idx[1]]
        
        if lambda1 < 1e-6:
            continue
            
        if (lambda1 - lambda2) / lambda1 > 0.8:
            indices = nbrs.kneighbors([points[i]], return_distance=False)[0]
            local_cloud = points[indices]
            vec_to_center = np.mean(local_cloud, axis=0) - points[i]
            shoot_dir = -v1 if np.dot(vec_to_center, v1) > 0 else v1
            
            if np.linalg.norm(vec_to_center) > (search_radius * 0.2):
                curr = points[i].copy()
                for _ in range(vote_steps):
                    curr += shoot_dir * step_size
                    new_points.append(curr.copy())
                    
    return np.array(new_points)

def webmerge_skeletonize(points: np.ndarray, search_radius: float = 20.0, vote_steps: int = 5, step_size: float = 2.5, lam: float = 0.4, iterations: int = 30) -> np.ndarray:
    """
    Applies the WebMerge skeletonization algorithm on raw points.
    Returns the skeleton node positions.
    """
    if len(points) == 0:
        return np.empty((0, 3))
        
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    dists, _ = nbrs.kneighbors(points)
    eps = np.median(dists[:, 1]) * 8
    labels = DBSCAN(eps=eps, min_samples=5).fit(points).labels_
    
    skel_pts = []
    for lbl in np.unique(labels):
        c_pts = points[labels == lbl]
        k = max(1, int(len(c_pts) * 0.02))
        fps_idx = farthest_point_sampling(c_pts, k)
        c_red = c_pts[fps_idx]
        if len(c_red) < 15:
            skel_pts.append(c_red)
            continue
        knn_idx = NearestNeighbors(n_neighbors=8).fit(c_red).kneighbors(c_red, return_distance=False)
        skel_pts.append(laplacian_contraction(c_red, knn_idx, lam=lam, iterations=iterations))
    
    base_points = np.vstack(skel_pts) if skel_pts else np.empty((0, 3))
    accum_new = np.empty((0, 3))
    
    for _ in range(8):
        ctx = np.vstack((base_points, accum_new)) if len(accum_new) > 0 else base_points
        growth = tensor_vote_extrapolate(ctx, search_radius=search_radius, vote_steps=vote_steps, step_size=step_size)
        if len(growth) == 0:
            break
        accum_new = np.vstack((accum_new, growth))
        p_temp = o3d.geometry.PointCloud()
        p_temp.points = o3d.utility.Vector3dVector(accum_new)
        accum_new = np.asarray(p_temp.voxel_down_sample(1.1).points)
    
    final_pts = np.vstack((base_points, accum_new)) if len(accum_new) > 0 else base_points
    return final_pts
