import numpy as np
from scipy.interpolate import RBFInterpolator
from app.data.webmerge import farthest_point_sampling

def get_stationary_pins(positions: np.ndarray, active_anchors: np.ndarray, n_pins: int = 50) -> np.ndarray:
    """Find points far from active anchors to use as stationary pins."""
    N = len(positions)
    if N > 5000:
        idx = np.random.choice(N, 5000, replace=False)
        sample = positions[idx]
    else:
        sample = positions
        
    fps_idx = farthest_point_sampling(sample, n_samples=min(n_pins * 2, len(sample)))
    candidate_pins = sample[fps_idx]
    
    if len(active_anchors) == 0:
        return candidate_pins[:n_pins]
        
    stationary = []
    extent = np.ptp(positions, axis=0).max()
    threshold = extent * 0.15  # Pins must be at least 15% extent away from any active anchor
    
    for pin in candidate_pins:
        dists = np.linalg.norm(active_anchors - pin, axis=1)
        if dists.min() > threshold:
            stationary.append(pin)
            if len(stationary) == n_pins:
                break
                
    return np.array(stationary)

def tps_warp(positions: np.ndarray, source_anchors: np.ndarray, target_anchors: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """
    Warp positions using Thin Plate Spline (TPS) given matched source and target anchors.
    Adds stationary edge pins to prevent global distortion.
    """
    if len(source_anchors) == 0:
        return positions.copy()
        
    stationary_pins = get_stationary_pins(positions, source_anchors, n_pins=100)
    
    if len(stationary_pins) > 0:
        all_sources = np.vstack((source_anchors, stationary_pins))
        all_targets = np.vstack((target_anchors, stationary_pins))
    else:
        all_sources = source_anchors
        all_targets = target_anchors
        
    if len(all_sources) < 4:
        delta = np.mean(target_anchors - source_anchors, axis=0)
        return positions + delta
        
    deltas = all_targets - all_sources
    
    rbf = RBFInterpolator(
        all_sources, 
        deltas, 
        kernel='thin_plate_spline',
        smoothing=smoothing,
        degree=1
    )
    
    warp_deltas = rbf(positions)
    return positions + warp_deltas
