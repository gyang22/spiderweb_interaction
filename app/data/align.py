"""
Point cloud alignment utilities.

  icp_align     — Rigid ICP with 90° rotation candidate search (pure numpy)
  cpd_align     — Non-rigid CPD, density-adaptive beta schedule, full-cloud warp
  euler_to_transform — 6-DOF → 4×4 rigid transform

icp_align tries 0°/90°/180°/270° rotations around the dominant axis first
to handle the common case where two scans of the same web are taken from
orthogonal angles.

cpd_align matches the approach in align_pcd.py:
  • Random subsample both clouds to ≤ SAMPLE_SIZE points
  • Compute minimum-safe beta from neighbour spacing (prevents tearing)
  • Run multi-stage CPD (coarse-to-fine beta schedule, clamped to min_safe_beta)
  • Store control-point arrays (Y) and weight matrices (W) from each stage
  • Apply the full RBF warp to ALL source points (not just the sample)
"""

from __future__ import annotations
import numpy as np
from app.data.strand_graph import _voxelize


# ── SVD-based rigid transform estimator ──────────────────────────────────────

def _estimate_rigid_svd(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """
    Estimate the 4×4 rigid transform that maps src→tgt via SVD.
    src, tgt: (N, 3) float64 point correspondences.
    """
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    H = (src - src_c).T @ (tgt - tgt_c)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = tgt_c - R @ src_c
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ── Rotation-around-centroid helpers ─────────────────────────────────────────

def _rotation_about_axis(axis: str, deg: float, center: np.ndarray) -> np.ndarray:
    """
    4×4 transform for a rotation of `deg` degrees about `axis` ('x', 'y', 'z'),
    rotating around `center` (so the centroid stays fixed).
    """
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    elif axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    else:  # z
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t = center - R @ center
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ── Single ICP iteration block (shared by candidate search + full run) ────────

def _run_icp_block(
    src: np.ndarray,
    tgt: np.ndarray,
    T_init: np.ndarray,
    max_iter: int,
    convergence_tol: float,
    max_correspondence_dist,
    max_nn_per_iter: int,
) -> tuple[np.ndarray, float, int]:
    """Run ICP from T_init; return (T, rmse, n_inliers)."""
    T = T_init.copy().astype(np.float64)
    prev_rmse = np.inf
    rmse = np.inf
    n_inliers = 0

    for _ in range(max_iter):
        src_h = np.hstack([src, np.ones((len(src), 1))])
        src_t = (T @ src_h.T).T[:, :3]

        if len(src_t) > max_nn_per_iter:
            idx_sub = np.random.choice(len(src_t), max_nn_per_iter, replace=False)
            src_sub = src_t[idx_sub]
        else:
            src_sub = src_t

        diff   = src_sub[:, None, :] - tgt[None, :, :]
        dists  = np.sqrt((diff ** 2).sum(axis=-1))
        nn_idx = dists.argmin(axis=1)
        nn_d   = dists[np.arange(len(src_sub)), nn_idx]

        thresh = max_correspondence_dist if max_correspondence_dist is not None \
            else nn_d.mean() + nn_d.std()
        valid = nn_d < thresh

        n_inliers = int(valid.sum())
        if n_inliers < 4:
            break

        src_corr = src_sub[valid]
        tgt_corr = tgt[nn_idx[valid]]
        rmse     = float(nn_d[valid].mean())

        T_delta = _estimate_rigid_svd(src_corr, tgt_corr)
        T = T_delta @ T

        if abs(prev_rmse - rmse) < convergence_tol:
            break
        prev_rmse = rmse

    return T, rmse, n_inliers


# ── ICP with 90° rotation candidate search ────────────────────────────────────

def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: np.ndarray | None = None,
    voxel_size: float | None = None,
    max_iter: int = 50,
    convergence_tol: float = 1e-6,
    max_correspondence_dist: float | None = None,
    max_nn_per_iter: int = 3000,
    try_rotations: bool = True,
    rotation_probe_iter: int = 10,
) -> tuple[np.ndarray, float, int]:
    """
    Point-to-point ICP alignment with optional 90° rotation candidate search.

    Before running the full ICP, optionally tests four 90° rotations of the
    source cloud around each principal axis (12 candidates total) and picks
    the one with the lowest probe RMSE as the initial transform.  This handles
    the common situation where two spiderweb scans are taken from orthogonal
    camera angles.

    Parameters
    ----------
    source, target         : (N, 3) / (M, 3) float32 point positions
    init_transform         : 4×4 starting transform (identity if None)
    voxel_size             : downsample before alignment (auto = extent/50 if None)
    max_iter               : maximum ICP iterations for the final run
    convergence_tol        : stop when ΔRMSE < tol
    max_correspondence_dist: reject pairs farther than this (auto if None)
    max_nn_per_iter        : subsample source points per iteration for speed
    try_rotations          : if True, probe 0°/90°/180°/270° about X, Y, Z first
    rotation_probe_iter    : short ICP iterations used during candidate probing

    Returns
    -------
    transform : (4, 4) float32 — maps source → target
    rmse      : float — mean inlier correspondence distance
    n_inliers : int   — number of inlier correspondences at convergence
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Auto voxel size
    if voxel_size is None:
        ext = float(max(np.ptp(source, axis=0).max(), np.ptp(target, axis=0).max()))
        voxel_size = max(ext / 50.0, 1e-9)

    src = _voxelize(source.astype(np.float32), voxel_size).astype(np.float64)
    tgt = _voxelize(target.astype(np.float32), voxel_size).astype(np.float64)

    T_base = init_transform.astype(np.float64) if init_transform is not None \
        else np.eye(4, dtype=np.float64)

    if try_rotations:
        src_center = source.mean(axis=0)
        best_T   = T_base.copy()
        best_rmse = np.inf

        candidates = [T_base]
        for axis in ('x', 'y', 'z'):
            for deg in (90.0, 180.0, 270.0):
                R_cand = _rotation_about_axis(axis, deg, src_center)
                candidates.append(T_base @ R_cand)

        for T_cand in candidates:
            T_probe, rmse_probe, n_probe = _run_icp_block(
                src, tgt, T_cand,
                max_iter=rotation_probe_iter,
                convergence_tol=convergence_tol,
                max_correspondence_dist=max_correspondence_dist,
                max_nn_per_iter=max_nn_per_iter,
            )
            if n_probe >= 4 and rmse_probe < best_rmse:
                best_rmse = rmse_probe
                best_T    = T_probe

        T_base = best_T

    T, rmse, n_inliers = _run_icp_block(
        src, tgt, T_base,
        max_iter=max_iter,
        convergence_tol=convergence_tol,
        max_correspondence_dist=max_correspondence_dist,
        max_nn_per_iter=max_nn_per_iter,
    )

    return T.astype(np.float32), rmse, n_inliers


# ── Non-rigid CPD alignment ───────────────────────────────────────────────────

_CPD_SAMPLE_SIZE = 15_000   # random subsample used for CPD fitting (matches align_pcd.py)
_CPD_BATCH       = 2_000    # chunk size for full-cloud warp application


def _cpd_min_safe_beta(points: np.ndarray) -> float:
    """
    Compute the minimum safe beta for CPD from the average nearest-neighbour
    distance of the control-point cloud.  Mirrors get_control_point_spacing()
    from align_pcd.py: safe_floor = avg_nn_dist × 2.
    """
    # Pairwise O(N²) is too slow for large N; subsample if needed
    n = len(points)
    idx = np.random.choice(n, min(n, 2000), replace=False)
    pts = points[idx]
    diff  = pts[:, None, :] - pts[None, :, :]          # (M, M, 3)
    dists = np.sqrt((diff ** 2).sum(axis=-1))           # (M, M)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)                        # (M,)
    avg_spacing = float(nn_dists[np.isfinite(nn_dists)].mean())
    return avg_spacing * 2.0


def _apply_cpd_warp(
    points: np.ndarray,
    control_points: np.ndarray,
    weights: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Apply CPD Gaussian RBF displacement to an arbitrary set of points.

    displacement(x) = G(x, Y) @ W   where G_ij = exp(-||x_i - y_j||² / 2β²)

    Batched to avoid allocating an N×M matrix at once.
    """
    warped = np.zeros_like(points)
    for i in range(0, len(points), _CPD_BATCH):
        chunk = points[i:i + _CPD_BATCH]
        diff    = chunk[:, None, :] - control_points[None, :, :]   # (B, M, 3)
        dist_sq = (diff ** 2).sum(axis=-1)                          # (B, M)
        G       = np.exp(-dist_sq / (2.0 * beta ** 2))             # (B, M)
        warped[i:i + _CPD_BATCH] = chunk + G @ weights
    return warped


def cpd_align(
    source: np.ndarray,
    target: np.ndarray,
    beta_schedule: list[float] | None = None,
    alpha: float = 1e-6,
    sample_size: int = _CPD_SAMPLE_SIZE,
    max_iter: int = 150,
    progress_callback=None,
) -> np.ndarray:
    """
    Non-rigid CPD (Coherent Point Drift) alignment, full-cloud output.

    Procedure (matches align_pcd.py):
      1. Randomly subsample both clouds to ≤ `sample_size` points.
      2. Compute min-safe beta from control-point neighbour spacing.
      3. Run multi-stage CPD with a coarse-to-fine beta schedule, clamping
         each stage so beta ≥ min_safe_beta (prevents tearing).
      4. After fitting each stage, store the control points Y and weight
         matrix W, then advance the sample via reg.transform_point_cloud().
      5. Apply all stored (Y, W, β) stages to ALL source points via the
         Gaussian RBF function — not just the subsample.

    Parameters
    ----------
    source           : (N, 3) float32 — source cloud (already roughly aligned via ICP)
    target           : (M, 3) float32 — target / reference cloud
    beta_schedule    : beta values, coarsest first (default [60, 15, 5, 2])
    alpha            : CPD regularisation weight (matches align_pcd.py default 1e-6)
    sample_size      : max points used for CPD fitting (default 15 000)
    max_iter         : max CPD iterations per stage
    progress_callback: optional callable(stage_idx, n_stages, msg: str)

    Returns
    -------
    warped : (N, 3) float32 — ALL source points after applying the deformation field
    """
    try:
        from pycpd import DeformableRegistration
    except ImportError as exc:
        raise ImportError(
            "pycpd is required for non-rigid CPD alignment.\n"
            "Install it with:  pip install pycpd"
        ) from exc

    if beta_schedule is None:
        beta_schedule = [60.0, 15.0, 5.0, 2.0]

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if len(source) < 2 or len(target) < 2:
        raise ValueError("Too few points for CPD alignment.")

    # ── Random subsampling for CPD fitting ────────────────────────────────────
    idx_s = np.random.choice(len(source), min(len(source), sample_size), replace=False)
    idx_t = np.random.choice(len(target), min(len(target), sample_size), replace=False)
    src_sample = source[idx_s].copy()
    tgt_sample = target[idx_t].copy()

    # ── Density-adaptive beta clamping ────────────────────────────────────────
    min_safe_beta = _cpd_min_safe_beta(src_sample)

    final_schedule: list[float] = []
    for b in beta_schedule:
        safe_b = max(b, min_safe_beta)
        if not final_schedule or safe_b < final_schedule[-1]:
            final_schedule.append(safe_b)
    if not final_schedule:
        final_schedule = [min_safe_beta]

    n_stages = len(final_schedule)

    # ── Multi-stage CPD fitting (on subsample) ────────────────────────────────
    stage_params: list[tuple[np.ndarray, np.ndarray, float]] = []  # (Y, W, beta)
    current_sample = src_sample.copy()

    for stage_idx, beta in enumerate(final_schedule):
        if progress_callback is not None:
            progress_callback(stage_idx, n_stages, f"Stage {stage_idx+1}/{n_stages}  β={beta:.1f}")

        reg = DeformableRegistration(
            X=tgt_sample,
            Y=current_sample,
            alpha=float(alpha),
            beta=float(beta),
            max_iterations=max_iter,
            tolerance=1e-5,
        )
        reg.register()

        stage_params.append((current_sample.copy(), np.array(reg.W), float(reg.beta)))
        current_sample = reg.transform_point_cloud(current_sample)

    # ── Apply full deformation field to ALL source points ─────────────────────
    if progress_callback is not None:
        progress_callback(n_stages, n_stages, "Applying warp to full cloud…")

    warped = source.copy()
    for Y, W, beta in stage_params:
        warped = _apply_cpd_warp(warped, Y, W, beta)

    return warped.astype(np.float32)


# ── Euler angles → 4×4 transform ─────────────────────────────────────────────

def euler_to_transform(
    tx: float, ty: float, tz: float,
    yaw_deg: float, pitch_deg: float, roll_deg: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a 4×4 float32 rigid transform.

    Rotation order: Rz(yaw) @ Ry(pitch) @ Rx(roll), applied around *center*
    (defaults to origin if None).  Translation (tx, ty, tz) is applied after.

    This means: first spin the cloud around its own center, then translate it.
    """
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,             cp*cr  ],
    ], dtype=np.float64)

    c = np.asarray(center, dtype=np.float64) if center is not None \
        else np.zeros(3, dtype=np.float64)

    t_rot = c - R @ c

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3]  = t_rot + np.array([tx, ty, tz], dtype=np.float64)
    return T
