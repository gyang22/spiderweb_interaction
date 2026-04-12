"""Voxel-grid downsampling for PointCloud objects."""

from __future__ import annotations
import numpy as np

from app.data.point_cloud import PointCloud


def voxel_downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    """
    Return a new PointCloud with one representative point per voxel cell.

    Only alive points are considered. The representative position is the
    centroid of all original points in the cell; the representative color
    is the per-channel average of those same points.

    Parameters
    ----------
    pc         : source PointCloud (only alive points are downsampled)
    voxel_size : edge length of each cubic voxel cell

    Returns
    -------
    PointCloud with alive_mask all-True and no selection
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    alive_idx = np.where(pc.alive_mask)[0]
    if len(alive_idx) == 0:
        return PointCloud(np.empty((0, 3), dtype=np.float32))

    positions = pc.positions[alive_idx]   # (N, 3)
    colors    = pc.colors[alive_idx]      # (N, 4)

    # ── Voxel grouping ────────────────────────────────────────────────────────
    shifted = positions - positions.min(axis=0)
    keys    = np.floor(shifted / voxel_size).astype(np.int64)

    key_struct = np.ascontiguousarray(keys).view(
        np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])
    ).ravel()

    _, first_idx, inverse = np.unique(
        key_struct, return_index=True, return_inverse=True
    )
    M = len(first_idx)

    # ── Centroid positions and averaged colors ────────────────────────────────
    new_pos    = np.zeros((M, 3), dtype=np.float64)
    new_colors = np.zeros((M, 4), dtype=np.float64)
    counts     = np.zeros(M, dtype=np.int64)

    np.add.at(new_pos,    inverse, positions)
    np.add.at(new_colors, inverse, colors)
    np.add.at(counts,     inverse, 1)

    new_pos    /= counts[:, None]
    new_colors /= counts[:, None]
    new_colors  = np.clip(new_colors, 0.0, 1.0)

    return PointCloud(
        new_pos.astype(np.float32),
        new_colors.astype(np.float32),
    )
