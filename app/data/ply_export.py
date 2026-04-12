"""PLY export using plyfile (lightweight, no open3d dependency)."""

from pathlib import Path
import numpy as np
from app.data.point_cloud import PointCloud


def export_ply(pc: PointCloud, path: str | Path) -> None:
    """Export alive points to a binary PLY file."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise ImportError("plyfile is required for PLY export. Install: pip install plyfile")

    path = Path(path)
    mask = pc.alive_mask
    positions = pc.positions[mask]
    colors = pc.colors[mask]
    n = len(positions)

    r = (np.clip(colors[:, 0], 0, 1) * 255).astype(np.uint8)
    g = (np.clip(colors[:, 1], 0, 1) * 255).astype(np.uint8)
    b = (np.clip(colors[:, 2], 0, 1) * 255).astype(np.uint8)

    vertex_data = np.zeros(n, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('red', np.uint8), ('green', np.uint8), ('blue', np.uint8),
    ])
    vertex_data['x'] = positions[:, 0]
    vertex_data['y'] = positions[:, 1]
    vertex_data['z'] = positions[:, 2]
    vertex_data['red'] = r
    vertex_data['green'] = g
    vertex_data['blue'] = b

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], byte_order='<').write(str(path))
