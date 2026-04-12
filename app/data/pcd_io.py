"""
PCD file I/O — supports ASCII, binary, and binary_compressed formats.
"""

import io
import struct
import numpy as np
from pathlib import Path
from app.data.point_cloud import PointCloud


# ── header parsing ────────────────────────────────────────────────────────────

def _parse_header(f) -> dict:
    """Read PCD header lines until DATA line. Returns header dict."""
    header = {}
    while True:
        raw = f.readline()
        if isinstance(raw, bytes):
            line = raw.decode('ascii', errors='replace').strip()
        else:
            line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        key = parts[0].upper()
        val = parts[1:]
        if key == 'VERSION':
            header['VERSION'] = val[0]
        elif key == 'FIELDS':
            header['FIELDS'] = val
        elif key == 'SIZE':
            header['SIZE'] = [int(v) for v in val]
        elif key == 'TYPE':
            header['TYPE'] = val
        elif key == 'COUNT':
            header['COUNT'] = [int(v) for v in val]
        elif key == 'WIDTH':
            header['WIDTH'] = int(val[0])
        elif key == 'HEIGHT':
            header['HEIGHT'] = int(val[0])
        elif key == 'POINTS':
            header['POINTS'] = int(val[0])
        elif key == 'DATA':
            header['DATA'] = val[0].lower()
            break
    return header


def _build_dtype(header: dict):
    """Build numpy dtype from PCD FIELDS/SIZE/TYPE/COUNT."""
    type_map = {
        ('F', 4): np.float32,
        ('F', 8): np.float64,
        ('I', 1): np.int8,
        ('I', 2): np.int16,
        ('I', 4): np.int32,
        ('I', 8): np.int64,
        ('U', 1): np.uint8,
        ('U', 2): np.uint16,
        ('U', 4): np.uint32,
        ('U', 8): np.uint64,
    }
    dt_fields = []
    for field, size, typ, count in zip(
        header['FIELDS'], header['SIZE'], header['TYPE'], header['COUNT']
    ):
        np_type = type_map.get((typ.upper(), size), np.uint8)
        if count == 1:
            dt_fields.append((field, np_type))
        else:
            dt_fields.append((field, np_type, (count,)))
    return np.dtype(dt_fields)


# ── RGB/RGBA unpacking from packed float ─────────────────────────────────────

def _unpack_rgb(packed: np.ndarray) -> np.ndarray:
    """Unpack float-encoded RGB (common in PCL output) → (N,3) float32 [0,1]."""
    rgb_int = packed.view(np.uint32)
    r = ((rgb_int >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_int >> 8)  & 0xFF).astype(np.float32) / 255.0
    b = ( rgb_int        & 0xFF).astype(np.float32) / 255.0
    return np.stack([r, g, b], axis=1)


def _extract_colors(data: np.ndarray, fields: list[str]) -> np.ndarray | None:
    """Try to extract RGBA color from structured array. Returns (N,4) float32 or None."""
    fields_lower = [f.lower() for f in fields]
    n = len(data)

    # rgba packed float
    if 'rgba' in fields_lower:
        rgba_int = data[fields[fields_lower.index('rgba')]].view(np.uint32)
        r = ((rgba_int >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgba_int >> 8)  & 0xFF).astype(np.float32) / 255.0
        b = ( rgba_int        & 0xFF).astype(np.float32) / 255.0
        a = ((rgba_int >> 24) & 0xFF).astype(np.float32) / 255.0
        return np.stack([r, g, b, a], axis=1)

    # rgb packed float
    if 'rgb' in fields_lower:
        rgb = _unpack_rgb(data[fields[fields_lower.index('rgb')]])
        alpha = np.ones((n, 1), dtype=np.float32)
        return np.concatenate([rgb, alpha], axis=1)

    # separate r g b channels (normalized 0–1 float)
    has_r = 'r' in fields_lower
    has_g = 'g' in fields_lower
    has_b = 'b' in fields_lower
    if has_r and has_g and has_b:
        r = data[fields[fields_lower.index('r')]].astype(np.float32)
        g = data[fields[fields_lower.index('g')]].astype(np.float32)
        b = data[fields[fields_lower.index('b')]].astype(np.float32)
        # Detect uint8 channels and normalize
        if r.max() > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        alpha = np.ones((n, 1), dtype=np.float32)
        return np.concatenate([r[:, None], g[:, None], b[:, None], alpha], axis=1)

    return None


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_ascii(f, header: dict) -> np.ndarray:
    n = header['POINTS']
    fields = header['FIELDS']
    data_lines = []
    for _ in range(n):
        line = f.readline()
        if isinstance(line, bytes):
            line = line.decode('ascii', errors='replace')
        data_lines.append(line)
    raw = np.loadtxt(io.StringIO(''.join(data_lines)), dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    # Build structured array manually for consistent downstream handling
    dtype = _build_dtype(header)
    struct_data = np.zeros(n, dtype=dtype)
    for i, field in enumerate(fields):
        struct_data[field] = raw[:, i]
    return struct_data


def _load_binary(f, header: dict) -> np.ndarray:
    dtype = _build_dtype(header)
    n = header['POINTS']
    raw = f.read(n * dtype.itemsize)
    return np.frombuffer(raw, dtype=dtype).copy()


def _load_binary_compressed(f, header: dict) -> np.ndarray:
    try:
        import lzf
    except ImportError:
        raise ImportError(
            "binary_compressed PCD requires 'python-lzf'. Install: pip install python-lzf"
        )
    compressed_size = struct.unpack('<I', f.read(4))[0]
    uncompressed_size = struct.unpack('<I', f.read(4))[0]
    compressed_data = f.read(compressed_size)
    raw = lzf.decompress(compressed_data, uncompressed_size)
    dtype = _build_dtype(header)
    return np.frombuffer(raw, dtype=dtype).copy()


# ── public API ────────────────────────────────────────────────────────────────

def load_pcd(path: str | Path) -> PointCloud:
    """Load a .pcd file and return a PointCloud."""
    path = Path(path)
    with open(path, 'rb') as f:
        header = _parse_header(f)

    data_format = header.get('DATA', 'ascii')
    fields = header.get('FIELDS', [])
    fields_lower = [fld.lower() for fld in fields]

    with open(path, 'rb') as f:
        _parse_header(f)  # advance past header to DATA position
        if data_format == 'ascii':
            data = _load_ascii(f, header)
        elif data_format == 'binary':
            data = _load_binary(f, header)
        elif data_format == 'binary_compressed':
            data = _load_binary_compressed(f, header)
        else:
            raise ValueError(f"Unknown PCD DATA format: {data_format}")

    # Extract XYZ
    if 'x' not in fields_lower or 'y' not in fields_lower or 'z' not in fields_lower:
        raise ValueError("PCD file must contain x, y, z fields.")
    xi = fields_lower.index('x')
    yi = fields_lower.index('y')
    zi = fields_lower.index('z')
    positions = np.stack([
        data[fields[xi]].astype(np.float32),
        data[fields[yi]].astype(np.float32),
        data[fields[zi]].astype(np.float32),
    ], axis=1)

    # Remove NaN/Inf points
    valid = np.isfinite(positions).all(axis=1)
    positions = positions[valid]

    # Extract colors
    colors = _extract_colors(data[valid], fields)

    return PointCloud(positions, colors)


def save_pcd(pc: PointCloud, path: str | Path) -> None:
    """Save alive points as a binary PCD file."""
    path = Path(path)
    mask = pc.alive_mask
    positions = pc.positions[mask]
    colors = pc.colors[mask]
    n = len(positions)

    # Build structured array: x y z rgb (packed float)
    dt = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)])
    out = np.zeros(n, dtype=dt)
    out['x'] = positions[:, 0]
    out['y'] = positions[:, 1]
    out['z'] = positions[:, 2]

    # Pack RGB into float
    r = (colors[:, 0] * 255).astype(np.uint32)
    g = (colors[:, 1] * 255).astype(np.uint32)
    b = (colors[:, 2] * 255).astype(np.uint32)
    rgb_int = (r << 16) | (g << 8) | b
    out['rgb'] = rgb_int.view(np.float32)

    header = (
        f"# .PCD v0.7 - Point Cloud Data\n"
        f"VERSION 0.7\n"
        f"FIELDS x y z rgb\n"
        f"SIZE 4 4 4 4\n"
        f"TYPE F F F F\n"
        f"COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    )

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(out.tobytes())
