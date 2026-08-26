"""
I/O for the organelle segmentation pipeline: ND2 loading, PNG saving,
uint8 scaling, and samplesheet <-> image-path resolution.

Named image_io (not io) to avoid shadowing Python's stdlib `io` module when
these modules live flat in one directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import nd2  # replaces tifffile

from PIL import Image


def _imsave(path: Path, arr: np.ndarray):
    """Save a uint8 array as PNG using Pillow (avoids imageio/tifffile backend issues)."""
    Image.fromarray(arr).save(path)


def to_uint8(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    """Percentile-stretch to uint8 for visualization."""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (255 * x).astype(np.uint8)


def build_img_lookup(img_dir: Path, pattern: str = "*.nd2") -> Dict[str, Path]:
    imgs = sorted(img_dir.glob(pattern))
    return {p.name: p for p in imgs}


def add_image_paths(samplesheet: pd.DataFrame, img_lookup: Dict[str, Path]) -> pd.DataFrame:
    out = samplesheet.copy()
    out["image_path"] = out["filename"].map(lambda fn: img_lookup.get(fn))
    return out


def load_nd2_as_czyx(img_path: Path) -> np.ndarray:
    """
    Open an ND2 file and return a (C, H, W) array via max projection over Z.

    nd2.imread returns an array whose axis order is reported by the file's
    dimension labels (accessible via nd2.ND2File).  We normalise to (C, Z, H, W)
    before projecting so the rest of the pipeline receives the same (C, H, W)
    shape it previously got from tifffile.

    Supported axis orderings emitted by the nd2 package:
      CZYX, ZCYX, TCZYX (T is dropped by taking index 0), and the degenerate
      CYX / ZYX cases (no-Z or no-C, handled gracefully).
    """
    with nd2.ND2File(img_path) as f:
        # axes string, e.g. 'CZYX', 'ZCYX', 'TCZYX' …
        axes: str = "".join(f.sizes.keys()).upper()
        arr: np.ndarray = f.asarray()          # numpy array, dtype preserved

    # --- drop T if present (take first time-point) ---
    if "T" in axes:
        t_idx = axes.index("T")
        arr = arr.take(0, axis=t_idx)
        axes = axes.replace("T", "")

    # arr is now one of: CZYX, ZCYX, CYX, ZYX, YX …
    # Normalise to (C, Z, Y, X) or (C, Y, X) so we can max-project cleanly.

    if axes == "CZYX":
        pass                                   # already (C, Z, Y, X)
    elif axes == "ZCYX":
        arr = np.moveaxis(arr, 0, 1)           # → (C, Z, Y, X)
        axes = "CZYX"
    elif axes == "ZYX":
        arr = arr[np.newaxis]                  # → (1, Z, Y, X)  — single channel
        axes = "CZYX"
    elif axes == "CYX":
        # no Z dimension — nothing to project, just return
        return arr                             # (C, H, W)
    elif axes == "YX":
        return arr[np.newaxis]                 # (1, H, W)
    else:
        raise ValueError(
            f"Unexpected axis order '{axes}' in {img_path.name}. "
            "Please extend load_nd2_as_czyx() to handle it."
        )

    # Max-project over Z (axis 1) → (C, H, W)
    max_proj: np.ndarray = arr.max(axis=1)
    return max_proj