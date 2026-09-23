"""
Per-cell measurements for the organelle pipeline: geometry, colocalization,
area metrics, spatial (perinuclear) metrics, edge flags, nucleus-overlap
diagnostics, and a safe merge helper.

Pure numpy/pandas/skimage — no cellpose, no I/O. Downstream analysis scripts
can import from here without pulling in the segmentation stack.
"""
from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from scipy import ndimage as ndi  # noqa: F401  (kept for parity / future use)

from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table


def find_edge_touching_labels(label_image: np.ndarray, margin: int = 0) -> set:
    h, w = label_image.shape
    labels = set()
    if margin <= 0:
        labels |= set(np.unique(label_image[0, :]))
        labels |= set(np.unique(label_image[h-1, :]))
        labels |= set(np.unique(label_image[:, 0]))
        labels |= set(np.unique(label_image[:, w-1]))
    else:
        labels |= set(np.unique(label_image[:margin, :]))
        labels |= set(np.unique(label_image[h-margin:, :]))
        labels |= set(np.unique(label_image[:, :margin]))
        labels |= set(np.unique(label_image[:, w-margin:]))
    labels.discard(0)
    return labels


def compute_edge_flags_bbox(cell_masks: np.ndarray, margin: int = 0) -> Dict[int, bool]:
    h, w = cell_masks.shape
    props = regionprops_table(cell_masks, properties=("label", "bbox"))
    if not props or len(props.get("label", [])) == 0:
        return {}

    labels = np.asarray(props["label"], dtype=int)
    ymin = np.asarray(props["bbox-0"], dtype=int)
    xmin = np.asarray(props["bbox-1"], dtype=int)
    ymax = np.asarray(props["bbox-2"], dtype=int)
    xmax = np.asarray(props["bbox-3"], dtype=int)

    flags: Dict[int, bool] = {}
    for lab, y0, x0, y1, x1 in zip(labels, ymin, xmin, ymax, xmax):
        touches = (y0 <= margin) or (x0 <= margin) or (y1 >= (h - margin)) or (x1 >= (w - margin))
        flags[int(lab)] = bool(touches)
    return flags


def compute_nuc_overlap_and_area_ratio(
    cell_masks: np.ndarray,
    nuc_masks: np.ndarray,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    flat_cells = cell_masks.ravel()
    flat_nucs = nuc_masks.ravel()
    max_cell = int(flat_cells.max())
    max_nuc = int(flat_nucs.max())

    cell_area = np.bincount(flat_cells, minlength=max_cell + 1)
    nuc_area = np.bincount(flat_nucs, minlength=max_nuc + 1)

    mask_both = (flat_cells > 0) & (flat_nucs > 0)
    if mask_both.any():
        overlap_counts = np.bincount(flat_cells[mask_both], minlength=max_cell + 1)
    else:
        overlap_counts = np.zeros(max_cell + 1, dtype=int)

    overlap_pairs_cells = flat_cells[mask_both]
    overlap_pairs_nucs = flat_nucs[mask_both]

    overlap_frac = {}
    area_ratio = {}

    if overlap_pairs_cells.size == 0:
        for cid in np.unique(flat_cells):
            if cid == 0:
                continue
            a = int(cell_area[cid])
            overlap_frac[int(cid)] = 0.0
            area_ratio[int(cid)] = np.inf
        return overlap_frac, area_ratio

    cell_to_nucs = {}
    for c, n in zip(overlap_pairs_cells, overlap_pairs_nucs):
        cell_to_nucs.setdefault(int(c), set()).add(int(n))

    unique_cells = np.unique(flat_cells)
    unique_cells = unique_cells[unique_cells > 0]
    for cid in unique_cells:
        cid = int(cid)
        c_area = int(cell_area[cid]) if cid <= max_cell else 0
        ov = int(overlap_counts[cid]) if cid <= max_cell else 0
        overlap_frac[cid] = float(ov) / float(c_area) if c_area > 0 else 0.0

        nucs = cell_to_nucs.get(cid, set())
        if not nucs:
            area_ratio[cid] = np.inf
        else:
            nuc_areas = [int(nuc_area[nl]) for nl in nucs if nl <= max_nuc]
            largest = max(nuc_areas) if nuc_areas else 0
            area_ratio[cid] = float(c_area) / float(largest) if largest > 0 else np.inf

    return overlap_frac, area_ratio


def safe_merge_on_id(left: pd.DataFrame, right: pd.DataFrame, on: str = "cell_id", keep_on_left: bool = True) -> pd.DataFrame:
    if right is None or right.empty:
        return left.copy()

    left_cols = set(left.columns)
    right_cols = [c for c in right.columns if c != on]

    to_take = [c for c in right_cols if c not in left_cols]
    if not to_take:
        return left.copy()

    right_sel = right[[on] + to_take].copy()
    merged = left.merge(right_sel, on=on, how="left", validate="one_to_one")
    return merged


def compute_cell_coloc(
    cell_masks: np.ndarray,
    lys_int: np.ndarray,
    prab_int: np.ndarray,
    min_pixels: int = 50,
) -> pd.DataFrame:
    cell_ids = np.unique(cell_masks)
    cell_ids = cell_ids[cell_ids != 0]

    rows: List[dict] = []

    for cid in cell_ids:
        m = (cell_masks == cid)
        npx = int(m.sum())
        if npx < min_pixels:
            continue

        a = lys_int[m].astype(np.float64)
        b = prab_int[m].astype(np.float64)

        if a.std() == 0 or b.std() == 0:
            pearson_r = np.nan
        else:
            pearson_r = float(np.corrcoef(a, b)[0, 1])

        try:
            tA = float(threshold_otsu(a)) if np.unique(a).size > 1 else 0.0
        except Exception:
            tA = 0.0
        try:
            tB = float(threshold_otsu(b)) if np.unique(b).size > 1 else 0.0
        except Exception:
            tB = 0.0

        a_pos = a > tA
        b_pos = b > tB

        denomA = float(a[a_pos].sum())
        denomB = float(b[b_pos].sum())

        M1 = float(a[a_pos & b_pos].sum() / denomA) if denomA > 0 else np.nan
        M2 = float(b[b_pos & a_pos].sum() / denomB) if denomB > 0 else np.nan

        rows.append({
            "cell_id": int(cid),
            "cell_pixels": npx,
            "pearson_r": pearson_r,
            "manders_M1_lys_in_prab": M1,
            "manders_M2_prab_in_lys": M2,
            "threshold_cutoff_lys": tA,
            "threshold_cutoff_prab": tB,
        })

    return pd.DataFrame(rows)


def compute_cell_geometry(cell_masks: np.ndarray) -> pd.DataFrame:
    props = regionprops_table(
        cell_masks,
        properties=("label", "area", "centroid", "bbox")
    )
    df = pd.DataFrame(props).rename(columns={
        "label": "cell_id",
        "area": "cell_area_px_geom",
        "centroid-0": "cell_centroid_y",
        "centroid-1": "cell_centroid_x",
        "bbox-0": "bbox_ymin",
        "bbox-1": "bbox_xmin",
        "bbox-2": "bbox_ymax",
        "bbox-3": "bbox_xmax",
    })
    df["cell_id"] = df["cell_id"].astype(int)
    return df


def compute_per_cell_area_metrics(
    cell_masks: np.ndarray,
    nuc_masks: np.ndarray,
    lys_mask: np.ndarray,
    prab_mask: np.ndarray,
) -> pd.DataFrame:
    flat_cells = cell_masks.ravel()
    max_lab = int(flat_cells.max())
    counts = np.bincount(flat_cells, minlength=max_lab + 1)
    nuc_overlap = np.bincount(flat_cells, weights=(nuc_masks.ravel() > 0).astype(int), minlength=max_lab + 1)
    lys_overlap = np.bincount(flat_cells, weights=lys_mask.ravel().astype(int), minlength=max_lab + 1)
    prab_overlap = np.bincount(flat_cells, weights=prab_mask.ravel().astype(int), minlength=max_lab + 1)

    rows = []
    for cid in range(1, max_lab + 1):
        if counts[cid] == 0:
            continue
        rows.append({
            "cell_id": int(cid),
            "cell_area_px": int(counts[cid]),
            "nuc_area_px_in_cell": int(nuc_overlap[cid]),
            "lys_segmented_area_px_in_cell": int(lys_overlap[cid]),
            "prab_segmented_area_px_in_cell": int(prab_overlap[cid]),
        })
    return pd.DataFrame(rows)


def compute_per_cell_signal_spatial_metrics(
    cell_masks: np.ndarray,
    nuc_masks: np.ndarray,
    signal_mask: np.ndarray,
    signal_int: np.ndarray,
    prefix: str,
    perinuclear_radius_px: float = 10.0,
) -> pd.DataFrame:
    cell_ids = np.unique(cell_masks)
    cell_ids = cell_ids[cell_ids != 0]

    rows = []
    r = float(perinuclear_radius_px)

    for cid in cell_ids:
        cell_m = (cell_masks == cid)

        nuc_m = (nuc_masks > 0) & cell_m
        if nuc_m.sum() == 0:
            rows.append({
                "cell_id": int(cid),
                "nuc_centroid_y": np.nan,
                "nuc_centroid_x": np.nan,
                f"{prefix}_centroid_y": np.nan,
                f"{prefix}_centroid_x": np.nan,
                f"dist_nuc_to_{prefix}_centroid_px": np.nan,
                f"{prefix}_dist_mean_px": np.nan,
                f"{prefix}_dist_median_px": np.nan,
                f"{prefix}_dist_p90_px": np.nan,
                f"{prefix}_perinuclear_frac_r{int(r)}px": np.nan,
                f"{prefix}_pixels_in_cell": int((signal_mask & cell_m).sum()),
            })
            continue

        nuc_yx = np.argwhere(nuc_m)
        nuc_cy, nuc_cx = nuc_yx.mean(axis=0)

        sig_m = signal_mask & cell_m
        n_sig_px = int(sig_m.sum())
        if n_sig_px == 0:
            rows.append({
                "cell_id": int(cid),
                "nuc_centroid_y": float(nuc_cy),
                "nuc_centroid_x": float(nuc_cx),
                f"{prefix}_centroid_y": np.nan,
                f"{prefix}_centroid_x": np.nan,
                f"dist_nuc_to_{prefix}_centroid_px": np.nan,
                f"{prefix}_dist_mean_px": np.nan,
                f"{prefix}_dist_median_px": np.nan,
                f"{prefix}_dist_p90_px": np.nan,
                f"{prefix}_perinuclear_frac_r{int(r)}px": 0.0,
                f"{prefix}_pixels_in_cell": 0,
            })
            continue

        sig_coords = np.argwhere(sig_m)
        weights = signal_int[sig_m].astype(np.float64)
        wsum = weights.sum()

        if wsum > 0:
            sig_cy = float((sig_coords[:, 0] * weights).sum() / wsum)
            sig_cx = float((sig_coords[:, 1] * weights).sum() / wsum)
        else:
            sig_cy, sig_cx = sig_coords.mean(axis=0).astype(float)

        d_centroid = float(np.hypot(sig_cy - nuc_cy, sig_cx - nuc_cx))

        dy = sig_coords[:, 0].astype(np.float64) - nuc_cy
        dx = sig_coords[:, 1].astype(np.float64) - nuc_cx
        dists = np.hypot(dy, dx)

        perinu_frac = float((dists <= r).mean()) if dists.size else np.nan

        rows.append({
            "cell_id": int(cid),
            "nuc_centroid_y": float(nuc_cy),
            "nuc_centroid_x": float(nuc_cx),
            f"{prefix}_centroid_y": float(sig_cy),
            f"{prefix}_centroid_x": float(sig_cx),
            f"dist_nuc_to_{prefix}_centroid_px": d_centroid,
            f"{prefix}_dist_mean_px": float(dists.mean()),
            f"{prefix}_dist_median_px": float(np.median(dists)),
            f"{prefix}_dist_p90_px": float(np.percentile(dists, 90)),
            f"{prefix}_perinuclear_frac_r{int(r)}px": perinu_frac,
            f"{prefix}_pixels_in_cell": n_sig_px,
        })

    return pd.DataFrame(rows)