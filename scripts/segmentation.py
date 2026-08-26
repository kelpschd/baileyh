"""
Segmentation for the organelle pipeline: Cellpose wrappers (cells + nuclei),
structure/lysosome segmentation, and per-cell object labelling.

IMPORTANT: `import _env` MUST be the first import — it sets
CELLPOSE_LOCAL_MODELS_PATH before `from cellpose import models` runs below.
Do not reorder these imports.
"""
from __future__ import annotations

import _env  # noqa: F401  — side effect: sets CELLPOSE_LOCAL_MODELS_PATH. Keep first.

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from cellpose import models
from scipy.ndimage import gaussian_laplace
from scipy import ndimage as ndi
from scipy.stats import norm

from skimage import filters, morphology


# ----------------------------
# Structure segmentation params
# ----------------------------
@dataclass
class StructureSegParams:
    intensity_scaling_param: Tuple[float, float]
    min_area: int
    blur_sigma: float
    log_sigma_1: float
    log_cutoff_1: float
    log_sigma_2: float
    log_cutoff_2: float
    log_sigma_3: float
    log_cutoff_3: float
    vesselness_sigma: Tuple[float, ...]
    vesselness_cutoff: float


def segment_structures_acis_style(channel_img: np.ndarray, params: StructureSegParams) -> np.ndarray:
    ch = channel_img.astype(float)

    m, s = norm.fit(ch.flatten())
    stretch_min = max(m - params.intensity_scaling_param[0] * s, float(ch.min()))
    stretch_max = min(m + params.intensity_scaling_param[1] * s, float(ch.max()))
    if stretch_max <= stretch_min:
        return np.zeros_like(ch, dtype=bool)

    ch_n = np.clip(ch, stretch_min, stretch_max)
    image_norm = (ch_n - stretch_min) / (stretch_max - stretch_min)

    blurred = filters.gaussian(image_norm, sigma=params.blur_sigma)

    log_1 = -1.0 * (params.log_sigma_1**2) * gaussian_laplace(blurred, sigma=params.log_sigma_1)
    log_2 = -1.0 * (params.log_sigma_2**2) * gaussian_laplace(blurred, sigma=params.log_sigma_2)
    log_3 = -1.0 * (params.log_sigma_3**2) * gaussian_laplace(blurred, sigma=params.log_sigma_3)

    log_mask = (log_1 > params.log_cutoff_1) | (log_2 > params.log_cutoff_2) | (log_3 > params.log_cutoff_3)

    vesselness = filters.frangi(
        blurred, sigmas=list(params.vesselness_sigma), black_ridges=False
    ) > params.vesselness_cutoff

    combined = log_mask | vesselness
    filled = ndi.binary_fill_holes(combined)
    cleaned = morphology.remove_small_objects(filled, min_size=params.min_area)

    return cleaned.astype(bool)


def segment_lysosomes(channel_img: np.ndarray, params: StructureSegParams) -> np.ndarray:
    lys_ch = channel_img.astype(float)

    m, s = norm.fit(lys_ch.flatten())
    stretch_min = max(m - params.intensity_scaling_param[0] * s, float(lys_ch.min()))
    stretch_max = min(m + params.intensity_scaling_param[1] * s, float(lys_ch.max()))
    if stretch_max <= stretch_min:
        return np.zeros_like(lys_ch, dtype=bool)

    lys_ch_n = np.clip(lys_ch, stretch_min, stretch_max)
    image_norm = (lys_ch_n - stretch_min) / (stretch_max - stretch_min)

    blurred = filters.gaussian(image_norm, sigma=params.blur_sigma)

    triangle_cutoff = filters.threshold_triangle(blurred)
    global_median_cutoff = np.percentile(blurred, 50)
    th_low_cutoff = (triangle_cutoff + global_median_cutoff) / 2.0
    img_low_level = blurred > th_low_cutoff

    img_low_level_small = morphology.remove_small_objects(img_low_level, min_size=int(params.min_area), connectivity=1)
    img_low_level_small_grow = morphology.dilation(img_low_level_small, footprint=morphology.disk(2))

    otsu_cutoff = 0.333 * filters.threshold_otsu(blurred)
    img_high_level = np.zeros_like(img_low_level_small_grow, dtype=bool)

    lab_low, num_obj = morphology.label(img_low_level_small_grow, return_num=True, connectivity=1)
    for idx in range(num_obj):
        single_obj = lab_low == (idx + 1)
        if np.count_nonzero(single_obj) == 0:
            continue
        try:
            local_otsu = filters.threshold_otsu(blurred[single_obj])
        except Exception:
            local_otsu = 0.0
        if local_otsu > otsu_cutoff:
            mask_condition = np.logical_and(blurred > 0.98 * local_otsu, single_obj)
            img_high_level[mask_condition] = True

    log_sigma = params.log_sigma_1
    log_response = -1.0 * (log_sigma**2) * gaussian_laplace(blurred, sigma=log_sigma)
    bw_extra = log_response > 0.09
    bw_extra[~img_low_level_small_grow] = False

    bw_final = np.logical_or(bw_extra, img_high_level)

    filled = ndi.binary_fill_holes(bw_final)
    labeled_filled = morphology.label(filled, connectivity=1)
    lysosome_mask = morphology.remove_small_objects(labeled_filled, min_size=int(params.min_area)) > 0

    return lysosome_mask.astype(bool)


# ----------------------------
# Object assignment per cell
# ----------------------------
def label_objects_within_cells(
    cell_masks: np.ndarray,
    obj_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    cell_ids = np.unique(cell_masks)
    cell_ids = cell_ids[cell_ids != 0]

    labels_global = np.zeros_like(obj_mask, dtype=np.int32)
    parent_cell: Dict[int, int] = {}
    current_label = 1

    for cid in cell_ids:
        single_cell_mask = (cell_masks == cid)
        obj_in_cell = obj_mask & single_cell_mask
        labeled_in_cell, n = ndi.label(obj_in_cell)

        if n == 0:
            continue

        for ll in np.unique(labeled_in_cell)[1:]:
            labels_global[labeled_in_cell == ll] = current_label
            parent_cell[current_label] = int(cid)
            current_label += 1

    return labels_global, parent_cell


# ----------------------------
# Cellpose wrappers
# ----------------------------
@dataclass
class CellposeParams:
    diameter: float = 120
    batch_size: int = 32
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    tile_norm_blocksize: int = 0


def run_cellpose_cells(
    model: models.CellposeModel,
    dapi: np.ndarray,
    gfap: np.ndarray,
    params: CellposeParams,
) -> np.ndarray:
    stack = np.stack([dapi, gfap], axis=0)
    masks, flows, styles = model.eval(
        stack,
        batch_size=params.batch_size,
        diameter=params.diameter,
        flow_threshold=params.flow_threshold,
        cellprob_threshold=params.cellprob_threshold,
        normalize={"tile_norm_blocksize": params.tile_norm_blocksize},
    )
    return masks


def run_cellpose_nuclei(
    model: models.CellposeModel,
    dapi: np.ndarray,
    params: CellposeParams,
) -> np.ndarray:
    masks, flows, styles = model.eval(
        dapi,
        batch_size=params.batch_size,
        diameter=params.diameter,
        flow_threshold=params.flow_threshold,
        cellprob_threshold=params.cellprob_threshold,
        normalize={"tile_norm_blocksize": params.tile_norm_blocksize},
    )
    return masks