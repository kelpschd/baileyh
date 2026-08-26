"""
Visualization outputs for the organelle pipeline: per-cell crops, merged
pseudocolor RGB, yellow outline overlays, bbox overlays, and binary mask PNGs.

Depends on image_io for uint8 scaling and PNG saving.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from skimage import morphology
from skimage.segmentation import find_boundaries
from skimage.draw import rectangle_perimeter

from image_io import _imsave, to_uint8


def square_crop_coords(ymin, xmin, ymax, xmax, H, W, pad: int = 8):
    ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(H, ymax + pad)
    xmax = min(W, xmax + pad)

    h = ymax - ymin
    w = xmax - xmin
    side = max(h, w)

    cy = (ymin + ymax) // 2
    cx = (xmin + xmax) // 2

    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    y1 = min(H, y0 + side)
    x1 = min(W, x0 + side)

    y0 = max(0, y1 - side)
    x0 = max(0, x1 - side)

    return y0, x0, y1, x1


def make_merged_rgb(dapi_u8, gfap_u8, lamp1_u8, prab_u8):
    """
    Pseudocolor mapping:
      DAPI -> blue
      GFAP -> green
      LAMP1 -> red
      pRAB10 -> magenta (red + blue)
    """
    H, W = dapi_u8.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    rgb[..., 0] = np.maximum(rgb[..., 0], lamp1_u8)
    rgb[..., 0] = np.maximum(rgb[..., 0], prab_u8)
    rgb[..., 1] = np.maximum(rgb[..., 1], gfap_u8)
    rgb[..., 2] = np.maximum(rgb[..., 2], dapi_u8)
    rgb[..., 2] = np.maximum(rgb[..., 2], prab_u8)

    return rgb


def cellpose_outline_from_labels(
    cell_masks: np.ndarray,
    cid: int,
    thickness: int = 2,
    mode: str = "outer",
) -> np.ndarray:
    cell_bin = (cell_masks == cid)
    outline = find_boundaries(cell_bin, mode=mode)
    if thickness and thickness > 1:
        outline = morphology.dilation(outline, footprint=morphology.disk(int(thickness)))
    return outline


def overlay_outline_yellow(gray_u8: np.ndarray, outline: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).copy()
    rgb[outline] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def save_cell_pngs_for_image(
    img: np.ndarray,
    cell_masks: np.ndarray,
    cell_geom_df: pd.DataFrame,
    ch_index: dict,
    out_base: Path,
    filename_stem: str,
    pad: int = 8,
    min_cell_area: int = 0,
    bbox_on_crop_edge: bool = True,
    write_outlines: bool = True,
    outline_thickness: int = 2,
):
    out_base.mkdir(parents=True, exist_ok=True)

    dapi = img[ch_index["DAPI"]]
    gfap = img[ch_index["GFAP"]]
    lamp1 = img[ch_index["LAMP1"]]
    prab = img[ch_index["pRAB10"]]

    H, W = dapi.shape

    dapi_u8 = to_uint8(dapi)
    gfap_u8 = to_uint8(gfap)
    lamp1_u8 = to_uint8(lamp1)
    prab_u8 = to_uint8(prab)

    for _, r in cell_geom_df.iterrows():
        cid = int(r["cell_id"])
        area = int(r["cell_area_px_geom"])
        if area < min_cell_area:
            continue

        y0, x0, y1, x1 = square_crop_coords(
            r["bbox_ymin"], r["bbox_xmin"], r["bbox_ymax"], r["bbox_xmax"],
            H=H, W=W, pad=pad
        )

        cell_dir = out_base / f"{filename_stem}_cell{cid:04d}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        dapi_crop = dapi_u8[y0:y1, x0:x1]
        gfap_crop = gfap_u8[y0:y1, x0:x1]
        lamp1_crop = lamp1_u8[y0:y1, x0:x1]
        prab_crop = prab_u8[y0:y1, x0:x1]

        _imsave(cell_dir / "DAPI_gray.png",  dapi_crop)
        _imsave(cell_dir / "GFAP_gray.png",  gfap_crop)
        _imsave(cell_dir / "LAMP1_gray.png", lamp1_crop)
        _imsave(cell_dir / "pRAB10_gray.png", prab_crop)

        merged = make_merged_rgb(dapi_crop, gfap_crop, lamp1_crop, prab_crop)
        _imsave(cell_dir / "merged_color.png", merged)

        if write_outlines:
            outline_crop = cellpose_outline_from_labels(
                cell_masks=cell_masks[y0:y1, x0:x1],
                cid=cid,
                thickness=outline_thickness,
                mode="outer",
            )

            _imsave(cell_dir / "DAPI_outline_yellow.png",  overlay_outline_yellow(dapi_crop, outline_crop))
            _imsave(cell_dir / "GFAP_outline_yellow.png",  overlay_outline_yellow(gfap_crop, outline_crop))
            _imsave(cell_dir / "LAMP1_outline_yellow.png", overlay_outline_yellow(lamp1_crop, outline_crop))
            _imsave(cell_dir / "pRAB10_outline_yellow.png", overlay_outline_yellow(prab_crop, outline_crop))

            merged_outline = merged.copy()
            merged_outline[outline_crop] = np.array([255, 255, 0], dtype=np.uint8)
            _imsave(cell_dir / "merged_color_outline_yellow.png", merged_outline)

        overlay = merged.copy()

        if bbox_on_crop_edge:
            rr, cc = rectangle_perimeter(
                start=(0, 0),
                end=(overlay.shape[0] - 1, overlay.shape[1] - 1),
                shape=overlay.shape[:2],
            )
        else:
            by0 = int(r["bbox_ymin"]) - y0
            bx0 = int(r["bbox_xmin"]) - x0
            by1 = int(r["bbox_ymax"]) - y0 - 1
            bx1 = int(r["bbox_xmax"]) - x0 - 1
            by0 = max(0, min(by0, overlay.shape[0] - 1))
            bx0 = max(0, min(bx0, overlay.shape[1] - 1))
            by1 = max(0, min(by1, overlay.shape[0] - 1))
            bx1 = max(0, min(bx1, overlay.shape[1] - 1))

            rr, cc = rectangle_perimeter(
                start=(by0, bx0),
                end=(by1, bx1),
                shape=overlay.shape[:2],
            )

        overlay[rr, cc] = 255
        _imsave(cell_dir / "merged_color_bbox.png", overlay)


def save_cell_mask_pngs_for_image(
    cell_masks: np.ndarray,
    cell_geom_df: pd.DataFrame,
    lys_mask: np.ndarray,
    prab_mask: np.ndarray,
    out_base: Path,
    filename_stem: str,
    pad: int = 8,
    min_cell_area: int = 0,
    restrict_to_cell: bool = True,
):
    out_base.mkdir(parents=True, exist_ok=True)
    H, W = cell_masks.shape

    for _, r in cell_geom_df.iterrows():
        cid = int(r["cell_id"])
        area = int(r["cell_area_px_geom"])
        if area < min_cell_area:
            continue

        y0, x0, y1, x1 = square_crop_coords(
            r["bbox_ymin"], r["bbox_xmin"], r["bbox_ymax"], r["bbox_xmax"],
            H=H, W=W, pad=pad
        )

        cell_dir = out_base / f"{filename_stem}_cell{cid:04d}"
        masks_dir = cell_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        lys_crop = lys_mask[y0:y1, x0:x1]
        prab_crop = prab_mask[y0:y1, x0:x1]

        if restrict_to_cell:
            cell_crop = (cell_masks[y0:y1, x0:x1] == cid)
            lys_crop = lys_crop & cell_crop
            prab_crop = prab_crop & cell_crop

        _imsave(masks_dir / "lys_mask.png",  (lys_crop.astype(np.uint8) * 255))
        _imsave(masks_dir / "prab_mask.png", (prab_crop.astype(np.uint8) * 255))