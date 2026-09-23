"""
Orchestration layer for the organelle segmentation pipeline.

Holds PipelineConfig and the two entry functions that call across the leaf
modules (image_io, segmentation, metrics, viz):
  - process_one_image : run the full pipeline on a single samplesheet row
  - run_on_folder     : serial batch over a whole samplesheet (local runs)

For Biowulf array jobs, biowulf_runner.py imports process_one_image from here.

Note: importing `segmentation` triggers `import _env`, which sets
CELLPOSE_LOCAL_MODELS_PATH before cellpose is imported. Keeping segmentation in
the import list below is what guarantees the model path is set for any run that
goes through this module.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from skimage.measure import regionprops_table

# --- leaf modules ---
from image_io import (
    build_img_lookup,
    add_image_paths,
    load_nd2_as_czyx,
)
from segmentation import (
    StructureSegParams,
    CellposeParams,
    segment_structures_acis_style,
    segment_lysosomes,
    label_objects_within_cells,
    run_cellpose_cells,
    run_cellpose_nuclei,
    models,  # re-exported cellpose.models, used for type + model construction
)
from metrics import (
    compute_edge_flags_bbox,
    compute_nuc_overlap_and_area_ratio,
    compute_cell_coloc,
    compute_cell_geometry,
    compute_per_cell_area_metrics,
    compute_per_cell_signal_spatial_metrics,
)
from viz import (
    save_cell_pngs_for_image,
    save_cell_mask_pngs_for_image,
)


# ----------------------------
# Channel index helper
# ----------------------------
def get_channel_indices(row: pd.Series, n_channels: int = 4) -> Dict[str, int]:
    """Map channel name -> channel index based on columns ch0..ch{n-1}."""
    channel_map: Dict[str, int] = {}
    for i in range(n_channels):
        name = row.get(f"ch{i}")
        if isinstance(name, str) and name.strip():
            channel_map[name.strip()] = i
    return channel_map


# ----------------------------
# Per-image pipeline config
# ----------------------------
@dataclass
class PipelineConfig:
    n_channels: int = 4
    channel_names: Tuple[str, str, str, str] = ("DAPI", "GFAP", "LAMP1", "pRAB10")

    cellpose: CellposeParams = field(default_factory=CellposeParams)

    edge_filter_mode: str = "bbox"
    edge_margin_px: int = 0

    lys_params: StructureSegParams = field(default_factory=lambda: StructureSegParams(
        intensity_scaling_param=(3, 19),
        min_area=5,
        blur_sigma=1,
        log_sigma_1=3, log_cutoff_1=0.13,
        log_sigma_2=2, log_cutoff_2=0.08,
        log_sigma_3=1, log_cutoff_3=0.06,
        vesselness_sigma=(1,),
        vesselness_cutoff=0.3,
    ))

    prab_params: StructureSegParams = field(default_factory=lambda: StructureSegParams(
        intensity_scaling_param=(4, 9),
        min_area=5,
        blur_sigma=2,
        log_sigma_1=3, log_cutoff_1=0.12,
        log_sigma_2=2, log_cutoff_2=0.10,
        log_sigma_3=1, log_cutoff_3=0.08,
        vesselness_sigma=(1,),
        vesselness_cutoff=0.5,
    ))

    coloc_min_pixels: int = 50

    write_cell_pngs: bool = True
    cell_png_pad: int = 8
    cell_png_min_area: int = 0
    bbox_on_crop_edge: bool = True

    write_outlines: bool = True
    outline_thickness: int = 2

    write_mask_pngs: bool = True
    restrict_mask_to_cell: bool = True


# ----------------------------
# Per-image processing
# ----------------------------
def process_one_image(
    row: pd.Series,
    cfg: PipelineConfig,
    cellpose_model: "models.CellposeModel",
    out_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:

    def safe_merge_on_id(left: pd.DataFrame, right: pd.DataFrame, on: str = "cell_id") -> pd.DataFrame:
        if right is None or right.empty:
            return left.copy()
        left_cols = set(left.columns)
        right_cols = [c for c in right.columns if c != on]
        to_take = [c for c in right_cols if c not in left_cols]
        if not to_take:
            return left.copy()
        right_sel = right[[on] + to_take].copy()
        return left.merge(right_sel, on=on, how="left", validate="one_to_one")

    img_path = row.get("image_path", None)
    if img_path is None or str(img_path) == "nan":
        raise FileNotFoundError(f"No image_path for filename={row.get('filename')}")

    img_path = Path(img_path)

    # --- ND2 load + max-projection over Z → (C, H, W) ---
    img = load_nd2_as_czyx(img_path)

    ch_index = get_channel_indices(row, n_channels=cfg.n_channels)
    for chname in cfg.channel_names:
        if chname not in ch_index:
            raise KeyError(f"Missing channel '{chname}' for image {img_path.name}. Got: {list(ch_index.keys())}")

    dapi = img[ch_index["DAPI"]]
    gfap = img[ch_index["GFAP"]]
    lys  = img[ch_index["LAMP1"]]
    prab = img[ch_index["pRAB10"]]

    cell_masks = run_cellpose_cells(cellpose_model, dapi, gfap, cfg.cellpose)
    nuc_masks  = run_cellpose_nuclei(cellpose_model, dapi, cfg.cellpose)

    cell_geom_df = compute_cell_geometry(cell_masks)
    cell_geom_df.insert(0, "filename", img_path.name)

    filename_stem = img_path.stem

    lys_mask  = segment_lysosomes(lys, cfg.lys_params)
    prab_mask = segment_structures_acis_style(prab, cfg.prab_params)

    per_cell_area_df = compute_per_cell_area_metrics(
        cell_masks=cell_masks,
        nuc_masks=nuc_masks,
        lys_mask=lys_mask,
        prab_mask=prab_mask,
    )

    edge_margin_px    = getattr(cfg, "edge_margin_px", 0)
    overlap_thresh    = getattr(cfg, "nuc_overlap_thresh", 0.9)
    area_ratio_thresh = getattr(cfg, "nuc_area_ratio_thresh", 1.25)

    edge_flags = compute_edge_flags_bbox(cell_masks, margin=edge_margin_px)
    overlap_frac_dict, area_ratio_dict = compute_nuc_overlap_and_area_ratio(cell_masks, nuc_masks)

    if not per_cell_area_df.empty:
        per_cell_area_df["flag_edge"] = per_cell_area_df["cell_id"].map(edge_flags).fillna(False).astype(bool)
        per_cell_area_df["nuc_overlap_frac"] = per_cell_area_df["cell_id"].map(overlap_frac_dict).fillna(0.0).astype(float)
        per_cell_area_df["cell_to_nuc_area_ratio"] = per_cell_area_df["cell_id"].map(area_ratio_dict).fillna(np.inf).astype(float)
        per_cell_area_df["flag_nucleus_only"] = (
            (per_cell_area_df["nuc_overlap_frac"] >= overlap_thresh)
            & (per_cell_area_df["cell_to_nuc_area_ratio"] <= area_ratio_thresh)
        )

    _lys_r       = lys_mask.ravel().astype(int)
    _prab_r      = prab_mask.ravel().astype(int)
    _lys_int_r   = lys.ravel().astype(np.float64)
    _prab_int_r  = prab.ravel().astype(np.float64)
    _flat_cells  = cell_masks.ravel()
    _max_cell    = int(_flat_cells.max())

    _cell_area_bc    = np.bincount(_flat_cells, minlength=_max_cell + 1)
    _overlap_px_bc   = np.bincount(_flat_cells, weights=_lys_r * _prab_r,              minlength=_max_cell + 1)
    _lys_px_bc       = np.bincount(_flat_cells, weights=_lys_r,                         minlength=_max_cell + 1)
    _prab_px_bc      = np.bincount(_flat_cells, weights=_prab_r,                        minlength=_max_cell + 1)
    _prab_in_lys_bc  = np.bincount(_flat_cells, weights=_prab_int_r * _lys_r,          minlength=_max_cell + 1)
    _prab_out_lys_bc = np.bincount(_flat_cells, weights=_prab_int_r * (1 - _lys_r),    minlength=_max_cell + 1)
    _lys_in_prab_bc  = np.bincount(_flat_cells, weights=_lys_int_r  * _prab_r,         minlength=_max_cell + 1)
    _lys_out_prab_bc = np.bincount(_flat_cells, weights=_lys_int_r  * (1 - _prab_r),   minlength=_max_cell + 1)

    _overlap_rows = []
    for _cid in range(1, _max_cell + 1):
        if _cell_area_bc[_cid] == 0:
            continue
        _n_lys      = _lys_px_bc[_cid]
        _n_prab     = _prab_px_bc[_cid]
        _n_out_lys  = _cell_area_bc[_cid] - _n_lys
        _n_out_prab = _cell_area_bc[_cid] - _n_prab

        _mean_prab_in  = _prab_in_lys_bc[_cid]  / _n_lys      if _n_lys      > 0 else np.nan
        _mean_prab_out = _prab_out_lys_bc[_cid] / _n_out_lys  if _n_out_lys  > 0 else np.nan
        _mean_lys_in   = _lys_in_prab_bc[_cid]  / _n_prab     if _n_prab     > 0 else np.nan
        _mean_lys_out  = _lys_out_prab_bc[_cid] / _n_out_prab if _n_out_prab > 0 else np.nan

        _overlap_rows.append({
            "cell_id":                    int(_cid),
            "mask_overlap_px":            int(_overlap_px_bc[_cid]),
            "mean_prab_int_in_lys_mask":  _mean_prab_in,
            "mean_prab_int_out_lys_mask": _mean_prab_out,
            "mean_lys_int_in_prab_mask":  _mean_lys_in,
            "mean_lys_int_out_lys_mask": _mean_lys_out,
            "prab_enrichment_in_lys": _mean_prab_in  / _mean_prab_out if (_mean_prab_out and _mean_prab_out > 0 and not np.isnan(_mean_prab_in))  else np.nan,
            "lys_enrichment_in_prab": _mean_lys_in   / _mean_lys_out  if (_mean_lys_out  and _mean_lys_out  > 0 and not np.isnan(_mean_lys_in))   else np.nan,
        })

    mask_overlap_df = pd.DataFrame(_overlap_rows)

    lys_labels_global, lys_parent_cell   = label_objects_within_cells(cell_masks, lys_mask)
    prab_labels_global, prab_parent_cell = label_objects_within_cells(cell_masks, prab_mask)

    lys_props = regionprops_table(
        lys_labels_global,
        intensity_image=lys,
        properties=("label", "area", "min_intensity", "mean_intensity", "max_intensity"),
    )
    lys_df = pd.DataFrame(lys_props)
    if not lys_df.empty:
        lys_df["cell_id"] = lys_df["label"].map(lys_parent_cell)
        lys_df.insert(0, "filename", img_path.name)

    prab_props = regionprops_table(
        prab_labels_global,
        intensity_image=prab,
        properties=("label", "area", "min_intensity", "mean_intensity", "max_intensity"),
    )
    prab_df = pd.DataFrame(prab_props)
    if not prab_df.empty:
        prab_df["cell_id"] = prab_df["label"].map(prab_parent_cell)
        prab_df.insert(0, "filename", img_path.name)

    cell_coloc_df = compute_cell_coloc(cell_masks, lys, prab, min_pixels=cfg.coloc_min_pixels)
    if not cell_coloc_df.empty:
        cell_coloc_df.insert(0, "filename", img_path.name)

    lys_spatial_df = compute_per_cell_signal_spatial_metrics(
        cell_masks=cell_masks,
        nuc_masks=nuc_masks,
        signal_mask=lys_mask,
        signal_int=lys,
        prefix="lys",
        perinuclear_radius_px=10.0,
    )

    prab_spatial_df = compute_per_cell_signal_spatial_metrics(
        cell_masks=cell_masks,
        nuc_masks=nuc_masks,
        signal_mask=prab_mask,
        signal_int=prab,
        prefix="prab",
        perinuclear_radius_px=10.0,
    )

    if not cell_geom_df.empty:
        cell_df = cell_geom_df.copy()
    elif not cell_coloc_df.empty:
        cell_df = cell_coloc_df.copy()
    else:
        cell_df = pd.DataFrame(columns=["filename", "cell_id"])

    if "filename" not in cell_df.columns:
        cell_df.insert(0, "filename", img_path.name)

    if not per_cell_area_df.empty:
        cell_df = safe_merge_on_id(cell_df, per_cell_area_df, on="cell_id")

    if not cell_coloc_df.empty:
        geometry_cols = {
            "filename", "cell_id",
            "bbox_ymin", "bbox_xmin", "bbox_ymax", "bbox_xmax",
            "cell_centroid_y", "cell_centroid_x", "cell_area_px_geom"
        }
        coloc_cols = [c for c in cell_coloc_df.columns if c not in geometry_cols]
        if coloc_cols:
            coloc_sel = cell_coloc_df[["cell_id"] + coloc_cols].copy()
            cell_df = safe_merge_on_id(cell_df, coloc_sel, on="cell_id")

    if not lys_spatial_df.empty:
        cell_df = safe_merge_on_id(cell_df, lys_spatial_df, on="cell_id")

    if not prab_spatial_df.empty:
        prab_sel = prab_spatial_df.drop(columns=["nuc_centroid_y", "nuc_centroid_x"], errors="ignore")
        cell_df = safe_merge_on_id(cell_df, prab_sel, on="cell_id")

    if not mask_overlap_df.empty:
        cell_df = safe_merge_on_id(cell_df, mask_overlap_df, on="cell_id")

    if not lys_df.empty:
        lys_agg = lys_df.groupby("cell_id").agg(
            lys_obj_count=("label", "count"),
            lys_obj_area_sum_px=("area", "sum"),
            lys_obj_area_mean_px=("area", "mean"),
        ).reset_index()
        cell_df = safe_merge_on_id(cell_df, lys_agg, on="cell_id")
    else:
        cell_df["lys_obj_count"]        = cell_df.get("lys_obj_count", 0)
        cell_df["lys_obj_area_sum_px"]  = cell_df.get("lys_obj_area_sum_px", 0.0)
        cell_df["lys_obj_area_mean_px"] = cell_df.get("lys_obj_area_mean_px", 0.0)

    if not prab_df.empty:
        prab_agg = prab_df.groupby("cell_id").agg(
            prab_obj_count=("label", "count"),
            prab_obj_area_sum_px=("area", "sum"),
            prab_obj_area_mean_px=("area", "mean"),
        ).reset_index()
        cell_df = safe_merge_on_id(cell_df, prab_agg, on="cell_id")
    else:
        cell_df["prab_obj_count"]        = cell_df.get("prab_obj_count", 0)
        cell_df["prab_obj_area_sum_px"]  = cell_df.get("prab_obj_area_sum_px", 0.0)
        cell_df["prab_obj_area_mean_px"] = cell_df.get("prab_obj_area_mean_px", 0.0)

    if not cell_df.empty:
        for c in ["lys_obj_count", "prab_obj_count"]:
            if c in cell_df.columns:
                cell_df[c] = cell_df[c].fillna(0).astype(int)
        for c in ["lys_obj_area_sum_px", "prab_obj_area_sum_px", "lys_obj_area_mean_px", "prab_obj_area_mean_px"]:
            if c in cell_df.columns:
                cell_df[c] = cell_df[c].fillna(0.0)

    flag_cols = ["flag_edge", "nuc_overlap_frac", "cell_to_nuc_area_ratio", "flag_nucleus_only"]
    for fc in flag_cols:
        if fc not in cell_df.columns:
            cell_df[fc] = np.nan if ("frac" in fc or "ratio" in fc) else False

    cols = list(cell_df.columns)
    if "filename" in cols and "cell_id" in cols:
        front = ["filename", "cell_id"]
        rest  = [c for c in cols if c not in front]
        cell_df = cell_df[front + rest]

    if cfg.write_cell_pngs and out_dir is not None:
        save_cell_pngs_for_image(
            img=img,
            cell_masks=cell_masks,
            cell_geom_df=cell_df,
            ch_index=ch_index,
            out_base=out_dir / "cell_pngs" / filename_stem,
            filename_stem=filename_stem,
            pad=cfg.cell_png_pad,
            min_cell_area=cfg.cell_png_min_area,
            bbox_on_crop_edge=cfg.bbox_on_crop_edge,
            write_outlines=cfg.write_outlines,
            outline_thickness=cfg.outline_thickness,
        )

    if cfg.write_mask_pngs and out_dir is not None:
        save_cell_mask_pngs_for_image(
            cell_masks=cell_masks,
            cell_geom_df=cell_df,
            lys_mask=lys_mask,
            prab_mask=prab_mask,
            out_base=out_dir / "cell_pngs" / filename_stem,
            filename_stem=filename_stem,
            pad=cfg.cell_png_pad,
            min_cell_area=cfg.cell_png_min_area,
            restrict_to_cell=cfg.restrict_mask_to_cell,
        )

    return {
        "cell_df": cell_df,
        "lys_df":  lys_df,
        "prab_df": prab_df,
    }


# ----------------------------
# Batch runner (local, serial)
# ----------------------------
def run_on_folder(
    samplesheet_csv: Path,
    img_dir: Path,
    out_dir: Path,
    pattern: str = "*.nd2",
    use_gpu: bool = True,
    cfg: Optional[PipelineConfig] = None,
) -> Dict[str, Path]:
    cfg = cfg or PipelineConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    samplesheet = pd.read_csv(samplesheet_csv)

    img_lookup = build_img_lookup(img_dir, pattern=pattern)
    samplesheet = add_image_paths(samplesheet, img_lookup)

    present = samplesheet["image_path"].notna()
    missing = samplesheet.loc[~present, "filename"].tolist()
    if missing:
        print(f"[WARN] {len(missing)} filenames not found in folder; skipping first few: {missing[:5]}")
    samplesheet = samplesheet.loc[present].reset_index(drop=True)

    cellpose_model = models.CellposeModel(gpu=use_gpu)

    cells_all = []
    lys_all = []
    prab_all = []
    failures = []

    for i, row in samplesheet.iterrows():
        try:
            res = process_one_image(row, cfg, cellpose_model, out_dir=out_dir)

            cell_df = res.get("cell_df")
            if cell_df is not None and not cell_df.empty:
                cells_all.append(cell_df)

            if res.get("lys_df") is not None and not res["lys_df"].empty:
                lys_all.append(res["lys_df"])
            if res.get("prab_df") is not None and not res["prab_df"].empty:
                prab_all.append(res["prab_df"])

        except Exception as e:
            fname = str(row.get("filename", "UNKNOWN"))
            tb = traceback.format_exc()
            failures.append((fname, type(e).__name__, str(e), tb))
            print(f"[FAIL] {fname}: {type(e).__name__}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(samplesheet)} images so far")

    out_paths: Dict[str, Path] = {}

    cells_all_df = pd.concat(cells_all, ignore_index=True) if cells_all else pd.DataFrame()
    lys_all_df   = pd.concat(lys_all,   ignore_index=True) if lys_all   else pd.DataFrame()
    prab_all_df  = pd.concat(prab_all,  ignore_index=True) if prab_all  else pd.DataFrame()

    out_paths["cells_all"] = out_dir / "cells_all.csv"
    out_paths["lys_objects_all"] = out_dir / "lys_objects_all.csv"
    out_paths["prab_objects_all"] = out_dir / "prab_objects_all.csv"

    cells_all_df.to_csv(out_paths["cells_all"], index=False)
    lys_all_df.to_csv(out_paths["lys_objects_all"], index=False)
    prab_all_df.to_csv(out_paths["prab_objects_all"], index=False)

    if failures:
        fail_path = out_dir / "failures.csv"
        pd.DataFrame(failures, columns=["filename", "error_type", "error_message", "traceback"]).to_csv(
            fail_path, index=False
        )
        out_paths["failures"] = fail_path
        print(f"[WARN] {len(failures)} failures written to {fail_path}")

    print("Done.")
    return out_paths