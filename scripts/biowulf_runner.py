#!/usr/bin/env python
"""
Biowulf entry point for the organelle segmentation pipeline.

Two modes:
  run       process one chunk of the samplesheet (called per Slurm array task)
  aggregate concatenate all per-task part files into final CSVs

CRITICAL: CELLPOSE_LOCAL_MODELS_PATH must be set BEFORE cellpose is imported.
We set it here, at the top, before importing the pipeline module (which imports
cellpose at module load time). You can also set it in the sbatch script; setting
it in both places is harmless.
"""
from __future__ import annotations

import os

# --- MUST come before any import that pulls in cellpose ---
os.environ.setdefault(
    "CELLPOSE_LOCAL_MODELS_PATH",
    "/data/kelpschdj/cellpose/models",
)

import argparse
import sys
from pathlib import Path

import pandas as pd

# Import AFTER the env var is set. Rename your existing script's module here.
# e.g. if the file is `pipeline.py`, this pulls in process_one_image etc.
import pipeline as P


# ----------------------------
# Samplesheet -> resolved, present-only rows
# ----------------------------
def load_resolved_samplesheet(samplesheet_csv: Path, img_dir: Path, pattern: str) -> pd.DataFrame:
    """Same resolution logic as run_on_folder, factored out so 'run' and
    'aggregate' both see an identical, deterministically-ordered row set."""
    ss = pd.read_csv(samplesheet_csv)
    img_lookup = P.build_img_lookup(img_dir, pattern=pattern)
    ss = P.add_image_paths(ss, img_lookup)

    present = ss["image_path"].notna()
    missing = ss.loc[~present, "filename"].tolist()
    if missing:
        print(f"[WARN] {len(missing)} filenames not found; skipping first few: {missing[:5]}")

    # reset_index gives a stable global row order that chunking relies on
    return ss.loc[present].reset_index(drop=True)


def chunk_bounds(n_rows: int, chunk_size: int, task_id: int) -> tuple[int, int]:
    """Contiguous slice [start, end) for this array task."""
    start = task_id * chunk_size
    end = min(start + chunk_size, n_rows)
    return start, end


# ----------------------------
# GPU sanity check
# ----------------------------
def assert_gpu(require_gpu: bool):
    try:
        import torch
    except Exception as e:
        print(f"[WARN] could not import torch to check GPU: {e}")
        return
    avail = torch.cuda.is_available()
    if avail:
        print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        msg = "[GPU] CUDA NOT available — would fall back to CPU (very slow)."
        if require_gpu:
            print(msg, file=sys.stderr)
            sys.exit(2)  # fail the task loudly so Slurm records it
        print(msg)


# ----------------------------
# run one chunk
# ----------------------------
def cmd_run(args):
    assert_gpu(require_gpu=not args.allow_cpu)

    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    ss = load_resolved_samplesheet(Path(args.samplesheet), Path(args.img_dir), args.pattern)
    n_rows = len(ss)

    task_id = args.task_id
    if task_id is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            print("[ERR] no --task-id and no SLURM_ARRAY_TASK_ID in env", file=sys.stderr)
            sys.exit(2)
        task_id = int(env)

    start, end = chunk_bounds(n_rows, args.chunk_size, task_id)
    if start >= n_rows:
        print(f"[SKIP] task {task_id}: start {start} >= n_rows {n_rows}; nothing to do.")
        return

    chunk = ss.iloc[start:end].reset_index(drop=True)
    print(f"[TASK {task_id}] rows [{start}:{end}) of {n_rows}  ({len(chunk)} images)")

    # Build config; let CLI turn PNG output off for this run.
    cfg = P.PipelineConfig()
    if args.no_pngs:
        cfg.write_cell_pngs = False
        cfg.write_mask_pngs = False

    # Model loaded once per task and reused across the chunk.
    cellpose_model = P.models.CellposeModel(gpu=not args.allow_cpu)

    cells_all, lys_all, prab_all, failures = [], [], [], []

    for i, row in chunk.iterrows():
        fname = str(row.get("filename", "UNKNOWN"))
        try:
            res = P.process_one_image(row, cfg, cellpose_model, out_dir=out_dir)
            cell_df = res.get("cell_df")
            if cell_df is not None and not cell_df.empty:
                cells_all.append(cell_df)
            if res.get("lys_df") is not None and not res["lys_df"].empty:
                lys_all.append(res["lys_df"])
            if res.get("prab_df") is not None and not res["prab_df"].empty:
                prab_all.append(res["prab_df"])
        except Exception as e:
            import traceback
            failures.append((fname, type(e).__name__, str(e), traceback.format_exc()))
            print(f"[FAIL] {fname}: {type(e).__name__}: {e}")

    # Per-task outputs, tagged by task id so aggregation can glob them.
    tag = f"part{task_id:05d}"

    def _write(dfs, name):
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        path = parts_dir / f"{name}.{tag}.csv"
        df.to_csv(path, index=False)
        print(f"[WRITE] {path}  ({len(df)} rows)")

    _write(cells_all, "cells_all")
    _write(lys_all, "lys_objects_all")
    _write(prab_all, "prab_objects_all")

    if failures:
        fp = parts_dir / f"failures.{tag}.csv"
        pd.DataFrame(
            failures, columns=["filename", "error_type", "error_message", "traceback"]
        ).to_csv(fp, index=False)
        print(f"[WARN] {len(failures)} failures -> {fp}")
        if args.fail_on_error:
            # Non-zero exit lets you requeue just this array index.
            sys.exit(1)


# ----------------------------
# aggregate parts -> final CSVs
# ----------------------------
def cmd_aggregate(args):
    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"

    def _combine(name):
        files = sorted(parts_dir.glob(f"{name}.part*.csv"))
        if not files:
            print(f"[AGG] no parts for {name}")
            df = pd.DataFrame()
        else:
            df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"[AGG] {path}  ({len(df)} rows from {len(files)} parts)")

    _combine("cells_all")
    _combine("lys_objects_all")
    _combine("prab_objects_all")

    fail_files = sorted(parts_dir.glob("failures.part*.csv"))
    if fail_files:
        fdf = pd.concat((pd.read_csv(f) for f in fail_files), ignore_index=True)
        fpath = out_dir / "failures.csv"
        fdf.to_csv(fpath, index=False)
        print(f"[AGG] {fpath}  ({len(fdf)} failures)")


# ----------------------------
# how many array tasks do I need?
# ----------------------------
def cmd_count(args):
    ss = load_resolved_samplesheet(Path(args.samplesheet), Path(args.img_dir), args.pattern)
    n_rows = len(ss)
    n_tasks = (n_rows + args.chunk_size - 1) // args.chunk_size
    # Print just the max array index (0-based) for use in sbatch --array=0-N
    print(f"n_images={n_rows} chunk_size={args.chunk_size} "
          f"n_tasks={n_tasks} array_max_index={max(n_tasks - 1, 0)}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--samplesheet", required=True)
    common.add_argument("--img-dir", required=True)
    common.add_argument("--out-dir", required=True)
    common.add_argument("--pattern", default="*.nd2")
    common.add_argument("--chunk-size", type=int, default=10)

    pr = sub.add_parser("run", parents=[common])
    pr.add_argument("--task-id", type=int, default=None,
                    help="override SLURM_ARRAY_TASK_ID (for local testing)")
    pr.add_argument("--no-pngs", action="store_true",
                    help="disable all PNG output for this run")
    pr.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU fallback instead of failing when no GPU")
    pr.add_argument("--fail-on-error", action="store_true",
                    help="exit non-zero if any image in the chunk failed")
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("aggregate", parents=[common])
    pa.set_defaults(func=cmd_aggregate)

    pc = sub.add_parser("count", parents=[common])
    pc.set_defaults(func=cmd_count)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)