# Organelle Segmentation Pipeline

Cell-by-cell segmentation and analysis of **lysosomes (LAMP1)** and **pRAB10-positive vesicles** in astrocytes, from multi-channel ND2 microscopy images.

The pipeline segments individual astrocytes (via Cellpose), segments organelle structures within each cell, and produces per-cell and per-object measurements including geometry, area metrics, colocalization, and perinuclear spatial distribution. It is designed to run either serially on a local machine or as a Slurm array job on the NIH Biowulf cluster.

## What it does

For each image, the pipeline:

1. Loads an ND2 file and max-projects over Z to a `(C, H, W)` array.
2. Segments **cells** (from DAPI + GFAP) and **nuclei** (from DAPI) using Cellpose.
3. Segments **lysosomes** (LAMP1 channel) and **pRAB10 vesicles** (pRAB10 channel) using intensity-normalized LoG / Frangi / Otsu-based structure segmentation.
4. Assigns each segmented organelle to its parent cell.
5. Computes per-cell measurements and writes results to CSV.
6. Optionally writes per-cell PNG crops (grayscale channels, pseudocolor merge, yellow outline overlays, bbox overlays, and binary organelle masks).

### Expected channels

By default the pipeline expects four channels, mapped per samplesheet row:

| Channel | Role |
|---------|------|
| DAPI    | nuclei / cell segmentation |
| GFAP    | astrocyte marker / cell segmentation |
| LAMP1   | lysosomes |
| pRAB10  | pRAB10-positive vesicles |

## Repository structure

| File | Purpose |
|------|---------|
| `pipeline.py` | Orchestration layer. Defines `PipelineConfig`, `process_one_image` (single image), and `run_on_folder` (local serial batch). |
| `image_io.py` | ND2 loading, max-projection, uint8 percentile scaling, PNG saving, and samplesheet-to-image-path resolution. |
| `segmentation.py` | Cellpose wrappers (cells + nuclei), lysosome and structure segmentation, and per-cell object labelling. |
| `metrics.py` | Pure numpy/pandas/skimage per-cell measurements: geometry, colocalization, area metrics, perinuclear spatial metrics, edge flags, nucleus-overlap diagnostics. |
| `viz.py` | Per-cell crop PNGs, pseudocolor RGB merges, outline/bbox overlays, and binary mask PNGs. |
| `biowulf_runner.py` | CLI entry point for cluster runs. Subcommands: `count`, `run`, `aggregate`. |
| `submit_seg.sh` | Slurm array job script (GPU) that runs one chunk per array task. |
| `aggregate_seg.sh` | Slurm job that concatenates per-task part CSVs into final outputs (CPU only). |
| `run_seg.sh` | Convenience wrapper: counts tasks, submits the array job, and submits aggregation with a dependency. |
| `_env.py` | Sets `CELLPOSE_LOCAL_MODELS_PATH` on import (before Cellpose is imported). Must be imported first by any module touching Cellpose. See below. |

## Requirements

- Python 3.14 (per the conda environment)
- A CUDA-capable GPU is strongly recommended (CPU fallback exists but is very slow)
- Cellpose 4.x with model weights available at `CELLPOSE_LOCAL_MODELS_PATH`

Core scientific stack: `numpy`, `pandas`, `scipy`, `scikit-image`, `nd2`, `Pillow`, `torch` (CUDA build), and `cellpose`.

The `nd2` package is used for image loading (replacing tifffile).

### The `_env.py` file

`segmentation.py` requires `import _env` to be its **first** import, because that import sets the Cellpose model path *before* Cellpose is imported. `_env.py` does this via:

```python
import os
os.environ.setdefault(
    "CELLPOSE_LOCAL_MODELS_PATH",
    "/data/kelpschdj/cellpose/models",
)
```

Because it uses `setdefault`, a value set externally (exported in the sbatch script, or set at the top of `biowulf_runner.py`) still takes precedence — `_env.py` only provides the fallback. Setting the variable in multiple places is harmless. Edit the fallback path to point at your own Cellpose model directory.

## Input format

**Samplesheet (CSV):** one row per image. Required columns:

- `filename` — the ND2 file name (matched against files in the image directory)
- `ch0`, `ch1`, `ch2`, `ch3` — channel name for each index (e.g. `DAPI`, `GFAP`, `LAMP1`, `pRAB10`)

Rows whose `filename` is not found in the image directory are skipped with a warning.

## Usage

### Local (serial) run

```python
from pathlib import Path
from pipeline import run_on_folder

run_on_folder(
    samplesheet_csv=Path("samples.csv"),
    img_dir=Path("/path/to/nd2"),
    out_dir=Path("/path/to/out"),
    pattern="*.nd2",
    use_gpu=True,
)
```

### Biowulf (Slurm) run

The simplest path is the one-command wrapper. Edit the paths at the top of `run_seg.sh` (or override them via environment variables), then:

```bash
./run_seg.sh
```

This will:
1. Run `biowulf_runner.py count` to determine how many array tasks are needed.
2. Submit the GPU array job (`submit_seg.sh`).
3. Submit the aggregation job (`aggregate_seg.sh`) to run automatically after the array succeeds.

Override paths without editing the file:

```bash
SAMPLES=/path/to/samplesheet.csv \
IMGDIR=/path/to/nd2 \
OUTDIR=/path/to/out \
CHUNK=40 \
./run_seg.sh
```

#### Running the steps manually

```bash
# 1. Determine array size
python biowulf_runner.py count \
  --samplesheet samples.csv --img-dir /path/to/nd2 \
  --out-dir /path/to/out --chunk-size 40
# -> prints array_max_index=N

# 2. Submit the array job
sbatch --array=0-N submit_seg.sh

# 3. Aggregate after the array finishes
sbatch --dependency=afterok:<arrayjobid> aggregate_seg.sh
```

Each array task processes a contiguous chunk of `chunk-size` images and writes per-task "part" files; aggregation concatenates them.

### `biowulf_runner.py` subcommands

| Command | Description |
|---------|-------------|
| `count` | Print image count, chunk size, number of tasks, and the max 0-based array index. |
| `run` | Process one chunk (uses `--task-id` or `SLURM_ARRAY_TASK_ID`). |
| `aggregate` | Concatenate all per-task part CSVs into final outputs. |

Common options for `run`:

- `--task-id N` — override `SLURM_ARRAY_TASK_ID` (useful for local testing)
- `--no-pngs` — disable all PNG output for the run
- `--allow-cpu` — permit CPU fallback instead of failing when no GPU is present
- `--fail-on-error` — exit non-zero if any image in the chunk failed (lets you requeue just that array index)

## Outputs

Written to the output directory:

| File | Contents |
|------|----------|
| `cells_all.csv` | One row per cell: geometry, area metrics, colocalization (Pearson, Manders M1/M2), mask-overlap enrichment, perinuclear spatial metrics, object counts, and QC flags. |
| `lys_objects_all.csv` | One row per lysosome object: area and intensity stats, with parent `cell_id`. |
| `prab_objects_all.csv` | One row per pRAB10 object: area and intensity stats, with parent `cell_id`. |
| `failures.csv` | Any images that errored, with error type, message, and traceback. |
| `cell_pngs/` | Per-cell crop images (if PNG output is enabled). |

During cluster runs, intermediate per-task files are written to `out_dir/parts/` and combined by the `aggregate` step.

### Per-cell QC flags

- `flag_edge` — cell bounding box touches the image edge
- `nuc_overlap_frac` — fraction of the cell overlapped by nuclei
- `cell_to_nuc_area_ratio` — cell area relative to largest overlapping nucleus
- `flag_nucleus_only` — heuristic flag for cells that are essentially just a nucleus (high nuclear overlap and low cell-to-nucleus area ratio)

## Configuration

Segmentation and output behavior are controlled by `PipelineConfig` in `pipeline.py`, including channel names, Cellpose parameters (`CellposeParams`), lysosome and pRAB10 structure-segmentation parameters (`StructureSegParams`), colocalization minimum pixel count, and toggles for PNG/outline/mask output. Edit the defaults there to tune the pipeline.

## Notes

- Images are max-projected over Z before analysis; `load_nd2_as_czyx` handles several ND2 axis orderings (CZYX, ZCYX, TCZYX, and degenerate CYX/ZYX cases).
- The module is named `image_io` (not `io`) to avoid shadowing Python's standard library.
- `metrics.py` has no Cellpose or I/O dependencies, so downstream analysis scripts can import from it without pulling in the segmentation stack.
