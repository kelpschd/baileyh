#!/bin/bash
# aggregate_seg.sh — concatenate per-task part CSVs into final outputs.
# No GPU needed. Run after the array job completes, e.g.:
#   sbatch --dependency=afterok:<arrayjobid> aggregate_seg.sh

#SBATCH --job-name=orgseg_agg
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=1:00:00
#SBATCH --output=logs/orgseg_agg_%j.out
#SBATCH --error=logs/orgseg_agg_%j.err

set -euo pipefail

mkdir -p logs

# Cellpose path not needed for aggregate (no segmentation), but harmless.
export CELLPOSE_LOCAL_MODELS_PATH="/data/kelpschdj/cellpose/models"

# --- Environment (match submit_seg.sh) ---
source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate baileyh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# --- Paths (must match submit_seg.sh) ---
SAMPLES="/data/Baileyhm_img/raw_data/LRRK2KOmAC/20260717_LRRK2KOmAC_2_samplesheet.csv"
IMGDIR="/data/Baileyhm_img/raw_data/LRRK2KOmAC"
OUTDIR="/data/Baileyhm_img/out/LRRK2KOmAC"
CHUNK=10

python biowulf_runner.py aggregate \
  --samplesheet "$SAMPLES" \
  --img-dir "$IMGDIR" \
  --out-dir "$OUTDIR" \
  --chunk-size "$CHUNK"

echo "Aggregation complete -> $OUTDIR"