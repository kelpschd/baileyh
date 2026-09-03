#!/bin/bash
# submit_seg.sh — Biowulf array submission for organelle segmentation
#
# Usage:
#   1. Find how many array tasks you need:
#        python biowulf_runner.py count \
#          --samplesheet samples.csv --img-dir /data/kelpschdj/nd2 \
#          --out-dir /data/kelpschdj/seg_out --chunk-size 10
#      -> note array_max_index
#
#   2. Submit (set --array=0-<array_max_index>):
#        sbatch --array=0-42 submit_seg.sh
#
#   3. After the array finishes, aggregate:
#        sbatch --dependency=afterok:<arrayjobid> aggregate_seg.sh
#      (or just run the aggregate command on an interactive node)

#SBATCH --job-name=orgseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=1:00:00
#SBATCH --array=0-0                  # override on the command line with --array
#SBATCH --output=logs/orgseg_%A_%a.out
#SBATCH --error=logs/orgseg_%A_%a.err

set -euo pipefail

mkdir -p logs

# --- Cellpose local model path (also set inside the Python entry point) ---
export CELLPOSE_LOCAL_MODELS_PATH="/data/kelpschdj/cellpose/models"

# --- Environment: pick ONE of these paradigms ---
# (a) conda env
#   source /data/kelpschdj/conda/etc/profile.d/conda.sh
#   conda activate seg
# (b) Biowulf module + venv
#   module load python/3.11
#   source /data/kelpschdj/venvs/seg/bin/activate
source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate baileyh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# --- Paths (read from env if exported by run_seg.sh, else these defaults) ---
SAMPLES="${SAMPLES:-/data/Baileyhm_img/raw_data/LRRK2KOmAC/20260717_LRRK2KOmAC_2_samplesheet.csv}"
IMGDIR="${IMGDIR:-/data/Baileyhm_img/raw_data/LRRK2KOmAC}"
OUTDIR="${OUTDIR:-/data/Baileyhm_img/out/LRRK2KOmAC}"
CHUNK="${CHUNK:-10}"

echo "Host: $(hostname)  Task: ${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

python biowulf_runner.py run \
  --samplesheet "$SAMPLES" \
  --img-dir "$IMGDIR" \
  --out-dir "$OUTDIR" \
  --chunk-size "$CHUNK" \
  --fail-on-error
  # add --no-pngs to skip PNG output for a run