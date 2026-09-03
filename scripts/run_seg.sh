#!/bin/bash
# run_seg.sh — one command to submit the whole workflow.
#   1. computes how many array tasks are needed from the samplesheet
#   2. submits the GPU array job
#   3. submits the aggregation job to run automatically after the array succeeds
#
# Usage:
#   ./run_seg.sh
# (edit the paths below, or override via environment variables)

set -euo pipefail

mkdir -p logs

# --- Paths (override by exporting before calling, e.g. OUTDIR=... ./run_seg.sh) ---
SAMPLES="${SAMPLES:-/data/Baileyhm_img/raw_data/LRRK2KOmAC/20260717_LRRK2KOmAC_2_samplesheet.csv}"
IMGDIR="${IMGDIR:-/data/Baileyhm_img/raw_data/LRRK2KOmAC}"
OUTDIR="${OUTDIR:-/data/Baileyhm_img/out/LRRK2KOmAC}"
CHUNK="${CHUNK:-10}"

# --- Environment (needed for the `count` step, which runs here on the login node) ---
source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate baileyh

# --- Step 1: how many array tasks? ---
# biowulf_runner.py count prints: n_images=... chunk_size=... n_tasks=... array_max_index=N
COUNT_LINE="$(python biowulf_runner.py count \
  --samplesheet "$SAMPLES" --img-dir "$IMGDIR" \
  --out-dir "$OUTDIR" --chunk-size "$CHUNK")"
echo "$COUNT_LINE"

ARRAY_MAX="$(echo "$COUNT_LINE" | sed -n 's/.*array_max_index=\([0-9]\+\).*/\1/p')"
if [ -z "$ARRAY_MAX" ]; then
  echo "ERROR: could not parse array_max_index from count output" >&2
  exit 1
fi

# --- Step 2: submit the array job, capture its job id ---
ARRAY_JOBID="$(sbatch --parsable --array=0-"${ARRAY_MAX}" \
  --export=ALL,SAMPLES="$SAMPLES",IMGDIR="$IMGDIR",OUTDIR="$OUTDIR",CHUNK="$CHUNK" \
  submit_seg.sh)"
echo "Submitted array job: ${ARRAY_JOBID}  (tasks 0-${ARRAY_MAX})"

# --- Step 3: submit aggregation, dependent on the whole array finishing OK ---
AGG_JOBID="$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" \
  --export=ALL,SAMPLES="$SAMPLES",IMGDIR="$IMGDIR",OUTDIR="$OUTDIR",CHUNK="$CHUNK" \
  aggregate_seg.sh)"
echo "Submitted aggregation job: ${AGG_JOBID}  (runs after ${ARRAY_JOBID} succeeds)"

echo
echo "Monitor with:  squeue -u \$USER"
echo "Array logs:    logs/orgseg_<jobid>_<taskid>.out"
echo "Agg log:       logs/orgseg_agg_<jobid>.out"


####
# run as ./run_seg.sh and jobs will submit