#!/bin/bash

# submit this script as: sbatch --array <min>-<max> submit.sh
# min, max are integers; max is included

#SBATCH -o ./logs/job.%A.out
#SBATCH -e ./logs/job.%A.out
#SBATCH -D ./
#SBATCH -p gpu-interactive
#SBATCH -J BSW
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00

echo "Job is running on partition: $SLURM_JOB_PARTITION"

mkdir -p gpu_logs

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv,noheader,nounits -l 10 >> gpu_logs/nvidia-smi_$SLURM_JOB_ID.csv &
GPU_MON_PID=$!

set -euo pipefail

# 1) Per-job tmp/cache dirs under your scratch
export RUN_TAG="${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
export TMP_BASE="/var/tmp/holtsamu"
mkdir -p "$TMP_BASE"

# 2) Point EVERYTHING at your scratch tmp
export TMPDIR="$TMP_BASE"
export TMP="$TMP_BASE"
export TEMP="$TMP_BASE"
export XDG_CACHE_HOME="$TMP_BASE/cache"          # avoids $HOME quota

time pixi run python generate_data.py

kill $GPU_MON_PID   