#!/bin/bash -l
#SBATCH -o ./logs/job.%A.out
#SBATCH -e ./logs/job.%A.out
#SBATCH -D ./
#SBATCH -J wafer_data
#SBATCH --mail-type=ALL
#SBATCH -p p.ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#SBATCH -t 3:00:00

set -euo pipefail

module purge
module load gcc/14

echo "HOST=$(hostname)"
echo "Job is running on partition: $SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-}"

mkdir -p logs gpu_logs

# Live GPU logging
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 >> "gpu_logs/nvidia-smi_${SLURM_JOB_ID}.csv" &
GPU_MON_PID=$!
trap 'kill $GPU_MON_PID 2>/dev/null || true' EXIT

# Quick snapshot
nvidia-smi || true

# JAX test (prints backend/devices)
pixi run python -u - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY

# Run the actual workload
srun bash -lc '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  echo "RUN proc=$SLURM_PROCID local=$SLURM_LOCALID cuda=$CUDA_VISIBLE_DEVICES"
  pixi run python generate_data.py
'
