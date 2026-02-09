#!/bin/bash -l
#SBATCH -o ./logs/job.%A_%a.out
#SBATCH -e ./logs/job.%A_%a.out
#SBATCH -D ./
#SBATCH -J BSW
#SBATCH -p p.ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#SBATCH -t 4:00:00

set -euo pipefail

module purge
module load gcc/14

echo "HOST=$(hostname)"
echo "Job is running on partition: $SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-}"

# JAX test (prints backend/devices)
pixi run python -u - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY

# Run the actual workload
srun --kill-on-bad-exit=0 bash -lc '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  echo "RUN proc=$SLURM_PROCID local=$SLURM_LOCALID cuda=$CUDA_VISIBLE_DEVICES"
  time pixi run python generate_data.py
'
