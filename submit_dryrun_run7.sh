#!/bin/bash
#SBATCH -J onco_dry_run7
#SBATCH -o /cv/home/wangs278/scratch/fmi/run7_moco_out/logs/dryrun_%J.stdout
#SBATCH -e /cv/home/wangs278/scratch/fmi/run7_moco_out/logs/dryrun_%J.stderr
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:2
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 00:30:00

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))

OUT_DIR=/cv/home/wangs278/scratch/fmi/run7_moco_out
SCRIPT=/cv/home/wangs278/scratch/fmi/dryrun_run7.py

mkdir -p ${OUT_DIR}/logs
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate

echo "=== Dry-run job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"

# Dry-run reads the on-disk cache directly (no shm staging needed for 6 steps).
cd /cv/home/wangs278/scratch/fmi
python ${SCRIPT}
echo "=== Dry-run exit $? at $(date) ==="
