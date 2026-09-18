#!/bin/bash
#SBATCH -J onco_esmfeat
#SBATCH -o /cv/home/wangs278/scratch/fmi/esmfeat_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/esmfeat_%J.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 04:00:00

set -e
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
cd /cv/home/wangs278/scratch/fmi

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# Raw ESM+VAF per-sample features (same inputs Oncoformer ingests), train+val.
python -u extract_esm_features.py --run run6

echo "=== Done at $(date) ==="
