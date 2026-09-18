#!/bin/bash
#SBATCH -J onco_dxfeat
#SBATCH -o /cv/home/wangs278/scratch/fmi/dxfeat_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/dxfeat_%J.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00

set -e
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
cd /cv/home/wangs278/scratch/fmi

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# TRAIN + VAL disease features (emb + raw gene presence), one aligned pass per split.
python -u extract_disease_features.py --run run6

echo "=== Done at $(date) ==="
