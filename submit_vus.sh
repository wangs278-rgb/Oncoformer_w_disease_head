#!/bin/bash
#SBATCH -J onco_vus
#SBATCH -o /cv/home/wangs278/scratch/fmi/vus_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/vus_%J.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 03:00:00

set -e
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
cd /cv/home/wangs278/scratch/fmi
ANA=/cv/scratch/u/wangs278/oncoformer_test

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

echo "======== eval_vus (GPU) at $(date) ========"
python eval_vus.py --run run6

echo "======== analyze_vus (internal: anchor + ranking + context-vs-sequence) at $(date) ========"
python ${ANA}/analyze_vus.py || echo "WARN: analyze_vus failed; run6_vus_*.npz saved"

echo "=== Done at $(date) ==="
