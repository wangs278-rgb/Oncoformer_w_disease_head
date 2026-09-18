#!/bin/bash
#SBATCH -J onco_head_eval
#SBATCH -o /cv/home/wangs278/scratch/fmi/head_eval_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/head_eval_%J.err
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

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# run6 only (7 heads); moco module-swap isolated in its own process.
echo "======== head-competence eval run6 at $(date) ========"
python eval_head_competence.py --run run6

# CPU analysis (metrics + plots). Non-fatal: the npz is already saved.
echo "======== analysis at $(date) ========"
python /cv/scratch/u/wangs278/oncoformer_test/analyze_head_competence.py || \
    echo "WARNING: analysis failed; run6_head_eval.npz is saved, re-run analyze_head_competence.py."

echo "=== Done at $(date) ==="
