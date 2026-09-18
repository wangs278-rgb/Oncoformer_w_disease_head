#!/bin/bash
#SBATCH -J onco_embed_ep
#SBATCH -o /cv/home/wangs278/scratch/fmi/embed_epochs_%x_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/embed_epochs_%x_%J.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 03:00:00

# Usage: sbatch -J onco_ep_run4 submit_extract_epochs.sh run4
set -e
RUN=${1:?usage: submit_extract_epochs.sh <run4|run5|run6>}

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
cd /cv/home/wangs278/scratch/fmi

echo "=== Job ${SLURM_JOB_ID} RUN=${RUN} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python extract_embeddings_epochs.py --run "${RUN}"
echo "=== ${RUN} epochs done at $(date) ==="
