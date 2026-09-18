#!/bin/bash
#SBATCH --job-name=run8_dz_eval
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/eval_head_%j.out
#SBATCH --error=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/eval_head_%j.out

set -euo pipefail
cd /cv/home/wangs278/scratch/fmi
mkdir -p run8_disease_out/logs run8_disease_out/eval
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 eval_run8_disease_head.py
echo "EVAL DONE rc=$?"
