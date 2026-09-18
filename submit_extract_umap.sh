#!/bin/bash
#SBATCH --job-name=onco_umap
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=196G
#SBATCH --time=03:00:00
#SBATCH --output=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/umap_%j.out
#SBATCH --error=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/umap_%j.out

set -euo pipefail
cd /cv/home/wangs278/scratch/fmi
mkdir -p run8_disease_out/logs run8_disease_out/umap
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} maxb=${MAXB:-0} target=${TARGET:-pred}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 extract_umap_features.py --max-batches "${MAXB:-0}" --target "${TARGET:-pred}"
echo "UMAP extract rc=$?"
