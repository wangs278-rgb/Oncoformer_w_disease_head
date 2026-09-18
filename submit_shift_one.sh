#!/bin/bash
#SBATCH --job-name=onco_shift1
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=196G
#SBATCH --time=03:00:00
#SBATCH --output=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/shift1_%j.out
#SBATCH --error=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/shift1_%j.out

set -euo pipefail
cd /cv/home/wangs278/scratch/fmi
mkdir -p run8_disease_out/logs run8_disease_out/shift
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mode=${MODE:-ki} gene=${GENE} src='${SRC}' dst='${DST}'"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python3 insilico_shift.py --gene "${GENE}" --source-term "${SRC}" --dest-term "${DST}" \
  --mode "${MODE:-ki}" --max-batches "${MAXB:-0}"
echo "SHIFT ${MODE:-ki} ${GENE} rc=$?"
