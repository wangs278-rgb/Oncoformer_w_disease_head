#!/bin/bash
#SBATCH --job-name=onco_shift
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=196G
#SBATCH --time=03:00:00
#SBATCH --output=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/shift_%j.out
#SBATCH --error=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/shift_%j.out

set -euo pipefail
cd /cv/home/wangs278/scratch/fmi
mkdir -p run8_disease_out/logs run8_disease_out/shift
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} maxb=${MAXB:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 insilico_shift.py --gene CDH1 \
  --source-term "breast invasive ductal carcinoma (idc)" \
  --dest-term   "breast invasive lobular carcinoma (ilc)" --max-batches "${MAXB:-0}"

python3 insilico_shift.py --gene FOXL2 \
  --source-term "ovary serous carcinoma" \
  --dest-term   "ovary granulosa cell tumor" --max-batches "${MAXB:-0}"

echo "SHIFT rc=$?"
