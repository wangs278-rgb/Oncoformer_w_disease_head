#!/bin/bash
#SBATCH --job-name=onco_evo
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=196G
#SBATCH --time=03:00:00
#SBATCH --output=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/evo_%j.out
#SBATCH --error=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/evo_%j.out

set -euo pipefail
cd /cv/home/wangs278/scratch/fmi
mkdir -p run8_disease_out/logs run8_disease_out/evo
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} args=${EV_ARGS:-}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python3 insilico_evo.py ${EV_ARGS:-}
echo "EVO rc=$?"
