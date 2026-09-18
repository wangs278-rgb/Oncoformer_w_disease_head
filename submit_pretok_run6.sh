#!/bin/bash
#SBATCH -J onco_pretok_run6
#SBATCH -o /cv/home/wangs278/scratch/fmi/run6_moco_out/logs/pretok_%J.stdout
#SBATCH -e /cv/home/wangs278/scratch/fmi/run6_moco_out/logs/pretok_%J.stderr
#SBATCH --partition=defq
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH -t 04:00:00

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

DATA_DIR=/cv/home/wangs278/scratch/fmi
OUT_DIR=/cv/home/wangs278/scratch/fmi/run6_moco_out
SCRIPT=/cv/home/wangs278/scratch/fmi/pretokenize_run6.py

mkdir -p ${OUT_DIR}/logs

# Ensure the oncoformer venv is active (idempotent if already inherited)
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate

echo "=== Pretok job ${SLURM_JOB_ID} started on $(hostname) at $(date) ==="
echo "python: $(which python)"

cd ${DATA_DIR}
python ${SCRIPT}

echo "=== Pretok finished at $(date) ==="
