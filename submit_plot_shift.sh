#!/bin/bash
#SBATCH --job-name=onco_shift_plot
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/shiftplot_%j.out
#SBATCH --error=/cv/home/wangs278/scratch/fmi/run8_disease_out/logs/shiftplot_%j.out

set -euo pipefail
cd /cv/home/wangs278/scratch/fmi
mkdir -p run8_disease_out/logs run8_disease_out/shift
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
python3 plot_shift.py ${PLOT_ARGS:-}
echo "SHIFT PLOT rc=$?"
