#!/bin/bash
#SBATCH -J onco_gene_eval
#SBATCH -o /cv/home/wangs278/scratch/fmi/gene_eval_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/gene_eval_%J.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00

set -e
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
cd /cv/home/wangs278/scratch/fmi

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# One process per run (run6's MoCo module-swap must be isolated from fmi's editable install).
# Don't let one run's failure abort the others.
set +e
for RUN in run4 run5 run6; do
    echo "======== gene eval ${RUN} at $(date) ========"
    python eval_gene_stratified.py --run ${RUN}
    echo "---- ${RUN} exit=$? at $(date) ----"
done

echo "=== All gene evals done at $(date) ==="
