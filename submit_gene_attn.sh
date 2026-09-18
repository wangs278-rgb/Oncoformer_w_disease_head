#!/bin/bash
#SBATCH -J onco_gene_attn
#SBATCH -o /cv/home/wangs278/scratch/fmi/gene_attn_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/gene_attn_%J.err
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

# GPU extraction: one process per run (run6's MoCo module-swap must be isolated
# from fmi's editable install, exactly like extract_embeddings.py / eval_gene_stratified.py).
for RUN in run4 run5 run6; do
    echo "======== gene-attn extract ${RUN} at $(date) ========"
    python extract_gene_attn.py --run ${RUN}
done
echo "=== All extractions done at $(date) ==="

# Plotting (CPU UMAP). Non-fatal: extractions above are already saved to disk.
echo "======== plotting at $(date) ========"
python /cv/scratch/u/wangs278/oncoformer_test/plot_gene_attn.py || \
    echo "WARNING: plotting failed; npz are saved, re-run plot_gene_attn.py manually."

echo "=== Done at $(date) ==="
