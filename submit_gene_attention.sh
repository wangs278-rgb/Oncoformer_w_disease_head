#!/bin/bash
#SBATCH -J onco_geneattn_C
#SBATCH -o /cv/home/wangs278/scratch/fmi/geneattn_C_%J.out
#SBATCH -e /cv/home/wangs278/scratch/fmi/geneattn_C_%J.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 03:00:00

set -e
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
cd /cv/home/wangs278/scratch/fmi
ANA=/cv/scratch/u/wangs278/oncoformer_test

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# --- Job C GPU pass: per-gene/pathogenicity relative attention + gene-set presence ---
echo "======== eval_gene_attention (GPU) at $(date) ========"
python eval_gene_attention.py --run run6

# --- CPU analyses (non-fatal individually; npz are already saved) ---
echo "======== analyze C: gene/pathogenicity attention at $(date) ========"
python ${ANA}/analyze_gene_attention.py    || echo "WARN: analyze_gene_attention failed"
echo "======== analyze A: embedding structure at $(date) ========"
python ${ANA}/analyze_embedding_structure.py || echo "WARN: analyze_embedding_structure failed"
echo "======== analyze B: retrieval at $(date) ========"
python ${ANA}/analyze_retrieval.py         || echo "WARN: analyze_retrieval failed"
echo "======== regenerate gene-attention UMAP (size-robust metric) at $(date) ========"
python ${ANA}/plot_gene_attn.py            || echo "WARN: plot_gene_attn failed"

echo "=== Done at $(date) ==="
