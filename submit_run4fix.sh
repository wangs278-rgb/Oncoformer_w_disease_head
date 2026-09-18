#!/bin/bash
#SBATCH -J onco_r4fix
#SBATCH -o /cv/home/wangs278/scratch/fmi/run4_fixed_out/logs/train_%J.stdout
#SBATCH -e /cv/home/wangs278/scratch/fmi/run4_fixed_out/logs/train_%J.stderr
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:b200:4
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --requeue

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_P2P_DISABLE=1
# Unique DDP rendezvous port per job so two DDP jobs co-scheduled on the same node
# don't race on a shared port (root cause of the 2026-07-10 run4/run5 co-location failures).
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))

DATA_DIR=/cv/home/wangs278/scratch/fmi
OUT_DIR=/cv/home/wangs278/scratch/fmi/run4_fixed_out
SCRIPT=/cv/home/wangs278/scratch/fmi/train_run4fix.py

mkdir -p ${OUT_DIR}/logs

# Ensure the oncoformer venv is active (idempotent if already inherited)
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate

echo "=== Job ${SLURM_JOB_ID} started on $(hostname) at $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"

# Cache tokenized data to /dev/shm for fast multi-GPU reads
SHM_COPY=/dev/shm/onco_tokenized_dna_${SLURM_JOB_ID}.pt
echo "=== Caching tokenized_dna.pt to /dev/shm ($(du -sh ${DATA_DIR}/tokenized_dna.pt | cut -f1)) ==="
cp ${DATA_DIR}/tokenized_dna.pt ${SHM_COPY}
echo "=== Cache ready ==="
trap "rm -f ${SHM_COPY}" EXIT

# Preemption/timeout handling: SLURM sends SIGTERM before preempting. Requeue the
# SAME job (--requeue in the header) so it resumes from run4_out/checkpoints/last.ckpt.
# A genuine code/env crash exits non-zero WITHOUT a signal and is NOT requeued
# (prevents the earlier infinite crash-loop on the B200/torch incompatibility).
requeue_on_preempt() {
    echo "=== SIGTERM (preemption/timeout) at $(date); requeuing job ${SLURM_JOB_ID} to resume from last.ckpt ==="
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap requeue_on_preempt SIGTERM

echo "=== Starting training ==="
set +e
python ${SCRIPT}
TRAIN_EXIT=$?
set -e

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "=== Training crashed (exit ${TRAIN_EXIT}) at $(date). NOT resubmitting — code/env error, fix and relaunch manually. ==="
    exit ${TRAIN_EXIT}
fi

echo "=== Training finished at $(date) ==="
