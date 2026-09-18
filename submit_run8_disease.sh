#!/bin/bash
#SBATCH -J onco_mlm_run8
#SBATCH -o /cv/home/wangs278/scratch/fmi/run8_disease_out/logs/train_%J.stdout
#SBATCH -e /cv/home/wangs278/scratch/fmi/run8_disease_out/logs/train_%J.stderr
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
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_P2P_DISABLE=1
# Unique DDP rendezvous port per job so co-scheduled DDP jobs don't race on a shared port.
export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 40000) ))

DATA_DIR=/cv/home/wangs278/scratch/fmi
OUT_DIR=/cv/home/wangs278/scratch/fmi/run8_disease_out
SCRIPT=/cv/home/wangs278/scratch/fmi/train_run8_disease.py

mkdir -p ${OUT_DIR}/logs

# Ensure the oncoformer venv is active (idempotent if already inherited)
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate

echo "=== Job ${SLURM_JOB_ID} started on $(hostname) at $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"

# Reuse run6's DX1/DX2 MoCo tokenized cache (metadata doesn't change tokenization).
# Stage to /dev/shm for fast multi-GPU reads. run8-specific shm basename avoids collision
# with a co-scheduled run6/run7 job. Abort if the cache is missing (else 4 ranks race to tokenize).
RUN_CACHE=${DATA_DIR}/tokenized_dna_dx12_moco.pt
SHM_COPY=/dev/shm/onco_tokenized_dna_dx12_moco_run8_${SLURM_JOB_ID}.pt
if [ -f "${RUN_CACHE}" ]; then
    echo "=== Caching ${RUN_CACHE} to /dev/shm ($(du -sh ${RUN_CACHE} | cut -f1)) ==="
    cp "${RUN_CACHE}" "${SHM_COPY}"
    trap "rm -f ${SHM_COPY}" EXIT
    echo "=== Cache ready ==="
else
    echo "=== ERROR: ${RUN_CACHE} missing (run6's cache). Build it via submit_pretok_run6.sh first. ==="
    exit 1
fi

# Also require the disease targets + vocab (built by build_disease_metadata_run8.py).
for f in ${DATA_DIR}/metadata_disease_run8.pt ${DATA_DIR}/vocab_metadata_disease_run8.json; do
    if [ ! -f "${f}" ]; then
        echo "=== ERROR: ${f} missing. Run: python build_disease_metadata_run8.py ==="
        exit 1
    fi
done

# Preemption/timeout handling: SLURM sends SIGTERM before preempting. Requeue the SAME
# job (--requeue) so it resumes from run8_disease_out/checkpoints/last.ckpt. A genuine
# code/env crash exits non-zero WITHOUT a signal and is NOT requeued.
requeue_on_preempt() {
    echo "=== SIGTERM (preemption/timeout) at $(date); requeuing job ${SLURM_JOB_ID} to resume from last.ckpt ==="
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap requeue_on_preempt SIGTERM

echo "=== Starting training ==="
set +e
# -u: unbuffered stdout so startup prints (incl. warm-start load report) are visible live.
python -u ${SCRIPT}
TRAIN_EXIT=$?
set -e

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "=== Training crashed (exit ${TRAIN_EXIT}) at $(date). NOT resubmitting — code/env error, fix and relaunch manually. ==="
    exit ${TRAIN_EXIT}
fi

echo "=== Training finished at $(date) ==="
