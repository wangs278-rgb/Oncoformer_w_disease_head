#!/usr/bin/env python3
"""
run7 — Oncoformer DNA-only pretraining, FROM SCRATCH, MLM + MoCo, DX1/DX2 baitset,
with the SAME OUTPUT HEADS AS run5 (predict only: gene + aa_vaf_bin).

Identical to train_run6.py except OUT_DIR and the config builder (run7_config, which
reduces the predict heads to run5's set). Reuses run6's tokenized cache
(tokenized_dna_dx12_moco.pt) — predict flags don't affect tokenization.
  - fresh OUT_DIR (run7_moco_out) => no last.ckpt => starts from scratch
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import run7_config
run7_config.use_moco_oncoformer()   # force oncoformer.* -> Oncoformer_moco (over fmi editable)

DATA_DIR = run7_config.DATA_DIR
OUT_DIR  = '/cv/home/wangs278/scratch/fmi/run7_moco_out'

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics
assert 'Oncoformer_moco' in oncoformer.dataset.__file__, f'wrong oncoformer: {oncoformer.dataset.__file__}'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

# Reuse run6's DX1/DX2 MoCo tokenized cache (predict flags don't change tokenization).
# Prefer the /dev/shm copy staged by submit_run7.sh, else the on-disk cache.
DISK_CACHE = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
_jid = os.environ.get('SLURM_JOB_ID')
_shm = f'/dev/shm/onco_tokenized_dna_dx12_moco_run7_{_jid}.pt' if _jid else None
tokenized_data_path = _shm if (_shm and os.path.exists(_shm)) else DISK_CACHE

config = run7_config.build_config(tokenized_data_path)

print('Building dataloaders...')
print(f'  oncoformer from : {os.path.dirname(oncoformer.dataset.__file__)}')
print(f'  modalities      : {config["omics"]["architecture"]["modalities"]}  (DNA-only => MoCo single-mod fallback)')
print(f'  clip_meta_w     : {config["training"]["clip_meta_w"]}   recon_meta_w: {config["training"]["recon_meta_w"]}')
_cc = config['omics']['modalities']['dna']['architecture']['component_configs']
print(f'  predict heads   : {[k for k, v in _cc.items() if v.get("predict") is True]}  (run5 output set)')
L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
print(f'  train={len(dl.train_dataloader)} steps/epoch  val={len(dl.val_dataloader)} steps')

print('Building model...')
model = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')

epoch_ckpt_cb = ModelCheckpoint(
    dirpath=f'{OUT_DIR}/checkpoints',
    filename='epoch={epoch}',
    every_n_epochs=5,
    save_top_k=-1,
    save_last=False,
    monitor=None,
)
step_ckpt_cb = ModelCheckpoint(
    dirpath=f'{OUT_DIR}/checkpoints',
    filename='step_ckpt',
    every_n_train_steps=100,
    save_top_k=1,
    save_last=True,
    enable_version_counter=False,
)
lr_cb = LearningRateMonitor(logging_interval='epoch')
csv_logger = pl_loggers.CSVLogger(save_dir=OUT_DIR, name='logs')

trainer = L.Trainer(
    accelerator='gpu',
    devices=torch.cuda.device_count(),
    strategy=DDPStrategy(find_unused_parameters=True, broadcast_buffers=False) if torch.cuda.device_count() > 1 else 'auto',
    max_epochs=config['training']['num_epochs'],
    precision='bf16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm',
    callbacks=[epoch_ckpt_cb, step_ckpt_cb, lr_cb],
    logger=[csv_logger],
    log_every_n_steps=50,
    num_sanity_val_steps=0,
    use_distributed_sampler=False,
)

import glob, zipfile

def _ckpt_ok(path):
    """A Lightning/torch checkpoint is a zip archive. Preemption mid-write leaves a
    truncated file that fails to load. Reading the namelist touches the central
    directory at EOF => cheap integrity check without loading the whole file."""
    try:
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False

def pick_resume_ckpt(ckpt_dir):
    """Newest *valid* checkpoint, so a corrupt last.ckpt from a preempted job can't
    stall the requeue. Quarantines corrupt files (rank 0 only)."""
    cands = [p for p in [f'{ckpt_dir}/last.ckpt', f'{ckpt_dir}/step_ckpt.ckpt',
                         *sorted(glob.glob(f'{ckpt_dir}/epoch=*.ckpt'))] if os.path.exists(p)]
    cands.sort(key=os.path.getmtime, reverse=True)
    is_rank0 = os.environ.get('LOCAL_RANK', '0') == '0'
    for p in cands:
        if _ckpt_ok(p):
            return p
        if is_rank0:
            try:
                os.replace(p, f'{p}.corrupt.{os.environ.get("SLURM_JOB_ID", "x")}')
                print(f'WARNING: {p} is corrupt (truncated); quarantined, skipping.')
            except OSError:
                pass
    return None

ckpt_path = pick_resume_ckpt(f'{OUT_DIR}/checkpoints')
if ckpt_path:
    print(f'Resuming from {ckpt_path}')
else:
    print('Starting from scratch (no valid checkpoint found).')

print(f'Training on {torch.cuda.device_count()} GPU(s)...')
trainer.fit(model, dl.train_dataloader, dl.val_dataloader, ckpt_path=ckpt_path)
print('Done.')
