#!/usr/bin/env python3
"""
Oncoformer DNA-only pretraining — MLM-only — DX1/DX2 — BUGFIX of run5

Identical to run5 (train_dx12.py) EXCEPT the single-task uncertainty-weighting bug
is fixed: log_var_recon is frozen at 0 so the total loss == recon (MLM) loss.
  - trains on DX1/DX2 baitset records only (via dx12_config baitset_filter)
  - fresh OUT_DIR (run5_fixed_out) => no last.ckpt => starts from scratch
  - uses the DX1/DX2 tokenized cache (tokenized_dna_dx12.pt)
  - run5's original results are preserved in run5_dx12_out.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import dx12_config
sys.path.insert(0, dx12_config.REPO)

DATA_DIR = dx12_config.DATA_DIR
OUT_DIR  = '/cv/home/wangs278/scratch/fmi/run5_fixed_out'

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

# Tokenized cache: prefer the /dev/shm copy staged by submit_dx12.sh, else the
# on-disk DX1/DX2 cache built by pretokenize_dx12.py. The '_dx12_' basename keeps
# it distinct from run4's shm cache on shared nodes.
DISK_CACHE = f'{DATA_DIR}/tokenized_dna_dx12.pt'
_jid = os.environ.get('SLURM_JOB_ID')
_shm = f'/dev/shm/onco_tokenized_dna_dx12_{_jid}.pt' if _jid else None
tokenized_data_path = _shm if (_shm and os.path.exists(_shm)) else DISK_CACHE

config = dx12_config.build_config(tokenized_data_path)

print('Building dataloaders...')
L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
print(f'  train={len(dl.train_dataloader)} steps/epoch  val={len(dl.val_dataloader)} steps')

print('Building model...')
model = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')

# ---------------------------------------------------------------------------
# BUGFIX — single-task heteroscedastic uncertainty weighting (see train_run4fix.py).
# With DINO disabled (w_dino=0) the loss reduces to  exp(-s)*L_recon + s , s a free
# nn.Parameter that drifts, sliding the logged loss negative and inflating the
# effective MLM LR. Freeze s=0 => prec=1 => total loss == recon(MLM) loss.
# requires_grad=False also removes s from the optimizer (models.py:2008).
# ---------------------------------------------------------------------------
assert float(config['training'].get('w_dino', 0.0)) == 0.0, \
    'uncertainty-weighting fix assumes DINO is disabled (w_dino=0)'
with torch.no_grad():
    model.log_var_recon.zero_()
    model.log_var_dino.zero_()
model.log_var_recon.requires_grad_(False)
model.log_var_dino.requires_grad_(False)
assert model.log_var_recon.item() == 0.0 and not model.log_var_recon.requires_grad, \
    'log_var_recon freeze failed'
print('[bugfix] log_var_recon frozen at 0 (requires_grad=False) => '
      'prec_recon=exp(0)=1.0 ; total loss == recon(MLM) loss')

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

last_ckpt = f'{OUT_DIR}/checkpoints/last.ckpt'
ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
if ckpt_path:
    print(f'Resuming from {ckpt_path}')
else:
    print('Starting from scratch (no checkpoint found).')

print(f'Training on {torch.cuda.device_count()} GPU(s)...')
trainer.fit(model, dl.train_dataloader, dl.val_dataloader, ckpt_path=ckpt_path)
print('Done.')
