#!/usr/bin/env python3
"""
run8_disease — run6 (DNA-only, MLM + single-modality MoCo, DX1/DX2) + a supervised
DiseaseTerm classification head, trained JOINTLY during a warm-started continuation of
pretraining.

Model wiring (no shared model-code change):
  backbone = OncoformerOmics(config)              # MLM + MoCo (run6)
  model    = OncoformerPost(backbone, config)     # + DiseaseTerm head, pre_train_model=True
  OncoformerPost.training_step sums backbone MLM+MoCo loss and the disease CE loss.

Warm-start / resume policy:
  - COLD start (no valid run8 checkpoint): initialise `backbone` from run6's final
    checkpoint (run6_moco_out/checkpoints/last.ckpt) via load_state_dict(strict=False),
    then train the combined model with ckpt_path=None (fresh optimizer + cosine schedule
    over 20 epochs, with the disease head randomly initialised).
  - REQUEUE (preemption): resume the COMBINED OncoformerPost from run8's own last.ckpt
    (Lightning restores backbone + disease head + optimizer); no run6 warm-start then.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import run8_disease_config
run8_disease_config.use_moco_oncoformer()   # force oncoformer.* -> Oncoformer_moco (over fmi editable)

DATA_DIR = run8_disease_config.DATA_DIR
OUT_DIR  = '/cv/home/wangs278/scratch/fmi/run8_disease_out'
RUN6_CKPT = run8_disease_config.RUN6_CKPT

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
from oncoformer.models import OncoformerOmics, OncoformerPost
assert 'Oncoformer_moco' in oncoformer.dataset.__file__, f'wrong oncoformer: {oncoformer.dataset.__file__}'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

# Tokenized cache: prefer the /dev/shm copy staged by submit_run8_disease.sh, else the
# on-disk DX1/DX2 MoCo cache (reused from run6 — metadata doesn't affect tokenization).
DISK_CACHE = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
_jid = os.environ.get('SLURM_JOB_ID')
_shm = f'/dev/shm/onco_tokenized_dna_dx12_moco_run8_{_jid}.pt' if _jid else None
tokenized_data_path = _shm if (_shm and os.path.exists(_shm)) else DISK_CACHE

config = run8_disease_config.build_config(tokenized_data_path)

print('Building dataloaders...')
print(f'  oncoformer from : {os.path.dirname(oncoformer.dataset.__file__)}')
print(f'  modalities      : {config["omics"]["architecture"]["modalities"]}  (DNA-only => MoCo single-mod fallback)')
print(f'  clip_meta_w     : {config["training"]["clip_meta_w"]}   recon_meta_w: {config["training"]["recon_meta_w"]}')
print(f'  disease task    : {list(config["metadata"]["data"]["discrete"].keys())}  '
      f'meta_weight={config["metadata"]["data"]["discrete"]["disease_term"]["meta_weight"]}')
print(f'  num_epochs      : {config["training"]["num_epochs"]}  (warm-start continuation)')
L.seed_everything(42, workers=True)
# load_metadata=True: pulls in metadata_disease_run8.pt + vocab so disease targets enter each batch.
dl = OncoformerDataLoader(config, load_metadata=True)
tokenizers = dl.dataset.tokenizers
n_classes = len(dl.dataset.vocab_metadata['disease_term']['levels'])
print(f'  train={len(dl.train_dataloader)} steps/epoch  val={len(dl.val_dataloader)} steps  '
      f'disease_classes={n_classes}')

import glob, zipfile

def _ckpt_ok(path):
    """A Lightning/torch checkpoint is a zip archive. Preemption mid-write leaves a
    truncated file that fails to load. Reading the namelist touches the central directory
    at EOF => cheap integrity check without loading the whole 7 GB."""
    try:
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False

def pick_resume_ckpt(ckpt_dir):
    """Newest *valid* run8 checkpoint, so a corrupt last.ckpt from a preempted job can't
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

print('Building model...')
backbone = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')

resume_ckpt = pick_resume_ckpt(f'{OUT_DIR}/checkpoints')
if resume_ckpt is None:
    # COLD start: warm-start the backbone from run6's final checkpoint.
    if not os.path.exists(RUN6_CKPT):
        raise FileNotFoundError(f'run6 warm-start checkpoint missing: {RUN6_CKPT}')
    print(f'Cold start: warm-starting backbone from {RUN6_CKPT}')
    # weights_only=False: a full Lightning checkpoint holds non-tensor entries; torch 2.6+
    # defaults weights_only=True which would refuse to unpickle it.
    sd = torch.load(RUN6_CKPT, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    incompat = backbone.load_state_dict(sd, strict=False)
    missing = [k for k in incompat.missing_keys]
    unexpected = [k for k in incompat.unexpected_keys]
    print(f'  warm-start load: {len(sd)} ckpt tensors; '
          f'missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print('  first missing keys   :', missing[:8])
    if unexpected:
        print('  first unexpected keys:', unexpected[:8])
    # A near-total mismatch means a wrong/renamed checkpoint — fail loudly rather than
    # silently training a randomly-initialised backbone.
    if len(missing) > 0.5 * max(len(list(backbone.state_dict())), 1):
        raise RuntimeError('warm-start matched <50% of backbone params — check RUN6_CKPT / config lineage')
else:
    print(f'Requeue: resuming combined model from {resume_ckpt}')

model = OncoformerPost(backbone, config, checkpoint_dir=f'{OUT_DIR}/checkpoints')

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

print(f'Training on {torch.cuda.device_count()} GPU(s)...  (resume={resume_ckpt is not None})')
trainer.fit(model, dl.train_dataloader, dl.val_dataloader, ckpt_path=resume_ckpt)
print('Done.')
