#!/usr/bin/env python3
"""
run6 GPU dry-run: build the real model/data and run a handful of training steps to
prove the MoCo path fires end-to-end (forward + backward) with no OOM, BEFORE
committing the full 4x B200 job. Runs on whatever GPUs are allocated (use >=2 to
exercise the DDP all_gather queue fix). Writes nothing permanent.
"""
import os, sys, warnings, math
warnings.filterwarnings('ignore')

import run6_moco_config
run6_moco_config.use_moco_oncoformer()

DATA_DIR = run6_moco_config.DATA_DIR

import torch
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import Callback

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics
assert 'Oncoformer_moco' in oncoformer.models.__file__, oncoformer.models.__file__

CACHE = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
config = run6_moco_config.build_config(CACHE)

# tiny + fast: fewer steps, no checkpoints/loggers
config['training']['num_epochs'] = 1

captured = {}

class GrabLosses(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        m = {k: float(v) for k, v in trainer.callback_metrics.items()
             if any(t in k for t in ('loss', 'clip'))}
        if m:
            captured[batch_idx] = m
        if trainer.is_global_zero:
            print(f'[step {batch_idx}] ' + '  '.join(f'{k}={v:.4f}' for k, v in sorted(m.items())), flush=True)

print('Building dataloaders...', flush=True)
L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
print(f'  train={len(dl.train_dataloader)} steps/epoch  val={len(dl.val_dataloader)} steps', flush=True)

print('Building model...', flush=True)
model = OncoformerOmics(config, tokenizers, checkpoint_dir='/tmp/run6_dryrun_ckpt')

trainer = L.Trainer(
    accelerator='gpu',
    devices=torch.cuda.device_count(),
    strategy=DDPStrategy(find_unused_parameters=True, broadcast_buffers=False) if torch.cuda.device_count() > 1 else 'auto',
    max_steps=6,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    precision='bf16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm',
    enable_checkpointing=False,
    logger=False,
    log_every_n_steps=1,
    use_distributed_sampler=False,
)
print(f'Dry-run on {torch.cuda.device_count()} GPU(s)...', flush=True)
trainer.fit(model, dl.train_dataloader, dl.val_dataloader)

if trainer.is_global_zero:
    print('\n=== DRY-RUN SUMMARY ===', flush=True)
    last = captured.get(max(captured), {}) if captured else {}
    print('final metrics:', last, flush=True)
    recon = next((v for k, v in last.items() if 'recon' in k), None)
    clip  = next((v for k, v in last.items() if 'clip' in k and 'loss' in k), None)
    ok = (recon is not None and math.isfinite(recon)
          and clip is not None and math.isfinite(clip) and clip > 0)
    print(f'recon_loss={recon}  clip_loss={clip}  ->  {"DRYRUN PASSED" if ok else "DRYRUN FAILED"}', flush=True)
    sys.exit(0 if ok else 2)
