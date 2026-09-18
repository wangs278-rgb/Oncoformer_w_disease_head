#!/usr/bin/env python3
"""
run8_disease GPU dry-run: build the REAL combined model (OncoformerOmics backbone wrapped
in OncoformerPost with a DiseaseTerm head, pre_train_model=True) and run a handful of
training steps to prove the joint MLM + MoCo + disease path fires end-to-end (forward +
backward) BEFORE committing the full 4x B200 job. Also:
  - asserts the disease head exists and is sized to the vocab (511 classes),
  - cheaply validates that run6's checkpoint is key-compatible with the backbone (the
    warm-start the full run performs) WITHOUT materialising the 7 GB of tensors (mmap),
  - checks disease loss finite & > 0, backbone recon finite, MoCo clip loss > 0.
Runs on whatever GPUs are allocated (>=2 exercises the DDP all_gather queue). Writes nothing permanent.
"""
import os, sys, warnings, math
warnings.filterwarnings('ignore')

import run8_disease_config
run8_disease_config.use_moco_oncoformer()

DATA_DIR  = run8_disease_config.DATA_DIR
RUN6_CKPT = run8_disease_config.RUN6_CKPT

import torch
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import Callback

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics, OncoformerPost
assert 'Oncoformer_moco' in oncoformer.models.__file__, oncoformer.models.__file__

CACHE = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
config = run8_disease_config.build_config(CACHE)
config['training']['num_epochs'] = 1

captured = {}

class GrabLosses(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        m = {k: float(v) for k, v in trainer.callback_metrics.items()
             if any(t in k for t in ('loss', 'clip', 'disease'))}
        if m:
            captured[batch_idx] = m
        if trainer.is_global_zero:
            print(f'[step {batch_idx}] ' + '  '.join(f'{k}={v:.4f}' for k, v in sorted(m.items())), flush=True)

print('Building dataloaders (load_metadata=True)...', flush=True)
L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)
tokenizers = dl.dataset.tokenizers
n_classes = len(dl.dataset.vocab_metadata['disease_term']['levels'])
print(f'  train={len(dl.train_dataloader)} steps/epoch  val={len(dl.val_dataloader)} steps  '
      f'disease_classes={n_classes}', flush=True)

print('Building model...', flush=True)
backbone = OncoformerOmics(config, tokenizers, checkpoint_dir='/tmp/run8_dryrun_ckpt')

# Cheap warm-start key-compat check (mmap => no tensor materialisation).
if os.path.exists(RUN6_CKPT):
    try:
        _sd = torch.load(RUN6_CKPT, map_location='cpu', mmap=True, weights_only=False)
        _sd = _sd.get('state_dict', _sd)
        _bk = set(backbone.state_dict().keys())
        _ck = set(_sd.keys())
        _missing, _unexpected = _bk - _ck, _ck - _bk
        print(f'  warm-start key check: backbone={len(_bk)} ckpt={len(_ck)} '
              f'missing={len(_missing)} unexpected={len(_unexpected)}', flush=True)
        if _missing:
            print('    first missing:', list(sorted(_missing))[:6], flush=True)
        del _sd
    except Exception as e:
        print(f'  warm-start key check skipped (mmap load failed: {e})', flush=True)
else:
    print(f'  WARNING: run6 warm-start ckpt not found at {RUN6_CKPT}', flush=True)

model = OncoformerPost(backbone, config, checkpoint_dir='/tmp/run8_dryrun_ckpt')

# Hard asserts on the disease head before spending GPU time.
assert 'disease_term' in model.prediction_heads, f'no disease head: {list(model.prediction_heads.keys())}'
_out = model.prediction_heads['disease_term'].net[-1].out_features
assert _out == n_classes, f'disease head out_features {_out} != {n_classes}'
assert model.pre_train_model is True, 'pre_train_model must be True for joint pretraining'
print(f'disease head OK: out_features={_out}, pre_train_model={model.pre_train_model}', flush=True)

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
    callbacks=[GrabLosses()],
)
print(f'Dry-run on {torch.cuda.device_count()} GPU(s)...', flush=True)
trainer.fit(model, dl.train_dataloader, dl.val_dataloader)

if trainer.is_global_zero:
    print('\n=== DRY-RUN SUMMARY ===', flush=True)
    last = captured.get(max(captured), {}) if captured else {}
    print('final metrics:', last, flush=True)
    recon   = next((v for k, v in last.items() if 'recon' in k), None)
    clip    = next((v for k, v in last.items() if 'clip' in k and 'loss' in k), None)
    disease = next((v for k, v in last.items() if 'disease' in k and 'loss' in k), None)
    ok = (recon is not None and math.isfinite(recon)
          and clip is not None and math.isfinite(clip) and clip > 0
          and disease is not None and math.isfinite(disease) and disease > 0)
    print(f'recon_loss={recon}  clip_loss={clip}  disease_loss={disease}  ->  '
          f'{"DRYRUN PASSED" if ok else "DRYRUN FAILED"}', flush=True)
    sys.exit(0 if ok else 2)
