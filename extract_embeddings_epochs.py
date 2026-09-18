#!/usr/bin/env python3
"""Extract val-set DNA embeddings from a trained Oncoformer checkpoint at MULTIPLE
epochs (seq(5,40,5) -> 0-indexed epoch files 4,9,...,39). Data + model are built
ONCE and each epoch's state_dict is swapped in, so the 8 forward passes share the
same (deterministic, val-only) dataloader.

    python extract_embeddings_epochs.py --run run4

Writes <analysis_dir>/plots/epochs/<run>_emb_ep{E}.npz for E in 5,10,...,40
(E = 1-indexed epoch = ckpt epoch + 1), each with emb/baitset/group/term/sample_id.
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'

# 0-indexed checkpoint epochs saved every 5 epochs -> 1-indexed labels 5..40
CKPT_EPOCHS = [4, 9, 14, 19, 24, 29, 34, 39]

RUNS = {
    'run4': dict(out='run4_fixed_out', analysis='run4_converge_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna.pt',           moco=False, dx12=False),
    'run5': dict(out='run5_fixed_out', analysis='run5_dx12_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12.pt',      moco=False, dx12=True),
    'run6': dict(out='run6_moco_out',  analysis='run6_moco_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12_moco.pt', moco=True,  dx12=True),
}

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, choices=list(RUNS))
args = ap.parse_args()
R = RUNS[args.run]

# ---- config + correct oncoformer resolution ----
sys.path.insert(0, DATA_DIR)
if R['moco']:
    import run6_moco_config
    run6_moco_config.use_moco_oncoformer()
    config = run6_moco_config.build_config(R['cache'])
else:
    sys.path.insert(0, f'{DATA_DIR}/Oncoformer')
    import dx12_config
    config = dx12_config.build_config(R['cache'])
    if not R['dx12']:                       # run4 = all baitsets
        config['omics']['modalities']['dna']['data'].pop('baitset_filter', None)

import torch
import lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics
print(f'[{args.run}] oncoformer from {os.path.dirname(oncoformer.dataset.__file__)}')
if R['moco']:
    assert 'Oncoformer_moco' in oncoformer.dataset.__file__

L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)               # built ONCE
tokenizers = dl.dataset.tokenizers
print(f'[{args.run}] val batches = {len(dl.val_dataloader)}')

# ---- val-only guard ----
train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'[{args.run}] split: total={len(dl.dataset)}  train={len(train_ids)}  '
      f'val={len(val_ids)}  disjoint={train_ids.isdisjoint(val_ids)}')

ckpt_dir = f'{DATA_DIR}/{R["out"]}/checkpoints'
model = OncoformerOmics(config, tokenizers, checkpoint_dir=ckpt_dir)  # built ONCE
trainer = L.Trainer(accelerator='gpu', devices=1, precision='bf16-mixed',
                    logger=False, enable_progress_bar=True)

plots = os.path.join(BASE, R['analysis'], 'plots', 'epochs')
os.makedirs(plots, exist_ok=True)


def collect(preds):
    embs, baits, groups, terms, ids = [], [], [], [], []
    for p in preds:
        if p is None or 'sample_embeddings' not in p:
            continue
        emb = p['sample_embeddings']
        emb = emb.float().cpu().numpy() if hasattr(emb, 'float') else np.asarray(emb)
        md = p.get('sample_metadata')
        if md is None:
            continue
        embs.append(emb)
        baits  += list(md['BaitSet'].astype(str))
        groups += list(md['DiseaseGroup'].astype(str))
        terms  += list(md['DiseaseTerm'].astype(str))
        ids    += list(md.index.astype(str))
    emb = np.concatenate(embs, axis=0)
    baits = np.array(baits); groups = np.array(groups); terms = np.array(terms); ids = np.array(ids)
    n = min(len(emb), len(baits), len(ids))
    emb, baits, groups, terms, ids = emb[:n], baits[:n], groups[:n], terms[:n], ids[:n]
    if val_ids:
        leaked = sum(1 for s in ids if s in train_ids)
        keep = np.array([s in val_ids for s in ids])
        assert leaked == 0, f'{leaked} TRAIN samples leaked — split mismatch!'
        emb, baits, groups, terms, ids = emb[keep], baits[keep], groups[keep], terms[keep], ids[keep]
    return emb, baits, groups, terms, ids


for ep0 in CKPT_EPOCHS:
    label = ep0 + 1                                     # 1-indexed epoch (5,10,...,40)
    ckpt_path = f'{ckpt_dir}/epoch=epoch={ep0}.ckpt'
    outp = os.path.join(plots, f'{args.run}_emb_ep{label}.npz')
    if not os.path.exists(ckpt_path):
        print(f'[{args.run}] ep{label}: MISSING {ckpt_path}, skipping'); continue
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[{args.run}] ep{label}: loaded ckpt epoch={ck.get("epoch")} '
          f'(missing={len(missing)} unexpected={len(unexpected)})')
    emb, baits, groups, terms, ids = collect(trainer.predict(model, dl.val_dataloader))
    np.savez_compressed(outp, emb=emb, baitset=baits, group=groups, term=terms, sample_id=ids)
    print(f'[{args.run}] ep{label}: wrote {outp}  {emb.shape}')

print(f'[{args.run}] all epochs done.')
