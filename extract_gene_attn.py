#!/usr/bin/env python3
"""Extract per-sample pooled DNA embeddings + a per-sample GENE-ATTENTION scalar
from a trained Oncoformer checkpoint, over the full val set.

Gene-attention scalar (per sample):
    mean over the sample's real gene tokens of the CLS->token self-attention,
    where the attention is first averaged over ALL transformer layers and heads
    (exactly model._avg_self_attn('dna'), which uses pool_attn='mean',
    pool_attn_layer=None => mean over layers+heads -> [B, Lq, Lk]).
    We take the CLS row [:,0,:], drop CLS(pos 0) and padding, and average over the
    remaining (valid) gene-alteration tokens.  Independent of #mutations.

This mirrors OncoformerOmics.predict_step's core (enable attention capture ->
_encode_interleaved -> _pool -> _avg_self_attn) with a manual eval loop, so NO
edit to the shared model repo is needed (run6 uses a separate Oncoformer_moco copy).

Embedding + attention come from the SAME forward pass => aligned by construction.

Run ONE run per process (isolates run6's MoCo module-swap from fmi's editable
install):
    python extract_gene_attn.py --run run4
    python extract_gene_attn.py --run run5
    python extract_gene_attn.py --run run6

Writes <analysis_dir>/plots/<run>_gene_attn.npz
    (emb, gene_attn, n_gene_tokens, baitset, group, term, sample_id).
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'

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

# ---- config + correct oncoformer resolution (mirror extract_embeddings.py) ----
sys.path.insert(0, DATA_DIR)  # so dx12_config / run6_moco_config import
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
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
print(f'[{args.run}] val batches = {len(dl.val_dataloader)}')

# ---- val-only guard: predictions must come from the held-out split only ----
train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'[{args.run}] split: total={len(dl.dataset)}  train={len(train_ids)}  '
      f'val={len(val_ids)}  disjoint={train_ids.isdisjoint(val_ids)}')

model = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{DATA_DIR}/{R["out"]}/checkpoints')
ckpt_path = f'{DATA_DIR}/{R["out"]}/checkpoints/last.ckpt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'[{args.run}] loaded {ckpt_path} epoch={ck.get("epoch")}  '
      f'(missing={len(missing)} unexpected={len(unexpected)})')

device = torch.device('cuda')
model = model.to(device).eval()

# enable self-attention capture on every stream (predict_step does the same when
# save_self_attn=False); forward hooks overwrite .data each forward pass.
for m in model.modalities:
    stream = model.streams[m]
    if hasattr(stream, 'clear_attention_buffers'):
        stream.clear_attention_buffers()
    if not getattr(stream, 'attention_capture_enabled', False):
        stream.enable_attention_capture()


@torch.no_grad()
def gene_attn_for_batch(inputs, masks):
    """Return (gene_attn[B], n_gene_tokens[B]) as CPU float/long tensors."""
    A = model._avg_self_attn('dna')            # [B, Lq, Lk] mean over layers+heads, or None
    if A is None:
        raise RuntimeError('attention buffers empty -> capture not active')
    cls_row = A[:, 0, :].float()               # [B, Lk]  CLS -> token
    mk = masks['dna'].bool().clone()           # [B, Lk]  1 = valid token
    mk[:, 0] = False                           # exclude CLS self-attention
    n = mk.sum(dim=1)                           # [B] number of real gene tokens
    w = (cls_row * mk.float()).sum(dim=1)       # [B] total CLS mass on gene tokens
    gene_attn = w / n.clamp_min(1).float()      # [B] mean over gene tokens (0 if none)
    gene_attn = torch.where(n > 0, gene_attn, torch.zeros_like(gene_attn))
    return gene_attn.cpu(), n.cpu()


embs, gattn, ntok, baits, groups, terms, ids = [], [], [], [], [], [], []
for batch in dl.val_dataloader:
    batch = model.transfer_batch_to_device(batch, device, 0)
    # run6's moco copy: _unpack_batch -> (inputs, masks, graphs) and _encode_interleaved
    # takes graphs; the base repo returns/accepts 2. Mirror each repo's own predict_step.
    unpacked = model._unpack_batch(batch)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        if len(unpacked) == 3:
            inputs, masks, graphs = unpacked
            H = model._encode_interleaved(inputs, masks, graphs)   # populates attn hooks
        else:
            inputs, masks = unpacked
            H = model._encode_interleaved(inputs, masks)           # populates attn hooks
        pooled = model._pool(H, masks)                 # [B, D]  same as sample_embeddings
    ga, nt = gene_attn_for_batch(inputs, masks)
    md = batch['sample_metadata']['sample_metadata']

    embs.append(pooled.float().cpu().numpy())
    gattn.append(ga.numpy()); ntok.append(nt.numpy())
    baits  += list(md['BaitSet'].astype(str))
    groups += list(md['DiseaseGroup'].astype(str))
    terms  += list(md['DiseaseTerm'].astype(str))
    ids    += list(md.index.astype(str))               # SampleID per row

emb    = np.concatenate(embs, axis=0)
gene_attn = np.concatenate(gattn, axis=0)
n_gene = np.concatenate(ntok, axis=0)
baits  = np.array(baits); groups = np.array(groups); terms = np.array(terms)
ids    = np.array(ids)
n = min(len(emb), len(gene_attn), len(baits), len(ids))
emb, gene_attn, n_gene = emb[:n], gene_attn[:n], n_gene[:n]
baits, groups, terms, ids = baits[:n], groups[:n], terms[:n], ids[:n]

# ---- enforce val-only: drop anything not in the held-out val split ----
if val_ids:
    leaked = sum(1 for s in ids if s in train_ids)
    keep = np.array([s in val_ids for s in ids])
    print(f'[{args.run}] val-only check: {int(keep.sum())}/{len(ids)} in val split; '
          f'{leaked} train-leaked (dropping non-val)')
    assert leaked == 0, f'{leaked} TRAIN samples leaked into extraction — split mismatch!'
    emb, gene_attn, n_gene = emb[keep], gene_attn[keep], n_gene[keep]
    baits, groups, terms, ids = baits[keep], groups[keep], terms[keep], ids[keep]

print(f'[{args.run}] extracted {emb.shape} embeddings; gene_attn '
      f'mean={gene_attn.mean():.4g} min={gene_attn.min():.4g} max={gene_attn.max():.4g}; '
      f'{int((n_gene == 0).sum())} degenerate (0 gene-token) samples')

plots = os.path.join(BASE, R['analysis'], 'plots')
os.makedirs(plots, exist_ok=True)
outp = os.path.join(plots, f'{args.run}_gene_attn.npz')
np.savez_compressed(outp, emb=emb, gene_attn=gene_attn, n_gene_tokens=n_gene,
                    baitset=baits, group=groups, term=terms, sample_id=ids)
print(f'[{args.run}] wrote {outp}')
