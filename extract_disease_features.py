#!/usr/bin/env python3
"""Extract TRAIN + VAL disease-classification features from a trained Oncoformer
checkpoint, in a single aligned forward pass per split.

For every sample we record, aligned by construction (same forward pass):
  - emb            : [512]  pooled CLS sample embedding  (== sample_embeddings)
  - gene_presence  : [586]  multi-hot of mutated genes (raw genomic profile,
                     model-independent -- this is the canonical-baseline feature)
  - group / term   : DiseaseGroup / DiseaseTerm labels
  - baitset        : BaitSet (DX1/DX2 kept downstream)
  - n_gene_tokens  : number of real gene-alteration tokens (>0 filter downstream)
  - sample_id      : SampleID

Unlike the val-only extractors, this runs over BOTH dl.train_dataloader and
dl.val_dataloader so a downstream probe can be FIT on the train split and
EVALUATED on the held-out val split -- the standard downstream protocol
(Oncoformer pretraining is self-supervised, so fitting a classifier on train
embeddings is not a label leak).

    python extract_disease_features.py --run run6

Writes <analysis_dir>/plots/<run>_disease_train.npz and _disease_val.npz.
Mirrors extract_gene_attn.py for config/model/oncoformer resolution.
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'

# gene component vocab width (specials PAD=0,CLS=1,MASK=2,UNK=3 dropped -> 586 real)
GENE_VOCAB = 590
N_SPECIAL  = 4

RUNS = {
    'run4': dict(out='run4_fixed_out', analysis='run4_converge_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna.pt',           moco=False, dx12=False),
    'run5': dict(out='run5_fixed_out', analysis='run5_dx12_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12.pt',      moco=False, dx12=True),
    'run6': dict(out='run6_moco_out',  analysis='run6_moco_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12_moco.pt', moco=True,  dx12=True),
}

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='run6', choices=list(RUNS))
args = ap.parse_args()
R = RUNS[args.run]

# ---- config + correct oncoformer resolution (mirror extract_gene_attn.py) ----
sys.path.insert(0, DATA_DIR)
if R['moco']:
    import run6_moco_config
    run6_moco_config.use_moco_oncoformer()
    config = run6_moco_config.build_config(R['cache'])
else:
    sys.path.insert(0, f'{DATA_DIR}/Oncoformer')
    import dx12_config
    config = dx12_config.build_config(R['cache'])
    if not R['dx12']:
        config['omics']['modalities']['dna']['data'].pop('baitset_filter', None)

import torch
import lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics
from torch.utils.data import DataLoader
print(f'[{args.run}] oncoformer from {os.path.dirname(oncoformer.dataset.__file__)}')
if R['moco']:
    assert 'Oncoformer_moco' in oncoformer.dataset.__file__

L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers

train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'[{args.run}] split: total={len(dl.dataset)}  train={len(train_ids)}  '
      f'val={len(val_ids)}  disjoint={train_ids.isdisjoint(val_ids)}')
print(f'[{args.run}] train batches={len(dl.train_dataloader)}  '
      f'val batches={len(dl.val_dataloader)}')

# ---- try to resolve gene names for interpretability (best-effort) ----
def resolve_gene_names():
    try:
        gt = tokenizers['gene'] if 'gene' in getattr(tokenizers, 'keys', lambda: [])() else None
    except Exception:
        gt = None
    for attr in ('itos', 'id_to_token', 'ids_to_tokens', 'vocab', 'stoi'):
        obj = getattr(gt, attr, None) if gt is not None else None
        if obj is None:
            continue
        try:
            if isinstance(obj, dict):
                inv = {v: k for k, v in obj.items()} if all(isinstance(v, int) for v in obj.values()) else obj
                return [str(inv.get(i, f'gene_{i}')) for i in range(GENE_VOCAB)]
            return [str(obj[i]) for i in range(GENE_VOCAB)]
        except Exception:
            continue
    return [f'gene_{i}' for i in range(GENE_VOCAB)]

gene_names_full = resolve_gene_names()
gene_names = np.array(gene_names_full[N_SPECIAL:GENE_VOCAB])   # [586]
print(f'[{args.run}] gene feature width={len(gene_names)}  e.g. {list(gene_names[:5])}')

model = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{DATA_DIR}/{R["out"]}/checkpoints')
ckpt_path = f'{DATA_DIR}/{R["out"]}/checkpoints/last.ckpt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'[{args.run}] loaded {ckpt_path} epoch={ck.get("epoch")}  '
      f'(missing={len(missing)} unexpected={len(unexpected)})')

device = torch.device('cuda')
model = model.to(device).eval()


def gene_presence_from_batch(batch):
    """Model-independent multi-hot [B,586] of mutated genes + n_gene_tokens [B]."""
    gene_tok = batch['omics_inputs']['dna']['gene'].long()      # [B,128] raw gene ids
    B = gene_tok.shape[0]
    mh = torch.zeros(B, GENE_VOCAB, dtype=torch.int8)
    idx = gene_tok.clamp(0, GENE_VOCAB - 1)
    mh.scatter_(1, idx, torch.ones_like(idx, dtype=torch.int8))
    mh[:, :N_SPECIAL] = 0                                        # drop PAD/CLS/MASK/UNK
    n_gene = (gene_tok > (N_SPECIAL - 1)).sum(dim=1)             # real gene tokens
    return mh[:, N_SPECIAL:GENE_VOCAB].numpy(), n_gene.numpy()


@torch.no_grad()
def run_split(loader, keep_ids, tag):
    embs, pres, ntok, baits, groups, terms, ids = [], [], [], [], [], [], []
    for batch in loader:
        gp, nt = gene_presence_from_batch(batch)                # from CPU tensors
        batch = model.transfer_batch_to_device(batch, device, 0)
        unpacked = model._unpack_batch(batch)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if len(unpacked) == 3:
                inputs, masks, graphs = unpacked
                H = model._encode_interleaved(inputs, masks, graphs)
            else:
                inputs, masks = unpacked
                H = model._encode_interleaved(inputs, masks)
            pooled = model._pool(H, masks)                       # [B,D] sample embedding
        md = batch['sample_metadata']['sample_metadata']
        embs.append(pooled.float().cpu().numpy())
        pres.append(gp); ntok.append(nt)
        baits  += list(md['BaitSet'].astype(str))
        groups += list(md['DiseaseGroup'].astype(str))
        terms  += list(md['DiseaseTerm'].astype(str))
        ids    += list(md.index.astype(str))

    emb = np.concatenate(embs, axis=0)
    presence = np.concatenate(pres, axis=0)
    n_gene = np.concatenate(ntok, axis=0)
    baits = np.array(baits); groups = np.array(groups)
    terms = np.array(terms); ids = np.array(ids)
    n = min(len(emb), len(presence), len(baits), len(ids))
    emb, presence, n_gene = emb[:n], presence[:n], n_gene[:n]
    baits, groups, terms, ids = baits[:n], groups[:n], terms[:n], ids[:n]

    # enforce split purity: keep only ids belonging to THIS split
    if keep_ids:
        keep = np.array([s in keep_ids for s in ids])
        other = train_ids if keep_ids is val_ids else val_ids
        leaked = sum(1 for s in ids if s in other)
        print(f'[{args.run}] {tag}: {int(keep.sum())}/{len(ids)} in split; '
              f'{leaked} from other split (dropping)')
        assert leaked == 0, f'{leaked} cross-split samples in {tag} — split mismatch!'
        emb, presence, n_gene = emb[keep], presence[keep], n_gene[keep]
        baits, groups, terms, ids = baits[keep], groups[keep], terms[keep], ids[keep]

    # dedupe by sample_id (keep first) — clean, one-row-per-sample matrix
    _, uniq = np.unique(ids, return_index=True)
    uniq.sort()
    if len(uniq) != len(ids):
        print(f'[{args.run}] {tag}: deduped {len(ids)} -> {len(uniq)} unique sample_ids')
    emb, presence, n_gene = emb[uniq], presence[uniq], n_gene[uniq]
    baits, groups, terms, ids = baits[uniq], groups[uniq], terms[uniq], ids[uniq]
    return dict(emb=emb, gene_presence=presence, n_gene_tokens=n_gene,
                baitset=baits, group=groups, term=terms, sample_id=ids)


plots = os.path.join(BASE, R['analysis'], 'plots')
os.makedirs(plots, exist_ok=True)

# clean, deterministic, full-coverage loaders (train_dataloader uses a mixture
# sampler that does NOT visit each train sample exactly once -> build our own)
bs = int(config['training'].get('batch_size', 64))
train_loader = DataLoader(dl.train_dataset, batch_size=bs, shuffle=False,
                          collate_fn=dl.collate_fn)
val_loader   = DataLoader(dl.val_dataset, batch_size=bs, shuffle=False,
                          collate_fn=dl.collate_fn)

for tag, loader, keep in [('train', train_loader, train_ids),
                          ('val',   val_loader,   val_ids)]:
    d = run_split(loader, keep, tag)
    outp = os.path.join(plots, f'{args.run}_disease_{tag}.npz')
    np.savez_compressed(outp, gene_names=gene_names, **d)
    print(f'[{args.run}] {tag}: wrote {outp}  emb={d["emb"].shape} '
          f'presence={d["gene_presence"].shape} '
          f'degenerate(0-gene)={int((d["n_gene_tokens"] == 0).sum())}')

print(f'[{args.run}] disease-feature extraction done.')
