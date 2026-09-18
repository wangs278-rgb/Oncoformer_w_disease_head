#!/usr/bin/env python3
"""Extract per-sample RAW ESM + VAF features for the disease benchmark, so a
canonical classifier "sees the same inputs" as Oncoformer.

Oncoformer's encoder ingests exactly three components (encode:true in the arch
config): protein ESM embedding, mutation ESM embedding, and aa_vaf (VAF).
Everything else (gene, alt_type, ...) is an MLM target only, NOT an input.  This
script reconstructs the RAW inputs -- the frozen ESM table lookups + the VAF
value -- and pools them to a fixed-length per-sample vector, WITHOUT the
transformer, the learned 2560->512 projection, the trained CNV-atomic overlay,
or the Fourier VAF encoder (those are all Oncoformer's learned parameters).

Per sample, over its valid alteration tokens (gene id > 3):
  mean_prot       [1280]  mean of protein ESM
  mean_mut        [1280]  mean of mutation ESM (frozen base table only)
  vafw_prot       [1280]  VAF-weighted mean of protein ESM   <- VAF influence
  vafw_mut        [1280]  VAF-weighted mean of mutation ESM  <- VAF influence
  [mean_vaf, max_vaf]      VAF magnitude summary
  -> esm_feat [5122], fp16

ESM tables + valid-token ids are taken from the EXACT paths / tokens run6 uses
(resolved config), so the baseline and Oncoformer consume identical raw signal.

    python extract_esm_features.py --run run6

Writes <analysis_dir>/plots/<run>_esm_{train,val}.npz
  (esm_feat, n_gene_tokens, baitset, group, term, sample_id).
"""
import argparse, os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'
N_SPECIAL = 4     # PAD=0, CLS=1, MASK=2, UNK=3 -> real alteration tokens have gene id > 3

RUNS = {
    'run6': dict(analysis='run6_moco_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12_moco.pt', moco=True, dx12=True),
}

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='run6', choices=list(RUNS))
args = ap.parse_args()
R = RUNS[args.run]

sys.path.insert(0, DATA_DIR)
import run6_moco_config
run6_moco_config.use_moco_oncoformer()
config = run6_moco_config.build_config(R['cache'])

import torch
import lightning as L
from torch.utils.data import DataLoader
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
print(f'[{args.run}] oncoformer from {os.path.dirname(oncoformer.dataset.__file__)}')
assert 'Oncoformer_moco' in oncoformer.dataset.__file__

# ---- exact ESM tables run6 ingests (from resolved config) ----
cc = config['omics']['modalities']['dna']['architecture']['component_configs']
PROT_W = cc['protein']['params']['weight_path']
MUT_W  = cc['mutation']['params']['weight_path']
print(f'[{args.run}] protein ESM : {PROT_W}')
print(f'[{args.run}] mutation ESM: {MUT_W}')
for c in ('protein', 'mutation', 'aa_vaf'):
    assert cc[c].get('encode') is True, f'{c} is not an encoded input in this config!'


def load_table(path):
    W = torch.load(path, map_location='cpu')
    if isinstance(W, dict) and 'weight' in W:
        W = W['weight']
    return W.float()                     # pool in fp32 for accuracy


L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'[{args.run}] split: total={len(dl.dataset)}  train={len(train_ids)}  '
      f'val={len(val_ids)}  disjoint={train_ids.isdisjoint(val_ids)}')

device = torch.device('cuda')
Wp = load_table(PROT_W).to(device)       # [Vp,1280]
Wm = load_table(MUT_W).to(device)        # [Vm,1280]
Dp, Dm = Wp.shape[1], Wm.shape[1]
print(f'[{args.run}] protein table {tuple(Wp.shape)}  mutation table {tuple(Wm.shape)}')


@torch.no_grad()
def esm_feat_for_batch(batch):
    dna = batch['omics_inputs']['dna']
    gene = dna['gene'].long().to(device)                 # [B,128]
    prot = dna['protein'].long().to(device).clamp_(0, Wp.shape[0] - 1)
    mut  = dna['mutation'].long().to(device).clamp_(0, Wm.shape[0] - 1)
    vaf  = dna['aa_vaf'].float().to(device).clamp_(0, 1)  # [B,128]

    valid = (gene > (N_SPECIAL - 1)).float()             # [B,128] real alterations
    n = valid.sum(1).clamp_min(1.0)                       # [B]

    Pe = Wp[prot]                                         # [B,128,1280]
    Me = Wm[mut]                                          # [B,128,1280]
    vm = valid.unsqueeze(-1)                              # [B,128,1]

    mean_prot = (Pe * vm).sum(1) / n.unsqueeze(1)
    mean_mut  = (Me * vm).sum(1) / n.unsqueeze(1)

    vw = vaf * valid                                      # [B,128] VAF only on real tokens
    vwn = vw.sum(1).clamp_min(1e-6)                       # [B]
    vafw_prot = (Pe * vw.unsqueeze(-1)).sum(1) / vwn.unsqueeze(1)
    vafw_mut  = (Me * vw.unsqueeze(-1)).sum(1) / vwn.unsqueeze(1)

    mean_vaf = (vw.sum(1) / n).unsqueeze(1)               # [B,1]
    max_vaf  = torch.where(valid.bool(), vaf, torch.full_like(vaf, -1)).max(1).values
    max_vaf  = max_vaf.clamp_min(0).unsqueeze(1)          # [B,1]

    feat = torch.cat([mean_prot, mean_mut, vafw_prot, vafw_mut,
                      mean_vaf, max_vaf], dim=1)          # [B, 2*Dp+2*Dm+2]
    n_gene = (gene > (N_SPECIAL - 1)).sum(1)
    return feat.half().cpu().numpy(), n_gene.cpu().numpy()


@torch.no_grad()
def run_split(loader, keep_ids, tag):
    feats, ntok, baits, groups, terms, ids = [], [], [], [], [], []
    for batch in loader:
        f, nt = esm_feat_for_batch(batch)
        md = batch['sample_metadata']['sample_metadata']
        feats.append(f); ntok.append(nt)
        baits  += list(md['BaitSet'].astype(str))
        groups += list(md['DiseaseGroup'].astype(str))
        terms  += list(md['DiseaseTerm'].astype(str))
        ids    += list(md.index.astype(str))

    feat = np.concatenate(feats, axis=0)
    n_gene = np.concatenate(ntok, axis=0)
    baits = np.array(baits); groups = np.array(groups)
    terms = np.array(terms); ids = np.array(ids)
    n = min(len(feat), len(baits), len(ids))
    feat, n_gene = feat[:n], n_gene[:n]
    baits, groups, terms, ids = baits[:n], groups[:n], terms[:n], ids[:n]

    if keep_ids:
        keep = np.array([s in keep_ids for s in ids])
        other = train_ids if keep_ids is val_ids else val_ids
        leaked = sum(1 for s in ids if s in other)
        print(f'[{args.run}] {tag}: {int(keep.sum())}/{len(ids)} in split; '
              f'{leaked} from other split (dropping)')
        assert leaked == 0, f'{leaked} cross-split samples in {tag} — split mismatch!'
        feat, n_gene = feat[keep], n_gene[keep]
        baits, groups, terms, ids = baits[keep], groups[keep], terms[keep], ids[keep]

    _, uniq = np.unique(ids, return_index=True)
    uniq.sort()
    if len(uniq) != len(ids):
        print(f'[{args.run}] {tag}: deduped {len(ids)} -> {len(uniq)} unique')
    feat, n_gene = feat[uniq], n_gene[uniq]
    baits, groups, terms, ids = baits[uniq], groups[uniq], terms[uniq], ids[uniq]
    return dict(esm_feat=feat, n_gene_tokens=n_gene,
                baitset=baits, group=groups, term=terms, sample_id=ids)


bs = int(config['training'].get('batch_size', 64))
train_loader = DataLoader(dl.train_dataset, batch_size=bs, shuffle=False, collate_fn=dl.collate_fn)
val_loader   = DataLoader(dl.val_dataset, batch_size=bs, shuffle=False, collate_fn=dl.collate_fn)

plots = os.path.join(BASE, R['analysis'], 'plots')
os.makedirs(plots, exist_ok=True)
for tag, loader, keep in [('train', train_loader, train_ids),
                          ('val',   val_loader,   val_ids)]:
    d = run_split(loader, keep, tag)
    outp = os.path.join(plots, f'{args.run}_esm_{tag}.npz')
    np.savez_compressed(outp, **d)
    print(f'[{args.run}] {tag}: wrote {outp}  esm_feat={d["esm_feat"].shape} '
          f'degenerate(0-gene)={int((d["n_gene_tokens"] == 0).sum())}')

print(f'[{args.run}] ESM feature extraction done.')
