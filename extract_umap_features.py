#!/usr/bin/env python3
"""Extract per-patient features for a UMAP of the run8_disease embedding space.

For every DX1/DX2 patient with >=1 real gene we record, in a single aligned forward
pass (same code path as insilico_ko_disease.py):

  - emb          : [512]  pooled CLS embedding  (model.backbone.calculate_pooled_embedding)
  - pred_term    : disease-head argmax lineage (over the 426 supervised classes)
  - pred_prob    : its probability
  - true_term    : sample_metadata DiseaseTerm  (Panel A color)
  - true_group   : sample_metadata DiseaseGroup (coarse organ label; optional Panel A)
  - sample_id
  - top_gene     : Panel B color. The carried panel-gene whose KNOCKOUT most LOWERS the
                   patient's disease-head probability on its OWN predicted top lineage
                   (most-negative dP). '' when undefined (see rule below).
  - top_gene_dP  : that most-negative dP (nan when top_gene == '').

Panel-B rule (mirrors the KO measurability rule): a patient gets a top_gene only if it
carries >=2 distinct real genes (so a knockout never empties the sample) AND carries >=1
gene from the KO-selected panel set (the 316 genes with >=100 carriers, loaded from
ko_results.npz). Candidate knockouts = carried genes intersect that selected set.

Knockout == zeroing the DNA attention-mask at the gene's slot(s); this is exact removal for
the positionless set-encoder (proven by insilico_ko_disease.py --mode audit). Baseline and
knockout share the SAME gathered rows and differ ONLY by the masked slots, so dP is
confound-free (the bf16 forward is batch-composition sensitive).

Outputs run8_disease_out/umap/umap_features.npz (+ umap_meta.json).

Usage:  python extract_umap_features.py [--max-batches N] [--target {pred,true}]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
UMAP_DIR = f'{OUT_DIR}/umap'
KO_NPZ   = f'{OUT_DIR}/ko/ko_results.npz'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CKPT     = f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'

CHUNK = 512          # forward sub-batch (matches KO script)
SEED  = 42

ap = argparse.ArgumentParser()
ap.add_argument('--max-batches', type=int, default=0, help='0=all; >0 for a dry run')
ap.add_argument('--target', choices=['pred', 'true'], default='pred',
                help="disease column the KO dP is measured on: patient's predicted top "
                     "lineage (default) or its true DiseaseTerm")
args = ap.parse_args()

sys.path.insert(0, DATA_DIR)
import run8_disease_config
run8_disease_config.use_moco_oncoformer()
config = run8_disease_config.build_config(CACHE)

import torch
import lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics, OncoformerPost
from torch.utils.data import DataLoader
assert 'Oncoformer_moco' in oncoformer.models.__file__, oncoformer.models.__file__

os.makedirs(UMAP_DIR, exist_ok=True)
device = torch.device('cuda')

# ----------------------------------------------------------------------------- setup
L.seed_everything(SEED, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
g2i = tok.component_token2idx['gene']
i2g = {v: k for k, v in g2i.items()}
REAL_MIN = 4                                    # gene ids 0..3 are <pad><cls><mask><unk>

# supervised disease set (identical rule to training / eval / KO)
levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
term2id = {t: i for i, t in enumerate(levels)}
IGNORE = ['other', 'nos']
ignored_ids = np.array([i for i, l in enumerate(levels) if any(p in l.lower() for p in IGNORE)])
sup_classes = np.array([i for i in range(len(levels)) if i not in set(ignored_ids.tolist())])
sup_names = np.array([levels[i] for i in sup_classes])
n_sup = len(sup_classes)
supid2col = {int(c): j for j, c in enumerate(sup_classes)}          # class id -> supervised col
print(f'classes: total={len(levels)} ignored={len(ignored_ids)} supervised={n_sup}', flush=True)

# KO-selected panel genes (candidate knockouts for Panel B)
ko = np.load(KO_NPZ, allow_pickle=True)
sel_gene_ids = set(int(x) for x in ko['gene_ids'])
print(f'selected panel genes for Panel B: {len(sel_gene_ids)}', flush=True)

# model (identical to insilico_ko_disease.py)
backbone = OncoformerOmics(config, dl.dataset.tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')
model = OncoformerPost(backbone, config, checkpoint_dir=f'{OUT_DIR}/checkpoints')
assert 'disease_term' in model.prediction_heads
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'loaded {os.path.basename(CKPT)} epoch={ck.get("epoch")} '
      f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
assert len(missing) == 0 and len(unexpected) == 0, (missing[:5], unexpected[:5])
model.to(device).eval()

_negmask = torch.full((len(levels),), 0.0, device=device)
_negmask[torch.as_tensor(ignored_ids, device=device)] = float('-inf')
_sup_idx = torch.as_tensor(sup_classes, device=device)


def to_dev(x):
    if torch.is_tensor(x): return x.to(device, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_dev(v) for v in x)
    return x


@torch.no_grad()
def probs_sup(batch):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(batch)['disease_term'].float()
    z = logits + _negmask
    z = z - z.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    p = e / e.sum(dim=1, keepdim=True)
    return p.index_select(1, _sup_idx)                       # [B, n_sup]


@torch.no_grad()
def pooled_of(batch):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        return model.backbone.calculate_pooled_embedding(batch).float()


def sub_batch(dna, m):
    return {'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m}, 'omics_graphs': {}}


# ----------------------------------------------------------------------------- pass
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
emb_chunks = []
sample_ids, true_terms, true_groups, pred_terms = [], [], [], []
pred_probs, top_genes, top_dPs = [], [], []
n_kept = 0
n_panelB = 0

print('\n===== extracting embeddings + per-patient top gene =====', flush=True)
for bi, batch in enumerate(loader):
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    gf_np = batch['omics_inputs']['dna']['gene'].numpy()
    mk_np = batch['omics_masks']['dna'].numpy()
    valid = (gf_np >= REAL_MIN) & (mk_np > 0)
    n_distinct = np.array([np.unique(gf_np[r][valid[r]]).size for r in range(gf_np.shape[0])])
    keep = np.where(dx & (n_distinct >= 1))[0]
    if keep.size == 0:
        if args.max_batches and bi + 1 >= args.max_batches: break
        continue

    bd = to_dev(batch)
    inp = bd['omics_inputs']['dna']; mask = bd['omics_masks']['dna']
    ri = torch.as_tensor(keep, device=device)
    dna_k = {c: t.index_select(0, ri) for c, t in inp.items()}
    m_k = mask.index_select(0, ri)

    p_base = probs_sup(sub_batch(dna_k, m_k))            # [K, n_sup]
    emb = pooled_of(sub_batch(dna_k, m_k))              # [K, 512]
    pred_col = p_base.argmax(1)                          # [K] supervised col
    pred_prob = p_base.gather(1, pred_col[:, None])[:, 0]

    K = keep.size
    best_dP = np.full(K, np.inf, dtype=np.float64)
    best_gene = np.full(K, -1, dtype=np.int64)

    # target column per kept row (pred argmax, or the true DiseaseTerm's supervised col)
    terms_k = md['DiseaseTerm'].astype(str).values[keep]
    if args.target == 'true':
        tgt_col = pred_col.clone()
        for kr in range(K):
            col = supid2col.get(term2id.get(terms_k[kr], -1), None)
            if col is not None:
                tgt_col[kr] = col
    else:
        tgt_col = pred_col

    # Panel B candidate (kept_row, gene_id) pairs
    gf_k = gf_np[keep]; valid_k = valid[keep]
    pairs_kr, pairs_gid = [], []
    for kr in range(K):
        if n_distinct[keep[kr]] < 2:
            continue
        genes = np.unique(gf_k[kr][valid_k[kr]])
        cand = [int(g) for g in genes.tolist() if int(g) in sel_gene_ids]
        for g in cand:
            pairs_kr.append(kr); pairs_gid.append(g)

    if pairs_kr:
        kr_all = torch.as_tensor(pairs_kr, device=device)
        gid_all = torch.as_tensor(pairs_gid, device=device)
        # baseline prob at each pair's target column (per kept row)
        tgt_base = p_base.gather(1, tgt_col[:, None])[:, 0].gather(0, kr_all)
        for s in range(0, len(pairs_kr), CHUNK):
            kr_t = kr_all[s:s + CHUNK]; gi_t = gid_all[s:s + CHUNK]
            dna_p = {c: t.index_select(0, kr_t) for c, t in dna_k.items()}
            m_p = m_k.index_select(0, kr_t).clone()
            m_p[dna_p['gene'] == gi_t.unsqueeze(1)] = 0        # remove target gene's slot(s)
            p_ko = probs_sup(sub_batch(dna_p, m_p))            # [P, n_sup]
            col = tgt_col.index_select(0, kr_t)
            dP = (p_ko.gather(1, col[:, None])[:, 0] - tgt_base[s:s + CHUNK]).cpu().numpy()
            krs = np.asarray(pairs_kr[s:s + CHUNK]); gids = np.asarray(pairs_gid[s:s + CHUNK])
            for j in range(len(krs)):
                if dP[j] < best_dP[krs[j]]:
                    best_dP[krs[j]] = dP[j]; best_gene[krs[j]] = gids[j]

    # collect
    emb_chunks.append(emb.cpu().numpy().astype(np.float32))
    sample_ids.extend(md.index.astype(str).values[keep].tolist())
    true_terms.extend(terms_k.tolist())
    true_groups.extend(md['DiseaseGroup'].astype(str).values[keep].tolist())
    pc = pred_col.cpu().numpy()
    pred_terms.extend(sup_names[pc].tolist())
    pred_probs.extend(pred_prob.cpu().numpy().astype(np.float32).tolist())
    for kr in range(K):
        if best_gene[kr] >= 0:
            top_genes.append(i2g[best_gene[kr]]); top_dPs.append(float(best_dP[kr])); n_panelB += 1
        else:
            top_genes.append(''); top_dPs.append(np.nan)
    n_kept += K

    if args.max_batches and bi + 1 >= args.max_batches:
        break
    if (bi + 1) % 100 == 0:
        print(f'  batch {bi+1}/{len(loader)}  kept={n_kept}  panelB={n_panelB}', flush=True)

# ----------------------------------------------------------------------------- save
emb = np.concatenate(emb_chunks, 0) if emb_chunks else np.zeros((0, 512), np.float32)
np.savez_compressed(
    f'{UMAP_DIR}/umap_features.npz',
    emb=emb,
    sample_id=np.array(sample_ids),
    true_term=np.array(true_terms),
    true_group=np.array(true_groups),
    pred_term=np.array(pred_terms),
    pred_prob=np.array(pred_probs, dtype=np.float32),
    top_gene=np.array(top_genes),
    top_gene_dP=np.array(top_dPs, dtype=np.float64),
)
meta = dict(ckpt=os.path.basename(CKPT), target=args.target, n_patients=int(n_kept),
            n_with_top_gene=int(n_panelB), n_supervised=int(n_sup),
            emb_dim=int(emb.shape[1]) if emb.size else 512,
            max_batches=int(args.max_batches))
json.dump(meta, open(f'{UMAP_DIR}/umap_meta.json', 'w'), indent=2)
print(f'\nDONE  patients={n_kept}  with_top_gene={n_panelB}  '
      f'-> {UMAP_DIR}/umap_features.npz', flush=True)
if n_kept:
    from collections import Counter
    print('top-15 Panel-B genes:',
          Counter(g for g in top_genes if g).most_common(15), flush=True)
    print('top-10 Panel-A lineages:',
          Counter(true_terms).most_common(10), flush=True)
