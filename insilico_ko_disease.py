#!/usr/bin/env python3
"""In-silico gene knockout on the run8_disease head.

For every panel gene G, in each DX1/DX2 patient CARRYING a mutation in G, remove that
mutation (zero the DNA attention-mask at the slot(s) whose `gene` field == id(G)) and
measure how the disease-head probability landscape shifts:  dP = P_perturbed - P_baseline
over the 426 supervised disease classes.

Why masking == removal (verified): the DNA encoder is a positionless SET encoder
(initial_embed adds NO positional encoding; self-attn uses only src_key_padding_mask) and
pooling is `cls_token`.  So zeroing a slot's mask removes that mutation's entire
contribution (gene + ESM-protein + ESM-mutation + VAF) from the CLS with no position/order
confound.  `--mode audit` proves this numerically before any full run.

Population: ALL DX1/DX2 carriers (train+val), n_gene>0.
Genes: all panel genes with >= MIN_CARRIERS *measurable* carriers.
Measurable carrier of G = patient with >=1 slot of G AND >=2 distinct real genes (so removing
G does not empty the sample; single-gene-only carriers are excluded and counted).
Common genes are capped at CAP carriers via a stable per-(sample,gene) hash (unbiased,
stateless, reproducible); genes below CAP use ALL carriers. Capped genes are reported.

Outputs run8_disease_out/ko/ko_results.npz  (+ ko_meta.json) for the plotting script.

Usage:  python insilico_ko_disease.py --mode {audit,full}
"""
import os, sys, json, argparse, warnings, hashlib
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
KO_DIR   = f'{OUT_DIR}/ko'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CKPT     = f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'

MIN_CARRIERS = 100
CAP          = 20000       # per-gene carrier cap (stat-irrelevant above this); reported
CHUNK        = 512         # perturbation forward sub-batch
SEED         = 42

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['audit', 'full'], required=True)
ap.add_argument('--max-batches', type=int, default=0, help='0=all (full mode)')
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

os.makedirs(KO_DIR, exist_ok=True)
device = torch.device('cuda')

# ----------------------------------------------------------------------------- setup
L.seed_everything(SEED, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)
tokenizers = dl.dataset.tokenizers

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
g2i = tok.component_token2idx['gene']          # gene name -> id
i2g = {v: k for k, v in g2i.items()}
SPECIALS = set(tok.special_tokens)
REAL_MIN = 4                                    # gene ids 0..3 are <pad><cls><mask><unk>
real_gene_ids = np.array(sorted(v for k, v in g2i.items() if k not in SPECIALS))
assert real_gene_ids.min() == REAL_MIN

# disease vocab / supervised set (identical rule to training & eval)
levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
term2id = {t: i for i, t in enumerate(levels)}
IGNORE = ['other', 'nos']
ignored_ids = np.array([i for i, l in enumerate(levels) if any(p in l.lower() for p in IGNORE)])
sup_classes = np.array([i for i in range(len(levels)) if i not in set(ignored_ids.tolist())])
sup_names = [levels[i] for i in sup_classes]
n_sup = len(sup_classes)
print(f'classes: total={len(levels)} ignored={len(ignored_ids)} supervised={n_sup}', flush=True)

# model
backbone = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')
model = OncoformerPost(backbone, config, checkpoint_dir=f'{OUT_DIR}/checkpoints')
assert 'disease_term' in model.prediction_heads
assert not getattr(model, 'use_confounder', False), 'use_confounder unsupported by this KO script'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'loaded {os.path.basename(CKPT)} epoch={ck.get("epoch")} missing={len(missing)} unexpected={len(unexpected)}', flush=True)
assert len(missing) == 0 and len(unexpected) == 0, (missing[:5], unexpected[:5])
model.to(device).eval()

# supervised-softmax mask over 511 logits -> probs over 426 supervised cols
_negmask = torch.full((len(levels),), 0.0, device=device)
_negmask[torch.as_tensor(ignored_ids, device=device)] = float('-inf')
_sup_idx = torch.as_tensor(sup_classes, device=device)

def to_dev(x):
    if torch.is_tensor(x): return x.to(device, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_dev(v) for v in x)
    return x

def cast_fp32(x):
    """Cast every floating tensor to fp32 (audit only; leaves int/long ids and non-tensors)."""
    if torch.is_tensor(x): return x.float() if x.is_floating_point() else x
    if isinstance(x, dict): return {k: cast_fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(cast_fp32(v) for v in x)
    return x

@torch.no_grad()
def probs_sup(batch, use_ac=True):
    """batch(on device) -> [B, n_sup] supervised probabilities."""
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
        logits = model(batch)['disease_term'].float()       # [B,511]
    z = logits + _negmask
    z = z - z.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    p = e / e.sum(dim=1, keepdim=True)
    return p.index_select(1, _sup_idx)                       # [B,n_sup]

@torch.no_grad()
def pooled_of(batch, use_ac=True):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
        return model.backbone.calculate_pooled_embedding(batch).float()

def make_pert_batch(base_inputs, base_mask, row_idx, gene_ids):
    """Gather rows row_idx and zero-mask each row's slots where gene==gene_ids[r]."""
    ri = torch.as_tensor(row_idx, device=device)
    gi = torch.as_tensor(gene_ids, device=device)
    dna = {c: t.index_select(0, ri) for c, t in base_inputs.items()}
    m = base_mask.index_select(0, ri).clone()
    gf = dna['gene']
    m[gf == gi.unsqueeze(1)] = 0
    return {'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m}, 'omics_graphs': {}}

def batch_dna(batch):
    assert not batch.get('omics_graphs'), 'non-empty omics_graphs not supported'
    return batch['omics_inputs']['dna'], batch['omics_masks']['dna']

def keep_pair(sid, gene_name, count):
    """Stable unbiased subsample: keep all if count<=CAP else keep ~CAP via hash threshold."""
    if count <= CAP:
        return True
    h = int(hashlib.md5(f'{sid}|{gene_name}|{SEED}'.encode()).hexdigest()[:12], 16)
    return (h % 1_000_000) < int(1_000_000 * CAP / count)

# =============================================================================== AUDIT
if args.mode == 'audit':
    print('\n===== AUDIT: numerical ablation invariants =====', flush=True)
    # This model stores precomputed-embedding buffers in fp16 (reconciled with its fp32
    # LayerNorms only under autocast). The audit tests the ABLATION MATH, so we cast the
    # whole model to fp32 and run WITHOUT autocast for exact, tight tolerances. The full
    # run leaves the model as-loaded and uses bf16 autocast (matches training/eval).
    model.float()
    print('audit: model cast to fp32 (autocast disabled for invariant forwards)', flush=True)
    loader = DataLoader(dl.dataset, batch_size=256, shuffle=False, collate_fn=dl.collate_fn)
    it = iter(loader)
    # find a batch containing DX1/DX2 nondegenerate patients
    batch = None
    for _ in range(20):
        b = next(it)
        gf = b['omics_inputs']['dna']['gene']; mk = b['omics_masks']['dna']
        md = b['sample_metadata']['sample_metadata']
        dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
        nz = (((gf >= REAL_MIN) & (mk > 0)).sum(1).numpy() >= 2)
        if (dx & nz).sum() >= 4:
            batch = b; keep = np.where(dx & nz)[0]; break
    assert batch is not None, 'no suitable audit batch found'
    bd = cast_fp32(to_dev(batch))                    # fp32 inputs to match fp32 model
    inp, mask = batch_dna(bd)
    gf_np = batch['omics_inputs']['dna']['gene'].numpy()
    mk_np = batch['omics_masks']['dna'].numpy()
    print(f'audit batch: {len(keep)} DX1/DX2 multi-gene patients; L={mask.shape[1]}', flush=True)

    # (3) determinism: same batch twice -> identical
    p1 = probs_sup(bd, use_ac=False); p2 = probs_sup(bd, use_ac=False)
    d_det = (p1 - p2).abs().max().item()
    print(f'[3] determinism  max|dP| over identical reruns = {d_det:.2e}', flush=True)

    # (1) absent-gene isolation: ablate a gene the patient does NOT carry -> dP == 0.
    #     Baseline is a MATCHED single-row forward of b0 (same batch size as the perturbation),
    #     so the ONLY variable is the mask edit; this removes fp32 batch-size GEMM noise.
    b0 = int(keep[0])
    def single_row(b_idx):
        ri = torch.as_tensor([b_idx], device=device)
        dna = {c: t.index_select(0, ri) for c, t in inp.items()}
        m = mask.index_select(0, ri).clone()
        return {'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m}, 'omics_graphs': {}}
    p_b0 = probs_sup(single_row(b0), use_ac=False)[0]                     # matched 1-row baseline
    present0 = set(gf_np[b0][(gf_np[b0] >= REAL_MIN) & (mk_np[b0] > 0)].tolist())
    absent_gid = int(next(g for g in real_gene_ids.tolist() if g not in present0))
    pb = make_pert_batch(inp, mask, [b0], [absent_gid])
    dP_absent = (probs_sup(pb, use_ac=False)[0] - p_b0).abs().max().item()
    print(f'[1] absent-gene ({i2g[absent_gid]}) isolation  max|dP| = {dP_absent:.2e}', flush=True)

    # present-gene sanity: ablating a carried gene DOES move predictions
    present_gid = int(next(iter(present0)))
    pb2 = make_pert_batch(inp, mask, [b0], [present_gid])
    dP_present = (probs_sup(pb2, use_ac=False)[0] - p_b0).abs().max().item()
    print(f'    present-gene ({i2g[present_gid]}) effect     max|dP| = {dP_present:.2e} (expect >0)', flush=True)

    # (4) mask-ablation == physical prune-and-repad (proves set-encoder equivalence)
    def prune_pooled(b_idx, gid):
        L_ = mask.shape[1]
        row_keep = ((mask[b_idx] > 0) & (inp['gene'][b_idx] != gid))  # keep valid, non-target (incl CLS)
        idx = torch.where(row_keep)[0]
        nk = idx.numel()
        dna = {}
        for c, t in inp.items():
            row = t[b_idx].index_select(0, idx)
            pad = torch.zeros((L_ - nk,) + tuple(row.shape[1:]), dtype=row.dtype, device=device)
            dna[c] = torch.cat([row, pad], 0).unsqueeze(0)
        m = torch.cat([torch.ones(nk, device=device), torch.zeros(L_ - nk, device=device)]).unsqueeze(0)
        return pooled_of({'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m}, 'omics_graphs': {}}, use_ac=False)
    worst = 0.0
    for bi in keep[:4]:
        bi = int(bi)
        pres = set(gf_np[bi][(gf_np[bi] >= REAL_MIN) & (mk_np[bi] > 0)].tolist())
        gid = int(next(iter(pres)))
        pooled_mask = pooled_of(make_pert_batch(inp, mask, [bi], [gid]), use_ac=False)
        pooled_prune = prune_pooled(bi, gid)
        worst = max(worst, (pooled_mask - pooled_prune).abs().max().item())
    print(f'[4] mask==prune  max|pooled diff| over 4 patients = {worst:.2e}', flush=True)

    # (2) remove-all -> degenerate: fully-ablated pooled is input-independent (identical across patients)
    #     keep ONLY position 0 (CLS); any residual unmasked slot would leak per-patient content
    assert (inp['gene'][:, 0] < REAL_MIN).all(), 'position 0 is not a special/CLS token'
    def remove_all(b_idx):
        m = torch.zeros_like(mask[b_idx])
        m[0] = mask[b_idx][0]                                 # keep only CLS at position 0
        dna = {c: t[b_idx].unsqueeze(0) for c, t in inp.items()}
        return pooled_of({'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m.unsqueeze(0)}, 'omics_graphs': {}}, use_ac=False)
    degs = torch.cat([remove_all(int(bi)) for bi in keep[:5]], 0)
    d_deg = (degs - degs[0:1]).abs().max().item()
    print(f'[2] remove-all degeneracy  max|pooled diff| across 5 patients = {d_deg:.2e}', flush=True)

    ok = (d_det < 1e-5 and dP_absent < 1e-5 and dP_present > 1e-4
          and worst < 5e-3 and d_deg < 5e-3)
    print(f'\nAUDIT {"PASSED" if ok else "FAILED"}  '
          f'(det<1e-5:{d_det<1e-5} absent0:{dP_absent<1e-5} present>0:{dP_present>1e-4} '
          f'mask==prune:{worst<5e-3} degenerate:{d_deg<5e-3})', flush=True)
    sys.exit(0 if ok else 2)

# =============================================================================== FULL
print('\n===== FULL: pass 1 (carrier census) =====', flush=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
carrier_count = np.zeros(len(g2i), dtype=np.int64)   # measurable carriers per gene id
n_dx_nondeg = 0
for bi, batch in enumerate(loader):
    gf = batch['omics_inputs']['dna']['gene'].numpy()
    mk = batch['omics_masks']['dna'].numpy()
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    for r in np.where(dx)[0]:
        valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
        genes = np.unique(gf[r][valid])
        if genes.size < 2:                # single-gene-only -> not measurable
            continue
        n_dx_nondeg += 1
        carrier_count[genes] += 1
    if args.max_batches and bi + 1 >= args.max_batches:
        break
    if (bi + 1) % 200 == 0:
        print(f'  pass1 batch {bi+1}/{len(loader)}', flush=True)

selected = np.where(carrier_count >= MIN_CARRIERS)[0]
selected = selected[np.argsort(-carrier_count[selected])]        # by frequency desc
sel_names = [i2g[g] for g in selected]
G = len(selected)
planned = int(np.minimum(carrier_count[selected], CAP).sum())
capped = [(i2g[g], int(carrier_count[g])) for g in selected if carrier_count[g] > CAP]
print(f'pass1: DX1/DX2 measurable patients={n_dx_nondeg}  selected genes(>={MIN_CARRIERS})={G}  '
      f'planned perturbations~={planned}  capped(>{CAP}):{len(capped)}', flush=True)
if capped:
    print('  capped genes:', capped, flush=True)

gid2local = -np.ones(len(g2i), dtype=np.int64)
gid2local[selected] = np.arange(G)

# accumulators
acc_sum   = torch.zeros(G, n_sup, dtype=torch.float64, device=device)
acc_sumsq = torch.zeros(G, n_sup, dtype=torch.float64, device=device)
acc_cnt   = torch.zeros(G, dtype=torch.float64, device=device)
acc_tv    = torch.zeros(G, dtype=torch.float64, device=device)
acc_kl    = torch.zeros(G, dtype=torch.float64, device=device)
acc_flip  = torch.zeros(G, dtype=torch.float64, device=device)
disease_comp = np.zeros((G, n_sup), dtype=np.int64)   # carrier true-label composition (CPU; per-pair increment)
sup_pos = {int(c): j for j, c in enumerate(sup_classes)}                    # class id -> col

print('\n===== FULL: pass 2 (baseline + knockout) =====', flush=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
n_pairs = 0
for bi, batch in enumerate(loader):
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    if not dx.any():
        if args.max_batches and bi + 1 >= args.max_batches: break
        continue
    bd = to_dev(batch)
    inp, mask = batch_dna(bd)
    gf_np = batch['omics_inputs']['dna']['gene'].numpy()
    mk_np = batch['omics_masks']['dna'].numpy()
    sids = md.index.astype(str).tolist()
    terms = md['DiseaseTerm'].astype(str).tolist()

    row_idx, gene_ids, gloc = [], [], []
    for r in np.where(dx)[0]:
        valid = (gf_np[r] >= REAL_MIN) & (mk_np[r] > 0)
        genes = np.unique(gf_np[r][valid])
        if genes.size < 2:
            continue
        tid = term2id.get(terms[r], -1)
        tcol = sup_pos.get(tid, None)                        # supervised true-label col (or None)
        for g in genes.tolist():
            lc = gid2local[g]
            if lc < 0:
                continue
            if not keep_pair(sids[r], i2g[g], int(carrier_count[g])):
                continue
            row_idx.append(int(r)); gene_ids.append(int(g)); gloc.append(int(lc))
            if tcol is not None:
                disease_comp[lc, tcol] += 1
    if not row_idx:
        if args.max_batches and bi + 1 >= args.max_batches: break
        continue

    gloc_t = torch.as_tensor(gloc, device=device)
    # chunked forward: baseline and knockout share the SAME gathered rows (identical batch
    # composition), differing ONLY by the zeroed mask slots -> dP isolates the knockout with
    # no batch-composition confound (the forward is batch-composition-sensitive in bf16).
    for s in range(0, len(row_idx), CHUNK):
        ri = row_idx[s:s + CHUNK]; gi = gene_ids[s:s + CHUNK]; gl = gloc_t[s:s + CHUNK]
        ri_t = torch.as_tensor(ri, device=device)
        gi_t = torch.as_tensor(gi, device=device)
        dna = {c: t.index_select(0, ri_t) for c, t in inp.items()}   # shared inputs
        m_base = mask.index_select(0, ri_t)
        m_pert = m_base.clone()
        m_pert[dna['gene'] == gi_t.unsqueeze(1)] = 0                  # remove the target gene's slot(s)
        p_base = probs_sup({'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m_base}, 'omics_graphs': {}})
        p_pert = probs_sup({'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m_pert}, 'omics_graphs': {}})
        dP = p_pert - p_base                                 # [K,n_sup]
        acc_sum.index_add_(0, gl, dP.double())
        acc_sumsq.index_add_(0, gl, (dP * dP).double())
        acc_cnt.index_add_(0, gl, torch.ones(len(ri), dtype=torch.float64, device=device))
        acc_tv.index_add_(0, gl, (0.5 * dP.abs().sum(1)).double())
        kl = (p_base * ((p_base + 1e-9).log() - (p_pert + 1e-9).log())).sum(1)
        acc_kl.index_add_(0, gl, kl.double())
        flip = (p_pert.argmax(1) != p_base.argmax(1)).double()
        acc_flip.index_add_(0, gl, flip)
        n_pairs += len(ri)
    if args.max_batches and bi + 1 >= args.max_batches:
        break
    if (bi + 1) % 100 == 0:
        print(f'  pass2 batch {bi+1}/{len(loader)}  pairs={n_pairs}', flush=True)

# ----------------------------------------------------------------------------- reduce
cnt = acc_cnt.clamp_min(1).cpu().numpy()
n_used = acc_cnt.cpu().numpy().astype(np.int64)
meandP = (acc_sum.cpu().numpy() / cnt[:, None])
var = (acc_sumsq.cpu().numpy() / cnt[:, None]) - meandP ** 2
sd = np.sqrt(np.clip(var, 0, None))
se = sd / np.sqrt(cnt[:, None])
influence_tv = (acc_tv.cpu().numpy() / cnt)
influence_kl = (acc_kl.cpu().numpy() / cnt)
flip_rate = (acc_flip.cpu().numpy() / cnt)
comp = disease_comp

# one-sample t on dP vs 0, per cell; BH-FDR across all cells
from scipy import stats
with np.errstate(divide='ignore', invalid='ignore'):
    tval = np.where(se > 0, meandP / se, 0.0)
dfree = np.clip(cnt[:, None] - 1, 1, None)
pval = 2 * stats.t.sf(np.abs(tval), dfree)
flat = pval.ravel()
order = np.argsort(flat)
m = flat.size
bh = np.empty(m); ranked = flat[order]
bh_sorted = np.minimum.accumulate((ranked * m / (np.arange(m) + 1))[::-1])[::-1]
bh[order] = np.clip(bh_sorted, 0, 1)
qval = bh.reshape(pval.shape)

np.savez_compressed(
    f'{KO_DIR}/ko_results.npz',
    genes=np.array(sel_names), gene_ids=selected, n_used=n_used, carrier_count=carrier_count[selected],
    sup_classes=sup_classes, sup_names=np.array(sup_names),
    meandP=meandP, sd=sd, se=se, tval=tval, pval=pval, qval=qval,
    influence_tv=influence_tv, influence_kl=influence_kl, flip_rate=flip_rate,
    disease_comp=comp,
)
meta = dict(mode='full', ckpt=os.path.basename(CKPT), min_carriers=MIN_CARRIERS, cap=CAP,
            n_dx_nondeg=int(n_dx_nondeg), n_genes=int(G), n_pairs=int(n_pairs),
            n_supervised=int(n_sup), capped_genes=capped)
json.dump(meta, open(f'{KO_DIR}/ko_meta.json', 'w'), indent=2)
print(f'\nDONE  genes={G}  pairs={n_pairs}  -> {KO_DIR}/ko_results.npz', flush=True)
print('top-10 genes by mean TV influence:', flush=True)
top = np.argsort(-influence_tv)[:10]
for j in top:
    dd = meandP[j]
    dn = sup_names[int(np.argmin(dd))]
    print(f'  {sel_names[j]:8s} n={n_used[j]:6d} TV={influence_tv[j]:.4f} flip={flip_rate[j]:.3f} '
          f'| most-lowered lineage: {dn} ({dd.min():+.3f})', flush=True)
