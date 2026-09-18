#!/usr/bin/env python3
"""In-silico gene knock-IN (addition) on the run8_disease head.  Mirror of the knockout.

For every panel gene G, in each DX1/DX2 patient that does NOT carry G, INSERT G's modal
(most-frequent real) mutation token into a free slot and set its mask=1, then measure how
the disease-head probability landscape shifts:  dP = P_added - P_baseline over the 426
supervised disease classes.  Positive dP = adding G raises that lineage (model reads G as
evidence FOR it); negative dP = adding G lowers it.

Why free-slot insertion == genuine presence (mirror of the knockout argument): the DNA
encoder is a positionless SET encoder (no positional encoding; self-attn uses only
src_key_padding_mask) and pooling is `cls_token`.  A padding slot with mask=0 is inert
(proven by the knockout mask==prune audit); writing a real token's components there and
flipping mask->1 adds exactly that mutation's contribution to the CLS, independent of which
free slot is used.  `--mode audit` proves this numerically before any full run.

Injected mutation = the MODAL real mutation of G (the single most-frequent full token —
gene + ESM-protein + ESM-mutation + VAF — observed across all real carriers of G in the
cohort).  Deterministic, in-distribution, one canonical perturbation per gene.

Population (recipients): ALL DX1/DX2 patients with >=1 real gene, >=1 free slot, and NOT
carrying G.  Genes: the same panel genes as the knockout (>= MIN_CARRIERS real carriers).
Recipients per gene are capped at RECIP_CAP via the same stable per-(sample,gene) hash.

Outputs run8_disease_out/ki/ki_results.npz (+ ki_meta.json).

Usage:  python insilico_ki_disease.py --mode {audit,full}
"""
import os, sys, json, argparse, warnings, hashlib
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
KI_DIR   = f'{OUT_DIR}/ki'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CKPT     = f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'

MIN_CARRIERS = 100         # gene must have >= this many real carriers to be tested (same set as KO)
RECIP_CAP    = 10000       # per-gene recipient cap (n=10k gives SE ~ sd/100; reported)
CHUNK        = 512
SEED         = 42

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['audit', 'full'], required=True)
ap.add_argument('--max-batches', type=int, default=0, help='0=all (full mode)')
ap.add_argument('--audit-census-batches', type=int, default=80, help='batches to scan for modal tokens in audit')
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

os.makedirs(KI_DIR, exist_ok=True)
device = torch.device('cuda')

# ----------------------------------------------------------------------------- setup
L.seed_everything(SEED, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)
tokenizers = dl.dataset.tokenizers

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
g2i = tok.component_token2idx['gene']
i2g = {v: k for k, v in g2i.items()}
SPECIALS = set(tok.special_tokens)
REAL_MIN = 4
real_gene_ids = np.array(sorted(v for k, v in g2i.items() if k not in SPECIALS))
assert real_gene_ids.min() == REAL_MIN

levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
term2id = {t: i for i, t in enumerate(levels)}
IGNORE = ['other', 'nos']
ignored_ids = np.array([i for i, l in enumerate(levels) if any(p in l.lower() for p in IGNORE)])
sup_classes = np.array([i for i in range(len(levels)) if i not in set(ignored_ids.tolist())])
sup_names = [levels[i] for i in sup_classes]
n_sup = len(sup_classes)
print(f'classes: total={len(levels)} ignored={len(ignored_ids)} supervised={n_sup}', flush=True)

backbone = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')
model = OncoformerPost(backbone, config, checkpoint_dir=f'{OUT_DIR}/checkpoints')
assert 'disease_term' in model.prediction_heads
assert not getattr(model, 'use_confounder', False)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'loaded {os.path.basename(CKPT)} epoch={ck.get("epoch")} missing={len(missing)} unexpected={len(unexpected)}', flush=True)
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

def cast_fp32(x):
    if torch.is_tensor(x): return x.float() if x.is_floating_point() else x
    if isinstance(x, dict): return {k: cast_fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(cast_fp32(v) for v in x)
    return x

@torch.no_grad()
def probs_sup(batch, use_ac=True):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
        logits = model(batch)['disease_term'].float()
    z = logits + _negmask
    z = z - z.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    p = e / e.sum(dim=1, keepdim=True)
    return p.index_select(1, _sup_idx)

@torch.no_grad()
def pooled_of(batch, use_ac=True):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
        return model.backbone.calculate_pooled_embedding(batch).float()

def batch_dna(batch):
    assert not batch.get('omics_graphs')
    return batch['omics_inputs']['dna'], batch['omics_masks']['dna']

def keep_pair(sid, gene_name, count, cap):
    if count <= cap:
        return True
    h = int(hashlib.md5(f'{sid}|{gene_name}|{SEED}'.encode()).hexdigest()[:12], 16)
    return (h % 1_000_000) < int(1_000_000 * cap / count)

# ------- modal-token signature (identical mutations -> identical component bytes -> same sig)
COMP_KEYS = None
def set_comp_keys(dna):
    global COMP_KEYS
    if COMP_KEYS is None:
        COMP_KEYS = list(dna.keys())
        assert 'gene' in COMP_KEYS

def slot_sig(dna_np, r, s):
    return hash(b'|'.join(dna_np[c][r, s].tobytes() for c in COMP_KEYS))

def census_and_modal(loader, max_batches, capture_all=False):
    """One data pass: carrier_count (patients/gene), recipient pool size, and per-gene
    signature frequencies (+ first example components per sig). Returns everything needed
    to pick each gene's modal token."""
    carrier_count = np.zeros(len(g2i), dtype=np.int64)
    sig_count = {}                      # gene_id -> {sig: count}
    sig_example = {}                    # sig -> {comp: cpu tensor}  (captured lazily)
    n_recip_pool = 0                    # DX1/DX2 patients with >=1 real gene AND a free slot
    for bi, batch in enumerate(loader):
        dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
        set_comp_keys(dna)
        dna_np = {c: dna[c].numpy() for c in COMP_KEYS}
        mk = mask.numpy()
        gf = dna_np['gene']
        md = batch['sample_metadata']['sample_metadata']
        dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
        L_ = mask.shape[1]
        for r in np.where(dx)[0]:
            valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
            genes = np.unique(gf[r][valid])
            has_free = (mk[r] == 0).any()
            if genes.size >= 1 and has_free:
                n_recip_pool += 1
            if genes.size < 1:
                continue
            carrier_count[genes] += 1                 # patient counts once per distinct gene
            for s in np.where(valid)[0]:
                g = int(gf[r, s]); sig = slot_sig(dna_np, r, s)
                d = sig_count.setdefault(g, {})
                d[sig] = d.get(sig, 0) + 1
                if capture_all and sig not in sig_example:
                    sig_example[sig] = {c: dna[c][r, s].clone() for c in COMP_KEYS}
        if max_batches and bi + 1 >= max_batches:
            break
        if (bi + 1) % 200 == 0:
            print(f'  census batch {bi+1}/{len(loader)}', flush=True)
    return carrier_count, sig_count, sig_example, n_recip_pool

def modal_sig_of(sig_count, gid):
    d = sig_count[gid]
    return max(d.items(), key=lambda kv: kv[1])       # (sig, count)

def capture_examples(loader, want, max_batches=0):
    """Second pass: grab component values for the specific modal sigs in `want` (sig->key)."""
    got = {}
    for bi, batch in enumerate(loader):
        dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
        dna_np = {c: dna[c].numpy() for c in COMP_KEYS}
        mk = mask.numpy(); gf = dna_np['gene']
        md = batch['sample_metadata']['sample_metadata']
        dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
        for r in np.where(dx)[0]:
            valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
            for s in np.where(valid)[0]:
                sig = slot_sig(dna_np, r, s)
                if sig in want and want[sig] not in got:
                    got[want[sig]] = {c: dna[c][r, s].clone() for c in COMP_KEYS}
        if len(got) == len(set(want.values())):
            break
        if max_batches and bi + 1 >= max_batches:
            break
        if (bi + 1) % 200 == 0:
            print(f'  capture batch {bi+1}/{len(loader)}  got {len(got)}/{len(set(want.values()))}', flush=True)
    return got

def stack_modal(examples_by_local, G, fp32=False):
    """examples_by_local: dict local_idx -> {comp: tensor} -> device tensors [G,*feat] per comp."""
    modal_dev = {}
    for c in COMP_KEYS:
        ref = next(iter(examples_by_local.values()))[c]
        buf = torch.zeros((G,) + tuple(ref.shape), dtype=ref.dtype)
        for lc, ex in examples_by_local.items():
            buf[lc] = ex[c]
        buf = buf.to(device)
        if fp32 and buf.is_floating_point():
            buf = buf.float()
        modal_dev[c] = buf
    return modal_dev

def first_free(m_rows):
    """[K,L] mask -> [K] index of first slot with mask==0 (rows must have >=1 free slot)."""
    return (m_rows == 0).float().argmax(1)

def insert_modal(dna, m_base, gl_t, modal_dev, use_free=None):
    """Return (dna_add, m_add): clone dna, write each row's modal token (by local gene idx
    gl_t) into its first free slot and set that mask=1. m_base is NOT mutated."""
    K, Ln = m_base.shape
    f_idx = first_free(m_base) if use_free is None else use_free
    rows = torch.arange(K, device=device)
    dna_add = {c: dna[c].clone() for c in COMP_KEYS}
    m_add = m_base.clone()
    for c in COMP_KEYS:
        src = modal_dev[c].index_select(0, gl_t)         # [K,*feat]
        dna_add[c][rows, f_idx] = src
    m_add[rows, f_idx] = 1
    return dna_add, m_add, f_idx

# =============================================================================== AUDIT
if args.mode == 'audit':
    print('\n===== AUDIT: numerical insertion invariants =====', flush=True)
    model.float()
    print('audit: model cast to fp32 (autocast disabled for invariant forwards)', flush=True)
    loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
    print(f'audit: scanning {args.audit_census_batches} batches for modal tokens ...', flush=True)
    cc, sig_count, sig_example, n_recip_pool = census_and_modal(
        loader, args.audit_census_batches, capture_all=True)
    print(f'audit: component keys = {COMP_KEYS}', flush=True)
    for c in COMP_KEYS:
        ex = next(iter(sig_example.values()))[c]
        print(f'    comp {c:16s} shape/slot={tuple(ex.shape)} dtype={ex.dtype}', flush=True)
    gstar = int(np.argmax(cc))                              # most common gene in the scan
    msig, mcnt = modal_sig_of(sig_count, gstar)
    tot = sum(sig_count[gstar].values())
    print(f'audit target gene g*={i2g[gstar]} (id {gstar}): {cc[gstar]} carriers, '
          f'{len(sig_count[gstar])} distinct mutations, modal count {mcnt}/{tot} '
          f'({100*mcnt/tot:.1f}%)', flush=True)
    modal_dev = stack_modal({0: sig_example[msig]}, 1, fp32=True)   # local idx 0 == g*

    # find a DX1/DX2 recipient that does NOT carry g* and has a free slot
    it = iter(DataLoader(dl.dataset, batch_size=256, shuffle=False, collate_fn=dl.collate_fn))
    rb = None
    for _ in range(30):
        b = next(it)
        gf = b['omics_inputs']['dna']['gene'].numpy(); mk = b['omics_masks']['dna'].numpy()
        md = b['sample_metadata']['sample_metadata']
        dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
        cand = []
        for r in np.where(dx)[0]:
            valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
            if (np.unique(gf[r][valid]).size >= 1 and (mk[r] == 0).any()
                    and gstar not in set(gf[r][valid].tolist())):
                cand.append(int(r))
        if len(cand) >= 5:
            rb = b; keep = cand[:5]; break
    assert rb is not None, 'no suitable audit recipient found'
    bd = cast_fp32(to_dev(rb)); inp, mask = batch_dna(bd)
    print(f'audit recipients: {len(keep)} DX1/DX2 non-carriers of g* with a free slot; L={mask.shape[1]}', flush=True)

    def rows_of(idxs):
        ri = torch.as_tensor(idxs, device=device)
        dna = {c: inp[c].index_select(0, ri) for c in COMP_KEYS}
        m = mask.index_select(0, ri).clone()
        return dna, m
    gl0 = torch.zeros(len(keep), dtype=torch.long, device=device)   # all target g* (local 0)

    # (3) determinism
    dna, mb = rows_of(keep)
    p1 = probs_sup({'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': mb}, 'omics_graphs': {}}, use_ac=False)
    p2 = probs_sup({'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': mb}, 'omics_graphs': {}}, use_ac=False)
    d_det = (p1 - p2).abs().max().item()
    print(f'[3] determinism  max|dP| = {d_det:.2e}', flush=True)

    # (1) add-then-remove == baseline: insert modal g*, then flip its mask back to 0 -> baseline
    #     (proves a written-but-masked slot is inert; content never leaks through the pad mask).
    dna_add, m_add, f_idx = insert_modal(dna, mb, gl0, modal_dev)
    m_ar = m_add.clone(); rows = torch.arange(len(keep), device=device)
    m_ar[rows, f_idx] = 0                                    # remove the inserted slot again
    p_base = probs_sup({'omics_inputs': {'dna': dna},     'omics_masks': {'dna': mb},   'omics_graphs': {}}, use_ac=False)
    p_ar   = probs_sup({'omics_inputs': {'dna': dna_add}, 'omics_masks': {'dna': m_ar}, 'omics_graphs': {}}, use_ac=False)
    d_ar = (p_ar - p_base).abs().max().item()
    print(f'[1] add-then-remove == baseline  max|dP| = {d_ar:.2e}', flush=True)

    # present-gene sanity: adding g* DOES move predictions
    p_add = probs_sup({'omics_inputs': {'dna': dna_add}, 'omics_masks': {'dna': m_add}, 'omics_graphs': {}}, use_ac=False)
    d_eff = (p_add - p_base).abs().max().item()
    print(f'    add-effect  max|dP| = {d_eff:.2e} (expect >0)', flush=True)

    # (2) free-slot position invariance: insert at first free vs a DIFFERENT free slot -> identical
    b0 = [keep[0]]
    dna1, m1 = rows_of(b0)
    free_idxs = torch.where(m1[0] == 0)[0]
    assert free_idxs.numel() >= 2, 'need >=2 free slots for position-invariance test'
    fa = free_idxs[0:1]; fb = free_idxs[free_idxs.numel() // 2:free_idxs.numel() // 2 + 1]
    da, ma, _ = insert_modal(dna1, m1, gl0[:1], modal_dev, use_free=fa)
    db, mbx, _ = insert_modal(dna1, m1, gl0[:1], modal_dev, use_free=fb)
    poola = pooled_of({'omics_inputs': {'dna': da}, 'omics_masks': {'dna': ma},  'omics_graphs': {}}, use_ac=False)
    poolb = pooled_of({'omics_inputs': {'dna': db}, 'omics_masks': {'dna': mbx}, 'omics_graphs': {}}, use_ac=False)
    d_pos = (poola - poolb).abs().max().item()
    print(f'[2] free-slot position invariance  max|pooled diff| = {d_pos:.2e}', flush=True)

    # (4) mask-insert == physical-append (set-encoder equivalence, mirror of KO mask==prune)
    def append_pooled(b_idx):
        idx = torch.where(mask[b_idx] > 0)[0]; nk = idx.numel(); Ln = mask.shape[1]
        dna_a = {}
        for c in COMP_KEYS:
            row = inp[c][b_idx].index_select(0, idx)                 # [nk,*feat]
            add = modal_dev[c][0].unsqueeze(0)                       # [1,*feat]
            pad = torch.zeros((Ln - nk - 1,) + tuple(row.shape[1:]), dtype=row.dtype, device=device)
            dna_a[c] = torch.cat([row, add, pad], 0).unsqueeze(0)
        m = torch.cat([torch.ones(nk + 1, device=device), torch.zeros(Ln - nk - 1, device=device)]).unsqueeze(0)
        return pooled_of({'omics_inputs': {'dna': dna_a}, 'omics_masks': {'dna': m}, 'omics_graphs': {}}, use_ac=False)
    worst = 0.0
    for bi in keep[:4]:
        d1, m1b = rows_of([bi])
        da, ma, _ = insert_modal(d1, m1b, gl0[:1], modal_dev)
        pooled_ins = pooled_of({'omics_inputs': {'dna': da}, 'omics_masks': {'dna': ma}, 'omics_graphs': {}}, use_ac=False)
        pooled_app = append_pooled(bi)
        worst = max(worst, (pooled_ins - pooled_app).abs().max().item())
    print(f'[4] mask-insert == physical-append  max|pooled diff| over 4 = {worst:.2e}', flush=True)

    ok = (d_det < 1e-5 and d_ar < 1e-5 and d_eff > 1e-4 and d_pos < 5e-3 and worst < 5e-3)
    print(f'\nAUDIT {"PASSED" if ok else "FAILED"}  '
          f'(det<1e-5:{d_det<1e-5} add-then-remove0:{d_ar<1e-5} effect>0:{d_eff>1e-4} '
          f'pos-invariant:{d_pos<5e-3} insert==append:{worst<5e-3})', flush=True)
    sys.exit(0 if ok else 2)

# =============================================================================== FULL
print('\n===== FULL: pass 1 (census + modal-token signatures) =====', flush=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
carrier_count, sig_count, _, n_recip_pool = census_and_modal(loader, args.max_batches, capture_all=False)

selected = np.where(carrier_count >= MIN_CARRIERS)[0]
selected = selected[np.argsort(-carrier_count[selected])]
sel_names = [i2g[g] for g in selected]
G = len(selected)
gid2local = -np.ones(len(g2i), dtype=np.int64)
gid2local[selected] = np.arange(G)
recip_count = np.clip(n_recip_pool - carrier_count[selected], 1, None)     # ~recipients per gene
planned = int(np.minimum(recip_count, RECIP_CAP).sum())
capped = [(i2g[g], int(recip_count[k])) for k, g in enumerate(selected) if recip_count[k] > RECIP_CAP]
print(f'pass1: recipient pool={n_recip_pool}  selected genes(>={MIN_CARRIERS} carriers)={G}  '
      f'planned additions~={planned}  capped(>{RECIP_CAP}):{len(capped)}', flush=True)

# modal signature per selected gene, then capture its example components (pass C)
want = {}
modal_frac = np.zeros(G)
for k, g in enumerate(selected):
    msig, mcnt = modal_sig_of(sig_count, int(g))
    want[msig] = k                          # sig -> local idx
    modal_frac[k] = mcnt / max(1, sum(sig_count[int(g)].values()))
print(f'pass1: modal mutation fraction  median={np.median(modal_frac):.2f} '
      f'min={modal_frac.min():.2f} max={modal_frac.max():.2f}', flush=True)
print('\n===== FULL: pass C (capture modal token components) =====', flush=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
examples = capture_examples(loader, want)
assert len(examples) == G, f'captured {len(examples)}/{G} modal tokens'
modal_dev = stack_modal(examples, G, fp32=False)
print(f'captured {len(examples)} modal tokens', flush=True)

# accumulators
acc_sum   = torch.zeros(G, n_sup, dtype=torch.float64, device=device)
acc_sumsq = torch.zeros(G, n_sup, dtype=torch.float64, device=device)
acc_cnt   = torch.zeros(G, dtype=torch.float64, device=device)
acc_tv    = torch.zeros(G, dtype=torch.float64, device=device)
acc_kl    = torch.zeros(G, dtype=torch.float64, device=device)
acc_flip  = torch.zeros(G, dtype=torch.float64, device=device)

print('\n===== FULL: pass 2 (baseline + knock-in) =====', flush=True)
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

    row_idx, gene_loc = [], []
    for r in np.where(dx)[0]:
        valid = (gf_np[r] >= REAL_MIN) & (mk_np[r] > 0)
        present = set(np.unique(gf_np[r][valid]).tolist())
        if len(present) < 1 or not (mk_np[r] == 0).any():     # need a real gene + a free slot
            continue
        for k, g in enumerate(selected):                       # every selected gene NOT present
            if int(g) in present:
                continue
            if not keep_pair(sids[r], i2g[int(g)], int(recip_count[k]), RECIP_CAP):
                continue
            row_idx.append(int(r)); gene_loc.append(int(k))
    if not row_idx:
        if args.max_batches and bi + 1 >= args.max_batches: break
        continue

    gl_all = torch.as_tensor(gene_loc, device=device)
    for s in range(0, len(row_idx), CHUNK):
        ri = row_idx[s:s + CHUNK]; gl = gl_all[s:s + CHUNK]
        ri_t = torch.as_tensor(ri, device=device)
        dna = {c: inp[c].index_select(0, ri_t) for c in COMP_KEYS}
        m_base = mask.index_select(0, ri_t)
        dna_add, m_add, _ = insert_modal(dna, m_base, gl, modal_dev)
        p_base = probs_sup({'omics_inputs': {'dna': dna},     'omics_masks': {'dna': m_base}, 'omics_graphs': {}})
        p_add  = probs_sup({'omics_inputs': {'dna': dna_add}, 'omics_masks': {'dna': m_add},  'omics_graphs': {}})
        dP = p_add - p_base
        acc_sum.index_add_(0, gl, dP.double())
        acc_sumsq.index_add_(0, gl, (dP * dP).double())
        acc_cnt.index_add_(0, gl, torch.ones(len(ri), dtype=torch.float64, device=device))
        acc_tv.index_add_(0, gl, (0.5 * dP.abs().sum(1)).double())
        kl = (p_base * ((p_base + 1e-9).log() - (p_add + 1e-9).log())).sum(1)
        acc_kl.index_add_(0, gl, kl.double())
        acc_flip.index_add_(0, gl, (p_add.argmax(1) != p_base.argmax(1)).double())
        n_pairs += len(ri)
    if args.max_batches and bi + 1 >= args.max_batches:
        break
    if (bi + 1) % 100 == 0:
        print(f'  pass2 batch {bi+1}/{len(loader)}  additions={n_pairs}', flush=True)

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

from scipy import stats
with np.errstate(divide='ignore', invalid='ignore'):
    tval = np.where(se > 0, meandP / se, 0.0)
dfree = np.clip(cnt[:, None] - 1, 1, None)
pval = 2 * stats.t.sf(np.abs(tval), dfree)
flat = pval.ravel(); order = np.argsort(flat); m = flat.size
bh = np.empty(m); ranked = flat[order]
bh_sorted = np.minimum.accumulate((ranked * m / (np.arange(m) + 1))[::-1])[::-1]
bh[order] = np.clip(bh_sorted, 0, 1)
qval = bh.reshape(pval.shape)

np.savez_compressed(
    f'{KI_DIR}/ki_results.npz',
    genes=np.array(sel_names), gene_ids=selected, n_used=n_used, carrier_count=carrier_count[selected],
    recip_count=recip_count, modal_frac=modal_frac,
    sup_classes=sup_classes, sup_names=np.array(sup_names),
    meandP=meandP, sd=sd, se=se, tval=tval, pval=pval, qval=qval,
    influence_tv=influence_tv, influence_kl=influence_kl, flip_rate=flip_rate,
)
meta = dict(mode='full', analysis='knock-in', ckpt=os.path.basename(CKPT),
            min_carriers=MIN_CARRIERS, recip_cap=RECIP_CAP, n_recip_pool=int(n_recip_pool),
            n_genes=int(G), n_pairs=int(n_pairs), n_supervised=int(n_sup),
            modal_frac_median=float(np.median(modal_frac)), n_capped=len(capped))
json.dump(meta, open(f'{KI_DIR}/ki_meta.json', 'w'), indent=2)
print(f'\nDONE  genes={G}  additions={n_pairs}  -> {KI_DIR}/ki_results.npz', flush=True)
print('top-12 genes by mean TV influence (knock-in):', flush=True)
for j in np.argsort(-influence_tv)[:12]:
    dd = meandP[j]
    up = sup_names[int(np.argmax(dd))]; dn = sup_names[int(np.argmin(dd))]
    print(f'  {sel_names[j]:8s} n={n_used[j]:6d} TV={influence_tv[j]:.4f} flip={flip_rate[j]:.3f} '
          f'| most-RAISED: {up} ({dd.max():+.3f})  | most-lowered: {dn} ({dd.min():+.3f})', flush=True)
