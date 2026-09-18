#!/usr/bin/env python3
"""EVO  Audit of the hypothesis: 'cancer lineage identity is established EARLY, by truncal
(high-VAF/clonal) mutations'.

VAF as a molecular clock: within one bulk tumour, high-VAF mutations are clonal/truncal (early),
low-VAF are subclonal (late). We take REAL DX1/DX2 tumours of a known lineage L and reveal their
OWN mutations to the disease head one at a time, in three orders:
  - clonal    : high-VAF first (early->late)      <- the hypothesis's evolutionary order
  - subclonal : low-VAF first  (late->early)      <- reverse control
  - random    : shuffled                          <- null control
At each prefix we read P(L). Hiding a mutation == setting its mask to 0 (the audited KO mechanism);
only the MASK varies across prefixes, the tumour's tokens are fixed, so this is confound-free and
cheap. The set-encoder is order-blind, so the FULL-set P is identical across orders; any difference
in the TRAJECTORY tells us which mutations (clonal vs subclonal) carry the identity signal.

Hypothesis SUPPORTED if the clonal trajectory rises faster than subclonal/random and a large
fraction of the final P(L) is reached within the first 1-2 clonal mutations.

Writes run8_disease_out/evo/evo_results.npz (+ evo_meta.json).
    python insilico_evo.py [--dry] [--n-per 120] [--k-max 10] [--n-targets 15]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
EV_DIR   = f'{OUT_DIR}/evo'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CKPT     = f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'
CHUNK = 512
SEED  = 42
REAL_MIN = 4

ap = argparse.ArgumentParser()
ap.add_argument('--dry', action='store_true', help='tiny: 4 targets, 20 tumours, 6 steps, 60 scan batches')
ap.add_argument('--n-per', type=int, default=120, help='real tumours sampled per lineage')
ap.add_argument('--k-max', type=int, default=10, help='max mutations revealed along the trajectory')
ap.add_argument('--n-targets', type=int, default=15)
ap.add_argument('--min-mut', type=int, default=3, help='min real mutations for a tumour to qualify')
ap.add_argument('--scan-batches', type=int, default=800)
args = ap.parse_args()
if args.dry:
    args.n_targets, args.n_per, args.k_max, args.scan_batches = 4, 20, 6, 60
N_PER, K_MAX, N_TARGETS = args.n_per, args.k_max, args.n_targets

sys.path.insert(0, DATA_DIR)
import run8_disease_config
run8_disease_config.use_moco_oncoformer()
config = run8_disease_config.build_config(CACHE)

import torch, lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics, OncoformerPost
from torch.utils.data import DataLoader
assert 'Oncoformer_moco' in oncoformer.models.__file__

os.makedirs(EV_DIR, exist_ok=True)
device = torch.device('cuda')
L.seed_everything(SEED, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
vaf2i = tok.component_token2idx['aa_vaf_bin']
VAF_IDS = {vaf2i[str(k)]: k for k in range(1, 11)}     # token-id -> human VAF level 1..10

levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
IGNORE = ['other', 'nos']
ignored_ids = np.array([i for i, l in enumerate(levels) if any(p in l.lower() for p in IGNORE)])
sup_classes = np.array([i for i in range(len(levels)) if i not in set(ignored_ids.tolist())])
sup_names = [levels[i] for i in sup_classes]
n_sup = len(sup_classes)
sup_name2col = {s: i for i, s in enumerate(sup_names)}
print(f'classes: total={len(levels)} supervised={n_sup}', flush=True)

backbone = OncoformerOmics(config, dl.dataset.tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')
model = OncoformerPost(backbone, config, checkpoint_dir=f'{OUT_DIR}/checkpoints')
assert 'disease_term' in model.prediction_heads and not getattr(model, 'use_confounder', False)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(ck.get('state_dict', ck), strict=False)
assert len(missing) == 0 and len(unexpected) == 0
model.to(device).eval()
print(f'loaded {os.path.basename(CKPT)} epoch={ck.get("epoch")}', flush=True)

_negmask = torch.full((len(levels),), 0.0, device=device)
_negmask[torch.as_tensor(ignored_ids, device=device)] = float('-inf')
_sup_idx = torch.as_tensor(sup_classes, device=device)
COMP_KEYS = None


def to_dev(x):
    if torch.is_tensor(x): return x.to(device, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    return x


@torch.no_grad()
def probs_sup(dna, m):
    batch = {'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m}, 'omics_graphs': {}}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        logits = model(batch)['disease_term'].float()
    z = logits + _negmask
    z = z - z.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    return (e / e.sum(dim=1, keepdim=True)).index_select(1, _sup_idx)


@torch.no_grad()
def probs_sup_chunked(dna, m):
    B = m.shape[0]
    out = torch.empty((B, n_sup), device=device)
    for s in range(0, B, CHUNK):
        sl = slice(s, min(B, s + CHUNK))
        out[sl] = probs_sup({c: dna[c][sl] for c in COMP_KEYS}, m[sl])
    return out


# ---------- choose target lineages (frequent, well-defined) ----------
feat = np.load(f'{OUT_DIR}/umap/umap_features.npz', allow_pickle=True)
tt_all = feat['true_term'].astype(str); sid_all = feat['sample_id'].astype(str)
sid2true = dict(zip(sid_all, tt_all))
freq = Counter(tt_all)
def is_defined(s):
    s = s.lower()
    return not (('unknown primary' in s) or ('undifferentiated' in s) or ('(cup)' in s))
targets = [l for l, _ in freq.most_common() if l in sup_name2col and is_defined(l)][:N_TARGETS]
tgt_set = set(targets)
print(f'targets: {len(targets)} lineages', flush=True)

# ---------- gather N real tumours per lineage ----------
print('\n===== gather real tumours per lineage =====', flush=True)
rng = np.random.default_rng(SEED)
pool = {t: [] for t in targets}
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
for bi, batch in enumerate(loader):
    dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
    if COMP_KEYS is None:
        COMP_KEYS = list(dna.keys())
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    sids = md.index.astype(str).tolist()
    gf = dna['gene'].numpy(); mk = mask.numpy(); vf = dna['aa_vaf'].numpy()   # CONTINUOUS VAF
    for r in np.where(dx)[0]:
        tr = sid2true.get(sids[r])
        if tr not in tgt_set or len(pool[tr]) >= N_PER:
            continue
        real = np.where((gf[r] >= REAL_MIN) & (mk[r] > 0))[0]
        if real.size < args.min_mut:
            continue
        # VAF key per real slot: continuous aa_vaf in (0,1]; CN/missing (vaf<=0) -> random (unbiased)
        vkey = vf[r][real].astype(float)
        miss = ~(vkey > 0)
        vkey[miss] = rng.random(int(miss.sum()))
        rec = {c: dna[c][r].clone().numpy() for c in COMP_KEYS}
        rec['__mask__'] = mk[r].copy(); rec['__real__'] = real; rec['__vkey__'] = vkey
        rec['__nmiss__'] = int(miss.sum())
        pool[tr].append(rec)
    got = min(len(pool[t]) for t in targets)
    if got >= N_PER or bi + 1 >= args.scan_batches:
        break
    if (bi + 1) % 50 == 0:
        print(f'  batch {bi+1}: min per-lineage={got}/{N_PER}', flush=True)
for t in targets:
    print(f'  {t[:40]:40s} n={len(pool[t])}', flush=True)
targets = [t for t in targets if len(pool[t]) >= max(10, N_PER // 4)]
assert targets, 'no lineage reached enough tumours'


def order_slots(rec, mode):
    """return real-slot indices ordered: clonal=high VAF first, subclonal=low first, random."""
    real, vkey = rec['__real__'], rec['__vkey__']
    jitter = rng.random(real.size) * 1e-6                      # break ties randomly
    if mode == 'clonal':
        idx = np.argsort(-(vkey + jitter))
    elif mode == 'subclonal':
        idx = np.argsort(vkey + jitter)
    else:
        idx = rng.permutation(real.size)
    return real[idx]


ORDERS = ['clonal', 'subclonal', 'random']


@torch.no_grad()
def trajectories(recs, col):
    """P(L) as k mutations are revealed, for each order. Only the MASK changes per prefix.
    Returns dict order -> [n, K_MAX+1] and full-set P [n]."""
    n = len(recs)
    Lmax = max(r['__mask__'].shape[0] for r in recs)
    dna = {}
    for c in COMP_KEYS:
        ref = recs[0][c]
        dna[c] = torch.zeros((n, Lmax) + tuple(ref.shape[1:]), dtype=torch.as_tensor(ref).dtype)
    base_spec = torch.zeros((n, Lmax))                          # mask with ALL real slots hidden
    full_mask = torch.zeros((n, Lmax))
    ordered = {o: [] for o in ORDERS}
    for i, r in enumerate(recs):
        Lh = r['__mask__'].shape[0]
        for c in COMP_KEYS:
            dna[c][i, :Lh] = torch.as_tensor(r[c])
        m = r['__mask__'].astype(np.float32)
        full_mask[i, :Lh] = torch.as_tensor(m)
        spec = m.copy(); spec[r['__real__']] = 0.0             # keep specials, hide real muts
        base_spec[i, :Lh] = torch.as_tensor(spec)
        for o in ORDERS:
            ordered[o].append(order_slots(r, o))
    dna = {c: v.to(device) for c, v in dna.items()}
    base_spec = base_spec.to(device); full_mask = full_mask.to(device)
    out = {}
    for o in ORDERS:
        P = torch.empty((n, K_MAX + 1), device=device)
        for k in range(K_MAX + 1):
            m = base_spec.clone()
            for i in range(n):
                seq = ordered[o][i]
                rev = seq[:min(k, seq.size)]
                if rev.size:
                    m[i, torch.as_tensor(rev, dtype=torch.long, device=device)] = 1.0
            P[:, k] = probs_sup_chunked(dna, m)[:, col]
        out[o] = P.cpu().numpy()
    full_P = probs_sup_chunked(dna, full_mask)[:, col].cpu().numpy()
    return out, full_P


# =============================================================================== run
print('\n===== reveal trajectories per lineage =====', flush=True)
res = {}
for ti, t in enumerate(targets):
    col = sup_name2col[t]
    traj, full_P = trajectories(pool[t], col)
    res[t] = dict(clonal=traj['clonal'], subclonal=traj['subclonal'], random=traj['random'],
                  full=full_P, n=len(pool[t]),
                  nmiss=float(np.mean([r['__nmiss__'] / max(1, r['__real__'].size) for r in pool[t]])),
                  nmut=float(np.median([r['__real__'].size for r in pool[t]])))
    cl = np.median(traj['clonal'], 0); sb = np.median(traj['subclonal'], 0)
    fp = np.median(full_P)
    # fraction of final identity reached by the first clonal mutation, and gap at k=2
    frac1 = (cl[1] / fp) if fp > 0 else np.nan
    print(f'  [{ti+1}/{len(targets)}] {t[:34]:34s} fullP={fp:.2f} clonal@1={cl[1]:.2f}({frac1*100:.0f}%) '
          f'clonal@2={cl[2]:.2f} subclonal@2={sb[2]:.2f} gap@2={cl[2]-sb[2]:+.2f}', flush=True)

# ---------- overall verdict (pooled across all tumours, aligned to k) ----------
allc = np.concatenate([res[t]['clonal'] for t in targets], 0)
alls = np.concatenate([res[t]['subclonal'] for t in targets], 0)
allr = np.concatenate([res[t]['random'] for t in targets], 0)
allf = np.concatenate([res[t]['full'] for t in targets])
# normalise each tumour by its own full P to compare "fraction of identity reached"
norm = np.clip(allf, 1e-6, None)[:, None]
fc, fs, fr = allc / norm, alls / norm, allr / norm
print('\n===== VERDICT (pooled, fraction of final identity reached) =====', flush=True)
for k in range(1, min(5, K_MAX) + 1):
    print(f'  k={k}: clonal={np.median(fc[:,k]):.2f}  random={np.median(fr[:,k]):.2f}  '
          f'subclonal={np.median(fs[:,k]):.2f}', flush=True)

np.savez_compressed(
    f'{EV_DIR}/evo_results.npz',
    targets=np.array(targets),
    clonal=np.array([res[t]['clonal'] for t in targets], dtype=object),
    subclonal=np.array([res[t]['subclonal'] for t in targets], dtype=object),
    random=np.array([res[t]['random'] for t in targets], dtype=object),
    full=np.array([res[t]['full'] for t in targets], dtype=object),
    n=np.array([res[t]['n'] for t in targets]),
    nmut=np.array([res[t]['nmut'] for t in targets]),
    nmiss=np.array([res[t]['nmiss'] for t in targets]),
    k_max=K_MAX, sup_names=np.array(sup_names),
)
json.dump(dict(analysis='evo-identity-lockin', ckpt=os.path.basename(CKPT), n_targets=len(targets),
               n_per=N_PER, k_max=K_MAX, dry=bool(args.dry)),
          open(f'{EV_DIR}/evo_meta.json', 'w'), indent=2)
print(f'\nDONE -> {EV_DIR}/evo_results.npz  (lineages={len(targets)})', flush=True)
