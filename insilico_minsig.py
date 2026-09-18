#!/usr/bin/env python3
"""#4  Minimal sufficient signature - the shortest genomic 'recipe' the disease head needs to
call each cancer lineage, built by GREEDY knock-in over a POPULATION of realistic backgrounds.

Why a population (not one blank patient): the disease head reads ONLY the DNA mutation set
(DNA-only backbone, CLS pooling, confounder off). A single blank profile gives one deterministic,
out-of-distribution trajectory; and STRIPPING drivers from real DX1/DX2 hosts leaves ~nothing
(the panel detects almost only driver-panel genes) so the backgrounds collapse to identical
blanks. Instead we sample N DIVERSE REAL full-genome tumors (reservoir sampling across the
cohort). Greedy adds one driver at a time to ALL N backgrounds; each step's pick = the gene with
the highest MEAN dP(target) across the N heterogeneous backgrounds. The recipe is thus the
minimal driver set that makes the head call lineage L regardless of the tumour's original
context, and we get per step a population band (P spread), an agreement % (fraction of
backgrounds that individually pick the same gene), a branching distribution, and a convergence
check (recipe stability vs N).

Insertion == genuine presence is the same free-slot mechanism proven numerically by
insilico_ki_disease.py --mode audit (positionless set-encoder; written-but-masked slot inert).

Stop rule (annotation; we still run MAX_STEPS): first step where the target is the argmax for
>=50% of backgrounds AND median P(target) >= 0.8 x the real-patient median P(target).

Outputs run8_disease_out/minsig/minsig_results.npz (+ minsig_meta.json).
    python insilico_minsig.py [--dry] [--n-bg 200] [--n-targets 35] [--k-cand 25] [--max-steps 8]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
MS_DIR   = f'{OUT_DIR}/minsig'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CKPT     = f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'
MIN_CARRIERS = 100
CHUNK = 512
SEED  = 42

ap = argparse.ArgumentParser()
ap.add_argument('--dry', action='store_true', help='tiny run: 40 bg, 3 targets, 10 cand, 5 steps, no convergence')
ap.add_argument('--n-bg', type=int, default=200)
ap.add_argument('--n-targets', type=int, default=35)
ap.add_argument('--k-cand', type=int, default=25)
ap.add_argument('--max-steps', type=int, default=8)
ap.add_argument('--ref-min', type=int, default=150, help='min real-patient samples per target for ref median')
ap.add_argument('--scan-batches', type=int, default=500, help='max batches to scan for ref + backgrounds')
ap.add_argument('--candidates', choices=['suff', 'nec', 'all'], default='suff',
                help='candidate pool per lineage: suff=top-K KI sufficiency; nec=top-K KO necessity; '
                     'all=every driver gene (sensitivity check for the top-K restriction; ignores --k-cand)')
ap.add_argument('--targets-only', type=str, default='',
                help='comma-separated EXACT lineage names to restrict targets to (bounds compute for '
                     'the all-genes sensitivity run); overrides --n-targets')
args = ap.parse_args()
SUFFIX = '' if args.candidates == 'suff' else f'_{args.candidates}'
if args.dry:
    args.n_bg, args.n_targets, args.k_cand, args.max_steps, args.scan_batches = 40, 3, 10, 5, 60
N_BG, N_TARGETS, K_CAND, MAX_STEPS = args.n_bg, args.n_targets, args.k_cand, args.max_steps

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

os.makedirs(MS_DIR, exist_ok=True)
device = torch.device('cuda')
L.seed_everything(SEED, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)
tokenizers = dl.dataset.tokenizers

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
g2i = tok.component_token2idx['gene']; i2g = {v: k for k, v in g2i.items()}
SPECIALS = set(tok.special_tokens); REAL_MIN = 4

levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
IGNORE = ['other', 'nos']
ignored_ids = np.array([i for i, l in enumerate(levels) if any(p in l.lower() for p in IGNORE)])
sup_classes = np.array([i for i in range(len(levels)) if i not in set(ignored_ids.tolist())])
sup_names = [levels[i] for i in sup_classes]
n_sup = len(sup_classes)
sup_name2col = {s: i for i, s in enumerate(sup_names)}
print(f'classes: total={len(levels)} supervised={n_sup}', flush=True)

backbone = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')
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


def to_dev(x):
    if torch.is_tensor(x): return x.to(device, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_dev(v) for v in x)
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


@torch.no_grad()
def median_ci95(v, B=2000):
    """Bootstrap 95% CI of the MEDIAN of v (a [Nb] tensor). Shrinks with Nb (~1/sqrt(N))."""
    n = v.numel()
    idx = torch.randint(0, n, (B, n), device=device)
    meds = v[idx].median(dim=1).values
    return float(torch.quantile(meds, 0.025).item()), float(torch.quantile(meds, 0.975).item())


# ------- modal token machinery (copied from insilico_ki_disease.py; same audited insertion)
COMP_KEYS = None
def set_comp_keys(dna):
    global COMP_KEYS
    if COMP_KEYS is None:
        COMP_KEYS = list(dna.keys()); assert 'gene' in COMP_KEYS


def slot_sig(dna_np, r, s):
    return hash(b'|'.join(dna_np[c][r, s].tobytes() for c in COMP_KEYS))


def census_and_modal(loader, max_batches=0):
    carrier_count = np.zeros(len(g2i), dtype=np.int64)
    sig_count = {}
    for bi, batch in enumerate(loader):
        dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
        set_comp_keys(dna)
        dna_np = {c: dna[c].numpy() for c in COMP_KEYS}
        mk = mask.numpy(); gf = dna_np['gene']
        md = batch['sample_metadata']['sample_metadata']
        dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
        for r in np.where(dx)[0]:
            valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
            genes = np.unique(gf[r][valid])
            if genes.size < 1:
                continue
            carrier_count[genes] += 1
            for s in np.where(valid)[0]:
                g = int(gf[r, s]); sig = slot_sig(dna_np, r, s)
                d = sig_count.setdefault(g, {}); d[sig] = d.get(sig, 0) + 1
        if max_batches and bi + 1 >= max_batches:
            break
        if (bi + 1) % 200 == 0:
            print(f'  census batch {bi+1}/{len(loader)}', flush=True)
    return carrier_count, sig_count


def capture_examples(loader, want, max_batches=0):
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
    return got


def stack_modal(examples_by_local, G):
    modal_dev = {}
    for c in COMP_KEYS:
        ref = next(iter(examples_by_local.values()))[c]
        buf = torch.zeros((G,) + tuple(ref.shape), dtype=ref.dtype)
        for lc, ex in examples_by_local.items():
            buf[lc] = ex[c]
        modal_dev[c] = buf.to(device)
    return modal_dev


def first_free(m_rows):
    return (m_rows == 0).float().argmax(1)


def insert_modal(dna, m_base, gl_t):
    """clone dna, write each row's modal token (local idx gl_t) into its first free slot, mask->1."""
    K = m_base.shape[0]
    f_idx = first_free(m_base)
    rows = torch.arange(K, device=device)
    dna_add = {c: dna[c].clone() for c in COMP_KEYS}
    m_add = m_base.clone()
    for c in COMP_KEYS:
        dna_add[c][rows, f_idx] = modal_dev[c].index_select(0, gl_t)
    m_add[rows, f_idx] = 1
    return dna_add, m_add


# =============================================================================== pass 1+C
print('\n===== pass 1: census + modal signatures =====', flush=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
carrier_count, sig_count = census_and_modal(loader, max_batches=(args.scan_batches if args.dry else 0))
selected = np.where(carrier_count >= MIN_CARRIERS)[0]
selected = selected[np.argsort(-carrier_count[selected])]
sel_names = [i2g[g] for g in selected]
G = len(selected)
gid2local = -np.ones(len(g2i), dtype=np.int64); gid2local[selected] = np.arange(G)
name2local = {n: k for k, n in enumerate(sel_names)}
print(f'selected driver genes (>= {MIN_CARRIERS} carriers): {G}', flush=True)

want = {}
for k, g in enumerate(selected):
    msig = max(sig_count[int(g)].items(), key=lambda kv: kv[1])[0]
    want[msig] = k
print('\n===== pass C: capture modal tokens =====', flush=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
examples = capture_examples(loader, want, max_batches=(args.scan_batches if args.dry else 0))
assert len(examples) == G, f'captured {len(examples)}/{G}'
modal_dev = stack_modal(examples, G)
print(f'captured {G} modal tokens; comp keys={COMP_KEYS}', flush=True)

# ------- target lineages + candidate pools (from the already-computed KI/KO matrices)
ki = np.load(f'{OUT_DIR}/ki/ki_results.npz', allow_pickle=True)
ko = np.load(f'{OUT_DIR}/ko/ko_results.npz', allow_pickle=True)
assert (ki['sup_names'].astype(str) == np.array(sup_names)).all()
assert (ko['sup_names'].astype(str) == np.array(sup_names)).all()
ki_genes = ki['genes'].astype(str); ki_KI = ki['meandP'].astype(float)
ko_genes = ko['genes'].astype(str); ko_KO = ko['meandP'].astype(float)

feat = np.load(f'{OUT_DIR}/umap/umap_features.npz', allow_pickle=True)
tt_all = feat['true_term'].astype(str); sid_all = feat['sample_id'].astype(str)
sid2true = dict(zip(sid_all, tt_all))
from collections import Counter
freq = Counter(tt_all)
def is_defined(s):
    s = s.lower()
    return not (('unknown primary' in s) or ('undifferentiated' in s) or ('(cup)' in s))
targets = [l for l, _ in freq.most_common() if l in sup_name2col and is_defined(l)][:N_TARGETS]
if args.targets_only:
    want_t = [s.strip() for s in args.targets_only.split(',') if s.strip()]
    for w in want_t:
        assert w in sup_name2col, f'--targets-only {w!r} not in supervised names'
    targets = want_t
tgt_cols = np.array([sup_name2col[t] for t in targets])
print(f'targets: {len(targets)} lineages', flush=True)

# candidate pool per target: top-K genes, ranked by KI sufficiency (suff) or KO necessity (nec)
# suff: most POSITIVE add-effect (argsort -KI).   nec: most NEGATIVE knockout-effect (argsort +KO).
if args.candidates == 'suff':
    src_genes, src_score = ki_genes, -ki_KI
elif args.candidates == 'nec':
    src_genes, src_score = ko_genes, ko_KO
else:                                               # 'all' -> every driver gene, no top-K screen
    src_genes = src_score = None
if args.candidates == 'all':
    print(f'candidate pool = ALL {G} driver genes per lineage (sensitivity check)', flush=True)
else:
    print(f'candidate pool = top-{K_CAND} {args.candidates} genes per lineage', flush=True)
cand_local = {}
for t in targets:
    if args.candidates == 'all':
        cand_local[t] = np.arange(G, dtype=np.int64)
        continue
    col = sup_name2col[t]
    order = np.argsort(src_score[:, col])          # ascending -> best candidates first
    locs = []
    for r in order:
        gname = src_genes[r]
        if gname in name2local and name2local[gname] not in locs:
            locs.append(name2local[gname])
        if len(locs) >= K_CAND:
            break
    cand_local[t] = np.array(locs, dtype=np.int64)

# =============================================================================== pass R
# reference median P(target) among REAL patients truly of that lineage + gather N stripped hosts
print('\n===== pass R: reference medians + background hosts =====', flush=True)
sel_arr = np.array(sorted(sel_set := set(int(x) for x in selected)))
ref_vals = {t: [] for t in targets}
hosts = []            # reservoir of dict{comp: np[L,*], '__mask__': np[L]}
n_seen_host = 0
rng_host = np.random.default_rng(SEED)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
tgt_set = set(targets)
for bi, batch in enumerate(loader):
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    if not dx.any():
        if bi + 1 >= args.scan_batches:
            break
        continue
    dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
    gf = dna['gene'].numpy(); mk = mask.numpy(); sids = md.index.astype(str).tolist()
    # reference: forward this batch once, read P at each target column for matching-true patients
    need_ref = any(len(ref_vals[t]) < args.ref_min for t in targets)
    if need_ref:
        P = probs_sup_chunked(to_dev(dna), to_dev(mask)).cpu().numpy()
        for r in np.where(dx)[0]:
            tr = sid2true.get(sids[r])
            if tr in tgt_set and len(ref_vals[tr]) < args.ref_min:
                ref_vals[tr].append(float(P[r, sup_name2col[tr]]))
    # backgrounds: DIVERSE REAL full-genome tumors (NOT stripped -- DX panels detect almost only
    # driver-panel genes, so stripping leaves ~empty/identical blanks and the population collapses).
    # Reservoir-sample across the whole scan so the ensemble is diverse and order-unbiased; the
    # recipe is then the minimal driver set that makes the head call L across heterogeneous tumors.
    for r in np.where(dx)[0]:
        mr = mk[r].copy()
        if (mr == 0).sum() < MAX_STEPS + 1:              # need free slots to insert genes
            continue
        n_seen_host += 1
        j = len(hosts) if len(hosts) < N_BG else int(rng_host.integers(0, n_seen_host))
        if j < N_BG:
            h = {c: dna[c][r].clone().numpy() for c in COMP_KEYS}; h['__mask__'] = mr
            if j == len(hosts):
                hosts.append(h)
            else:
                hosts[j] = h
    if bi + 1 >= args.scan_batches:
        break
    if (bi + 1) % 50 == 0:
        got = min(len(ref_vals[t]) for t in targets)
        print(f'  passR batch {bi+1}: reservoir={len(hosts)}/{N_BG} seen={n_seen_host} '
              f'min-ref={got}/{args.ref_min}', flush=True)

assert len(hosts) >= min(N_BG, 10), f'only gathered {len(hosts)} hosts'
_tokc = np.array([(h['__mask__'] > 0).sum() for h in hosts])
print(f'background real-mutation load: min/med/max = {_tokc.min()}/{int(np.median(_tokc))}/{_tokc.max()} '
      f'(over {len(hosts)} diverse real hosts, {n_seen_host} seen)', flush=True)
N_BG = len(hosts)
ref_median = np.array([np.median(ref_vals[t]) if ref_vals[t] else np.nan for t in targets])
print(f'gathered {N_BG} backgrounds; ref median P(target): '
      f'min={np.nanmin(ref_median):.2f} med={np.nanmedian(ref_median):.2f} max={np.nanmax(ref_median):.2f}', flush=True)

# pad hosts to common L and build device tensors
Lmax = max(h['__mask__'].shape[0] for h in hosts)
bg_mask = torch.zeros((N_BG, Lmax))
bg_dna = {}
for c in COMP_KEYS:
    ref = hosts[0][c]
    bg_dna[c] = torch.zeros((N_BG, Lmax) + tuple(ref.shape[1:]), dtype=torch.as_tensor(ref).dtype)
for i, h in enumerate(hosts):
    Lh = h['__mask__'].shape[0]
    bg_mask[i, :Lh] = torch.as_tensor(h['__mask__'].astype(np.float32))
    for c in COMP_KEYS:
        bg_dna[c][i, :Lh] = torch.as_tensor(h[c])
bg_mask = bg_mask.to(device)
bg_dna = {c: v.to(device) for c, v in bg_dna.items()}
free_after = int((bg_mask == 0).sum(1).min().item())
print(f'Lmax={Lmax}  min free slots per background={free_after}', flush=True)
assert free_after >= MAX_STEPS + 1


# =============================================================================== greedy
def greedy(cand, col, bg_dna0, bg_mask0, max_steps):
    """One greedy build over the given backgrounds for target column `col`.
    Returns dict of per-step arrays. bg_dna0/bg_mask0 are cloned internally."""
    Nb = bg_mask0.shape[0]
    dna = {c: bg_dna0[c].clone() for c in COMP_KEYS}
    m = bg_mask0.clone()
    used = []
    rec = dict(recipe_local=[], recipe_name=[], mean_dP=[], p_q10=[], p_q50=[], p_q90=[],
               ci_lo=[], ci_hi=[], agree=[], argmax_frac=[], branch=[])
    # baseline P(target) per background (step 0, before any gene added)
    Pb = probs_sup_chunked(dna, m)
    base_L = Pb[:, col].clone()
    rec['p_base_q10'] = float(torch.quantile(base_L, 0.10).item())
    rec['p_base_q50'] = float(torch.quantile(base_L, 0.50).item())
    rec['p_base_q90'] = float(torch.quantile(base_L, 0.90).item())
    rec['ci_base_lo'], rec['ci_base_hi'] = median_ci95(base_L)
    rec['argmax_frac0'] = float((Pb.argmax(1) == col).float().mean().item())
    for step in range(max_steps):
        avail = [i for i, c in enumerate(cand.tolist()) if c not in used]
        av_t = torch.as_tensor([cand[i] for i in avail], device=device)
        Cav = len(avail)
        # score each available candidate on ALL Nb backgrounds (loop keeps peak memory ~Nb rows,
        # not Nb*Cav -- DNA slots can carry large per-slot ESM components)
        dP = torch.empty((Nb, Cav), device=device)
        for ci in range(Cav):
            gl = av_t[ci:ci + 1].expand(Nb).contiguous()
            dna_a, m_a = insert_modal(dna, m, gl)
            dP[:, ci] = probs_sup_chunked(dna_a, m_a)[:, col] - base_L
        mean_dP = dP.mean(0)
        best = int(mean_dP.argmax().item())
        gstar_local = int(av_t[best].item())
        # per-background own best (agreement / branching)
        own_best = dP.argmax(1)                                         # [Nb] index into avail
        agree = float((own_best == best).float().mean().item())
        branch = torch.bincount(own_best, minlength=Cav).float()
        branch = branch / branch.sum()
        # committed state = insert gstar into ALL backgrounds
        gstar_t = torch.full((Nb,), gstar_local, device=device, dtype=torch.long)
        dna, m = insert_modal(dna, m, gstar_t)
        Pc = probs_sup_chunked(dna, m)
        argmax_frac = float((Pc.argmax(1) == col).float().mean().item())
        pL = Pc[:, col]
        rec['recipe_local'].append(gstar_local)
        rec['recipe_name'].append(sel_names[gstar_local])
        rec['mean_dP'].append(float(mean_dP[best].item()))
        rec['p_q10'].append(float(torch.quantile(pL, 0.10).item()))
        rec['p_q50'].append(float(torch.quantile(pL, 0.50).item()))
        rec['p_q90'].append(float(torch.quantile(pL, 0.90).item()))
        _lo, _hi = median_ci95(pL)
        rec['ci_lo'].append(_lo); rec['ci_hi'].append(_hi)
        rec['agree'].append(agree)
        rec['argmax_frac'].append(argmax_frac)
        # branching: store top-4 (avail-local -> global name, frac)
        bt = torch.argsort(-branch)[:4]
        rec['branch'].append([(sel_names[int(av_t[int(i)].item())], float(branch[int(i)].item())) for i in bt])
        base_L = pL.clone()
        used.append(gstar_local)
    return rec


print('\n===== greedy: minimal sufficient signature per lineage =====', flush=True)
results = {}
for ti, t in enumerate(targets):
    rec = greedy(cand_local[t], int(tgt_cols[ti]), bg_dna, bg_mask, MAX_STEPS)
    # stop step: first where argmax_frac>=0.5 AND median P >= 0.8*ref
    thr = 0.8 * ref_median[ti]
    stop = MAX_STEPS
    for s in range(MAX_STEPS):
        if rec['argmax_frac'][s] >= 0.5 and rec['p_q50'][s] >= thr:
            stop = s + 1; break
    rec['stop_step'] = stop; rec['ref_median'] = float(ref_median[ti])
    results[t] = rec
    print(f'  [{ti+1}/{len(targets)}] {t[:44]:44s} recipe={"→".join(rec["recipe_name"][:stop])} '
          f'(stop@{stop}, P {rec["p_q50"][0]:.2f}→{rec["p_q50"][stop-1]:.2f}, ref {ref_median[ti]:.2f})', flush=True)

# ------- convergence: flagship lineages, recipe stability vs N backgrounds
conv = {}
if not args.dry:
    flagship = targets[:5]
    Ngrid = sorted(set(n for n in (25, 50, 100, 250, 500, N_BG) if n <= N_BG))
    print('\n===== convergence: recipe stability vs N =====', flush=True)
    ref_recipe = {t: results[t]['recipe_local'][:results[t]['stop_step']] for t in flagship}
    for t in flagship:
        col = sup_name2col[t]; jac = []
        for n in Ngrid:
            sub_dna = {c: bg_dna[c][:n] for c in COMP_KEYS}
            r = greedy(cand_local[t], col, sub_dna, bg_mask[:n], results[t]['stop_step'])
            a = set(r['recipe_local']); b = set(ref_recipe[t])
            jac.append(len(a & b) / max(1, len(a | b)))
        conv[t] = dict(Ngrid=Ngrid, jaccard=jac)
        print(f'  {t[:40]:40s} Jaccard vs full: {[f"{x:.2f}" for x in jac]}', flush=True)

# =============================================================================== save
np.savez_compressed(
    f'{MS_DIR}/minsig_results{SUFFIX}.npz',
    targets=np.array(targets),
    candidates=np.array(args.candidates),
    recipe_name=np.array([results[t]['recipe_name'] for t in targets], dtype=object),
    recipe_local=np.array([results[t]['recipe_local'] for t in targets], dtype=object),
    mean_dP=np.array([results[t]['mean_dP'] for t in targets]),
    p_q10=np.array([results[t]['p_q10'] for t in targets]),
    p_q50=np.array([results[t]['p_q50'] for t in targets]),
    p_q90=np.array([results[t]['p_q90'] for t in targets]),
    p_base_q10=np.array([results[t]['p_base_q10'] for t in targets]),
    p_base_q50=np.array([results[t]['p_base_q50'] for t in targets]),
    p_base_q90=np.array([results[t]['p_base_q90'] for t in targets]),
    ci_lo=np.array([results[t]['ci_lo'] for t in targets]),
    ci_hi=np.array([results[t]['ci_hi'] for t in targets]),
    ci_base_lo=np.array([results[t]['ci_base_lo'] for t in targets]),
    ci_base_hi=np.array([results[t]['ci_base_hi'] for t in targets]),
    argmax_frac0=np.array([results[t]['argmax_frac0'] for t in targets]),
    agree=np.array([results[t]['agree'] for t in targets]),
    argmax_frac=np.array([results[t]['argmax_frac'] for t in targets]),
    branch=np.array([results[t]['branch'] for t in targets], dtype=object),
    stop_step=np.array([results[t]['stop_step'] for t in targets]),
    ref_median=ref_median,
    conv_targets=np.array(list(conv.keys()), dtype=object),
    conv_Ngrid=np.array([conv[t]['Ngrid'] for t in conv], dtype=object),
    conv_jaccard=np.array([conv[t]['jaccard'] for t in conv], dtype=object),
    sel_names=np.array(sel_names),
)
meta = dict(analysis='minimal-sufficient-signature', ckpt=os.path.basename(CKPT),
            candidates=args.candidates, n_bg=int(N_BG), n_targets=int(len(targets)),
            k_cand=int(K_CAND), max_steps=int(MAX_STEPS), n_drivers=int(G), Lmax=int(Lmax), dry=bool(args.dry))
json.dump(meta, open(f'{MS_DIR}/minsig_meta{SUFFIX}.json', 'w'), indent=2)
print(f'\nDONE ({args.candidates}) -> {MS_DIR}/minsig_results{SUFFIX}.npz  '
      f'(backgrounds={N_BG}, targets={len(targets)})', flush=True)
