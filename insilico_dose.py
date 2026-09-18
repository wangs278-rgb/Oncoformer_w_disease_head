#!/usr/bin/env python3
"""#5  Dose / VAF response probe - does the disease head read HOW STRONG a mutation is, or only
whether it is present?

Every mutation token carries, besides the gene: aa_vaf_bin (VAF in 10 ordered bins 1..10, low->high
= subclonal->clonal) and zygosity (heterozygous / homozygous). We insert a known driver into a
POPULATION of real backgrounds (as in insilico_minsig.py) but sweep ONE dose knob at a time,
holding the gene AND the specific mutation fixed:
  - VAF sweep : same gene+variant, real token at each observed VAF bin -> P(target) vs VAF.
  - Zygosity  : same gene+variant, real token het vs hom (VAF fixed at the variant's modal bin).

Design decision (auditability): we use REAL OBSERVED tokens at each level (not synthetic edits) so
every inserted token is an in-distribution combination a real patient actually had - only the dose
axis moves. The free-slot insertion == genuine presence is the same mechanism proven numerically by
insilico_ki_disease.py --mode audit.

Reads:  ki/ki_results.npz (to set each gene's target lineage = arg-max KI sufficiency, data-driven).
Writes: run8_disease_out/dose/dose_results.npz (+ dose_meta.json).
    python insilico_dose.py [--dry] [--n-bg 1000] [--min-level 5]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
DS_DIR   = f'{OUT_DIR}/dose'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CKPT     = f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'
MIN_CARRIERS = 100
CHUNK = 512
SEED  = 42

# test genes: classic oncogenes / lineage-identity TFs (dose UP -> P UP expected) and tumour
# suppressors (two-hit: hom > het expected). Each gene's TARGET lineage is set data-driven below
# (arg-max KI sufficiency), so the sweep is measured where the gene most 'builds' its lineage.
ONCO = ['KRAS', 'BRAF', 'EGFR', 'PIK3CA', 'NKX2-1', 'AR', 'PPARG', 'SOX2', 'GATA3']
TSG  = ['APC', 'PTEN', 'SMAD4', 'VHL', 'TP53']
TEST_GENES = ONCO + TSG
GENE_CLASS = {**{g: 'oncogene' for g in ONCO}, **{g: 'suppressor' for g in TSG}}

ap = argparse.ArgumentParser()
ap.add_argument('--dry', action='store_true', help='tiny: 5 genes, 40 bg, 60 scan batches')
ap.add_argument('--n-bg', type=int, default=1000)
ap.add_argument('--min-level', type=int, default=5, help='min real carriers of a (variant,level) to sweep it')
ap.add_argument('--ref-min', type=int, default=150)
ap.add_argument('--scan-batches', type=int, default=500, help='max batches to scan for ref + backgrounds')
args = ap.parse_args()
if args.dry:
    TEST_GENES = ['KRAS', 'NKX2-1', 'AR', 'APC', 'TP53']
    args.n_bg, args.scan_batches = 40, 60
N_BG = args.n_bg
MAX_INSERT = 1  # we insert exactly one dose token; need >=1 free slot (+margin)

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

os.makedirs(DS_DIR, exist_ok=True)
device = torch.device('cuda')
L.seed_everything(SEED, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
g2i = tok.component_token2idx['gene']; i2g = {v: k for k, v in g2i.items()}
vaf2i = tok.component_token2idx['aa_vaf_bin']          # '1'..'10' -> 4..13
zyg2i = tok.component_token2idx['zygosity']            # heterozygous->4 homozygous->5
VAF_IDS = {vaf2i[str(k)]: k for k in range(1, 11)}     # token-id -> human VAF level 1..10
HET, HOM = zyg2i['heterozygous'], zyg2i['homozygous']
REAL_MIN = 4

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
    n = v.numel()
    idx = torch.randint(0, n, (B, n), device=device)
    meds = v[idx].median(dim=1).values
    return float(torch.quantile(meds, 0.025).item()), float(torch.quantile(meds, 0.975).item())


# ---------------- token machinery -------------------------------------------------------------
COMP_KEYS = None
MUT_COMPS = ['gene', 'alt_type', 'aa_ref', 'aa_mut', 'aa_pos']   # fixes the specific variant
def set_comp_keys(dna):
    global COMP_KEYS
    if COMP_KEYS is None:
        COMP_KEYS = list(dna.keys()); assert 'gene' in COMP_KEYS
        for c in MUT_COMPS + ['aa_vaf_bin', 'zygosity']:
            assert c in COMP_KEYS, f'missing component {c}'


def mut_sig(dna_np, r, s):
    return hash(b'|'.join(dna_np[c][r, s].tobytes() for c in MUT_COMPS))


def slot_scalar(dna_np, c, r, s):
    v = dna_np[c][r, s]
    return int(v) if np.ndim(v) == 0 else int(np.asarray(v).ravel()[0])


def first_free(m_rows):
    return (m_rows == 0).float().argmax(1)


def insert_token(dna, m_base, token):
    """clone dna, write `token` (one real captured slot, dict of per-comp values) into every row's
    first free slot, mask->1.  Same free-slot insertion audited in insilico_ki_disease.py."""
    K = m_base.shape[0]
    f_idx = first_free(m_base)
    rows = torch.arange(K, device=device)
    dna_add = {c: dna[c].clone() for c in COMP_KEYS}
    m_add = m_base.clone()
    for c in COMP_KEYS:
        val = token[c].to(device)
        dna_add[c][rows, f_idx] = val.unsqueeze(0).expand((K,) + tuple(val.shape))
    m_add[rows, f_idx] = 1
    return dna_add, m_add


# =============================================================================== pass A: census
# For each test gene, over slots that HAVE a real VAF bin, count (a) the specific variant and
# (b) per-variant coverage over (zygosity, vaf_bin). Only variants with real VAF/zyg qualify, so
# the canonical variant we sweep is a point/indel with dose data (not a CN event without VAF).
print('\n===== pass A: census of dose coverage for test genes =====', flush=True)
test_gids = {g2i[g] for g in TEST_GENES if g in g2i}
missing_genes = [g for g in TEST_GENES if g not in g2i]
if missing_genes:
    print(f'  WARNING: genes not in vocab, dropped: {missing_genes}', flush=True)
var_count = defaultdict(Counter)                 # gene_id -> Counter(mut_sig)
var_levels = defaultdict(Counter)                # (gene_id,mut_sig) -> Counter((zyg,vaf))
carrier_count = np.zeros(len(g2i), dtype=np.int64)

loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
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
        carrier_count[genes[genes < len(carrier_count)]] += 1
        for s in np.where(valid)[0]:
            g = int(gf[r, s])
            if g not in test_gids:
                continue
            vb = slot_scalar(dna_np, 'aa_vaf_bin', r, s)
            if vb not in VAF_IDS:                 # skip slots without a real VAF bin (e.g. CN events)
                continue
            zy = slot_scalar(dna_np, 'zygosity', r, s)
            ms = mut_sig(dna_np, r, s)
            var_count[g][ms] += 1
            var_levels[(g, ms)][(zy, vb)] += 1
    if (bi + 1) >= (args.scan_batches if args.dry else 10**9):
        break
    if (bi + 1) % 200 == 0:
        print(f'  census batch {bi+1}/{len(loader)}', flush=True)

# choose canonical variant per gene + which levels to sweep
plan = {}          # gene_name -> dict(gid, mut_sig, modal_vaf, modal_zyg, vaf_bins[], zygs[])
wants = {}         # (mut_sig, zyg, vaf) -> (gene_name, axis, level_key)
for g in TEST_GENES:
    if g not in g2i or g2i[g] not in var_count or not var_count[g2i[g]]:
        print(f'  {g}: no dose-bearing variant found, skipped', flush=True); continue
    gid = g2i[g]
    ms, _ = var_count[gid].most_common(1)[0]
    lv = var_levels[(gid, ms)]
    modal_zyg, modal_vaf = max(lv.items(), key=lambda kv: kv[1])[0]
    # VAF sweep: fix zygosity at modal_zyg, take every VAF bin with enough carriers
    vaf_bins = sorted([vb for (zy, vb), n in lv.items() if zy == modal_zyg and n >= args.min_level],
                      key=lambda vb: VAF_IDS[vb])
    # zygosity sweep: fix VAF at modal_vaf, take het & hom if both present with enough carriers
    zygs = [z for z in (HET, HOM) if lv.get((z, modal_vaf), 0) >= args.min_level]
    plan[g] = dict(gid=gid, mut_sig=ms, modal_zyg=modal_zyg, modal_vaf=modal_vaf,
                   vaf_bins=vaf_bins, zygs=zygs)
    for vb in vaf_bins:
        wants[(ms, modal_zyg, vb)] = (g, 'vaf', vb)
    for z in zygs:
        wants[(ms, z, modal_vaf)] = (g, 'zyg', z)
    print(f'  {g:7s} carriers={carrier_count[gid]:6d} VAF bins={[VAF_IDS[b] for b in vaf_bins]} '
          f'zyg={"".join("H" if z==HET else "O" for z in zygs)} (modalVAF={VAF_IDS[modal_vaf]})', flush=True)
assert plan, 'no test gene had dose-bearing variants'

# =============================================================================== pass C: capture
print('\n===== pass C: capture one real token per (variant, level) =====', flush=True)
captured = {}      # want_key -> {comp: tensor slot}
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
for bi, batch in enumerate(loader):
    dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
    dna_np = {c: dna[c].numpy() for c in COMP_KEYS}
    mk = mask.numpy(); gf = dna_np['gene']
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    for r in np.where(dx)[0]:
        valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
        for s in np.where(valid)[0]:
            if int(gf[r, s]) not in test_gids:
                continue
            vb = slot_scalar(dna_np, 'aa_vaf_bin', r, s)
            if vb not in VAF_IDS:
                continue
            key = (mut_sig(dna_np, r, s), slot_scalar(dna_np, 'zygosity', r, s), vb)
            if key in wants and key not in captured:
                captured[key] = {c: dna[c][r, s].clone() for c in COMP_KEYS}
    if len(captured) == len(wants):
        break
    if (bi + 1) >= (args.scan_batches if args.dry else 10**9):
        break
print(f'captured {len(captured)}/{len(wants)} tokens', flush=True)

# ---- audit: within each gene's VAF series the variant is identical and only aa_vaf_bin moves
print('\n----- AUDIT: token construction -----', flush=True)
for g in list(plan)[:3]:
    ms = plan[g]['mut_sig']; zy = plan[g]['modal_zyg']
    got = [(vb, captured.get((ms, zy, vb))) for vb in plan[g]['vaf_bins']]
    got = [(vb, t) for vb, t in got if t is not None]
    if not got:
        print(f'  {g}: no VAF tokens captured'); continue
    vafs_in_token = [int(t['aa_vaf_bin'].reshape(-1)[0].item()) for _, t in got]
    same_gene = len({int(t['gene'].reshape(-1)[0].item()) for _, t in got}) == 1
    same_variant = len({t['aa_pos'].numpy().tobytes() for _, t in got}) == 1
    print(f'  {g}: requested VAF {[VAF_IDS[vb] for vb,_ in got]} -> token VAF {[VAF_IDS[v] for v in vafs_in_token]} '
          f'| same_gene={same_gene} same_variant_pos={same_variant}', flush=True)
    assert vafs_in_token == [vb for vb, _ in got], f'{g}: captured VAF bins do not match request'
    assert same_gene, f'{g}: VAF series gene mismatch'

# =============================================================================== pass R: bg + ref
print('\n===== pass R: reference medians + background hosts =====', flush=True)
feat = np.load(f'{OUT_DIR}/umap/umap_features.npz', allow_pickle=True)
sid2true = dict(zip(feat['sample_id'].astype(str), feat['true_term'].astype(str)))

# target lineage per gene = arg-max KI sufficiency (data-driven, self-consistent with #2/#4)
ki = np.load(f'{OUT_DIR}/ki/ki_results.npz', allow_pickle=True)
assert (ki['sup_names'].astype(str) == np.array(sup_names)).all()
ki_genes = list(ki['genes'].astype(str)); ki_KI = ki['meandP'].astype(float)
gene_target, gene_ki = {}, {}
for g in plan:
    if g in ki_genes:
        row = ki_KI[ki_genes.index(g)]
        col = int(np.argmax(row)); gene_target[g] = sup_names[col]; gene_ki[g] = float(row[col])
    else:
        gene_target[g] = None; gene_ki[g] = float('nan')
        print(f'  WARNING: {g} not in KI genes; no target', flush=True)
plan = {g: p for g, p in plan.items() if gene_target.get(g) in sup_name2col}
tgt_set = set(gene_target[g] for g in plan)
print(f'gene targets: ' + ', '.join(f'{g}->{gene_target[g][:22]}(KI{gene_ki[g]:+.2f})' for g in plan), flush=True)

ref_vals = {t: [] for t in tgt_set}
hosts = []; n_seen = 0
rng_host = np.random.default_rng(SEED)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
for bi, batch in enumerate(loader):
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    if not dx.any():
        if bi + 1 >= args.scan_batches: break
        continue
    dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
    mk = mask.numpy(); sids = md.index.astype(str).tolist()
    if any(len(ref_vals[t]) < args.ref_min for t in tgt_set):
        P = probs_sup_chunked(to_dev(dna), to_dev(mask)).cpu().numpy()
        for r in np.where(dx)[0]:
            tr = sid2true.get(sids[r])
            if tr in tgt_set and len(ref_vals[tr]) < args.ref_min:
                ref_vals[tr].append(float(P[r, sup_name2col[tr]]))
    for r in np.where(dx)[0]:
        mr = mk[r].copy()
        if (mr == 0).sum() < MAX_INSERT + 1:
            continue
        n_seen += 1
        j = len(hosts) if len(hosts) < N_BG else int(rng_host.integers(0, n_seen))
        if j < N_BG:
            h = {c: dna[c][r].clone().numpy() for c in COMP_KEYS}; h['__mask__'] = mr
            hosts.append(h) if j == len(hosts) else hosts.__setitem__(j, h)
    if bi + 1 >= args.scan_batches:
        break
    if (bi + 1) % 50 == 0:
        print(f'  passR batch {bi+1}: reservoir={len(hosts)}/{N_BG} seen={n_seen}', flush=True)
N_BG = len(hosts)
ref_median = {t: (float(np.median(ref_vals[t])) if ref_vals[t] else float('nan')) for t in tgt_set}
print(f'gathered {N_BG} backgrounds; ref medians: '
      + ', '.join(f'{t[:16]}={ref_median[t]:.2f}' for t in tgt_set), flush=True)

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
bg_mask = bg_mask.to(device); bg_dna = {c: v.to(device) for c, v in bg_dna.items()}
assert int((bg_mask == 0).sum(1).min().item()) >= MAX_INSERT + 1


# =============================================================================== sweeps
@torch.no_grad()
def sweep_token(token, col):
    """insert one captured token into all backgrounds; return P(target) tensor [N] and full P mean [n_sup]."""
    dna_a, m_a = insert_token(bg_dna, bg_mask, token)
    P = probs_sup_chunked(dna_a, m_a)
    return P[:, col], P.median(0).values         # per-bg target P ; median-over-bg full profile


print('\n===== sweeps: VAF dose + zygosity =====', flush=True)
res = {}
base_P = probs_sup_chunked(bg_dna, bg_mask)       # [N, n_sup] baseline (no insertion)
for g in plan:
    col = sup_name2col[gene_target[g]]; ms = plan[g]['mut_sig']
    r = dict(target=gene_target[g], gclass=GENE_CLASS[g], ki=gene_ki[g], col=col,
             base_p50=float(base_P[:, col].median().item()), ref=ref_median[gene_target[g]])
    # ---- VAF sweep
    vb_list, q10, q50, q90, clo, chi = [], [], [], [], [], []
    prof_lo = prof_hi = None
    for vb in plan[g]['vaf_bins']:
        t = captured.get((ms, plan[g]['modal_zyg'], vb))
        if t is None: continue
        pL, prof = sweep_token(t, col)
        vb_list.append(VAF_IDS[vb])
        q10.append(float(pL.quantile(0.10))); q50.append(float(pL.quantile(0.50))); q90.append(float(pL.quantile(0.90)))
        lo, hi = median_ci95(pL); clo.append(lo); chi.append(hi)
        if prof_lo is None: prof_lo = prof.cpu().numpy()
        prof_hi = prof.cpu().numpy()
    r.update(vaf_level=vb_list, vaf_q10=q10, vaf_q50=q50, vaf_q90=q90, vaf_ci_lo=clo, vaf_ci_hi=chi,
             vaf_prof_lo=prof_lo, vaf_prof_hi=prof_hi)
    # ---- zygosity sweep
    zlab, zq50, zclo, zchi = [], [], [], []
    for z in plan[g]['zygs']:
        t = captured.get((ms, z, plan[g]['modal_vaf']))
        if t is None: continue
        pL, _ = sweep_token(t, col)
        zlab.append('hom' if z == HOM else 'het'); zq50.append(float(pL.quantile(0.50)))
        lo, hi = median_ci95(pL); zclo.append(lo); zchi.append(hi)
    r.update(zyg_label=zlab, zyg_q50=zq50, zyg_ci_lo=zclo, zyg_ci_hi=zchi)
    res[g] = r
    slope = (q50[-1] - q50[0]) if len(q50) >= 2 else float('nan')
    zt = (f' zyg het->hom {zq50[0]:.2f}->{zq50[-1]:.2f}' if len(zq50) == 2 else '')
    print(f'  {g:7s} -> {gene_target[g][:26]:26s} VAF P {q50[0] if q50 else float("nan"):.2f}'
          f'->{q50[-1] if q50 else float("nan"):.2f} (slope {slope:+.2f}){zt}', flush=True)

# =============================================================================== save
genes_out = list(res)
np.savez_compressed(
    f'{DS_DIR}/dose_results.npz',
    genes=np.array(genes_out),
    gclass=np.array([res[g]['gclass'] for g in genes_out]),
    target=np.array([res[g]['target'] for g in genes_out]),
    ki=np.array([res[g]['ki'] for g in genes_out]),
    base_p50=np.array([res[g]['base_p50'] for g in genes_out]),
    ref=np.array([res[g]['ref'] for g in genes_out]),
    vaf_level=np.array([res[g]['vaf_level'] for g in genes_out], dtype=object),
    vaf_q10=np.array([res[g]['vaf_q10'] for g in genes_out], dtype=object),
    vaf_q50=np.array([res[g]['vaf_q50'] for g in genes_out], dtype=object),
    vaf_q90=np.array([res[g]['vaf_q90'] for g in genes_out], dtype=object),
    vaf_ci_lo=np.array([res[g]['vaf_ci_lo'] for g in genes_out], dtype=object),
    vaf_ci_hi=np.array([res[g]['vaf_ci_hi'] for g in genes_out], dtype=object),
    vaf_prof_lo=np.array([res[g]['vaf_prof_lo'] for g in genes_out], dtype=object),
    vaf_prof_hi=np.array([res[g]['vaf_prof_hi'] for g in genes_out], dtype=object),
    zyg_label=np.array([res[g]['zyg_label'] for g in genes_out], dtype=object),
    zyg_q50=np.array([res[g]['zyg_q50'] for g in genes_out], dtype=object),
    zyg_ci_lo=np.array([res[g]['zyg_ci_lo'] for g in genes_out], dtype=object),
    zyg_ci_hi=np.array([res[g]['zyg_ci_hi'] for g in genes_out], dtype=object),
    sup_names=np.array(sup_names),
)
meta = dict(analysis='dose-vaf-zygosity', ckpt=os.path.basename(CKPT), n_bg=int(N_BG),
            n_genes=int(len(genes_out)), min_level=int(args.min_level), dry=bool(args.dry))
json.dump(meta, open(f'{DS_DIR}/dose_meta.json', 'w'), indent=2)
print(f'\nDONE -> {DS_DIR}/dose_results.npz  (genes={len(genes_out)}, backgrounds={N_BG})', flush=True)
