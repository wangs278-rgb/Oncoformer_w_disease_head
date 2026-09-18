#!/usr/bin/env python3
"""EVO #3  Per-gene clonality census (NO model / NO GPU): for every driver gene, the distribution
of its VAF levels (1=subclonal .. 10=clonal) across real DX1/DX2 tumours - overall and within each
of the top lineages. Cross this with KO necessity / KI sufficiency to test the evolution claim:
'lineage-defining (necessity) genes are TRUNCAL (high VAF); toggles/modifiers are subclonal'.

Pure tabulation over the tokenised data - runs on CPU (defq).
Writes run8_disease_out/evo/clonality_results.npz.
    python census_clonality.py [--n-lineages 20] [--min-carriers 100] [--scan-batches 0]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
EV_DIR   = f'{OUT_DIR}/evo'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CHUNK = 512
REAL_MIN = 4

ap = argparse.ArgumentParser()
ap.add_argument('--n-lineages', type=int, default=20)
ap.add_argument('--min-carriers', type=int, default=100)
ap.add_argument('--scan-batches', type=int, default=0, help='0 = full cohort')
args = ap.parse_args()

sys.path.insert(0, DATA_DIR)
import run8_disease_config
run8_disease_config.use_moco_oncoformer()
config = run8_disease_config.build_config(CACHE)
from oncoformer.dataset import OncoformerDataLoader
from torch.utils.data import DataLoader
os.makedirs(EV_DIR, exist_ok=True)

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
g2i = tok.component_token2idx['gene']; i2g = {v: k for k, v in g2i.items()}
vaf2i = tok.component_token2idx['aa_vaf_bin']
VAF_IDS = {vaf2i[str(k)]: k for k in range(1, 11)}     # token-id -> level 1..10
NG = max(g2i.values()) + 1

feat = np.load(f'{OUT_DIR}/umap/umap_features.npz', allow_pickle=True)
sid2true = dict(zip(feat['sample_id'].astype(str), feat['true_term'].astype(str)))
levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
def is_defined(s):
    s = s.lower(); return not (('unknown primary' in s) or ('undifferentiated' in s) or ('(cup)' in s))
freq = Counter(feat['true_term'].astype(str))
top_lin = [l for l, _ in freq.most_common() if is_defined(l)][:args.n_lineages]
lin2idx = {l: i for i, l in enumerate(top_lin)}
nTL = len(top_lin)
print(f'genes vocab={NG}  top lineages={nTL}', flush=True)

dl = OncoformerDataLoader(config, load_metadata=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)

NBINS = 20                                                 # continuous VAF binned over [0,1]
CLONAL_BIN = int(0.8 * NBINS)                              # VAF >= 0.8 counts as high-clonality
CENTERS = (np.arange(NBINS) + 0.5) / NBINS
hist_overall = np.zeros((NG, NBINS), dtype=np.int64)       # gene -> continuous-VAF histogram
hist_bylin = np.zeros((NG, nTL, NBINS), dtype=np.int64)    # gene x lineage -> VAF histogram
carrier = np.zeros(NG, dtype=np.int64)
n_tumours = 0
for bi, batch in enumerate(loader):
    dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    sids = md.index.astype(str).tolist()
    gf = dna['gene'].numpy(); mk = mask.numpy(); vf = dna['aa_vaf'].numpy()   # CONTINUOUS VAF
    for r in np.where(dx)[0]:
        valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
        if not valid.any():
            continue
        n_tumours += 1
        li = lin2idx.get(sid2true.get(sids[r]), -1)
        seen = set()
        for s in np.where(valid)[0]:
            g = int(gf[r, s])
            if g >= NG:
                continue
            if g not in seen:
                carrier[g] += 1; seen.add(g)
            val = float(vf[r, s])
            if not (val > 0):                              # skip CN/missing (no VAF)
                continue
            b = min(NBINS - 1, int(val * NBINS))
            hist_overall[g, b] += 1
            if li >= 0:
                hist_bylin[g, li, b] += 1
    if args.scan_batches and bi + 1 >= args.scan_batches:
        break
    if (bi + 1) % 200 == 0:
        print(f'  batch {bi+1}/{len(loader)}  tumours={n_tumours}', flush=True)

sel = np.where(carrier >= args.min_carriers)[0]
sel = sel[np.argsort(-carrier[sel])]
names = [i2g[g] for g in sel]
print(f'genes with >= {args.min_carriers} carriers: {len(sel)}  (tumours scanned={n_tumours})', flush=True)


def summ(h):
    """weighted median + mean continuous VAF (0..1) from a [NBINS] histogram; nan if empty."""
    tot = h.sum()
    if tot == 0:
        return np.nan, np.nan
    mean = float((h * CENTERS).sum() / tot)
    cdf = np.cumsum(h) / tot
    med = float(CENTERS[np.searchsorted(cdf, 0.5)])
    return med, mean


med_overall = np.array([summ(hist_overall[g])[0] for g in sel])
mean_overall = np.array([summ(hist_overall[g])[1] for g in sel])
# clonal fraction = fraction of a gene's mutations at VAF >= 0.8 (high clonality)
clonal_frac = np.array([hist_overall[g, CLONAL_BIN:].sum() / max(1, hist_overall[g].sum()) for g in sel])
med_bylin = np.array([[summ(hist_bylin[g, li])[0] for li in range(nTL)] for g in sel])

np.savez_compressed(
    f'{EV_DIR}/clonality_results.npz',
    genes=np.array(names), carriers=carrier[sel],
    hist_overall=hist_overall[sel], hist_bylin=hist_bylin[sel],
    med_vaf=med_overall, mean_vaf=mean_overall, clonal_frac=clonal_frac,
    med_vaf_bylin=med_bylin, lineages=np.array(top_lin), n_tumours=n_tumours,
)
# quick peek: clonality of some canonical drivers
for gname in ['KRAS', 'TP53', 'APC', 'BRAF', 'EGFR', 'PIK3CA', 'NKX2-1', 'AR', 'VHL', 'PTEN']:
    if gname in names:
        k = names.index(gname)
        print(f'  {gname:7s} carriers={carrier[sel][k]:6d} medVAF={med_overall[k]:.2f} '
              f'meanVAF={mean_overall[k]:.2f} clonalFrac={clonal_frac[k]:.2f}', flush=True)
print(f'\nDONE -> {EV_DIR}/clonality_results.npz', flush=True)
