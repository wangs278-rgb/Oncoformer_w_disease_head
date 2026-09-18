#!/usr/bin/env python3
"""EVO #2  Does the model's greedy BUILD ORDER match the real EVOLUTIONARY order?

The greedy recipe (insilico_minsig) is a model-predicted order to ACQUIRE a lineage's drivers -
and because greedy inserts each gene at its MODAL VAF, that order is driven by identity-efficiency,
NOT clonality, so it is an independent prediction we can test against real mutation timing.

Real timing (single-sample, standard method): within a tumour, a higher-VAF mutation is earlier
(clonal/truncal) than a lower-VAF one. For each lineage we take real DX1/DX2 tumours of that
lineage and, for every pair of recipe genes that CO-OCCUR, ask whether the greedy-EARLIER gene is
the more clonal (higher VAF) one. Concordance > 0.5 => evolution follows the identity-efficient
order. Pure tabulation (no model) -> CPU.

Timing signal = raw aa_vaf_bin (conservative; CCF is the publication follow-up). A gene's VAF in a
tumour = its highest-VAF occurrence (most clonal instance).

Reads minsig/minsig_results_all.npz (recipe + stop + final P).
Writes run8_disease_out/evo/order_match_results.npz.
    python order_match.py [--max-genes 8 --min-co 10 --scan-batches 0]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from itertools import combinations
from collections import defaultdict

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
EV_DIR   = f'{OUT_DIR}/evo'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CHUNK = 512
REAL_MIN = 4

ap = argparse.ArgumentParser()
ap.add_argument('--max-genes', type=int, default=8, help='cap recipe length used for ordering')
ap.add_argument('--min-co', type=int, default=10, help='min co-occurring tumours to score a pair')
ap.add_argument('--scan-batches', type=int, default=0, help='0 = full cohort')
ap.add_argument('--tag', default='_all', help='which minsig recipe file to use')
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
g2i = tok.component_token2idx['gene']
vaf2i = tok.component_token2idx['aa_vaf_bin']
VAF_IDS = {vaf2i[str(k)]: k for k in range(1, 11)}

feat = np.load(f'{OUT_DIR}/umap/umap_features.npz', allow_pickle=True)
sid2true = dict(zip(feat['sample_id'].astype(str), feat['true_term'].astype(str)))

# ---- recipes (order per lineage) ----
ms = np.load(f'{OUT_DIR}/minsig/minsig_results{args.tag}.npz', allow_pickle=True)
targets = ms['targets'].astype(str)
recipe_name = ms['recipe_name']; stop_step = ms['stop_step']; p_q50 = ms['p_q50']; ref_med = ms['ref_median']
recipe = {}          # lineage -> list of (gene_name, rank)
gene_rank = {}       # lineage -> {gene_id: rank}
final_P = {}
for i, t in enumerate(targets):
    k = min(int(stop_step[i]), args.max_genes)
    genes = [g for g in list(recipe_name[i])[:k] if g in g2i]
    recipe[t] = genes
    gene_rank[t] = {g2i[g]: r for r, g in enumerate(genes)}
    final_P[t] = float(p_q50[i][int(stop_step[i]) - 1])
tgt_set = set(targets)
print(f'{len(targets)} lineages; recipe lengths '
      f'{np.median([len(recipe[t]) for t in targets]):.0f} median', flush=True)

# ---- scan: accumulate per-lineage pairwise co-occurrence + "greedy-earlier is more clonal" ----
co = {t: defaultdict(int) for t in targets}       # (rank_a<rank_b) -> co-occurring tumours
conc = {t: defaultdict(float) for t in targets}   # -> count where greedy-earlier gene more clonal
vaf_sum = {t: defaultdict(float) for t in targets}
vaf_cnt = {t: defaultdict(int) for t in targets}

dl = OncoformerDataLoader(config, load_metadata=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
n_tum = 0
for bi, batch in enumerate(loader):
    dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    sids = md.index.astype(str).tolist()
    gf = dna['gene'].numpy(); mk = mask.numpy(); vf = dna['aa_vaf'].numpy()   # CONTINUOUS VAF
    for r in np.where(dx)[0]:
        t = sid2true.get(sids[r])
        if t not in tgt_set:
            continue
        ranks = gene_rank[t]
        if len(ranks) < 2:
            continue
        # gene -> highest continuous VAF carried in this tumour (recipe genes with a real VAF>0)
        gv = {}
        valid = (gf[r] >= REAL_MIN) & (mk[r] > 0)
        for s in np.where(valid)[0]:
            g = int(gf[r, s])
            if g not in ranks:
                continue
            val = float(vf[r, s])
            if not (val > 0):
                continue
            gv[g] = max(gv.get(g, 0.0), val)
        if len(gv) < 2:
            continue
        n_tum += 1
        for g, lvl in gv.items():
            vaf_sum[t][g] += lvl; vaf_cnt[t][g] += 1
        for a, b in combinations(gv.keys(), 2):
            # order the pair by greedy rank so key is (earlier_gene, later_gene)
            ea, eb = (a, b) if ranks[a] < ranks[b] else (b, a)
            key = (ea, eb)
            co[t][key] += 1
            if gv[ea] > gv[eb]:
                conc[t][key] += 1.0          # greedy-earlier is more clonal (concordant)
            elif gv[ea] == gv[eb]:
                conc[t][key] += 0.5          # tie
    if args.scan_batches and bi + 1 >= args.scan_batches:
        break
    if (bi + 1) % 200 == 0:
        print(f'  batch {bi+1}/{len(loader)} tumours={n_tum}', flush=True)

# ---- summarise ----
i2g = {v: k for k, v in g2i.items()}
rows = []                 # per-pair records
lin_conc = {}             # lineage -> weighted concordance over pairs with enough co-occurrence
lin_npair = {}; lin_ninst = {}
for t in targets:
    tot_inst = 0.0; tot_w = 0.0; npair = 0
    for key, n in co[t].items():
        if n < args.min_co:
            continue
        frac = conc[t][key] / n
        ea, eb = key
        rows.append((t, i2g[ea], i2g[eb], gene_rank[t][ea], gene_rank[t][eb], int(n), float(frac)))
        tot_inst += conc[t][key]; tot_w += n; npair += 1
    lin_conc[t] = (tot_inst / tot_w) if tot_w > 0 else np.nan
    lin_npair[t] = npair; lin_ninst[t] = int(tot_w)

# pooled concordance (instance-weighted, pairs with enough co-occurrence)
all_inst = sum(conc[t][k] for t in targets for k in co[t] if co[t][k] >= args.min_co)
all_w = sum(co[t][k] for t in targets for k in co[t] if co[t][k] >= args.min_co)
pooled = all_inst / all_w if all_w else np.nan

print(f'\n===== ORDER-MATCH VERDICT (raw VAF) =====', flush=True)
print(f'tumours used={n_tum}  scored pairs={len(rows)}  pooled concordance={pooled:.3f} (null 0.5)', flush=True)
for t in sorted(targets, key=lambda x: -(lin_conc[x] if np.isfinite(lin_conc[x]) else -1)):
    if lin_npair[t] == 0:
        continue
    print(f'  {t[:36]:36s} conc={lin_conc[t]:.2f}  pairs={lin_npair[t]:2d}  '
          f'inst={lin_ninst[t]:5d}  fullP={final_P[t]:.2f}  recipe={"→".join(recipe[t][:4])}', flush=True)

np.savez_compressed(
    f'{EV_DIR}/order_match_results.npz',
    targets=np.array(targets),
    lin_conc=np.array([lin_conc[t] for t in targets]),
    lin_npair=np.array([lin_npair[t] for t in targets]),
    lin_ninst=np.array([lin_ninst[t] for t in targets]),
    final_P=np.array([final_P[t] for t in targets]),
    ref_median=ref_med,
    pooled=pooled, n_tumours=n_tum, min_co=args.min_co,
    recipe=np.array([recipe[t] for t in targets], dtype=object),
    pairs=np.array(rows, dtype=object),      # (lineage, earlier, later, rank_e, rank_l, co_n, conc_frac)
)
print(f'\nDONE -> {EV_DIR}/order_match_results.npz', flush=True)
