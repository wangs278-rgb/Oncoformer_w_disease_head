#!/usr/bin/env python3
"""CCF decoder, data prep v2 ("armloss"): hemizygous correction of homozygous short variants on a lost arm.
Rule (decided 2026-11-20, blanket, all diseases): a class-2 (FMI 'homozygous') short variant whose gene's OWN autosomal arm is called lost
(B1_samples chr<arm>_loss == 'positive') has one copy left in the tumour, so  CCF = VAF (2 - p) / p  instead of the copy-neutral VAF / p
(old ccf = VAF / p  ->  new = old * (2 - p)), then capped at 1 by ccf_to_bin as before.  Flag negative / unknown / sample not in B1: unchanged.
Everything else (classes 1, 3, untimed, copy-number) is copied unchanged from the v1 targets (ccf_prep.py output).
Reads  --src {ccf_targets,prefix_labels}.npz (v1, read only), run9 sequences.npz + vocab.npz, B1_samples.parquet, the samples table (purity),
RefSeq (gene arms).  Writes --out {ccf_targets.npz (same keys + 'armloss' bool), prefix_labels.npz (copy), armloss_stats.json}.
Use:  CCFData(run9_data, <--out>).
Usage: python3 -I ccf_prep_armloss.py [--src data/v1] [--out data] [--data-dir $RUN9_DATA] [--b1 $FMI_PARQUET/B1_samples.parquet]
                                      [--samples $SAMPLES_PARQUET] [--refseq $REFSEQ]"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ccf_data import env_dir                                      # noqa: E402
from prep_common import purity, ccf_to_bin, gene_positions        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(HERE, 'data', 'v1'), help='v1 targets (ccf_prep.py --out), read only')
ap.add_argument('--out', default=os.path.join(HERE, 'data'))
ap.add_argument('--data-dir', default=None, help='run9 data dir; default $RUN9_DATA')
ap.add_argument('--b1', default=None, help='FMI B1_samples.parquet; default $FMI_PARQUET/B1_samples.parquet')
ap.add_argument('--samples', default=None, help='samples table with xrn + computationalTumorPurity; default $SAMPLES_PARQUET')
ap.add_argument('--refseq', default=None, help='UCSC ncbiRefSeq.txt.gz; default $REFSEQ')
args = ap.parse_args()
SRC, OUT = args.src, args.out
DATA = env_dir('RUN9_DATA', args.data_dir)
B1 = args.b1 or os.path.join(env_dir('FMI_PARQUET'), 'B1_samples.parquet')
SAMPLES = env_dir('SAMPLES_PARQUET', args.samples)
REFSEQ = env_dir('REFSEQ', args.refseq)

os.makedirs(OUT, exist_ok=True)
seq = np.load(f'{DATA}/sequences.npz')
V = dict(np.load(f'{DATA}/vocab.npz'))
T = np.load(f'{SRC}/ccf_targets.npz')
sid = seq['sample_ids']
assert np.array_equal(T['sample_ids'], sid)
gene, offsets = seq['gene_idx'].astype(np.int64), seq['offsets'].astype(np.int64)
pat = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
ccf, cbin, tt = T['ccf'].astype(np.float64), T['cbin'].astype(np.int64), T['ttype']
GN = V['gene_names'].astype(str)
lv = [str(x) for x in V['levels']]
dis = seq['disease'].astype(np.int64)

GP = gene_positions(REFSEQ)
garm = np.array([(GP[g][0] + GP[g][1]) if (g in GP and GP[g][0] not in ('X', 'Y')) else '' for g in GN])
arm = garm[gene]
cand = np.nonzero((tt == 2) & (arm != ''))[0]
have = set(pq.read_schema(B1).names)
arms = sorted({a for a in arm[cand] if f'chr{a}_loss' in have})
B = pd.read_parquet(B1, columns=['xrn'] + [f'chr{a}_loss' for a in arms]).drop_duplicates('xrn').set_index('xrn')
P = purity(sid.astype(str), SAMPLES)[pat]                                     # per alteration
flag = np.zeros(len(tt), bool)
for a in arms:
    k = cand[arm[cand] == a]
    flag[k] = (B[f'chr{a}_loss'].reindex(sid.astype(str)[pat[k]]).to_numpy() == 'positive')
sel = flag & np.isfinite(P) & (tt == 2)
assert np.isfinite(ccf[sel]).all()
print(f'class-2 stored ccf: max {np.nanmax(ccf[tt == 2]):.2f}, share > 1 {np.mean(ccf[tt == 2] > 1):.3f}')

new_ccf, new_bin = ccf.copy(), cbin.copy()
new_ccf[sel] = ccf[sel] * (2 - P[sel])
new_bin[sel] = ccf_to_bin(new_ccf[sel])
np.savez(f'{OUT}/ccf_targets.npz', ccf=new_ccf.astype(np.float32), cbin=new_bin.astype(np.int8), ttype=tt, sample_ids=sid, armloss=sel)
shutil.copy(f'{SRC}/prefix_labels.npz', f'{OUT}/prefix_labels.npz')

top_old, top_new = (cbin[sel] == 20), (new_bin[sel] == 20)
stats = dict(class2=int((tt == 2).sum()), corrected=int(sel.sum()), share_of_timed=float(sel.sum() / ((tt >= 1) & (tt <= 3)).sum()),
             changed_bin=int((new_bin[sel] != cbin[sel]).sum()), top_bin_before=float(top_old.mean()), top_bin_after=float(top_new.mean()),
             into_top_bin=int((~top_old & top_new).sum()))
by_gene = pd.Series(GN[gene[sel]]).value_counts().head(10).to_dict()
stats['top_genes'] = {k: int(v) for k, v in by_gene.items()}
json.dump(stats, open(f'{OUT}/armloss_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))


def show(d, g):
    m = (dis[pat] == lv.index(d)) & (gene == GN.tolist().index(g)) & (tt == 2)
    if m.sum():
        print(f'  {d[:12]:12s} {g:6s} hom n={m.sum():5d} corrected {int(sel[m].sum()):5d} | median bin {np.median(cbin[m]):.0f} -> {np.median(new_bin[m]):.0f} | share top bin {np.mean(cbin[m] == 20):.2f} -> {np.mean(new_bin[m] == 20):.2f}')


for d, g in (('kidney clear cell carcinoma', 'VHL'), ('kidney clear cell carcinoma', 'PBRM1'), ('kidney clear cell carcinoma', 'SETD2'), ('kidney clear cell carcinoma', 'BAP1'),
             ('colon adenocarcinoma (crc)', 'TP53'), ('colon adenocarcinoma (crc)', 'APC'), ('colon adenocarcinoma (crc)', 'SMAD4')):
    show(d, g)
