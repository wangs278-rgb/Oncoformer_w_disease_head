"""CCF decoder, data prep v1 (input of ccf_prep_armloss.py).  Writes --out/{ccf_targets.npz, prefix_labels.npz, prep_stats.json}.
Needs the upstream run9 pipeline outputs, which are NOT in this folder: $RUN9_ROOT/code (seqdata.py, timing_model.py),
$RUN9_ROOT/evolution (copy_states.py + copy_states/copy_state_fit.json) and run9 data/{sequences,vocab,purity3,timing_targets_cs}.npz.

Per run9 alteration (aligned with run9 data/sequences.npz):
  ccf   CCF of the alteration, NaN if it has none.  EXACTLY the definition behind evolution_2/trees and step0_order_audit
        (lines copied unchanged from run9 step0_build.py, itself a verified copy of evolution_2/real_trees.py):
          class 1 (het, no arm gain)   2 VAF / p
          class 2 (homozygous)         VAF / p      (copy-neutral LOH, the working assumption)
          class 3 (het, gained arm)    MAP copy state of evolution/copy_states
        classes 4/5 and copy-number events have NO CCF ("untimed"): no cutoff, filter or class of my own is added.
  cbin  CCF capped at 1, 0.05 bins, 0..20 (bin 20 = CCF >= 1), -1 = untimed   (prep_common.ccf_to_bin, as in the trees)
Per patient prefix labels (besides the disease): purity4 = 0 '<20% or unknown', 1 '20-40%', 2 '40-60%', 3 '>=60%' (decided 2026-11-13);
site = FMI A1_samples.local_met_status as given (5 values, no regrouping).
Usage: python3 -I ccf_prep.py [--out data/v1] [--run9-root $RUN9_ROOT] [--data-dir $RUN9_DATA] [--a1 $FMI_PARQUET/A1_samples.parquet]
                              [--samples $SAMPLES_PARQUET]"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ccf_data import env_dir                       # noqa: E402
from prep_common import purity, ccf_to_bin         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(HERE, 'data', 'v1'))
ap.add_argument('--run9-root', default=None, help='run9 root (code/, evolution/); default $RUN9_ROOT')
ap.add_argument('--data-dir', default=None, help='run9 data dir; default $RUN9_DATA')
ap.add_argument('--a1', default=None, help='FMI A1_samples.parquet; default $FMI_PARQUET/A1_samples.parquet')
ap.add_argument('--samples', default=None, help='samples table with xrn + computationalTumorPurity; default $SAMPLES_PARQUET')
args = ap.parse_args()
OUT = args.out
ROOT = env_dir('RUN9_ROOT', args.run9_root)
DATA = env_dir('RUN9_DATA', args.data_dir)
A1 = args.a1 or os.path.join(env_dir('FMI_PARQUET'), 'A1_samples.parquet')
SAMPLES = env_dir('SAMPLES_PARQUET', args.samples)
sys.path.insert(0, f'{ROOT}/code')
sys.path.insert(0, f'{ROOT}/evolution')
from seqdata import SeqData                        # noqa: E402  (upstream run9)
import copy_states as cs                           # noqa: E402  (upstream run9)

os.makedirs(OUT, exist_ok=True)
D = SeqData(DATA, signature=True, sig_file='purity3.npz', timing_file='timing_targets_cs.npz')     # = run9 evolution_2/common2.load_data()
V = D.vocab
SID = D.sample_ids.astype(str)
TT = np.load(f'{DATA}/timing_targets_cs.npz', allow_pickle=True)
CCF = TT['ccf']
assert list(TT['prior_states']) == [f'CN{cn}_m{m}' for cn, m in cs.ST3.cands]
PUR = purity(SID, SAMPLES)
pat_all = np.repeat(np.arange(len(D.n_alt)), D.n_alt)

# ---- copied unchanged from step0_build.py lines 33-56 (= evolution_2/real_trees.py lines 39-68) ----
r3 = np.nonzero(D.ttype == 3)[0]
p3 = PUR[pat_all[r3]]
v3 = CCF[r3, 0] * p3 / 2                                                  # candidate CN2_m1 = 2 VAF / p
s3 = (V['conc_path'][D.conc[r3]] == 6).astype(int)
fitd = json.load(open(f'{ROOT}/evolution/copy_states/copy_state_fit.json'))['bins']
ccf3 = np.full(len(r3), np.nan); map3 = np.full(len(r3), -1)
for lo, hi, b in cs.BINS:
    m = (p3 >= lo) & (p3 < hi)
    f3, f1 = fitd[b]['class3'], fitd[b]['class1']
    th = dict(rho=np.asarray(f3['rho_CN2_CN3_CN4']), q=np.asarray(f3['q_clonal_driver_VUS']),
              mu={3: np.asarray(f3['m_given_CN3']), 4: np.asarray(f3['m_given_CN4'])}, g=np.asarray(f1['g_subclonal']))
    o = cs.Obs(v3[m], p3[m], s3[m], cs.ST3)
    _, Pc, _ = cs.posterior(o, cs.ST3, th, f3['kappa'])
    mp = Pc.argmax(1)
    map3[m] = mp
    ccf3[m] = CCF[r3[m], mp]
assert np.isfinite(ccf3).all()
r2 = np.nonzero(D.ttype == 2)[0]
ccf2 = CCF[r2, 0]
row_ccf = np.full(len(D.ttype), np.nan)
r1 = np.nonzero(D.ttype == 1)[0]
row_ccf[r1] = CCF[r1, 0]
row_ccf[r2] = ccf2
row_ccf[r3] = ccf3
# ---------------------------------------------------------------------------------------------------

timed = (D.ttype >= 1) & (D.ttype <= 3)
assert np.isfinite(row_ccf[timed]).all() and np.isnan(row_ccf[~timed]).all()
cbin = np.full(len(D.ttype), -1, np.int8)
cbin[timed] = ccf_to_bin(row_ccf[timed])
assert cbin[timed].min() >= 0 and cbin[timed].max() <= 20
np.savez(os.path.join(OUT, 'ccf_targets.npz'), ccf=row_ccf.astype(np.float32), cbin=cbin, ttype=D.ttype.astype(np.int8), sample_ids=D.sample_ids)

# ---- prefix labels
pur4 = np.select([PUR >= 0.6, PUR >= 0.4, PUR >= 0.2], [3, 2, 1], 0).astype(np.int8)            # NaN -> 0
PL = ['purity <20% or unknown', 'purity 20-40%', 'purity 40-60%', 'purity >=60%']
a1 = pd.read_parquet(A1, columns=['xrn', 'local_met_status'])
assert a1.xrn.is_unique
site_raw = pd.Series(SID).map(dict(zip(a1.xrn, a1.local_met_status)))
assert site_raw.notna().all(), 'site must exist for every run9 patient'
SL = sorted(site_raw.unique())
site = site_raw.map({s: i for i, s in enumerate(SL)}).to_numpy().astype(np.int8)
np.savez(os.path.join(OUT, 'prefix_labels.npz'), purity=pur4, purity_labels=np.array(PL), site=site, site_labels=np.array(SL), sample_ids=D.sample_ids)

nal = D.n_alt
nt = np.bincount(pat_all, weights=timed, minlength=len(nal))
stats = dict(patients=int(len(nal)), alterations=int(len(timed)), timed=int(timed.sum()), timed_share=float(timed.mean()),
             patients_with_timed=int((nt > 0).sum()), bin_counts=np.bincount(cbin[timed], minlength=21).tolist(),
             purity4={PL[i]: int((pur4 == i).sum()) for i in range(4)}, site={SL[i]: int((site == i).sum()) for i in range(len(SL))},
             timed_share_by_purity={PL[i]: float(nt[pur4 == i].sum() / nal[pur4 == i].sum()) for i in range(4)})
json.dump(stats, open(os.path.join(OUT, 'prep_stats.json'), 'w'), indent=1)
print(json.dumps(stats, indent=1))
