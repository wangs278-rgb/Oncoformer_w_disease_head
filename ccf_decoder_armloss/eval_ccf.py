#!/usr/bin/env python3
"""Compare synthetic patients (generate_ccf.py) with the REAL held-out patients they were conditioned on (same disease, same purity4 + site, one-to-one).
All numbers are written to <out>.json and printed.  Nothing is tuned: no cutoffs of mine except the ones already used for the trees/clusters
(bin gap >= 2 for a resolved order, as in the order graph).
M1 alterations per patient: median / mean, KS distance
M2 gene prevalence: Pearson r over genes; top-100 variant prevalence r
M3 gene co-occurrence: Pearson r of log odds ratios (50 most frequent genes)
M4 timed share of alterations, overall and by purity class; share of patients with >= 1 timed alteration
M5 CCF: bin distribution (KS, share at bin 20); median CCF bin per gene for the 8 most frequent timed genes
M6 order: for gene pairs among the most frequent timed genes, share of patients (both genes timed, |bin gap| >= 2) in which A has the higher CCF
M7 clusters: patients with >= 1 timed variant among the 50 genes are assigned to the 4 clusters of the 50-gene latent-class fit (top50_profile_fits.npz);
   cluster shares real vs synthetic, and gene-state encoding audited against top50_states.npz for the real patients that are in it
   (only if --top50-dir / $TOP50_DIR points to that fit; the fit is an external analysis, not part of this folder)
Usage: python3 -I eval_ccf.py gen/crc_epoch7.npz out.json [--data-dir <run9 data> (default $RUN9_DATA)] [--ccf-dir <ccf data>] [--top50-dir <dir>]"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
from scipy.stats import ks_2samp, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ccf_data import CCFData, CCF_DIR, env_dir       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('gen', help='generate_ccf.py output (.npz)')
ap.add_argument('out', help='metrics (.json)')
ap.add_argument('--data-dir', default=None, help='run9 data dir; default $RUN9_DATA')
ap.add_argument('--ccf-dir', default=CCF_DIR)
ap.add_argument('--top50-dir', default=os.environ.get('TOP50_DIR'), help='dir with top50_states.npz + top50_profile_fits.npz (M7); M7 skipped if unset')
args = ap.parse_args()
G_PATH, OUT = args.gen, args.out
D = CCFData(env_dir('RUN9_DATA', args.data_dir), args.ccf_dir)
V = D.vocab
GN = V['gene_names'].astype(str)
ALT = V['alt_names'][V['conc_alt']].astype(str)
NG = len(GN)
S = np.load(G_PATH)
ref = S['ref']
npat = len(ref)


def real_patient(i):
    a, e = D.offsets[i], D.offsets[i + 1]
    return D.gene[a:e], D.conc[a:e], D.cbin[a:e]


R = [real_patient(i) for i in ref]                                           # (gene, conc, cbin) per real patient
off = S['offsets']
Y = [(S['gene'][off[k]:off[k + 1]], S['conc'][off[k]:off[k + 1]], S['cbin'][off[k]:off[k + 1]]) for k in range(npat)]
res = {'n_patients': int(npat), 'truncated': int(S['truncated'].sum())}


def prevalence(P, key):
    c = np.zeros(NG if key == 'gene' else len(V['conc_gene_idx']))
    for g_, c_, b_ in P:
        np.add.at(c, np.unique(g_ if key == 'gene' else c_), 1)
    return c / len(P)


# M1
nr, ny = np.array([len(p[0]) for p in R]), np.array([len(p[0]) for p in Y])
res['M1'] = dict(median=[float(np.median(nr)), float(np.median(ny))], mean=[float(nr.mean()), float(ny.mean())], KS=float(ks_2samp(nr, ny).statistic))
# M2
pr, py = prevalence(R, 'gene'), prevalence(Y, 'gene')
vr, vy = prevalence(R, 'conc'), prevalence(Y, 'conc')
top = np.argsort(-vr)[:100]
res['M2'] = dict(gene_r=float(pearsonr(pr, py)[0]), top100_variant_r=float(pearsonr(vr[top], vy[top])[0]))


# M3
def presence(P, genes):
    X = np.zeros((len(P), len(genes)), bool)
    for i, (g_, _, _) in enumerate(P):
        X[i] = np.isin(genes, g_)
    return X


topg = np.argsort(-pr)[:50]


def logor(X):
    out = []
    for a in range(X.shape[1]):
        for b in range(a + 1, X.shape[1]):
            n11 = (X[:, a] & X[:, b]).sum(); n10 = (X[:, a] & ~X[:, b]).sum(); n01 = (~X[:, a] & X[:, b]).sum(); n00 = (~X[:, a] & ~X[:, b]).sum()
            out.append(np.log((n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5))))
    return np.array(out)


res['M3'] = dict(cooccurrence_logOR_r=float(pearsonr(logor(presence(R, topg)), logor(presence(Y, topg)))[0]))


# M4 / M5
def timed_table(P):
    pid = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(P)])
    return pid, np.concatenate([p[0] for p in P]), np.concatenate([p[1] for p in P]), np.concatenate([p[2] for p in P])


tr_, ty_ = timed_table(R), timed_table(Y)
pur = D.pur[ref]
share = lambda T, m=None: float((T[3][m] >= 0).mean()) if m is not None else float((T[3] >= 0).mean())
res['M4'] = dict(timed_share_of_alterations=[share(tr_), share(ty_)],
                 patients_with_timed=[float(np.mean([np.any(p[2] >= 0) for p in R])), float(np.mean([np.any(p[2] >= 0) for p in Y]))],
                 by_purity={str(D.pur_labels[c]): [share(tr_, (pur == c)[tr_[0]]) if (pur == c).any() else None, share(ty_, (pur == c)[ty_[0]]) if (pur == c).any() else None]
                            for c in range(len(D.pur_labels))})
br, by_ = tr_[3][tr_[3] >= 0], ty_[3][ty_[3] >= 0]
tg = [int(g) for g in np.argsort(-np.bincount(tr_[1][tr_[3] >= 0], minlength=NG))[:8]]
res['M5'] = dict(bin_KS=float(ks_2samp(br, by_).statistic), share_bin20=[float((br == 20).mean()), float((by_ == 20).mean())],
                 median_bin={GN[g]: [float(np.median(tr_[3][(tr_[1] == g) & (tr_[3] >= 0)])), float(np.median(ty_[3][(ty_[1] == g) & (ty_[3] >= 0)]))] for g in tg})


# M6
def gene_bin(P):
    """per patient: highest timed bin of each gene (-1 none)"""
    B = np.full((len(P), NG), -1, np.int64)
    for i, (g_, _, b_) in enumerate(P):
        m = b_ >= 0
        if m.any():
            np.maximum.at(B[i], g_[m], b_[m])
    return B


Br, By = gene_bin(R), gene_bin(Y)
pairs = {}
for ia in range(len(tg)):
    for ib in range(ia + 1, len(tg)):
        a, b = tg[ia], tg[ib]
        row = []
        for B in (Br, By):
            both = (B[:, a] >= 0) & (B[:, b] >= 0)
            d_ = B[both, a] - B[both, b]
            rs = np.abs(d_) >= 2
            row += [float((d_[rs] > 0).mean()) if rs.sum() >= 30 else None, int(rs.sum())]
        pairs[f'{GN[a]}>{GN[b]}'] = dict(share_A_higher_real=row[0], n_real=row[1], share_A_higher_synth=row[2], n_synth=row[3])
res['M6'] = pairs
if args.top50_dir is None:
    print('M7 skipped (no --top50-dir / $TOP50_DIR)')
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0)

# M7
Z = np.load(f'{args.top50_dir}/top50_states.npz', allow_pickle=True)
G50 = [str(x) for x in Z['genes']]
gi = {g: j for j, g in enumerate(GN)}
idx50 = np.array([gi[g] for g in G50])
FIT = np.load(f'{args.top50_dir}/top50_profile_fits.npz')
logth, logpi = FIT['logth_4'], FIT['logpi_4']
labels_saved = FIT['labels_4'].astype(int)
order = np.argsort(-np.bincount(labels_saved, minlength=4))                       # clusters by size, as in the figure


def states(P):
    """22 states per gene exactly as trees_clusters/all_genes/build_states_all.py: 0 none; bin+1 (highest timed); -1 untimed short variant only"""
    X = np.zeros((len(P), NG), np.int16)
    for i, (g_, c_, b_) in enumerate(P):
        un = (b_ < 0) & ~np.isin(ALT[c_], ['amplification', 'deletion'])
        X[i, g_[un]] = -1
        m = b_ >= 0
        if m.any():
            t = np.zeros(NG, np.int16)
            np.maximum.at(t, g_[m], (b_[m] + 1).astype(np.int16))
            X[i] = np.where(t > 0, t, X[i])
    return X[:, idx50]


def assign(St):
    keep = (St > 0).any(1)
    ii, jj = np.nonzero(St[keep] >= 0)
    X = sp.csr_matrix((np.ones(len(ii), np.float32), (ii, jj * 22 + St[keep][ii, jj])), shape=(int(keep.sum()), 50 * 22))
    lr = logpi[None, :] + np.asarray(X @ logth.T)
    post = np.exp(lr - logsumexp(lr, 1)[:, None])
    lab = post.argmax(1)
    return keep, np.array([int((lab == c).sum()) for c in order]) / max(1, len(lab)), np.array([(post.max(1))]).mean()


Sr, Sy = states(R), states(Y)
# audit of the state encoding + assignment: real patients that are in the saved 50-gene table must reproduce their saved states and labels
pos = {int(p): k for k, p in enumerate(Z['pid'])}
common = [(i, pos[int(p)]) for i, p in enumerate(ref) if int(p) in pos]
if common:
    a_, b_ = zip(*common)
    same_state = float(np.mean(np.all(Sr[list(a_)] == Z['state'][list(b_)], 1)))
    St = Sr[list(a_)]
    ii, jj = np.nonzero(St >= 0)
    Xc = sp.csr_matrix((np.ones(len(ii), np.float32), (ii, jj * 22 + St[ii, jj])), shape=(len(St), 50 * 22))
    lab = (logpi[None, :] + np.asarray(Xc @ logth.T)).argmax(1)
    res['M7_audit'] = dict(real_patients_in_saved_table=len(common), identical_states=same_state, identical_labels=float(np.mean(lab == labels_saved[list(b_)])))
kr, sr_, cr = assign(Sr)
ky, sy_, cy = assign(Sy)
res['M7'] = dict(patients_clustered=[int(kr.sum()), int(ky.sum())], cluster_share_real=sr_.round(4).tolist(), cluster_share_synth=sy_.round(4).tolist(),
                 mean_max_posterior=[float(cr), float(cy)])
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
