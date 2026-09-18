#!/usr/bin/env python3
"""Publication figures (Arial) for the in-silico gene-knockout disease analysis.
Reads run8_disease_out/ko/ko_results.npz (no GPU). Emits PDF+PNG to ko/figures/.

Figures:
  1  fig1_gene_disease_heatmap  - clustered signed dP heatmap (top genes x top-affected lineages)
  2  fig2_gene_influence        - genes ranked by overall knockout influence (mean TV distance)
  3  fig3_driver_sanity         - recovered gene->lineage vs textbook driver biology
  4  fig4_gene_topbars          - per-gene diverging top lowered/raised lineages (exemplars)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.cluster.hierarchy import linkage, leaves_list

# ---- Arial ----
_arial = os.path.expanduser('~/.local/share/fonts/Arial.ttf')
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

KO_DIR = '/cv/home/wangs278/scratch/fmi/run8_disease_out/ko'
FIG    = f'{KO_DIR}/figures'
os.makedirs(FIG, exist_ok=True)
d = np.load(f'{KO_DIR}/ko_results.npz', allow_pickle=True)
meta = json.load(open(f'{KO_DIR}/ko_meta.json'))

genes   = d['genes'];        meandP = d['meandP']       # [G, S]
qval    = d['qval'];         n_used = d['n_used']
tv      = d['influence_tv']; flip = d['flip_rate']
dnames  = d['sup_names']
G, S = meandP.shape
print(f'loaded {G} genes x {S} lineages; {meta["n_pairs"]} knockouts')

def short(s, n=34):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + '…'

def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png')
    plt.close(fig); print('wrote', name)

# =========================================================== Fig 1: clustered heatmap
TOPG, TOPD = min(40, G), 32
g_ord = np.argsort(-tv)[:TOPG]
col_score = np.abs(meandP[g_ord]).max(0)
d_ord = np.argsort(-col_score)[:TOPD]
M = meandP[np.ix_(g_ord, d_ord)]                       # [TOPG, TOPD]
# cluster
gr = leaves_list(linkage(M, method='average', metric='euclidean')) if TOPG > 2 else np.arange(TOPG)
dr = leaves_list(linkage(M.T, method='average', metric='euclidean')) if TOPD > 2 else np.arange(TOPD)
M = M[np.ix_(gr, dr)]
rg = [genes[g_ord[i]] for i in gr]
cd = [short(dnames[d_ord[i]]) for i in dr]
Q = qval[np.ix_(g_ord, d_ord)][np.ix_(gr, dr)]
vmax = np.percentile(np.abs(M), 99); vmax = max(vmax, 1e-3)

fig, ax = plt.subplots(figsize=(0.42 * TOPD + 4.0, 0.36 * TOPG + 2.5))
im = ax.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(TOPD)); ax.set_xticklabels(cd, rotation=90, fontsize=10)
ax.set_yticks(range(TOPG)); ax.set_yticklabels(rg, fontsize=10)
ax.tick_params(length=3, width=0.8)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if Q[i, j] < 0.05 and abs(M[i, j]) >= 0.3 * vmax:
            ax.text(j, i, '*', ha='center', va='center', fontsize=11,
                    fontweight='bold', color='k')
ax.set_title('In-silico gene knockout: ΔP(disease) landscape\n'
             'blue = removal lowers lineage (gene supports it)   *FDR<0.05',
             fontsize=14)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label('mean ΔP', fontsize=12)
cb.ax.tick_params(labelsize=10)
save(fig, 'fig1_gene_disease_heatmap')

# =========================================================== Fig 2: influence ranking
TOPN = min(30, G)
o = np.argsort(-tv)[:TOPN][::-1]
fig, ax = plt.subplots(figsize=(5.2, 0.24 * TOPN + 1.0))
y = np.arange(TOPN)
bars = ax.barh(y, tv[o], color=plt.cm.viridis(flip[o] / max(flip.max(), 1e-9)), edgecolor='none')
ax.set_yticks(y); ax.set_yticklabels([f'{genes[i]}' for i in o], fontsize=7)
ax.set_xlabel('mean total-variation distance  |P$_{KO}$ − P$_{base}$|', fontsize=8)
ax.set_title('Gene knockout influence on disease head', fontsize=9)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, max(flip.max(), 1e-9)))
cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02); cb.set_label('argmax-flip rate', fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.tick_params(axis='x', labelsize=7)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
save(fig, 'fig2_gene_influence')

# =========================================================== Fig 3: driver sanity panel
EXPECTED = {  # gene -> keywords that should appear in the lineage most-lowered by its knockout
    'EGFR': ['lung'], 'KRAS': ['pancrea', 'lung', 'colon', 'colorect'],
    'BRAF': ['melanoma', 'thyroid'], 'APC': ['colon', 'colorect'],
    'VHL': ['renal', 'kidney'], 'IDH1': ['glioma', 'astrocyt', 'glial'],
    'PTEN': ['endometr', 'uter', 'prostate'], 'CDKN2A': ['melanoma', 'pancrea'],
    'NF1': ['nerve', 'neurofibro', 'glioma'], 'GNAS': ['pancrea', 'appendix', 'pituit'],
    'CTNNB1': ['hepato', 'liver', 'endometr'], 'KIT': ['gist', 'stromal', 'melanoma'],
    'CDH1': ['breast', 'gastric', 'stomach'], 'SMAD4': ['pancrea', 'colon'],
    'RB1': ['retino', 'small cell', 'lung'], 'MTAP': ['pancrea', 'mesotheli'],
    'STK11': ['lung'], 'KEAP1': ['lung'], 'NFE2L2': ['lung', 'esoph'],
    'ATRX': ['glioma', 'astrocyt'], 'SPOP': ['prostate'], 'FOXA1': ['prostate', 'breast'],
    'AR': ['prostate'], 'GATA3': ['breast'], 'ESR1': ['breast'], 'PBRM1': ['renal', 'kidney'],
}
rows = []
for g, kws in EXPECTED.items():
    idx = np.where(genes == g)[0]
    if not idx.size: continue
    gi = idx[0]; dp = meandP[gi]
    cols = [j for j, nm in enumerate(dnames) if any(k in str(nm).lower() for k in kws)]
    if not cols: continue
    jbest = cols[int(np.argmin(dp[cols]))]               # expected lineage most-lowered
    rank = int((dp < dp[jbest]).sum()) + 1               # rank among all lowered (1=most lowered)
    rows.append((g, dp[jbest], rank, short(dnames[jbest], 26), n_used[gi]))
rows.sort(key=lambda r: r[1])                            # most-negative first
if rows:
    gg = [r[0] for r in rows]; vv = [r[1] for r in rows]; rk = [r[2] for r in rows]; ln = [r[3] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(rows) + 1.0))
    cols = ['#2166ac' if r <= 3 else ('#92c5de' if r <= 10 else '#bbbbbb') for r in rk]
    ax.barh(y, vv, color=cols, edgecolor='none')
    ax.set_yticks(y); ax.set_yticklabels(gg, fontsize=7.5)
    vmin = min(vv)
    gutter0 = -vmin * 0.03                        # labels start just right of the 0 line
    ax.set_xlim(vmin * 1.10, -vmin * 1.40)        # right gutter holds the lineage labels (no y-label collision)
    ax.axvline(0, color='k', lw=0.6)
    for i, (v, r, l) in enumerate(zip(vv, rk, ln)):
        ax.text(gutter0, i, f'{l}  (rank {r})', va='center', ha='left', fontsize=6.2,
                color='k' if r <= 10 else '#888888')
    ax.set_xlabel('mean ΔP on expected lineage (negative = knockout lowers it, as expected)', fontsize=8)
    ax.set_title('Known-driver validation: does knockout lower the textbook lineage?', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    save(fig, 'fig3_driver_sanity')
else:
    print('fig3 skipped: no expected drivers matched')

# =========================================================== Fig 4: per-gene top bars
EX = [g for g in ['EGFR', 'BRAF', 'APC', 'VHL', 'IDH1', 'KRAS', 'CDH1', 'KIT', 'GNAS']
      if g in set(genes.tolist())][:9]
if EX:
    nr = (len(EX) + 2) // 3
    fig, axes = plt.subplots(nr, 3, figsize=(15, 2.6 * nr))
    axes = np.atleast_1d(axes).ravel()
    for k, g in enumerate(EX):
        gi = np.where(genes == g)[0][0]; dp = meandP[gi]
        low = np.argsort(dp)[:6]                          # most lowered (neg)
        high = np.argsort(-dp)[:3]                         # most raised (pos)
        sel = list(low) + list(high[::-1])
        vals = dp[sel]; labs = [short(dnames[j], 24) for j in sel]
        ax = axes[k]; y = np.arange(len(sel))
        ax.barh(y, vals, color=['#2166ac' if v < 0 else '#b2182b' for v in vals], edgecolor='none')
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6)
        ax.axvline(0, color='k', lw=0.6); ax.invert_yaxis()
        ax.set_title(f'{g}  (n={n_used[gi]})', fontsize=9)
        ax.tick_params(axis='x', labelsize=6.5)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for k in range(len(EX), len(axes)): axes[k].axis('off')
    fig.suptitle('Per-gene knockout: top lowered (blue) and raised (red) lineages  —  mean ΔP',
                 fontsize=11, y=1.0)
    fig.tight_layout()
    save(fig, 'fig4_gene_topbars')
else:
    print('fig4 skipped: no exemplar genes present')

print('all figures in', FIG)
