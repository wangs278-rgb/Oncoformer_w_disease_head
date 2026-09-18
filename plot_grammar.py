#!/usr/bin/env python3
"""#2  Lineage grammar - the disease head's genetic definition of each cancer lineage.

Reuses the ALREADY-COMPUTED gene x lineage perturbation matrices:
  ki/ki_results.npz : meandP[gene,lineage] = P_added - P_baseline  (SUFFICIENCY; +ve = adding
                      the gene raises that lineage)
  ko/ko_results.npz : meandP[gene,lineage] = P_knockout - P_baseline (NECESSITY; -ve = removing
                      the gene lowers that lineage)
Gene ROW order differs between the two files (each sorted by its own carrier count) but the
gene SET and the 426 sup_names are identical -> we index genes by NAME.

One publication figure (Arial), four panels:
  A  Sufficiency heatmap  (KI meandP)  rows=drivers, cols=lineages, clustered.
  B  Necessity  heatmap  (KO meandP)  SAME row/col order -> A vs B reads as "add vs remove".
  C  Toggle vs master-driver: per lineage, sufficiency of its top driver (x) vs that driver's
     necessity (y). Diagonal = true toggles (CDH1-like); high-x/low-y = master drivers whose
     removal erases rather than converts.
  D  Top drivers as a labeled lollipop: each gene that is the #1 sufficiency driver for >=1
     lineage, sorted by peak sufficiency; color = how many cancers it is the #1 driver for
     (specialist vs promiscuous).

    python plot_grammar.py [--n-lineages 34] [--per-lineage 2] [--max-genes 46]
Output: run8_disease_out/grammar/fig_lineage_grammar.{png,pdf}
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from adjustText import adjust_text
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

_arial = os.path.expanduser('~/.local/share/fonts/Arial.ttf')
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT = '/cv/home/wangs278/scratch/fmi/run8_disease_out'
GDIR = f'{OUT}/grammar'
os.makedirs(GDIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--n-lineages', type=int, default=34, help='top lineages (by patient count) to show')
ap.add_argument('--per-lineage', type=int, default=2, help='top sufficiency genes taken per lineage')
ap.add_argument('--max-genes', type=int, default=46, help='cap on distinct driver rows')
ap.add_argument('--min-carriers', type=int, default=300, help='necessity needs enough carriers to be meaningful')
args = ap.parse_args()

# ------------------------------------------------------------------- load matrices
ki = np.load(f'{OUT}/ki/ki_results.npz', allow_pickle=True)
ko = np.load(f'{OUT}/ko/ko_results.npz', allow_pickle=True)
sup = ki['sup_names'].astype(str)
assert (sup == ko['sup_names'].astype(str)).all(), 'sup_names misaligned'
gk = ki['genes'].astype(str); go = ko['genes'].astype(str)
assert set(gk) == set(go)
# align KO rows onto KI gene order
io = {g: i for i, g in enumerate(go)}
perm = np.array([io[g] for g in gk])
GENES = gk
KI = ki['meandP'].astype(float)                 # [316,426] sufficiency
KO = ko['meandP'].astype(float)[perm]           # [316,426] necessity, now gene-aligned to GENES
KIq = ki['qval'].astype(float); KOq = ko['qval'].astype(float)[perm]
carr = ki['carrier_count'].astype(float)        # carriers per gene (KI file)
print(f'loaded {KI.shape[0]} genes x {KI.shape[1]} lineages', flush=True)

# ------------------------------------------------------------------- pick lineages
# rank lineages by real-patient frequency (from the UMAP feature dump)
feat = np.load(f'{OUT}/umap/umap_features.npz', allow_pickle=True)
tt = feat['true_term'].astype(str)
from collections import Counter
freq = Counter(tt)
supset = set(sup)
lin_by_freq = [l for l, _ in freq.most_common() if l in supset]
# drop the "unknown/undifferentiated" umbrella classes - grammar is about defined lineages
def is_defined(s):
    s = s.lower()
    return not (('unknown primary' in s) or ('undifferentiated' in s) or ('(cup)' in s))
lin_sel = [l for l in lin_by_freq if is_defined(l)][:args.n_lineages]
lin_idx = np.array([list(sup).index(l) for l in lin_sel])
print(f'selected {len(lin_sel)} lineages', flush=True)

# ------------------------------------------------------------------- pick driver genes
# per selected lineage, take its top sufficiency genes (by KI meandP); union -> row set
gene_score = {}
for c in lin_idx:
    order = np.argsort(-KI[:, c])
    for g in order[:args.per_lineage]:
        gene_score[g] = max(gene_score.get(g, -9), KI[g, c])
gene_rows = sorted(gene_score, key=lambda g: -gene_score[g])[:args.max_genes]
gene_rows = np.array(gene_rows)
print(f'selected {len(gene_rows)} driver genes', flush=True)

Ski = KI[np.ix_(gene_rows, lin_idx)]            # [G,L] sufficiency
Sko = KO[np.ix_(gene_rows, lin_idx)]            # [G,L] necessity
row_names = [GENES[g] for g in gene_rows]
col_names = lin_sel

# cluster rows & cols on the sufficiency matrix (the "positive" definition)
def order_axis(M):
    if M.shape[0] < 3:
        return np.arange(M.shape[0])
    Z = linkage(pdist(M, metric='correlation'), method='average')
    return leaves_list(Z)
ro = order_axis(Ski)
co = order_axis(Ski.T)
Ski, Sko = Ski[np.ix_(ro, co)], Sko[np.ix_(ro, co)]
row_names = [row_names[i] for i in ro]
col_names = [col_names[i] for i in co]

def short(s, n=30):
    return s if len(s) <= n else s[:n - 1] + '…'
col_short = [short(c, 20) for c in col_names]

# ------------------------------------------------------------------- panels C & D data
# C: per lineage top driver -> (sufficiency, necessity)
Cx, Cy, Clab = [], [], []
for c in lin_idx:
    g = int(np.argmax(KI[:, c]))
    if carr[g] < args.min_carriers:
        continue
    Cx.append(KI[g, c]); Cy.append(-KO[g, c]); Clab.append((GENES[g], sup[c]))
Cx, Cy = np.array(Cx), np.array(Cy)

# D: driver promiscuity - how many of the SELECTED lineages a gene is the #1 sufficiency driver for
top1_of = [int(np.argmax(KI[:, c])) for c in lin_idx]
promisc = Counter(top1_of)
Dx, Dy, Dlab = [], [], []
for g, n in promisc.items():
    Dx.append(n); Dy.append(float(KI[g, lin_idx].max())); Dlab.append(GENES[g])

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(26, 24.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.46, wspace=0.16,
                      left=0.11, right=0.98, top=0.93, bottom=0.08)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

def heat(ax, M, vlim, cmap, title, letter, cbar_label):
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(col_short)))
    ax.set_xticklabels(col_short, rotation=45, ha='right', rotation_mode='anchor', fontsize=13)
    ax.set_yticks(range(len(row_names))); ax.set_yticklabels(row_names, fontsize=13)
    ax.set_title(title, fontsize=23, pad=10)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=30, fontweight='bold',
            va='bottom', ha='right')
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label(cbar_label, fontsize=16); cb.ax.tick_params(labelsize=14)

vA = np.percentile(np.abs(Ski), 98)
vB = np.percentile(np.abs(Sko), 98)
heat(axA, Ski, vA, 'RdBu_r', 'A   Sufficiency: ΔP when the gene is ADDED', 'A', 'mean ΔP (add)')
heat(axB, Sko, vB, 'RdBu_r', 'B   Necessity: ΔP when the gene is KNOCKED OUT', 'B', 'mean ΔP (knockout)')

# C toggle vs master
axC.axline((0, 0), slope=1, color='#b0b0b0', lw=1.0, ls='--', zorder=0)
axC.scatter(Cx, Cy, s=70, c='#2a78d6', alpha=0.8, linewidths=0, zorder=3)
axC.tick_params(labelsize=16)
# label the most extreme / most interesting points, then ggrepel-style de-overlap
imp = np.argsort(-(Cx + Cy))[:16]
texts = [axC.text(Cx[i], Cy[i], f'{Clab[i][0]}·{short(Clab[i][1],16)}', fontsize=13, alpha=0.95)
         for i in imp]
adjust_text(texts, x=Cx[imp], y=Cy[imp], ax=axC, expand=(1.25, 1.6),
            arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.7))
axC.set_xlabel('sufficiency of top driver  (ΔP when added)', fontsize=19)
axC.set_ylabel('necessity of that driver  (−ΔP when knocked out)', fontsize=19)
axC.set_title('C   Toggle vs master driver (per lineage top gene)', fontsize=23, pad=10)
axC.text(-0.02, 1.02, 'C', transform=axC.transAxes, fontsize=30, fontweight='bold', va='bottom', ha='right')
for sp in ('top', 'right'):
    axC.spines[sp].set_visible(False)
axC.text(0.97, 0.06, 'diagonal = true toggle\n(add creates / remove erases)', transform=axC.transAxes,
         fontsize=15, ha='right', va='bottom', color='#666666')

# D top drivers as a labeled lollipop: sorted by peak sufficiency; color = # cancers driven
order = np.argsort(Dy)                          # ascending -> strongest ends up at the top
gl = [Dlab[i] for i in order]; pk = [float(Dy[i]) for i in order]; ct = [int(Dx[i]) for i in order]
MAXROWS = 24
if len(order) > MAXROWS:
    gl, pk, ct = gl[-MAXROWS:], pk[-MAXROWS:], ct[-MAXROWS:]
def ccol(c):
    return '#2a78d6' if c == 1 else ('#eb6834' if c == 2 else '#e34948')
ypos = np.arange(len(gl))
for yi, p, c in zip(ypos, pk, ct):
    col = ccol(c)
    axD.hlines(yi, 0, p, color=col, lw=2.8, zorder=1)
    axD.plot(p, yi, 'o', ms=12, color=col, zorder=2)
    if c > 1:
        axD.text(p, yi, f'  ×{c}', va='center', ha='left', fontsize=13, color=col)
axD.set_yticks(ypos); axD.set_yticklabels(gl, fontsize=13)
axD.set_ylim(-0.6, len(gl) - 0.4)
axD.set_xlim(0, max(pk) * 1.16)
axD.tick_params(labelsize=16)
axD.set_xlabel('peak sufficiency  (max ΔP when the gene is ADDED)', fontsize=19)
axD.set_title('D   Top drivers ranked by sufficiency (color = # cancers it defines)', fontsize=22, pad=10)
axD.text(-0.02, 1.02, 'D', transform=axD.transAxes, fontsize=30, fontweight='bold', va='bottom', ha='right')
for sp in ('top', 'right'):
    axD.spines[sp].set_visible(False)
handles = [Line2D([0], [0], marker='o', color=ccol(k), lw=0, ms=11, label=lab)
           for k, lab in [(1, 'defines 1 cancer (specialist)'), (2, 'defines 2 cancers'),
                          (3, 'defines ≥3 cancers')]]
axD.legend(handles=handles, fontsize=13, frameon=False, loc='lower right')

fig.suptitle('The disease head\'s genetic grammar: which genes are sufficient (add) and necessary (knock out) for each cancer lineage',
             fontsize=24, y=0.985)
for ext in ('png', 'pdf'):
    fig.savefig(f'{GDIR}/fig_lineage_grammar.{ext}')
plt.close(fig)
print(f'wrote fig_lineage_grammar to {GDIR}', flush=True)

# ------------------------------------------------------------------- text audit
print('\n--- validity audit ---', flush=True)
print(f'sufficiency: median(|KI| over shown cells)={np.median(np.abs(Ski)):.3f}  '
      f'frac cells q<0.05 = {(KIq[np.ix_(gene_rows, lin_idx)]<0.05).mean():.2f}')
print(f'necessity : median(|KO| over shown cells)={np.median(np.abs(Sko)):.3f}  '
      f'frac cells q<0.05 = {(KOq[np.ix_(gene_rows, lin_idx)]<0.05).mean():.2f}')
print('per-lineage top driver (sufficiency, necessity):')
for i in np.argsort(-(Cx))[:12]:
    g, l = Clab[i]
    print(f'  {short(l,42):42s} {g:8s} suff={Cx[i]:+.3f} nec={Cy[i]:+.3f}')
