#!/usr/bin/env python3
"""Violin plot (Arial) of in-silico gene knock-IN ΔP for the top-influence genes.

x-axis : top genes (by mean TV influence)
y-axis : ΔP = P_added - P_baseline  (one value per disease lineage, mean over recipients)
violin : distribution of ΔP across all 426 supervised disease classes for that gene
dots   : one per lineage; colored by lineage for the set of top-RAISED lineages
         (each gene's single most-raised cancer), light-grey otherwise.
         Positive ΔP = adding the gene RAISES that cancer's probability.

Genes whose modal mutation is unrepresentative (modal_frac < FLAG) are marked with a
dagger (†) — for these dispersed-spectrum genes the injected hotspot is only a small
slice of how the gene is really mutated.

Reads run8_disease_out/ki/ki_results.npz (no GPU). Emits PDF+PNG to ki/figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

_arial = os.path.expanduser('~/.local/share/fonts/Arial.ttf')
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

KI_DIR = '/cv/home/wangs278/scratch/fmi/run8_disease_out/ki'
FIG = f'{KI_DIR}/figures'
os.makedirs(FIG, exist_ok=True)
d = np.load(f'{KI_DIR}/ki_results.npz', allow_pickle=True)
genes = d['genes']
meandP = d['meandP']
tv = d['influence_tv']
modal_frac = d['modal_frac'] if 'modal_frac' in d.files else np.ones(len(genes))
dnames = np.array([str(x) for x in d['sup_names']])
G, S = meandP.shape

TOPN = 25
FLAG = 0.20                                     # modal_frac below this -> dagger
o = np.argsort(-tv)[:TOPN]
gsel = genes[o]
mfrac = modal_frac[o]
M = meandP[o]

def short(s, n=30):
    return s if len(s) <= n else s[:n - 1] + '…'

# top-RAISED lineage per selected gene (mirror of KO's most-lowered)
top_line_idx = []
for r in range(TOPN):
    j = int(np.argmax(M[r]))
    if j not in top_line_idx:
        top_line_idx.append(j)
palette = ['#e6194B', '#3cb44b', '#f58231', '#4363d8', '#911eb4', '#42d4f4',
           '#f032e6', '#bfef45', '#fabed4', '#469990', '#9A6324', '#800000',
           '#000075', '#808000', '#e6ab02', '#1b9e77', '#d95f02', '#7570b3']
color_of = {j: palette[k % len(palette)] for k, j in enumerate(top_line_idx)}
top_set = set(top_line_idx)

fig, ax = plt.subplots(figsize=(0.66 * TOPN + 3.0, 7.2))
x = np.arange(TOPN)
parts = ax.violinplot([M[r] for r in range(TOPN)], positions=x,
                      widths=0.82, showextrema=False, showmeans=False)
for b in parts['bodies']:
    b.set_facecolor('#dfe6ee'); b.set_edgecolor('#9aa7b4')
    b.set_alpha(0.9); b.set_linewidth(0.6)

rng = np.random.default_rng(0)
for r in range(TOPN):
    vals = M[r]
    jit = rng.uniform(-0.30, 0.30, size=S)
    is_top = np.array([j in top_set for j in range(S)])
    ax.scatter(x[r] + jit[~is_top], vals[~is_top], s=5, c='#c9c9c9',
               alpha=0.30, linewidths=0, zorder=2)
    tj = np.where(is_top)[0]
    ax.scatter(x[r] + jit[tj], vals[tj], s=22, c=[color_of[j] for j in tj],
               alpha=0.95, edgecolors='white', linewidths=0.4, zorder=4)

ax.axhline(0, color='k', lw=0.7, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f'{g}†' if mf < FLAG else g for g, mf in zip(gsel, mfrac)],
                   rotation=90, fontsize=8)
ax.set_ylabel('ΔP  =  P$_{knock-in}$ − P$_{baseline}$   (per disease lineage)', fontsize=9)
ax.set_title('In-silico gene knock-IN: per-lineage ΔP distribution across the top-influence genes\n'
             'each dot = one of 426 disease classes; colored dots = the top-raised cancers '
             '(positive = adding the gene raises that cancer).  † modal mutation <20% representative',
             fontsize=10)
ax.tick_params(axis='y', labelsize=8)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.set_xlim(-0.7, TOPN - 0.3)

handles = [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                  markerfacecolor=color_of[j], markeredgecolor='white',
                  markeredgewidth=0.4, label=short(dnames[j], 34))
           for j in top_line_idx]
ax.legend(handles=handles, title='Top-raised cancer lineages',
          fontsize=6.6, title_fontsize=7.5, loc='center left',
          bbox_to_anchor=(1.005, 0.5), frameon=False, labelspacing=0.35)

fig.savefig(f'{FIG}/fig_ki_gene_violin.pdf')
fig.savefig(f'{FIG}/fig_ki_gene_violin.png')
plt.close(fig)
print(f'wrote fig_ki_gene_violin (PDF+PNG) — {TOPN} genes, {S} lineages, '
      f'{len(top_line_idx)} highlighted lineages')
