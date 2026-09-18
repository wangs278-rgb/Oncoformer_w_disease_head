#!/usr/bin/env python3
"""#4  Minimal sufficient signature - one publication figure (Arial), panels A-D.

Reads minsig/minsig_results.npz (from insilico_minsig.py).
  A  climb-with-band staircases (small multiples): mean P(target) as genes are greedily added,
     with the population 10-90% band, per-step agreement %, and the stop step marked.
  B  consensus recipe matrix: lineages x genes, cell = order added, opacity = agreement.
  C  convergence: recipe stability (Jaccard vs full-N recipe) as N backgrounds grows.
  D  branching for one flagship lineage: per step, which gene each background chose (top-4).

    python plot_minsig.py
Output: run8_disease_out/minsig/fig_minsig.{png,pdf}
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

_arial = os.path.expanduser('~/.local/share/fonts/Arial.ttf')
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT = '/cv/home/wangs278/scratch/fmi/run8_disease_out'
MS = f'{OUT}/minsig'
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='', help="'' for sufficiency run, '_nec' for necessity run")
pa = ap.parse_args()
TAG = pa.tag

d = np.load(f'{MS}/minsig_results{TAG}.npz', allow_pickle=True)
targets = d['targets'].astype(str)
CANDS = str(d['candidates']) if 'candidates' in d.files else 'suff'
recipe_name = d['recipe_name']; p_q10 = d['p_q10']; p_q50 = d['p_q50']; p_q90 = d['p_q90']
agree = d['agree']; argmax_frac = d['argmax_frac']; stop_step = d['stop_step']
ref_median = d['ref_median']; branch = d['branch']
pb10 = d['p_base_q10']; pb50 = d['p_base_q50']; pb90 = d['p_base_q90']
ci_lo = d['ci_lo']; ci_hi = d['ci_hi']; cib_lo = d['ci_base_lo']; cib_hi = d['ci_base_hi']
conv_t = d['conv_targets'].astype(str) if d['conv_targets'].size else np.array([])
conv_N = d['conv_Ngrid']; conv_J = d['conv_jaccard']
nT, nS = p_q50.shape
print(f'{nT} lineages, {nS} steps', flush=True)


def short(s, n=26):
    return s if len(s) <= n else s[:n - 1] + '…'


fig = plt.figure(figsize=(27, 30))
outer = fig.add_gridspec(3, 1, height_ratios=[1.32, 1.32, 1.0], hspace=0.50,
                         left=0.06, right=0.985, top=0.905, bottom=0.045)
nA = min(8, nT)
x = np.arange(0, nS + 1)                                       # 0 = baseline (no genes)


def draw_climbs(sub_gs, lo_base, lo_step, hi_base, hi_step, show_agree):
    """2x4 small-multiple staircases; ribbon runs from (lo_base|lo_step) to (hi_base|hi_step).
    Returns the list of axes so the caller can place a header above the grid."""
    axs = []
    for i in range(nA):
        ax = fig.add_subplot(sub_gs[i // 4, i % 4])
        axs.append(ax)
        q50 = np.concatenate([[pb50[i]], p_q50[i]])
        lo = np.concatenate([[lo_base[i]], lo_step[i]])
        hi = np.concatenate([[hi_base[i]], hi_step[i]])
        ax.fill_between(x, lo, hi, color=PAL[0], alpha=0.20, linewidth=0)
        ax.plot(x, q50, '-o', color=PAL[0], ms=6, lw=2.2)
        if not np.isnan(ref_median[i]):
            ax.axhline(ref_median[i], color='#888888', ls='--', lw=1.1)
            ax.text(nS, ref_median[i], ' real', color='#888888', fontsize=10, va='center')
        ss = int(stop_step[i])
        if 1 <= ss <= nS:
            ax.axvline(ss, color=PAL[1], ls=':', lw=1.6)
        for s in range(nS):
            ax.annotate(f'+{recipe_name[i][s]}', (x[s + 1], q50[s + 1]), fontsize=12, rotation=40,
                        xytext=(3, 5), textcoords='offset points', color='#222222')
            if show_agree:
                yb = np.concatenate([[pb10[i]], p_q10[i]])[s + 1]
                ax.annotate(f'{int(round(agree[i][s]*100))}%', (x[s + 1], yb), fontsize=8,
                            xytext=(0, -11), textcoords='offset points', ha='center', color='#999999')
        _hi = float(np.nanmax(np.concatenate([[pb90[i]], p_q90[i]])))
        ax.set_ylim(0, min(1.18, max(0.42, _hi * 1.45)))      # headroom so gene labels clear the title
        ax.set_xlim(-0.4, nS + 0.9); ax.set_xticks(x)
        ax.set_title(short(targets[i], 30), fontsize=14, pad=8)
        ax.tick_params(labelsize=11)
        if i % 4 == 0:
            ax.set_ylabel('P(target)', fontsize=14)
        if i // 4 == 1:
            ax.set_xlabel('genes added (greedy)', fontsize=14)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    return axs


def row_header(axs, letter_ch, subtitle):
    """Place the panel letter + subtitle in the clear band ABOVE the grid's top row of subplots."""
    top = max(axs[j].get_position().y1 for j in range(4))      # top edge of the top-row subplots
    y = top + 0.030                                            # clears the subplot titles
    fig.text(0.055, y, letter_ch, fontsize=26, fontweight='bold', ha='right', va='center')
    fig.text(0.52, y, subtitle, fontsize=15, ha='center', va='center')


# ---------- A: population spread (10–90% of tumors)
gsA = outer[0].subgridspec(2, 4, hspace=0.85, wspace=0.32)
axsA = draw_climbs(gsA, pb10, p_q10, pb90, p_q90, show_agree=True)
row_header(axsA, 'A', 'Climb per lineage — band = 10–90% spread ACROSS tumors (population heterogeneity)   ·   %=agreement, dotted=stop, dashed=real-patient level')

# ---------- E: 95% CI of the median (shrinks with N)
gsE = outer[1].subgridspec(2, 4, hspace=0.85, wspace=0.32)
axsE = draw_climbs(gsE, cib_lo, ci_lo, cib_hi, ci_hi, show_agree=False)
row_header(axsE, 'E', 'Same climbs — band = 95% bootstrap CI of the MEDIAN (precision of the typical trajectory; tightens with N)')

# ---------- B recipe matrix, C convergence, D branching
gsB = outer[2].subgridspec(1, 3, width_ratios=[3.9, 1.0, 1.1], wspace=0.22)
axB = fig.add_subplot(gsB[0, 0]); axC = fig.add_subplot(gsB[0, 1]); axD = fig.add_subplot(gsB[0, 2])


def letter(ax, ch):
    ax.text(-0.06, 1.03, ch, transform=ax.transAxes, fontsize=25, fontweight='bold', va='bottom', ha='right')


# B: rows = lineages (all), cols = union of genes appearing within stop_step
nB = min(nT, 22)
gene_cols = []
for i in range(nB):
    for s in range(int(stop_step[i])):
        g = recipe_name[i][s]
        if g not in gene_cols:
            gene_cols.append(g)
col_ix = {g: j for j, g in enumerate(gene_cols)}
import matplotlib.colors as mcolors
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=1, vmax=max(2, int(stop_step[:nB].max())))
for i in range(nB):
    for s in range(int(stop_step[i])):
        g = recipe_name[i][s]; j = col_ix[g]
        rgba = list(cmap(norm(s + 1)))
        rgba[3] = 0.35 + 0.65 * float(agree[i][s])            # opacity encodes agreement
        axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=rgba, edgecolor='white', lw=0.5))
        axB.text(j, i, str(s + 1), ha='center', va='center', fontsize=9,
                 color='white' if norm(s + 1) < 0.6 else 'black')
axB.set_xlim(-0.5, len(gene_cols) - 0.5); axB.set_ylim(nB - 0.5, -0.5)
axB.set_xticks(range(len(gene_cols)))
axB.set_xticklabels(gene_cols, rotation=45, ha='right', rotation_mode='anchor', fontsize=11)
axB.set_yticks(range(nB)); axB.set_yticklabels([short(targets[i], 34) for i in range(nB)], fontsize=11)
axB.tick_params(length=0)
axB.set_title('B   Consensus recipes (number = order added, opacity = agreement)', fontsize=17, pad=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = fig.colorbar(sm, ax=axB, fraction=0.03, pad=0.01); cb.set_label('order added', fontsize=13)
cb.ax.tick_params(labelsize=12)
letter(axB, 'B')

# C: convergence
if conv_t.size:
    for k, t in enumerate(conv_t):
        axC.plot(list(conv_N[k]), list(conv_J[k]), '-o', ms=8, lw=2.2, color=PAL[k % len(PAL)], label=short(t, 24))
    axC.legend(fontsize=12, frameon=False, loc='lower right')
    axC.set_ylim(0, 1.05)
axC.tick_params(labelsize=13)
axC.set_xlabel('# background patients (N)', fontsize=16)
axC.set_ylabel('recipe Jaccard vs full-N', fontsize=16)
axC.set_title('C   Convergence: recipe stabilises as N grows', fontsize=17, pad=8)
for sp in ('top', 'right'):
    axC.spines[sp].set_visible(False)
letter(axC, 'C')

# D: branching for one flagship (first target)
fi = 0
ss = int(stop_step[fi]) if stop_step[fi] >= 1 else nS
steps = list(range(min(ss, nS)))
gene_color = {}
cyc = 0
for s in steps:
    bottom = 0.0
    for (g, frac) in branch[fi][s]:
        if g not in gene_color:
            gene_color[g] = PAL[cyc % len(PAL)]; cyc += 1
        axD.bar(s + 1, frac, bottom=bottom, width=0.7, color=gene_color[g], edgecolor='white', lw=0.4)
        if frac > 0.08:
            axD.text(s + 1, bottom + frac / 2, g, ha='center', va='center', fontsize=10,
                     color='white')
        bottom += frac
axD.set_xticks([s + 1 for s in steps])
axD.tick_params(labelsize=13)
axD.set_xlabel('greedy step', fontsize=16)
axD.set_ylabel('fraction of backgrounds picking gene', fontsize=15)
axD.set_ylim(0, 1)
axD.set_title(f'D   Per-step branching · {short(targets[fi], 24)}', fontsize=17, pad=8)
for sp in ('top', 'right'):
    axD.spines[sp].set_visible(False)
letter(axD, 'D')

_pool = 'sufficiency-gene' if CANDS == 'suff' else 'necessity-gene'
fig.suptitle(f'Minimal sufficient signature ({_pool} candidates): the shortest genomic recipe the disease head needs to call each cancer lineage',
             fontsize=20, y=0.972)
for ext in ('png', 'pdf'):
    fig.savefig(f'{MS}/fig_minsig{TAG}.{ext}')
plt.close(fig)
print(f'wrote fig_minsig{TAG} to {MS}', flush=True)

# ------- text audit
print('\n--- validity audit ---', flush=True)
for i in range(nT):
    ss = int(stop_step[i])
    rec = '→'.join(recipe_name[i][:ss])
    print(f'  {short(targets[i],40):40s} [{ss}] {rec:50s} P {p_q50[i][0]:.2f}→{p_q50[i][ss-1]:.2f} '
          f'(real {ref_median[i]:.2f}) agree1={agree[i][0]:.2f} argmax@stop={argmax_frac[i][ss-1]:.2f}')
