#!/usr/bin/env python3
"""#5  Dose / VAF response - one publication figure (Arial), panels A-D.

Reads dose/dose_results.npz (from insilico_dose.py).
  A  VAF dose-response small multiples: median P(target) vs VAF level (1..10) with the 10-90%
     population band; baseline (no insertion) and real-patient level marked. One mini per gene.
  B  Two-hit test: heterozygous -> homozygous median P(target), dumbbell per gene, grouped by
     oncogene vs tumour suppressor.
  C  VAF dose slope per gene = P(highest VAF) - P(lowest VAF), sorted, coloured by class.
  D  Specificity: on-target VAF effect vs the largest off-target VAF effect (below y=x = specific).

    python plot_dose.py
Output: run8_disease_out/dose/fig_dose.{png,pdf}
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

OUT = '/cv/home/wangs278/scratch/fmi/run8_disease_out'
DS = f'{OUT}/dose'
C_ONCO, C_TSG = '#2a78d6', '#eb6834'          # colourblind-safe blue / orange
def ccol(k): return C_ONCO if k == 'oncogene' else C_TSG

d = np.load(f'{DS}/dose_results.npz', allow_pickle=True)
genes = d['genes'].astype(str)
gclass = d['gclass'].astype(str)
target = d['target'].astype(str)
ki = d['ki'].astype(float)
base = d['base_p50'].astype(float)
ref = d['ref'].astype(float)
vaf_level = d['vaf_level']; vaf_q10 = d['vaf_q10']; vaf_q50 = d['vaf_q50']; vaf_q90 = d['vaf_q90']
vaf_prof_lo = d['vaf_prof_lo']; vaf_prof_hi = d['vaf_prof_hi']
zyg_label = d['zyg_label']; zyg_q50 = d['zyg_q50']; zyg_lo = d['zyg_ci_lo']; zyg_hi = d['zyg_ci_hi']
sup_names = list(d['sup_names'].astype(str))
name2col = {s: i for i, s in enumerate(sup_names)}
nG = len(genes)
print(f'{nG} genes', flush=True)


def short(s, n=24):
    return s if len(s) <= n else s[:n - 1] + '…'


fig = plt.figure(figsize=(26, 24))
outer = fig.add_gridspec(2, 1, height_ratios=[1.30, 1.0], hspace=0.30,
                         left=0.055, right=0.985, top=0.905, bottom=0.055)
fig.suptitle('Dose response of the disease head: does it read HOW STRONG a mutation is, '
             'not just whether it is present?', fontsize=25, y=0.965, fontweight='bold')

# ================================================================= Panel A: VAF small multiples
ncol = 5
nrow = int(np.ceil(nG / ncol))
gsA = outer[0].subgridspec(nrow, ncol, hspace=0.72, wspace=0.30)
axsA = []
for i in range(nG):
    ax = fig.add_subplot(gsA[i // ncol, i % ncol])
    axsA.append(ax)
    col = ccol(gclass[i])
    lv = np.array(vaf_level[i], dtype=float)
    q10 = np.array(vaf_q10[i], dtype=float); q50 = np.array(vaf_q50[i], dtype=float)
    q90 = np.array(vaf_q90[i], dtype=float)
    if lv.size:
        ax.fill_between(lv, q10, q90, color=col, alpha=0.16, linewidth=0)
        ax.plot(lv, q50, '-o', color=col, lw=2.4, ms=6, zorder=3)
    ax.axhline(base[i], ls=':', lw=1.6, color='#888888')          # baseline (no insertion)
    if np.isfinite(ref[i]):
        ax.axhline(ref[i], ls='--', lw=1.6, color='#c0392b', alpha=0.8)   # real-patient level
    ax.set_xlim(0.5, 10.5); ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_ylim(0, max(0.35, (q90.max() if lv.size else 0.3) * 1.18, ref[i] * 1.1 if np.isfinite(ref[i]) else 0))
    ax.tick_params(labelsize=13)
    ax.set_title(f'{genes[i]} → {short(target[i], 22)}', fontsize=14.5, color=col, pad=5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if i % ncol == 0:
        ax.set_ylabel('P(target)', fontsize=14)
    if i // ncol == nrow - 1:
        ax.set_xlabel('VAF level (1=subclonal → 10=clonal)', fontsize=13)
# header + legend for A
topA = max(a.get_position().y1 for a in axsA)
fig.text(0.055, topA + 0.028, 'A', fontsize=30, fontweight='bold', va='bottom')
fig.text(0.30, topA + 0.030, 'VAF dose-response per driver — band = 10–90% across 1,000 tumours',
         fontsize=17, va='bottom')
legA = [Line2D([0], [0], color='#333', lw=2.4, marker='o', ms=6, label='median P(target) vs VAF'),
        Line2D([0], [0], color='#888888', ls=':', lw=1.6, label='baseline (no insertion)'),
        Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label='real-patient level'),
        Line2D([0], [0], color=C_ONCO, lw=6, label='oncogene / identity TF'),
        Line2D([0], [0], color=C_TSG, lw=6, label='tumour suppressor')]
fig.legend(handles=legA, loc='upper right', bbox_to_anchor=(0.985, topA + 0.052),
           ncol=5, fontsize=13, frameon=False)

# ================================================================= bottom row: B, C, D
gsB = outer[1].subgridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.28)

# ---- Panel B: two-hit dumbbell (het -> hom), grouped by class
axB = fig.add_subplot(gsB[0, 0])
zg = [i for i in range(nG) if len(zyg_label[i]) == 2]          # need both het & hom
# order: suppressors first (grouped), then oncogenes; within group by hom-het effect
def zdelta(i):
    lab = list(zyg_label[i]); q = list(zyg_q50[i])
    h = dict(zip(lab, q)); return h.get('hom', np.nan) - h.get('het', np.nan)
zg = sorted(zg, key=lambda i: (gclass[i] != 'suppressor', zdelta(i)))
for y, i in enumerate(zg):
    lab = list(zyg_label[i]); q = list(zyg_q50[i])
    h = dict(zip(lab, q)); het, hom = h['het'], h['hom']
    col = ccol(gclass[i])
    axB.plot([het, hom], [y, y], '-', color=col, lw=2.2, zorder=1)
    axB.scatter([het], [y], s=70, facecolor='white', edgecolor=col, linewidth=2, zorder=2)
    axB.scatter([hom], [y], s=90, facecolor=col, edgecolor=col, zorder=2)
    axB.annotate('', xy=(hom, y), xytext=(het, y),
                 arrowprops=dict(arrowstyle='-|>', color=col, lw=0), zorder=1)
axB.set_yticks(range(len(zg)))
axB.set_yticklabels([f'{genes[i]}' for i in zg], fontsize=13)
for tick, i in zip(axB.get_yticklabels(), zg):
    tick.set_color(ccol(gclass[i]))
axB.set_xlabel('median P(target)', fontsize=14)
axB.set_title('B   Two-hit test: heterozygous → homozygous', fontsize=16, loc='left', pad=8)
axB.tick_params(labelsize=12)
for sp in ('top', 'right'):
    axB.spines[sp].set_visible(False)
axB.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                           markeredgecolor='#333', ms=9, label='heterozygous'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#333',
                           markeredgecolor='#333', ms=9, label='homozygous')],
           fontsize=12, loc='lower right', frameon=False)

# ---- Panel C: VAF slope per gene = P(high VAF) - P(low VAF)
axC = fig.add_subplot(gsB[0, 1])
slope = np.array([(vaf_q50[i][-1] - vaf_q50[i][0]) if len(vaf_q50[i]) >= 2 else np.nan
                  for i in range(nG)])
ok = np.where(np.isfinite(slope))[0]
ordc = ok[np.argsort(slope[ok])]
yv = np.arange(len(ordc))
axC.barh(yv, slope[ordc], color=[ccol(gclass[i]) for i in ordc], alpha=0.9)
axC.axvline(0, color='#333', lw=0.8)
axC.set_yticks(yv); axC.set_yticklabels([genes[i] for i in ordc], fontsize=13)
for tick, i in zip(axC.get_yticklabels(), ordc):
    tick.set_color(ccol(gclass[i]))
axC.set_xlabel('ΔP(target): clonal − subclonal', fontsize=14)
axC.set_title('C   Dose sensitivity (VAF slope)', fontsize=16, loc='left', pad=8)
axC.tick_params(labelsize=12)
for sp in ('top', 'right'):
    axC.spines[sp].set_visible(False)

# ---- Panel D: specificity - on-target vs largest off-target VAF effect
axD = fig.add_subplot(gsB[0, 2])
on, off, di = [], [], []
for i in range(nG):
    plo, phi = vaf_prof_lo[i], vaf_prof_hi[i]
    if plo is None or phi is None or target[i] not in name2col:
        continue
    plo = np.asarray(plo, dtype=float); phi = np.asarray(phi, dtype=float)
    c = name2col[target[i]]
    dprof = phi - plo
    on_t = dprof[c]
    off_mask = np.ones_like(dprof, dtype=bool); off_mask[c] = False
    off_t = dprof[off_mask].max()
    on.append(on_t); off.append(off_t); di.append(i)
on, off = np.array(on), np.array(off)
for k, i in enumerate(di):
    axD.scatter(on[k], off[k], s=90, color=ccol(gclass[i]), edgecolor='white', linewidth=0.8, zorder=3)
    axD.annotate(genes[i], (on[k], off[k]), fontsize=11, xytext=(4, 3),
                 textcoords='offset points', color=ccol(gclass[i]))
lim = max(0.05, np.nanmax(np.abs(np.concatenate([on, off]))) * 1.15) if len(on) else 0.3
axD.plot([-lim, lim], [-lim, lim], ls='--', color='#999', lw=1.2)
axD.axhline(0, color='#ccc', lw=0.8); axD.axvline(0, color='#ccc', lw=0.8)
axD.set_xlim(-lim * 0.2, lim); axD.set_ylim(-lim * 0.2, lim)
axD.set_xlabel('on-target VAF effect  ΔP(target)', fontsize=14)
axD.set_ylabel('largest off-target VAF effect', fontsize=14)
axD.set_title('D   Dose effect is lineage-specific', fontsize=16, loc='left', pad=8)
axD.tick_params(labelsize=12)
for sp in ('top', 'right'):
    axD.spines[sp].set_visible(False)
axD.text(0.97, 0.04, 'below dashed = specific', transform=axD.transAxes,
         ha='right', fontsize=12, color='#666', style='italic')

fig.savefig(f'{DS}/fig_dose.png')
fig.savefig(f'{DS}/fig_dose.pdf')
print(f'wrote fig_dose to {DS}', flush=True)
