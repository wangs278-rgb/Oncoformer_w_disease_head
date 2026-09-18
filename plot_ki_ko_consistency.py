#!/usr/bin/env python3
"""Publication figures (Arial) for KI vs KO in-silico consistency.

Reads run8_disease_out/{ki,ko}/*_results.npz (no GPU). Emits PDF+PNG to
run8_disease_out/ki_ko/figures/.

Consistency logic:
  KI = insert a gene's modal mutation into NON-carriers  -> should RAISE linked disease (dP>0)
  KO = remove a gene's mutation from CARRIERS            -> should LOWER linked disease (dP<0)
So per gene-disease pair, KI dP and KO dP should be ANTI-correlated.

Figures:
  1  fig_ki_ko_scatter_pairs  - all 316x426 pairs: KI dP vs KO dP (anti-correlation)
  2  fig_ki_ko_target_mirror  - top genes: KI(+)/KO(-) dP on each gene's target disease
  3  fig_ki_ko_influence      - per-gene KO vs KI influence (magnitude agreement + caveat)
  4  fig_ki_ko_gene_panels    - per-gene concordance: KI vs KO dP on that gene's top diseases
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr, linregress

# ---- Arial (match plot_ko_disease.py) ----
_arial = os.path.expanduser('~/.local/share/fonts/Arial.ttf')
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = '/cv/home/wangs278/scratch/fmi/run8_disease_out'
FIG  = f'{BASE}/ki_ko/figures'
os.makedirs(FIG, exist_ok=True)

RAISE = '#b2182b'   # KI / addition (warm)  = raises disease
LOWER = '#2166ac'   # KO / removal (cool)   = lowers disease
GOOD  = '#1b7837'   # concordant (opposite sign, as expected)
BAD   = '#e08214'   # discordant (same sign)
INK   = '#222222'; MUT = '#888888'

ki = np.load(f'{BASE}/ki/ki_results.npz', allow_pickle=True)
ko = np.load(f'{BASE}/ko/ko_results.npz', allow_pickle=True)

# ---- align (genes differ in order; sup_names identical, but reindex defensively) ----
gi, si = ki['genes'], ki['sup_names']
gpos = {g: j for j, g in enumerate(ko['genes'])}
spos = {s: j for j, s in enumerate(ko['sup_names'])}
gord = np.array([gpos[g] for g in gi])
sord = np.array([spos[s] for s in si])

genes  = gi
dnames = si
ki_dp  = ki['meandP']
ko_dp  = ko['meandP'][np.ix_(gord, sord)]
ki_tv  = ki['influence_tv']
ko_tv  = ko['influence_tv'][gord]
ki_n   = ki['n_used']
ko_n   = ko['n_used'][gord]
ko_car = ko['carrier_count'][gord]
G, S = ki_dp.shape
print(f'aligned {G} genes x {S} diseases')

def short(s, n=32):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + '…'

def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png')
    plt.close(fig); print('wrote', name)

# each gene's KI-target disease (disease KI most raises)
tgt = ki_dp.argmax(1)
ki_at = ki_dp[np.arange(G), tgt]
ko_at = ko_dp[np.arange(G), tgt]

# =============================================== Fig 1: per-pair anti-correlation
fki = ki_dp.ravel(); fko = ko_dp.ravel()
pr_all, _ = pearsonr(fki, fko); sr_all, _ = spearmanr(fki, fko)
m = np.abs(fki) > 0.02
pr_sig, _ = pearsonr(fki[m], fko[m])
opp = (np.sign(fki[m]) != np.sign(fko[m])).mean()

fig, ax = plt.subplots(figsize=(6.0, 5.6))
lim = 0.5
hb = ax.hexbin(fki, fko, gridsize=70, bins='log', cmap='Greys',
               mincnt=1, linewidths=0, extent=(-lim, lim, -lim, lim))
# guide lines
ax.axhline(0, color='#bbbbbb', lw=0.8, zorder=1)
ax.axvline(0, color='#bbbbbb', lw=0.8, zorder=1)
ax.plot([-lim, lim], [lim, -lim], ls='--', lw=1.0, color='#999999', zorder=1)
# overlay each gene's target-disease pair, colored by concordance
conc = (np.sign(ki_at) != np.sign(ko_at))
ax.scatter(ki_at[conc], ko_at[conc], s=26, c=GOOD, edgecolor='white',
           linewidth=0.4, zorder=3, label='concordant (add↑ / remove↓)')
ax.scatter(ki_at[~conc], ko_at[~conc], s=26, c=BAD, edgecolor='white',
           linewidth=0.4, zorder=3, label='discordant')
# label the corner genes (edge-aware) — union of strongest KI and strongest KO
lab = set(np.argsort(-ki_at)[:3]) | set(np.argsort(ko_at)[:3])
for i in lab:
    x, y = ki_at[i], ko_at[i]
    if x > 0.35:                      # near right edge -> anchor label to the left
        off, ha = (-5, 4), 'right'
    else:
        off, ha = (5, 4), 'left'
    ax.annotate(genes[i], (x, y), fontsize=7.5, color=INK, ha=ha,
                xytext=off, textcoords='offset points')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel('KI  ΔP  (insert modal mutation into non-carriers)', fontsize=9)
ax.set_ylabel('KO  ΔP  (remove mutation from carriers)', fontsize=9)
ax.set_title('KI and KO are anti-correlated across gene–disease pairs', fontsize=10)
txt = (f'all {G*S:,} pairs:  r = {pr_all:.2f}   ρ = {sr_all:.2f}\n'
       f'|ΔP$_{{KI}}$|>0.02 (n={m.sum()}):  r = {pr_sig:.2f}\n'
       f'opposite-sign: {opp*100:.0f}%')
ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.5, va='bottom',
        ha='left', bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc', lw=0.6))
ax.text(0.97, 0.03, 'expected\nquadrant', transform=ax.transAxes, fontsize=7,
        color=GOOD, ha='right', va='bottom', style='italic')
ax.tick_params(labelsize=8)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02); cb.set_label('pairs per bin (log)', fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.legend(loc='upper left', fontsize=7, frameon=False,
          title='gene target-disease', title_fontsize=7)
save(fig, 'fig_ki_ko_scatter_pairs')

# =============================================== Fig 2: target-disease mirror
TOPN = 20
o = np.argsort(-ki_tv)[:TOPN][::-1]      # bottom-to-top ascending so strongest on top
y = np.arange(TOPN); h = 0.38
fig, ax = plt.subplots(figsize=(7.2, 0.34 * TOPN + 1.2))
ax.barh(y + h/2, ki_at[o], height=h, color=RAISE, edgecolor='none', label='KI: insert → ΔP')
ax.barh(y - h/2, ko_at[o], height=h, color=LOWER, edgecolor='none', label='KO: remove → ΔP')
ax.axvline(0, color='k', lw=0.7)
ax.set_yticks(y); ax.set_yticklabels(genes[o], fontsize=8)
# disease label in the right gutter
xr = ki_at[o].max()
ax.set_xlim(ko_at[o].min() * 1.15, xr * 1.9)
for k, i in enumerate(o):
    ax.text(xr * 1.05, y[k], short(dnames[tgt[i]], 34), va='center', ha='left',
            fontsize=6.5, color=INK)
ax.set_xlabel('mean ΔP on the gene’s target disease', fontsize=9)
ax.set_title('Same disease: knock-in raises it, knock-out lowers it\n'
             '(top genes by KI influence; label = target disease)', fontsize=10)
ax.tick_params(axis='x', labelsize=8)
ax.legend(loc='lower right', fontsize=8, frameon=False)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
save(fig, 'fig_ki_ko_target_mirror')

# =============================================== Fig 3: per-gene influence agreement
rho, _ = spearmanr(ki_tv, ko_tv)
fig, ax = plt.subplots(figsize=(6.0, 5.6))
sc = ax.scatter(ko_tv, ki_tv, c=np.log10(ko_car + 1), cmap='viridis',
                s=22, edgecolor='white', linewidth=0.3)
# label the well-separated genes: top by KO (right side) + the two KI outliers.
# The dense KI-high / KO-mid cluster is left unlabelled to avoid overlap.
top = set(np.argsort(-ko_tv)[:14]) | set(np.argsort(-ki_tv)[:2])
for i in top:
    x, y = ko_tv[i], ki_tv[i]
    off = (4, -9) if genes[i] == 'FOXL2' else (4, 3)   # keep FOXL2 clear of the top edge
    ax.annotate(genes[i], (x, y), fontsize=7, color=INK,
                xytext=off, textcoords='offset points')
ax.margins(y=0.05)
ax.set_xlabel('KO influence  (mean TV distance)', fontsize=9)
ax.set_ylabel('KI influence  (mean TV distance)', fontsize=9)
ax.set_title('Per-gene influence: KI vs KO\n'
             f'Spearman ρ = {rho:.2f}  (magnitude differs by design — see caption)', fontsize=10)
ax.tick_params(labelsize=8)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('KO carrier count  (log$_{10}$)', fontsize=8); cb.ax.tick_params(labelsize=7)
cap = ('KI is capped at ~10k non-carriers/gene (modal mutation only); KO uses actual carriers, '
       'so high-frequency drivers (TP53, KRAS, APC) dominate the KO axis — the magnitude offset is by design.')
fig.text(0.5, -0.02, cap, ha='center', va='top', fontsize=7, color=MUT, style='italic')
save(fig, 'fig_ki_ko_influence')

# =============================================== Fig 4: per-gene concordance panels
EX = [g for g in ['FOXL2', 'APC', 'AR', 'CDH1', 'NKX2-1', 'PPARG',
                  'GATA4', 'VHL', 'TP53'] if g in set(genes.tolist())][:9]
nr = (len(EX) + 2) // 3
fig, axes = plt.subplots(nr, 3, figsize=(14, 3.0 * nr))
axes = np.atleast_1d(axes).ravel()
for k, g in enumerate(EX):
    idx = np.where(genes == g)[0][0]
    sel = np.argsort(-np.abs(ki_dp[idx]))[:6]        # diseases this gene moves most (by KI)
    sel = sel[np.argsort(-ki_dp[idx][sel])]          # order by KI dP desc
    yy = np.arange(len(sel))[::-1]; hh = 0.38
    ax = axes[k]
    ax.barh(yy + hh/2, ki_dp[idx][sel], height=hh, color=RAISE, edgecolor='none')
    ax.barh(yy - hh/2, ko_dp[idx][sel], height=hh, color=LOWER, edgecolor='none')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels([short(dnames[j], 26) for j in sel], fontsize=6.2)
    ax.set_title(f'{g}  (KI n={ki_n[idx]}, KO n={ko_n[idx]})', fontsize=9)
    ax.tick_params(axis='x', labelsize=6.5)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
for k in range(len(EX), len(axes)): axes[k].axis('off')
leg = [Patch(fc=RAISE, label='KI: insert mutation (ΔP)'),
       Patch(fc=LOWER, label='KO: remove mutation (ΔP)')]
fig.legend(handles=leg, loc='lower center', ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Per-gene concordance: KI and KO push the same diseases in opposite directions  —  mean ΔP',
             fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0.02, 1, 1))
save(fig, 'fig_ki_ko_gene_panels')

print('all figures in', FIG)

# =============================================== Supplementary CSV
import csv
ki_q = ki['qval']
ko_q = ko['qval'][np.ix_(gord, sord)]
ko_low = ko_dp.argmin(1)                          # disease KO most lowers
csv_path = f'{BASE}/ki_ko/ki_ko_consistency.csv'
order = np.argsort(-ki_tv)
with open(csv_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow([
        'gene', 'ki_influence_tv', 'ko_influence_tv',
        'ki_flip_rate', 'ko_flip_rate',
        'ki_n_recipients', 'ko_n_carriers', 'ko_carrier_count',
        'ki_target_disease',                      # disease KI most raises
        'ki_dP_target', 'ki_qval_target',
        'ko_dP_on_ki_target', 'ko_qval_on_ki_target',
        'concordant_on_target',                   # KI raises AND KO lowers the same disease
        'ko_top_lowered_disease', 'ko_dP_top_lowered',
        'target_argmatch',                        # argmax(KI) == argmin(KO)
    ])
    for i in order:
        t = int(tgt[i]); l = int(ko_low[i])
        w.writerow([
            genes[i], f'{ki_tv[i]:.5f}', f'{ko_tv[i]:.5f}',
            f'{ki["flip_rate"][i]:.4f}', f'{ko["flip_rate"][gord[i]]:.4f}',
            int(ki_n[i]), int(ko_n[i]), int(ko_car[i]),
            dnames[t],
            f'{ki_dp[i, t]:.5f}', f'{ki_q[i, t]:.3e}',
            f'{ko_dp[i, t]:.5f}', f'{ko_q[i, t]:.3e}',
            int(np.sign(ki_dp[i, t]) != np.sign(ko_dp[i, t])),
            dnames[l], f'{ko_dp[i, l]:.5f}',
            int(t == l),
        ])
print('wrote', csv_path, f'({G} genes)')
