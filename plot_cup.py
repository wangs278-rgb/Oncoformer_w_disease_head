#!/usr/bin/env python3
"""#1  CUP resolution - the disease head as a molecular tumor-of-origin classifier.

~28k patients carry an UNKNOWN-PRIMARY / undifferentiated label (true tissue never assignable).
The head predicts a supervised class for each; 86% resolve to a CONCRETE tissue. This script
characterises and VALIDATES those calls straight from the saved features (no new GPU pass):

  umap/umap_features.npz : emb[N,512], true_term, pred_term, pred_prob, top_gene
  umap/umap_coords.npz   : shared UMAP xy[N,2]

Validity has no ground truth for CUP, so we use INTERNAL CONSISTENCY: does a CUP patient the
model calls "lung" sit among REAL lung patients in embedding space? -> k-NN concordance
(fraction of a CUP patient's k nearest real non-CUP neighbours whose true tissue == the model's
call), stratified by confidence. Cached to cup/cup_concordance.npz.

One publication figure (Arial), six panels:
  A resolution overview (resolved-to-tissue vs stays-unknown, with confidence)
  B CUP subtype -> resolved tissue flow (row-normalised heatmap; histology consistency)
  C shared UMAP, CUP patients coloured by resolved tissue (land inside real clusters)
  D k-NN concordance vs confidence  (the validity curve)
  E confidence distribution, resolved vs unresolved
  F driver attribution: top knockout-driver gene for flagship resolved tissues

    python plot_cup.py [--knn 30] [--refit-knn]
Output: run8_disease_out/cup/fig_cup_resolution.{png,pdf}
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from collections import Counter

_arial = os.path.expanduser('~/.local/share/fonts/Arial.ttf')
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT = '/cv/home/wangs278/scratch/fmi/run8_disease_out'
UMAP = f'{OUT}/umap'
CDIR = f'{OUT}/cup'
os.makedirs(CDIR, exist_ok=True)

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948',
       '#16324f', '#8c5a2b', '#7a7a1f', '#59c7d6', '#b39ddb', '#9e0059', '#00767a', '#c98500']
OTHER = '#d9d9d9'

ap = argparse.ArgumentParser()
ap.add_argument('--knn', type=int, default=30)
ap.add_argument('--refit-knn', action='store_true')
ap.add_argument('--topsub', type=int, default=11, help='CUP subtypes shown in panel B')
ap.add_argument('--topdest', type=int, default=14, help='concrete tissues shown in panel B')
args = ap.parse_args()


def is_cup(s):
    s = s.lower()
    return ('unknown primary' in s) or ('undifferentiated' in s) or ('(cup)' in s) or ('occult' in s)


def short(s, n=30):
    return s if len(s) <= n else s[:n - 1] + '…'


# ------------------------------------------------------------------- load
d = np.load(f'{UMAP}/umap_features.npz', allow_pickle=True)
tt = d['true_term'].astype(str); pt = d['pred_term'].astype(str)
pp = d['pred_prob'].astype(float); tg = d['top_gene'].astype(str)
XY = np.load(f'{UMAP}/umap_coords.npz')['xy']
N = tt.shape[0]
cup = np.array([is_cup(x) for x in tt])
pred_cup = np.array([is_cup(x) for x in pt])
resolved = cup & ~pred_cup
unresolved = cup & pred_cup
print(f'N={N}  CUP={cup.sum()}  resolved={resolved.sum()} '
      f'({100*resolved.sum()/max(1,cup.sum()):.1f}% of CUP)  unresolved={unresolved.sum()}', flush=True)

# ------------------------------------------------------------------- k-NN concordance (cached)
cc_path = f'{CDIR}/cup_concordance_k{args.knn}.npz'
if os.path.exists(cc_path) and not args.refit_knn:
    cc = np.load(cc_path, allow_pickle=True)
    conc = cc['conc']; conc_idx = cc['conc_idx']
    print('loaded cached concordance', flush=True)
else:
    from pynndescent import NNDescent
    emb = d['emb'].astype(np.float32)
    mu = emb.mean(0, keepdims=True); sd = emb.std(0, keepdims=True) + 1e-6
    Xz = (emb - mu) / sd
    ref = np.where(~cup)[0]                       # real, non-CUP patients = the reference atlas
    qry = np.where(resolved)[0]                   # CUP patients that resolved to a concrete tissue
    print(f'building NNDescent index on {ref.size} real patients ...', flush=True)
    index = NNDescent(Xz[ref], metric='cosine', n_neighbors=max(40, args.knn + 5),
                      random_state=0, low_memory=True, verbose=True)
    nbr, _ = index.query(Xz[qry], k=args.knn)     # [Q,k] indices INTO ref
    nbr_true = tt[ref][nbr]                        # [Q,k] true tissue of each neighbour
    call = pt[qry][:, None]                        # [Q,1] the model's tissue call
    conc = (nbr_true == call).mean(1)             # [Q] fraction of neighbours matching the call
    conc_idx = qry
    np.savez_compressed(cc_path, conc=conc, conc_idx=conc_idx)
    print(f'wrote {cc_path}', flush=True)

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(27, 17))
gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.36, left=0.07, right=0.985, top=0.90, bottom=0.11)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 0]); axE = fig.add_subplot(gs[1, 1]); axF = fig.add_subplot(gs[1, 2])


def letter(ax, ch):
    ax.text(-0.05, 1.04, ch, transform=ax.transAxes, fontsize=28, fontweight='bold', va='bottom', ha='right')


# ---- A resolution overview
nres, nunr = int(resolved.sum()), int(unresolved.sum())
bars = axA.bar([0, 1], [nres, nunr], color=['#1baf7a', '#b8b8b8'], width=0.62)
axA.set_xticks([0, 1]); axA.set_xticklabels(['resolved to\na concrete tissue', 'stays unknown /\nundifferentiated'], fontsize=17)
axA.tick_params(axis='y', labelsize=15)
for b, n, cf in zip(bars, [nres, nunr], [pp[resolved].mean(), pp[unresolved].mean()]):
    axA.text(b.get_x() + b.get_width() / 2, n, f'{n:,}\n({100*n/cup.sum():.0f}%)\nconf {cf:.2f}',
             ha='center', va='bottom', fontsize=16)
axA.set_ylabel('CUP / unknown-primary patients', fontsize=18)
axA.set_ylim(0, nres * 1.28)
axA.set_title('A   Do unknown-primary tumours get resolved?', fontsize=19, pad=8)
for sp in ('top', 'right'):
    axA.spines[sp].set_visible(False)
letter(axA, 'A')

# ---- B subtype -> tissue flow
sub_top = [s for s, _ in Counter(tt[cup]).most_common() if is_cup(s)][:args.topsub]
dest_top = [t for t, _ in Counter(pt[resolved]).most_common(args.topdest)]
M = np.zeros((len(sub_top), len(dest_top)))
for i, s in enumerate(sub_top):
    m = resolved & (tt == s)
    tot = max(1, m.sum())
    for j, t in enumerate(dest_top):
        M[i, j] = (pt[m] == t).sum() / tot
im = axB.imshow(M, aspect='auto', cmap='magma_r', vmin=0, vmax=min(1.0, M.max()))
axB.set_xticks(range(len(dest_top)))
axB.set_xticklabels([short(x, 24) for x in dest_top], rotation=45, ha='right', rotation_mode='anchor', fontsize=13)
axB.set_yticks(range(len(sub_top))); axB.set_yticklabels([short(x, 30) for x in sub_top], fontsize=13)
axB.tick_params(length=0)
cb = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.02); cb.set_label('fraction of subtype', fontsize=15)
cb.ax.tick_params(labelsize=13)
axB.set_title('B   Where each CUP subtype resolves to', fontsize=19, pad=8)
letter(axB, 'B')

# ---- C UMAP placement of resolved CUP by tissue
dest_col = {t: PAL[i % len(PAL)] for i, t in enumerate(dest_top[:len(PAL)])}
axC.scatter(XY[~cup, 0], XY[~cup, 1], s=1.2, c=OTHER, alpha=0.25, linewidths=0, rasterized=True)
ridx = np.where(resolved)[0]
rng = np.random.default_rng(0); rng.shuffle(ridx)
cols = np.array([dest_col.get(pt[i], '#333333') for i in ridx])
axC.scatter(XY[ridx, 0], XY[ridx, 1], s=3.2, c=cols, alpha=0.75, linewidths=0, rasterized=True)
axC.set_xlim(*np.percentile(XY[:, 0], [0.5, 99.5])); axC.set_ylim(*np.percentile(XY[:, 1], [0.5, 99.5]))
axC.set_xticks([]); axC.set_yticks([])
for sp in ('top', 'right', 'left', 'bottom'):
    axC.spines[sp].set_visible(False)
axC.set_title('C   Resolved CUP patients on the atlas (color = called tissue)', fontsize=18, pad=8)
letter(axC, 'C')
handles = [Line2D([0], [0], marker='o', linestyle='', markersize=9, markerfacecolor=dest_col[t],
                  markeredgewidth=0, label=short(t, 26)) for t in list(dest_col)[:12]]
axC.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=13,
           frameon=False, handletextpad=0.3, labelspacing=0.45, borderaxespad=0.0)

# ---- D concordance vs confidence
cpp = pp[conc_idx]
bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
bctr, bmean, bse, bn = [], [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (cpp >= lo) & (cpp < hi if hi < 1.0 else cpp <= hi)
    if m.sum() < 20:
        continue
    bctr.append((lo + hi) / 2); bmean.append(conc[m].mean())
    bse.append(conc[m].std() / np.sqrt(m.sum())); bn.append(int(m.sum()))
bctr, bmean, bse = np.array(bctr), np.array(bmean), np.array(bse)
axD.errorbar(bctr, bmean, yerr=bse, marker='o', ms=11, color='#2a78d6', lw=2.8, capsize=5)
axD.tick_params(labelsize=15)
for x, y, n in zip(bctr, bmean, bn):
    axD.annotate(f'n={n:,}', (x, y), fontsize=14, xytext=(6, 9), textcoords='offset points', color='#555555')
axD.axhline(conc.mean(), color='#eb6834', lw=1.4, ls='--')
axD.text(0.02, conc.mean(), f'  overall {conc.mean():.2f}', color='#eb6834', fontsize=15, va='bottom')
axD.set_xlabel('model confidence  P(called tissue)', fontsize=18)
axD.set_ylabel(f'{args.knn}-NN concordance\n(frac. real neighbours = called tissue)', fontsize=17)
axD.set_ylim(0, 1); axD.set_xlim(0.1, 1.0)
axD.set_title('D   Validity: do resolved calls land among matching real tumours?', fontsize=18, pad=8)
for sp in ('top', 'right'):
    axD.spines[sp].set_visible(False)
letter(axD, 'D')

# ---- E confidence distributions
axE.hist(pp[resolved], bins=40, range=(0, 1), color='#1baf7a', alpha=0.75, label=f'resolved (n={nres:,})')
axE.hist(pp[unresolved], bins=40, range=(0, 1), color='#b8b8b8', alpha=0.75, label=f'unresolved (n={nunr:,})')
axE.tick_params(labelsize=15)
axE.set_xlabel('model confidence  P(top class)', fontsize=18)
axE.set_ylabel('CUP patients', fontsize=18)
axE.set_title('E   Confidence of resolved vs unresolved calls', fontsize=18, pad=8)
axE.legend(fontsize=15, frameon=False)
for sp in ('top', 'right'):
    axE.spines[sp].set_visible(False)
letter(axE, 'E')

# ---- F driver attribution for flagship resolved tissues
flagship = [t for t in dest_top if t in
            ('lung adenocarcinoma', 'skin melanoma', 'colon adenocarcinoma (crc)',
             'pancreas ductal adenocarcinoma', 'ovary serous carcinoma',
             'prostate acinar adenocarcinoma')][:5]
if len(flagship) < 4:
    flagship = dest_top[:5]
ng = 5
allbars = []
xticklab = []
base = 0
for t in flagship:
    m = resolved & (pt == t)
    genes = [g for g in tg[m] if g not in ('', 'none', 'None')]
    top = Counter(genes).most_common(ng)
    for k, (g, c) in enumerate(top):
        allbars.append((base + k, 100 * c / max(1, len(genes)), g))
    base += ng + 1
    xticklab.append((base - (ng + 1) / 2 - 0.5, short(t, 20)))
for x, h, g in allbars:
    axF.bar(x, h, color='#4a3aa7', width=0.85)
    axF.text(x, h, g, rotation=90, fontsize=11, ha='center', va='bottom')
axF.set_xticks([p for p, _ in xticklab]); axF.set_xticklabels([l for _, l in xticklab], fontsize=13)
axF.tick_params(axis='y', labelsize=15)
axF.set_ylabel('% of resolved patients\nwith this top driver', fontsize=16)
axF.set_title('F   Driver behind each flagship resolution', fontsize=18, pad=8)
axF.set_ylim(0, max(h for _, h, _ in allbars) * 1.22)
for sp in ('top', 'right'):
    axF.spines[sp].set_visible(False)
letter(axF, 'F')

fig.suptitle('Resolving cancers of unknown primary: the disease head predicts a tissue of origin from genomics alone',
             fontsize=22, y=0.98)
for ext in ('png', 'pdf'):
    fig.savefig(f'{CDIR}/fig_cup_resolution.{ext}')
plt.close(fig)
print(f'wrote fig_cup_resolution to {CDIR}', flush=True)

# ------------------------------------------------------------------- text audit
print('\n--- validity audit ---', flush=True)
print(f'resolution rate: {100*nres/cup.sum():.1f}%  mean conf resolved={pp[resolved].mean():.2f} unresolved={pp[unresolved].mean():.2f}')
print(f'overall {args.knn}-NN concordance (resolved CUP): {conc.mean():.3f}')
print(f'  high-conf (P>=0.6): {conc[cpp>=0.6].mean():.3f} (n={int((cpp>=0.6).sum()):,})')
print(f'  low-conf  (P<0.4): {conc[cpp<0.4].mean():.3f} (n={int((cpp<0.4).sum()):,})')
print('histology consistency (subtype -> modal resolved tissue):')
for s in sub_top[:8]:
    m = resolved & (tt == s)
    if m.sum():
        t, c = Counter(pt[m]).most_common(1)[0]
        print(f'  {short(s,44):44s} -> {short(t,34):34s} {100*c/m.sum():.0f}%')
