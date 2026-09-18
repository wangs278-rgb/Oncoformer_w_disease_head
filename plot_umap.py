#!/usr/bin/env python3
"""Two-panel UMAP of the run8_disease embedding space (publication style, Arial).

Panel A - color by top disease lineages (each patient's true DiseaseTerm).
Panel B - color by each patient's top gene: the carried panel-gene whose KNOCKOUT most
          lowers its disease-head probability (from extract_umap_features.py).

Reads run8_disease_out/umap/umap_features.npz. Fits UMAP once and caches the 2D coords to
umap_coords.npz so re-styling is instant. Emits PDF+PNG to umap/.

    python plot_umap.py [--topn 14] [--panelA {term,group}] [--refit]
"""
import os, argparse
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

UMAP_DIR = '/cv/home/wangs278/scratch/fmi/run8_disease_out/umap'

ap = argparse.ArgumentParser()
ap.add_argument('--topn', type=int, default=14, help='distinct colors per panel; rest -> Other')
ap.add_argument('--panelA', choices=['term', 'group'], default='term')
ap.add_argument('--refit', action='store_true', help='ignore cached coords and refit UMAP')
ap.add_argument('--n-neighbors', type=int, default=30)
ap.add_argument('--min-dist', type=float, default=0.10)
args = ap.parse_args()

# validated categorical palette (dataviz skill, light mode) + distinct extensions.
# In a UMAP, cluster POSITION is the primary identity channel; color is a secondary aid,
# so we use >8 hues with a legend and fold the long tail into a gray "Other".
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948',
       '#16324f', '#8c5a2b', '#7a7a1f', '#59c7d6', '#b39ddb', '#9e0059', '#00767a', '#c98500']
OTHER = '#d9d9d9'

# ------------------------------------------------------------------- load features
# npz arrays load lazily on access; we only touch the 765 MB `emb` when refitting.
d = np.load(f'{UMAP_DIR}/umap_features.npz', allow_pickle=True)
term = d['true_term'].astype(str) if args.panelA == 'term' else d['true_group'].astype(str)
gene = d['top_gene'].astype(str)
N = gene.shape[0]
print(f'loaded {N} patients', flush=True)

# ------------------------------------------------------------------- UMAP (cached)
coords_path = f'{UMAP_DIR}/umap_coords.npz'
if os.path.exists(coords_path) and not args.refit:
    XY = np.load(coords_path)['xy']
    assert XY.shape[0] == N, f'cached coords {XY.shape} != {N}; use --refit'
    print('loaded cached UMAP coords', flush=True)
else:
    import umap
    emb = d['emb'].astype(np.float32)
    mu = emb.mean(0, keepdims=True); sd = emb.std(0, keepdims=True) + 1e-6
    Xz = (emb - mu) / sd
    print('fitting UMAP ...', flush=True)
    # random_state left unset so numba runs multi-threaded (much faster at ~400k points);
    # the fitted 2D coords are cached to umap_coords.npz, so the saved figure is reproducible
    # even though the fit itself is not bit-identical run-to-run.
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        metric='cosine', verbose=True)
    XY = reducer.fit_transform(Xz).astype(np.float32)
    np.savez_compressed(coords_path, xy=XY)
    print('wrote', coords_path, flush=True)


def top_levels(labels, mask=None, topn=args.topn):
    v = labels if mask is None else labels[mask]
    from collections import Counter
    cnt = Counter(x for x in v if x != '')
    return [k for k, _ in cnt.most_common(topn)]


def short(s, n=34):
    return s if len(s) <= n else s[:n - 1] + '…'


def draw(ax, labels, keep_mask, title, letter):
    """Scatter with fixed-order categorical colors; 'Other'/unlabeled in gray behind."""
    tops = top_levels(labels, keep_mask)
    cmap = {lv: PAL[i % len(PAL)] for i, lv in enumerate(tops)}
    is_top = np.array([l in cmap for l in labels]) & keep_mask
    # background: everything not in the top set (incl. unlabeled), drawn first
    bg = ~is_top
    ax.scatter(XY[bg, 0], XY[bg, 1], s=2.0, c=OTHER, alpha=0.30, linewidths=0,
               rasterized=True)
    # foreground: top categories, shuffled so no class fully occludes another
    idx = np.where(is_top)[0]
    rng = np.random.default_rng(0); rng.shuffle(idx)
    cols = np.array([cmap[labels[i]] for i in idx])
    ax.scatter(XY[idx, 0], XY[idx, 1], s=4.0, c=cols, alpha=0.80, linewidths=0,
               rasterized=True)
    # crop outliers so the main cloud fills the panel
    ax.set_xlim(*np.percentile(XY[:, 0], [0.5, 99.5]))
    ax.set_ylim(*np.percentile(XY[:, 1], [0.5, 99.5]))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=12, pad=6)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='bottom', ha='right')
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                      markerfacecolor=cmap[lv], markeredgewidth=0, label=short(lv))
               for lv in tops]
    handles.append(Line2D([0], [0], marker='o', linestyle='', markersize=6,
                          markerfacecolor=OTHER, markeredgewidth=0, label='other'))
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=7, frameon=False, handletextpad=0.3, labelspacing=0.35,
              borderaxespad=0.0)


fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
lineage_name = 'lineage' if args.panelA == 'term' else 'organ group'
draw(axes[0], term, np.ones(N, bool),
     f'Embedding space colored by disease {lineage_name}', 'A')
draw(axes[1], gene, gene != '',
     'Embedding space colored by top knockout-driver gene', 'B')
fig.suptitle('UMAP of the run8 disease-model patient embeddings', fontsize=13, y=1.00)
fig.subplots_adjust(wspace=0.35)

for ext in ('png', 'pdf'):
    fig.savefig(f'{UMAP_DIR}/fig_umap_panels.{ext}')
plt.close(fig)
print(f'wrote fig_umap_panels (PDF+PNG) to {UMAP_DIR}', flush=True)
