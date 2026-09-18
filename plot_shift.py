#!/usr/bin/env python3
"""Visualize in-silico lineage shifts (from insilico_shift.py) on the shared run8 UMAP.

For each gene G with a shift/shift_<G>.npz, we show how adding G moves SOURCE-lineage
patients toward the DEST lineage:
  - a UMAP trajectory panel: source & dest clusters highlighted, arrows from each recipient's
    baseline position to its post-addition position (projected onto the SAME shared UMAP via
    KNN), colored by whether the argmax flipped to DEST;
  - a probability panel: P(dest) before vs after + the flip rate;
  - an animated GIF of the recipients gliding source -> dest.
Plus a 1xN overview and a flip-rate bar across all genes.

Perturbed points are placed on the existing full-cohort UMAP (umap_coords.npz) by averaging
the coords of each perturbed embedding's k nearest baseline neighbours (cosine, z-scored) --
so every gene shares one coordinate system and the overview is directly comparable.

    python plot_shift.py [--genes CDH1 FOXL2 NKX2-1 AR APC] [--knn 15] [--frames 42]
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

_arial = os.path.expanduser('~/.local/share/fonts/Arial.ttf')
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
plt.rcParams.update({
    'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'axes.edgecolor': '#333333',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

UMAP_DIR  = '/cv/home/wangs278/scratch/fmi/run8_disease_out/umap'
SHIFT_DIR = '/cv/home/wangs278/scratch/fmi/run8_disease_out/shift'

C_SRC   = '#2a78d6'   # source lineage / baseline
C_DST   = '#eb6834'   # destination lineage / perturbed
C_FLIP  = '#1baf7a'   # trajectory that flipped argmax -> dest
C_STAY  = '#b8b8b8'   # trajectory that did not flip
C_BG    = '#cfcfcd'   # background cloud
C_OLD   = '#2a78d6'   # recipient ORIGINAL position (blue)
C_NEW   = '#e34948'   # recipient position AFTER adding the gene (red)


def legend_handles():
    return [
        Line2D([0], [0], marker='o', ls='', mfc=C_SRC, mec='none', ms=8,
               label='source-lineage patient'),
        Line2D([0], [0], marker='o', ls='', mfc=C_DST, mec='none', ms=8,
               label='destination-lineage patient'),
        Line2D([0], [0], marker='o', ls='', mfc=C_BG, mec='none', ms=8,
               label='all other patients (background map)'),
        Line2D([0], [0], color=C_FLIP, lw=2.2, marker='>', markevery=[-1],
               mfc=C_FLIP, mec=C_FLIP, ms=7,
               label='trajectory of a recipient that flipped → destination'),
        Line2D([0], [0], marker='o', ls='', mfc=C_STAY, mec='none', ms=8,
               label='recipient that did NOT flip (endpoint)'),
    ]

ap = argparse.ArgumentParser()
ap.add_argument('--genes', nargs='+', default=['CDH1', 'FOXL2', 'NKX2-1', 'AR', 'APC'])
ap.add_argument('--knn', type=int, default=15)
ap.add_argument('--frames', type=int, default=42)
ap.add_argument('--max-arrows', type=int, default=350, help='arrows drawn in static panel')
ap.add_argument('--max-anim', type=int, default=1500, help='points animated in the gif')
ap.add_argument('--no-gif', action='store_true', help='skip GIFs (fast static-only re-render)')
ap.add_argument('--shift-dir', default=SHIFT_DIR, help='folder with shift_<gene>.npz + outputs')
args = ap.parse_args()
SHIFT_DIR = args.shift_dir

ABBR = {
    'breast invasive ductal carcinoma (idc)': 'breast IDC',
    'breast invasive lobular carcinoma (ilc)': 'breast ILC',
    'ovary serous carcinoma': 'ovary serous',
    'ovary granulosa cell tumor': 'ovary granulosa',
    'lung squamous cell carcinoma (scc)': 'lung SCC',
    'lung adenocarcinoma': 'lung adeno',
    'prostate neuroendocrine carcinoma': 'prostate NE',
    'prostate acinar adenocarcinoma': 'prostate acinar',
    'pancreas ductal adenocarcinoma': 'pancreas ductal',
    'colon adenocarcinoma (crc)': 'colon CRC',
}
def lab(term):
    return ABBR.get(term, term if len(term) <= 20 else term[:19] + '…')


def av(G):
    """verb phrase for the perturbation: 'adding' (ki) or 'removing' (ko)."""
    return 'adding' if G.get('mode', 'ki') == 'ki' else 'removing'

# ------------------------------------------------------------- shared map + KNN projector
feat = np.load(f'{UMAP_DIR}/umap_features.npz', allow_pickle=True)
emb = feat['emb'].astype(np.float32)
sid = feat['sample_id'].astype(str)
true_term = feat['true_term'].astype(str)
XY = np.load(f'{UMAP_DIR}/umap_coords.npz')['xy'].astype(np.float32)
assert XY.shape[0] == emb.shape[0] == sid.shape[0]
row_of = {s: i for i, s in enumerate(sid)}
mu = emb.mean(0, keepdims=True); sd = emb.std(0, keepdims=True) + 1e-6
embz = (emb - mu) / sd
print(f'shared map: {XY.shape[0]} patients', flush=True)

# lineage label anchors (median position of each well-populated lineage)
from collections import Counter
tcounts = Counter(true_term)
lineage_cent = {}
for t, c in tcounts.most_common(60):
    if t == '' or 'other' in t.lower() or 'nos' in t.lower() or c < 500:
        continue
    lineage_cent[t] = np.median(XY[true_term == t], axis=0)

# palette for the standalone colored landscape (dataviz categorical order + extensions)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948',
       '#16324f', '#8c5a2b', '#7a7a1f', '#59c7d6', '#b39ddb', '#9e0059', '#00767a', '#c98500']

print('building KNN index (pynndescent, cosine) ...', flush=True)
from pynndescent import NNDescent
index = NNDescent(embz, metric='cosine', n_neighbors=max(20, args.knn + 5), random_state=42)
index.prepare()


def project(emb_pert):
    """Place perturbed embeddings on the shared UMAP by similarity-weighted KNN average."""
    q = ((emb_pert - mu) / sd).astype(np.float32)
    nn, dist = index.query(q, k=args.knn)
    w = np.clip(1.0 - dist, 1e-6, None); w /= w.sum(1, keepdims=True)      # cosine sim weights
    return np.einsum('nk,nkd->nd', w, XY[nn])                              # [Nq, 2]


def roi(*xys, pad=0.06, q=(2, 98)):
    P = np.vstack(xys)
    xlo, xhi = np.percentile(P[:, 0], q); ylo, yhi = np.percentile(P[:, 1], q)
    dx, dy = (xhi - xlo) * pad, (yhi - ylo) * pad
    return (xlo - dx, xhi + dx), (ylo - dy, yhi + dy)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)


def load_gene(g):
    p = f'{SHIFT_DIR}/shift_{g}.npz'
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    ids = d['sample_id'].astype(str)
    rows = np.array([row_of[s] for s in ids if s in row_of])
    keep = np.array([s in row_of for s in ids])
    if len(rows) < 10:
        print(f'  {g}: only {len(rows)} recipients on the map — skipping', flush=True)
        return None
    base_xy = XY[rows]
    pert_xy = project(d['emb_pert'][keep])
    flipped = (d['argmax_pert'].astype(str)[keep] == str(d['dest_term']))
    meta = json.load(open(f'{SHIFT_DIR}/shift_{g}_meta.json'))
    return dict(g=g, base=base_xy, pert=pert_xy, flip=flipped,
                pdb=d['p_dest_base'][keep], pdp=d['p_dest_pert'][keep],
                psb=d['p_source_base'][keep], psp=d['p_source_pert'][keep],
                src=str(d['source_term']), dst=str(d['dest_term']), meta=meta,
                mode=meta.get('mode', 'ki'),
                src_mask=(true_term == str(d['source_term'])),
                dst_mask=(true_term == str(d['dest_term'])))


def _even(idx, k):
    return idx[np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)] if len(idx) else idx


def _labels_in_view(ax, view, src=None, dst=None, fs=7):
    """Place lineage-name labels at their centroids that fall inside the view (spaced)."""
    xl, yl = view; W = xl[1] - xl[0]; H = yl[1] - yl[0]
    placed = []
    for t in sorted(lineage_cent, key=lambda k: -tcounts[k]):
        x, y = lineage_cent[t]
        if not (xl[0] <= x <= xl[1] and yl[0] <= y <= yl[1]):
            continue
        if any(((x - px) / W) ** 2 + ((y - py) / H) ** 2 < 0.02 for px, py in placed) \
           and t not in (src, dst):
            continue
        col = C_SRC if t == src else C_DST if t == dst else '#222222'
        fw = 'bold' if t in (src, dst) else 'normal'
        ax.text(x, y, lab(t), fontsize=fs + (1 if t in (src, dst) else 0), ha='center',
                va='center', color=col, fontweight=fw, zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        placed.append((x, y))


def draw_landscape_colored(ax, view, src=None, dst=None, title_fs=11, focus=False):
    """Panel A landscape. focus=True: color ONLY the source & destination lineages (rest gray).
    focus=False: color all top lineages (standalone reference). Labels in both."""
    ax.scatter(XY[:, 0], XY[:, 1], s=0.6, c=C_BG, alpha=0.25, linewidths=0, rasterized=True)
    if focus:
        if src is not None:
            m = true_term == src
            ax.scatter(XY[m, 0], XY[m, 1], s=3, c=C_SRC, alpha=0.6, linewidths=0, rasterized=True)
        if dst is not None:
            m = true_term == dst
            ax.scatter(XY[m, 0], XY[m, 1], s=3.5, c=C_DST, alpha=0.8, linewidths=0, rasterized=True)
        title = (f'Cancer-lineage landscape\n'
                 f'blue = {lab(src)}    orange = {lab(dst)}')
    else:
        tops = [t for t, _ in tcounts.most_common(60)
                if t and 'other' not in t.lower() and 'nos' not in t.lower()][:16]
        for i, t in enumerate(tops):
            m = true_term == t
            ax.scatter(XY[m, 0], XY[m, 1], s=1.6, c=PAL[i % len(PAL)], alpha=0.5,
                       linewidths=0, rasterized=True)
        title = 'Cancer-lineage landscape (reference)'
    _labels_in_view(ax, view, src=src, dst=dst)
    ax.set_xlim(*view[0]); ax.set_ylim(*view[1]); clean(ax)
    ax.set_title(title, fontsize=title_fs, pad=6)


def draw_positions(ax, G, view, title_fs=11, legend=True):
    """Panel B: recipients' ORIGINAL (blue) vs AFTER-addition (red) positions. No arrows."""
    ax.scatter(XY[:, 0], XY[:, 1], s=0.6, c=C_BG, alpha=0.16, linewidths=0, rasterized=True)
    ax.scatter(G['base'][:, 0], G['base'][:, 1], s=3, c=C_OLD, alpha=0.5,
               linewidths=0, rasterized=True, zorder=3)
    ax.scatter(G['pert'][:, 0], G['pert'][:, 1], s=3, c=C_NEW, alpha=0.5,
               linewidths=0, rasterized=True, zorder=4)
    ax.set_xlim(*view[0]); ax.set_ylim(*view[1]); clean(ax)
    m = G['meta']
    ax.set_title(f"Recipient position: original → after {av(G)} {G['g']}\n"
                 f"argmax→dest {100*m['argmax_is_dest_before']:.0f}% → "
                 f"{100*m['argmax_is_dest_after']:.0f}%   (n={m['n_recipients']})",
                 fontsize=title_fs, pad=6)
    if legend:
        ax.legend(handles=[
            Line2D([0], [0], marker='o', ls='', mfc=C_OLD, mec='none', ms=8,
                   label=f'original position ({lab(G["src"])})'),
            Line2D([0], [0], marker='o', ls='', mfc=C_NEW, mec='none', ms=8,
                   label=f'after {av(G)} {G["g"]}'),
            Line2D([0], [0], marker='o', ls='', mfc=C_BG, mec='none', ms=8,
                   label='all other patients (background)'),
        ], loc='lower left', fontsize=8, frameon=True, facecolor='white',
           edgecolor='#cccccc', framealpha=0.9)


def draw_umap(ax, G, n_arrows, title=True, title_fs=11, view=None):
    xl, yl = view if view is not None else roi(G['base'], G['pert'],
                                               XY[G['src_mask']], XY[G['dst_mask']])
    ax.scatter(XY[:, 0], XY[:, 1], s=1.2, c=C_BG, alpha=0.18, linewidths=0, rasterized=True)
    ax.scatter(XY[G['src_mask'], 0], XY[G['src_mask'], 1], s=4, c=C_SRC, alpha=0.35,
               linewidths=0, rasterized=True)
    ax.scatter(XY[G['dst_mask'], 0], XY[G['dst_mask'], 1], s=9, c=C_DST, alpha=0.75,
               linewidths=0, rasterized=True)                         # emphasize destination
    fl = np.where(G['flip'])[0]; st = np.where(~G['flip'])[0]
    # flipped trajectories: green arrows with arrowheads, endpoints on top
    if len(fl):
        t = _even(fl, n_arrows); b = G['base'][t]; p = G['pert'][t]
        ax.quiver(b[:, 0], b[:, 1], p[:, 0] - b[:, 0], p[:, 1] - b[:, 1],
                  angles='xy', scale_units='xy', scale=1, color=C_FLIP, width=0.0022,
                  headwidth=4.5, headlength=5.5, alpha=0.6, zorder=4, rasterized=True)
        ax.scatter(p[:, 0], p[:, 1], s=9, c=C_FLIP, alpha=0.95, linewidths=0, zorder=5)
    # patients that did not flip: faint endpoints only (keeps the panel uncluttered)
    if len(st):
        s2 = _even(st, n_arrows); q = G['pert'][s2]
        ax.scatter(q[:, 0], q[:, 1], s=4, c=C_STAY, alpha=0.30, linewidths=0, zorder=3)
    ax.set_xlim(*xl); ax.set_ylim(*yl); clean(ax)
    if title:
        m = G['meta']
        ax.set_title(f"+{G['g']}:  {lab(G['src'])} → {lab(G['dst'])}\n"
                     f"argmax→dest {100*m['argmax_is_dest_before']:.0f}% → "
                     f"{100*m['argmax_is_dest_after']:.0f}%   (n={m['n_recipients']})",
                     fontsize=title_fs, pad=6)


def short(s, n=26):
    return s if len(s) <= n else s[:n - 1] + '…'


def draw_prob(ax, G):
    b, a = G['pdb'], G['pdp']
    bins = np.linspace(0, 1, 26)
    ax.hist(b, bins=bins, color=C_SRC, alpha=0.55, label=f'before {av(G)} {G["g"]}')
    ax.hist(a, bins=bins, color=C_DST, alpha=0.55, label=f'after {av(G)} {G["g"]}')
    ax.axvline(b.mean(), color=C_SRC, lw=1.5, ls='--'); ax.axvline(a.mean(), color=C_DST, lw=1.5, ls='--')
    ax.set_xlabel(f'P({short(G["dst"],22)})', fontsize=9)
    ax.set_ylabel('recipients', fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='upper center')
    m = G['meta']
    ax.set_title(f"mean P(dest) {m['mean_p_dest_before']:.2f}→{m['mean_p_dest_after']:.2f}   "
                 f"P(src) {m['mean_p_source_before']:.2f}→{m['mean_p_source_after']:.2f}",
                 fontsize=9)


def _sub(mask_or_idx, cap, rng):
    idx = np.where(mask_or_idx)[0] if mask_or_idx.dtype == bool else mask_or_idx
    return idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)


def make_gif(G):
    # FuncAnimation.save re-renders every artist per frame, so keep the static layers light:
    # subsample background + cluster points (they don't move; only recipients animate).
    rng = np.random.default_rng(0)
    xl, yl = roi(G['base'], G['pert'], XY[G['src_mask']], XY[G['dst_mask']])
    inroi = ((XY[:, 0] >= xl[0]) & (XY[:, 0] <= xl[1]) & (XY[:, 1] >= yl[0]) & (XY[:, 1] <= yl[1]))
    bg = _sub(inroi, 18000, rng)
    si = _sub(G['src_mask'], 5000, rng); di = _sub(G['dst_mask'], 5000, rng)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(XY[bg, 0], XY[bg, 1], s=2.5, c=C_BG, alpha=0.35, linewidths=0, rasterized=True)
    ax.scatter(XY[si, 0], XY[si, 1], s=6, c=C_SRC, alpha=0.30, linewidths=0, rasterized=True)
    ax.scatter(XY[di, 0], XY[di, 1], s=10, c=C_DST, alpha=0.55, linewidths=0, rasterized=True)
    ax.set_xlim(*xl); ax.set_ylim(*yl); clean(ax)
    ax.legend(handles=legend_handles(), loc='lower left', fontsize=7, frameon=True,
              facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    n = len(G['base'])
    sub = np.linspace(0, n - 1, min(args.max_anim, n)).astype(int) if n else np.array([], int)
    b, p = G['base'][sub], G['pert'][sub]
    fcol = np.array([C_FLIP if f else C_STAY for f in G['flip'][sub]])
    pts = ax.scatter(b[:, 0], b[:, 1], s=10, c=C_SRC, alpha=0.85, linewidths=0, zorder=5)
    ttl = ax.set_title('', fontsize=12, pad=6)
    m = G['meta']
    base_title = f"{'+' if G['mode'] == 'ki' else '−'}{G['g']}:  {short(G['src'])} → {short(G['dst'])}"

    def frame(i):
        t = i / (args.frames - 1)
        te = 0.5 - 0.5 * np.cos(np.pi * t)             # ease in/out
        pos = (1 - te) * b + te * p
        pts.set_offsets(pos)
        # color blends source->flip/stay as they move
        blend = np.array([_blend(C_SRC, fc, te) for fc in fcol])
        pts.set_color(blend)
        ttl.set_text(f"{base_title}      t={te:0.2f}")
        return pts, ttl

    anim = animation.FuncAnimation(fig, frame, frames=args.frames, interval=80, blit=False)
    out = f'{SHIFT_DIR}/anim_shift_{G["g"]}.gif'
    anim.save(out, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    print('wrote', out, flush=True)


def _blend(c1, c2, t):
    a = np.array(matplotlib.colors.to_rgb(c1)); b = np.array(matplotlib.colors.to_rgb(c2))
    return tuple((1 - t) * a + t * b)


# --------------------------------------------------------------------------- render
Gs = [load_gene(g) for g in args.genes]
Gs = [g for g in Gs if g is not None]
print(f'genes with data: {[g["g"] for g in Gs]}', flush=True)

FULL_VIEW = roi(XY, pad=0.02, q=(0.5, 99.5))          # whole map, shared by panels A & B
for G in Gs:
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(22, 7.2),
                                        gridspec_kw={'width_ratios': [1.35, 1.35, 1]})
    draw_landscape_colored(axA, FULL_VIEW, src=G['src'], dst=G['dst'], focus=True)  # A: source/dest only
    draw_positions(axB, G, FULL_VIEW)                                   # B: old vs new (same view)
    draw_prob(axC, G)                                                   # C: probability shift
    verb = 'adding' if G['mode'] == 'ki' else 'knocking out'
    fig.suptitle(f"In-silico lineage shift by {verb} {G['g']}:  "
                 f"{lab(G['src'])} → {lab(G['dst'])}", fontsize=13, y=1.01)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{SHIFT_DIR}/fig_shift_{G["g"]}.{ext}')
    plt.close(fig)
    print(f'wrote fig_shift_{G["g"]}', flush=True)
    if not args.no_gif:
        make_gif(G)

# standalone annotated landscape (full map, colored + labeled) — the global reference
fig, ax = plt.subplots(figsize=(12, 11))
lview = roi(XY, pad=0.02, q=(0.5, 99.5))
ax.scatter(XY[:, 0], XY[:, 1], s=0.6, c=C_BG, alpha=0.28, linewidths=0, rasterized=True)
tops = [t for t, _ in tcounts.most_common(60)
        if t and 'other' not in t.lower() and 'nos' not in t.lower()][:16]
for i, t in enumerate(tops):
    m = true_term == t
    ax.scatter(XY[m, 0], XY[m, 1], s=1.8, c=PAL[i % len(PAL)], alpha=0.5, linewidths=0, rasterized=True)
_labels_in_view(ax, lview, fs=8)
ax.set_xlim(*lview[0]); ax.set_ylim(*lview[1]); clean(ax)
ax.set_title('run8 disease-model embedding — cancer-lineage landscape (reference)', fontsize=13)
for ext in ('png', 'pdf'):
    fig.savefig(f'{SHIFT_DIR}/fig_lineage_landscape.{ext}')
plt.close(fig)
print('wrote fig_lineage_landscape', flush=True)

# overview: 1 x N trajectories
if Gs:
    ncol = len(Gs)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 5.0))
    if ncol == 1: axes = [axes]
    for ax, G in zip(axes, Gs):
        v = roi(G['base'], G['pert'], XY[G['src_mask']], XY[G['dst_mask']], pad=0.12)
        draw_positions(ax, G, v, title_fs=9, legend=False)
    fig.subplots_adjust(wspace=0.10, top=0.84, bottom=0.12)
    fig.legend(handles=[
        Line2D([0], [0], marker='o', ls='', mfc=C_OLD, mec='none', ms=8,
               label='recipient — original position (source lineage)'),
        Line2D([0], [0], marker='o', ls='', mfc=C_NEW, mec='none', ms=8,
               label='recipient — after adding the gene'),
        Line2D([0], [0], marker='o', ls='', mfc=C_BG, mec='none', ms=8,
               label='all other patients (background map)'),
    ], loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('In-silico lineage shifts: adding a gene moves patients toward its lineage',
                 fontsize=13, y=1.02)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{SHIFT_DIR}/fig_shift_overview.{ext}')
    plt.close(fig)
    print('wrote fig_shift_overview', flush=True)

    # flip-rate bar
    fig, ax = plt.subplots(figsize=(1.4 * len(Gs) + 2, 4.2))
    x = np.arange(len(Gs)); w = 0.38
    before = [g['meta']['argmax_is_dest_before'] * 100 for g in Gs]
    after = [g['meta']['argmax_is_dest_after'] * 100 for g in Gs]
    ax.bar(x - w / 2, before, w, color=C_SRC, label='before adding gene')
    ax.bar(x + w / 2, after, w, color=C_DST, label='after adding gene')
    for xi, a in zip(x, after):
        ax.text(xi + w / 2, a + 1, f'{a:.0f}%', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([g['g'] for g in Gs], fontsize=10)
    ax.set_ylabel('% of source patients with argmax = dest', fontsize=9)
    ax.set_title('Lineage conversion rate (disease-head argmax → destination)', fontsize=11)
    ax.tick_params(labelsize=8); ax.legend(fontsize=9, frameon=False)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{SHIFT_DIR}/fig_shift_fliprates.{ext}')
    plt.close(fig)
    print('wrote fig_shift_fliprates', flush=True)

print('ALL SHIFT FIGURES DONE', flush=True)
