#!/usr/bin/env python3
"""Generate synthetic patients with the CCFDecoder, ONE per real held-out patient of a disease (run8 validation split), conditioned on that
patient's own prefix (disease, purity4, site) -> matched real/synthetic comparison.  Temperature 1, repeat rule 'specific' (as the earlier decoder's
cohorts: a specific variant at most once per patient).  Every generated patient is checked against the grammar.
Writes <out>.npz: offsets (ragged arrays gene / tok / conc / cbin / seg, in generation order), n, truncated, ref (real patient index), dis, pur, site.
Usage: python generate_ccf.py --ckpt checkpoints/epoch_7.pt --out gen/crc_epoch7.npz [--disease 'colon adenocarcinoma (crc)'] [--seed 2026] [--dry]
       [--data-dir <run9 data> (default $RUN9_DATA)] [--ccf-dir <ccf data> (default $CCF_DATA or ./data)]"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ccf_data import CCFData, CCF_DIR, env_dir       # noqa: E402
from ccf_model import build_ccf_decoder, MAX_ALTS, N_BIN   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--disease', default='colon adenocarcinoma (crc)')
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--temperature', type=float, default=1.0)
ap.add_argument('--gen-bs', type=int, default=1024)
ap.add_argument('--dry', action='store_true', help='64 patients')
ap.add_argument('--data-dir', default=None, help='run9 data dir; default $RUN9_DATA')
ap.add_argument('--ccf-dir', default=CCF_DIR)
args = ap.parse_args()
DATA = env_dir('RUN9_DATA', args.data_dir)
T0 = time.time()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


D = CCFData(DATA, args.ccf_dir)
V = D.vocab
levels = [str(x) for x in V['levels']]
d = levels.index(args.disease)
ref = np.nonzero((D.split == 1) & D.labeled & (D.disease == d))[0]
if args.dry:
    ref = ref[:64]
log(f'{args.disease}: {len(ref)} real held-out patients -> one synthetic patient each; checkpoint {args.ckpt}')
emb = torch.load(f'{DATA}/embeddings.pt', map_location='cpu')
init = torch.load(f'{DATA}/run8_init.pt', map_location='cpu')
model = build_ccf_decoder(V, emb, init, dropout=0.0, n_sig=len(D.pur_labels), n_site=len(D.site_labels))
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
model.load_state_dict(ck['model'])
model.to(dev).eval()
log(f'checkpoint epoch(0-based)={ck.get("epoch")} val={ck.get("val")}')
AMBIG = {'unknown', '<unk>', 'splice', 'nonstart', 'nonstop'}
NOREP = ~np.isin(V['alt_names'][V['conc_alt']], list(AMBIG))
g = torch.Generator(device=dev).manual_seed(args.seed)
cols = {k: [] for k in ('gene', 'tok', 'conc', 'cbin', 'seg')}
ns, tr = [], []
for s in range(0, len(ref), args.gen_bs):
    ids = ref[s:s + args.gen_bs]
    out = model.generate(torch.as_tensor(D.dis_in[ids], device=dev), temperature=args.temperature, max_alts=MAX_ALTS, norep=NOREP, generator=g,
                         sig=torch.as_tensor(D.pur[ids], device=dev), site=torch.as_tensor(D.site[ids], device=dev))
    o = {k: v.cpu().numpy() for k, v in out.items()}
    for r in range(len(ids)):
        n = int(o['n'][r])
        p = {k: o[k][r, :n] for k in cols}
        assert ((p['cbin'] >= 0) == (p['seg'] == 0)).all() and (np.diff(p['seg']) >= 0).all(), 'segment structure'
        assert (np.diff(p['cbin'][p['seg'] == 0]) <= 0).all(), 'bins must be non-increasing in the timed block'
        assert (np.diff(p['gene'][p['seg'] == 1]) >= 0).all(), 'untimed block must keep the rank order'
        assert (V['conc_gene_idx'][p['conc']] == p['gene']).all() and (p['cbin'] < N_BIN).all()
        fl = p['conc'][NOREP[p['conc']]]
        assert len(np.unique(fl)) == len(fl), 'repeat rule violated'
        for k in cols:
            cols[k].append(p[k])
        ns.append(n); tr.append(bool(o['truncated'][r]))
    log(f'{s + len(ids)}/{len(ref)} generated')
ns = np.array(ns)
np.savez_compressed(args.out, offsets=np.concatenate([[0], np.cumsum(ns)]), n=ns, truncated=np.array(tr), ref=ref, dis=D.dis_in[ref], pur=D.pur[ref], site=D.site[ref],
                    **{k: np.concatenate(v) for k, v in cols.items()})
log(f'saved {args.out}: {len(ns)} patients, median {np.median(ns):.0f} alterations, truncated {int(np.sum(tr))}')
