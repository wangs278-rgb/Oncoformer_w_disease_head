#!/usr/bin/env python3
"""Diagnostic: is the continuous aa_vaf channel populated, and can we estimate per-sample tumour
purity from it? (purity ~= 2 x top VAF mode of diploid heterozygous SNVs). No model / CPU-ok.
    python vaf_diag.py [--scan-batches 40]
"""
import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
CACHE = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CHUNK = 512; REAL_MIN = 4
ap = argparse.ArgumentParser(); ap.add_argument('--scan-batches', type=int, default=40); a = ap.parse_args()

sys.path.insert(0, DATA_DIR)
import run8_disease_config
run8_disease_config.use_moco_oncoformer()
config = run8_disease_config.build_config(CACHE)
from oncoformer.dataset import OncoformerDataLoader
from torch.utils.data import DataLoader
import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
at2i = tok.component_token2idx['alt_type']; zy2i = tok.component_token2idx['zygosity']
SNV = {at2i[k] for k in ['missense', 'nonsense', 'frameshift', 'splice', 'nonframeshift',
                          'nonstart', 'nonstop', 'deleterious'] if k in at2i}
HET = zy2i['heterozygous']
print('SNV alt_type ids:', sorted(SNV), ' HET id:', HET, flush=True)

dl = OncoformerDataLoader(config, load_metadata=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
allvaf = []; per_purity = []; n_hetsnv = []; comp_keys = None
for bi, batch in enumerate(loader):
    dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
    if comp_keys is None:
        comp_keys = list(dna.keys()); print('comp keys:', comp_keys, flush=True)
        print('has aa_vaf channel:', 'aa_vaf' in comp_keys, flush=True)
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    gf = dna['gene'].numpy(); mk = mask.numpy()
    vaf = dna['aa_vaf'].numpy() if 'aa_vaf' in comp_keys else None
    at = dna['alt_type'].numpy(); zy = dna['zygosity'].numpy()
    if vaf is None:
        print('NO aa_vaf channel; cannot estimate CCF'); break
    for r in np.where(dx)[0]:
        real = (gf[r] >= REAL_MIN) & (mk[r] > 0)
        v_all = vaf[r][real]
        allvaf.extend(v_all[np.isfinite(v_all)].tolist())
        hs = real & np.isin(at[r], list(SNV)) & (zy[r] == HET)
        vh = vaf[r][hs]; vh = vh[np.isfinite(vh) & (vh > 0)]
        n_hetsnv.append(int(vh.size))
        if vh.size >= 3:
            per_purity.append(min(1.0, 2.0 * np.quantile(vh, 0.95)))
    if bi + 1 >= a.scan_batches:
        break
allvaf = np.array(allvaf); per_purity = np.array(per_purity); n_hetsnv = np.array(n_hetsnv)
print(f'\n=== continuous VAF over {allvaf.size} real mutations ===', flush=True)
if allvaf.size:
    qs = np.quantile(allvaf, [0, .05, .25, .5, .75, .95, 1])
    print('VAF quantiles [0,5,25,50,75,95,100]:', np.round(qs, 3).tolist(), flush=True)
    print('distinct values (first 15):', np.round(np.unique(allvaf)[:15], 4).tolist(), flush=True)
print(f'\n=== per-sample het-SNV counts ===', flush=True)
print(f'samples with >=3 het SNVs: {(n_hetsnv>=3).mean()*100:.0f}%  median count={np.median(n_hetsnv):.0f}', flush=True)
if per_purity.size:
    pq = np.quantile(per_purity, [0, .25, .5, .75, 1])
    print(f'estimated purity quantiles [0,25,50,75,100]: {np.round(pq,2).tolist()}  (n={per_purity.size})', flush=True)
print('\nDIAG DONE', flush=True)
