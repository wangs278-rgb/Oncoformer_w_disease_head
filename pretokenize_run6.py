#!/usr/bin/env python3
"""
One-time pre-tokenization for run6 (DNA-only MLM + MoCo, DX1/DX2), using MICHAL's
Oncoformer code (Oncoformer_moco/). Runs in a SINGLE process (no Trainer, no GPU):
constructing OncoformerDataLoader tokenizes the DX1/DX2-filtered DNA data and saves
the cache to RUN6_CACHE inside the dataset constructor (dataset.py:605). Built with
Michal's diverged dataset.py (do NOT reuse run5's fmi-built cache). Must run once
before the multi-GPU job so all 4 DDP ranks load the cache instead of racing.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import run6_moco_config
run6_moco_config.use_moco_oncoformer()   # force oncoformer.* -> Oncoformer_moco (over fmi editable)

import lightning as L
import oncoformer
from oncoformer.dataset import OncoformerDataLoader
assert 'Oncoformer_moco' in oncoformer.dataset.__file__, f'wrong oncoformer: {oncoformer.dataset.__file__}'

RUN6_CACHE = f'{run6_moco_config.DATA_DIR}/tokenized_dna_dx12_moco.pt'

if os.path.exists(RUN6_CACHE):
    print(f'ERROR: {RUN6_CACHE} already exists. Delete it to rebuild (the loader '
          f'has no cache validation and would silently reuse it).')
    sys.exit(1)

config = run6_moco_config.build_config(RUN6_CACHE)

print('=== Pre-tokenizing DX1/DX2 DNA data for run6/MoCo (single process) ===')
print('oncoformer from:', os.path.dirname(oncoformer.dataset.__file__))   # must be Oncoformer_moco
L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)

n = len(dl.dataset.tokenized_by_mod['dna'])
print(f'=== Done. DX1/DX2 tokenized samples: {n} ===')
print(f'=== Cache written to: {RUN6_CACHE} ===')
assert os.path.exists(RUN6_CACHE), 'cache file was not written!'
