#!/usr/bin/env python3
"""
One-time pre-tokenization for the DX1/DX2-only Oncoformer variant.

Runs in a SINGLE process (no Trainer, no GPU). Constructing OncoformerDataLoader
tokenizes the DX1/DX2-filtered DNA data and saves the cache to DX12_CACHE inside
the dataset constructor (dataset.py:405). This must be done once before the
multi-GPU training job, so all 4 DDP ranks load the cache on existence instead
of racing to build it.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import dx12_config
sys.path.insert(0, dx12_config.REPO)

import lightning as L
from oncoformer.dataset import OncoformerDataLoader

DX12_CACHE = f'{dx12_config.DATA_DIR}/tokenized_dna_dx12.pt'

if os.path.exists(DX12_CACHE):
    print(f'ERROR: {DX12_CACHE} already exists. Delete it to rebuild (the loader '
          f'has no cache validation and would silently reuse it).')
    sys.exit(1)

config = dx12_config.build_config(DX12_CACHE)

print('=== Pre-tokenizing DX1/DX2 DNA data (single process) ===')
L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)

n = len(dl.dataset.tokenized_by_mod['dna'])
print(f'=== Done. DX1/DX2 tokenized samples: {n} ===')
print(f'=== Cache written to: {DX12_CACHE} ===')
assert os.path.exists(DX12_CACHE), 'cache file was not written!'
