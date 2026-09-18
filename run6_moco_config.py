#!/usr/bin/env python3
"""
Shared config builder for run6 — DNA-only, from-scratch, MLM + MoCo (DX1/DX2).

run6 is built on MICHAL's Oncoformer code (copied to Oncoformer_moco/), which has
CLIP-MoCo. For a single-modality (DNA-only) model the CLIP-MoCo loss automatically
falls back to standard MoCo (two augmented views). This module loads Michal's Hydra
`dna_fmi` config and rewrites all paths to the converge data files, restricts to the
DX1/DX2 baitset, sets a conservative MoCo weight, and disables the metadata task and
wandb. Imported by both pretokenize_run6.py and train_run6.py so they can never drift.
"""
import os
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO       = '/cv/home/wangs278/scratch/fmi/Oncoformer_moco'
CONFIG_DIR = os.path.join(REPO, 'config')
DATA_DIR   = '/cv/home/wangs278/scratch/fmi'


def use_moco_oncoformer():
    """Force `import oncoformer` (and its submodules) to resolve to Oncoformer_moco.

    The fmi package is installed as a PEP-660 editable, whose meta-path finder
    (__editable___oncoformer_0_0_1_finder) maps `oncoformer.*` submodules to
    fmi/Oncoformer regardless of sys.path. Left in place, run6 would silently import
    fmi's code (no MoCo). We drop that finder, clear any cached oncoformer modules,
    and put our copy first on sys.path. Call BEFORE importing oncoformer.
    """
    import sys
    for name in [n for n in list(sys.modules) if n == 'oncoformer' or n.startswith('oncoformer.')]:
        del sys.modules[name]
    sys.meta_path = [f for f in sys.meta_path
                     if getattr(f, '__module__', '') != '__editable___oncoformer_0_0_1_finder']
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def _remap_datasets_paths(obj):
    """Recursively rewrite './datasets/<f>' -> '<DATA_DIR>/<f>' (basename remap).

    The converge ESM embeddings, tokenizer, vocab and sample_metadata share the same
    basenames as Michal's ./datasets/ files, so a basename remap resolves them. Files
    whose basename differs (anndata) or are run6-specific (tokenized/raw cache) are set
    explicitly afterwards.
    """
    if isinstance(obj, dict):
        return {k: _remap_datasets_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_remap_datasets_paths(v) for v in obj]
    if isinstance(obj, str) and obj.startswith('./datasets/'):
        return os.path.join(DATA_DIR, os.path.basename(obj))
    return obj


def build_config(tokenized_data_path, clip_meta_w=0.3):
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        # metadata=empty: run6 is pure MLM + MoCo (no disease-prediction task, no metadata.pt)
        cfg = compose(config_name='dna_fmi', overrides=['metadata=empty'])
    config = OmegaConf.to_container(cfg, resolve=True)

    # 1) Use paths verbatim (disable the 'dna_fmi_' filename prefix).
    config['data_file_prefix'] = ''

    # 2) Generic basename remap of every ./datasets/* path to the converge files.
    config = _remap_datasets_paths(config)

    # 3) Explicit overrides for files whose basename differs or are run6-specific.
    config['anndata'] = f'{DATA_DIR}/pretraining_data_full.h5'
    config['sample_metadata_path'] = f'{DATA_DIR}/sample_metadata.pkl'
    dna_data = config['omics']['modalities']['dna']['data']
    dna_data['tokenized_data_path'] = tokenized_data_path
    dna_data['raw_data_path'] = f'{DATA_DIR}/raw_dna_run6.pt'
    dna_data['baitset_filter'] = ['DX1', 'DX2']

    # 4) run6 loss weights, disable wandb, normalise empty metadata.
    config['training']['clip_meta_w'] = clip_meta_w   # conservative MoCo weight
    config['training']['recon_meta_w'] = 1.0          # MLM stays primary
    if isinstance(config['training'].get('wandb'), dict):
        config['training']['wandb']['enabled'] = False
    if config.get('metadata') is None:
        config['metadata'] = {}

    return config


if __name__ == '__main__':
    # Quick self-check (no data touched): resolves the config and prints key fields.
    c = build_config('/tmp/_run6_probe.pt')
    print('modalities        :', c['omics']['architecture']['modalities'])
    print('anndata           :', c['anndata'])
    print('tokenized_data    :', c['omics']['modalities']['dna']['data']['tokenized_data_path'])
    print('baitset_filter    :', c['omics']['modalities']['dna']['data']['baitset_filter'])
    print('clip_meta_w       :', c['training']['clip_meta_w'])
    print('recon_meta_w      :', c['training']['recon_meta_w'])
    print('clip_queue_size   :', c['training']['clip_queue_size'])
    print('dna_contrast_mask :', c['training']['dna_contrast_mask_rate'])
    print('wandb.enabled     :', c['training'].get('wandb', {}).get('enabled'))
    print('metadata          :', c['metadata'])
    prot = c['omics']['modalities']['dna']['architecture']['component_configs']['protein']['params']['weight_path']
    print('protein ESM path  :', prot)
