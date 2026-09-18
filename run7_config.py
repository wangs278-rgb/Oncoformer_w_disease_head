#!/usr/bin/env python3
"""
Shared config builder for run7 — DNA-only, from-scratch, MLM + MoCo (DX1/DX2),
but with the SAME OUTPUT HEADS AS run5: only `gene` and `aa_vaf_bin` are predicted.

run7 = run6 (Michal's Oncoformer_moco CLIP-MoCo, DX1/DX2, clip_meta_w=0.3) with the
MLM output restricted to run5's two heads. run6 inherited ~10 predict heads from
Michal's Hydra `dna_fmi` config; run5 (dx12_config.py) predicts ONLY gene + aa_vaf_bin.
This module reuses run6_moco_config.build_config() and then flips the component
`predict` flags so the reconstruction output exactly matches run5, keeping everything
else (architecture, MoCo, data subset, hyperparameters) identical to run6.

NOTE: run5's output has NO pathogenicity head, so run7 does NOT produce the VUS
driver-score (that was a run6-only head). This is intentional ("same output as run5").

Tokenization is unaffected by predict flags, so run7 REUSES run6's tokenized cache
(tokenized_dna_dx12_moco.pt) — no separate pre-tokenization job is needed.
"""
import run6_moco_config

# Re-export so train/dryrun scripts can call these off run7_config directly.
use_moco_oncoformer = run6_moco_config.use_moco_oncoformer
DATA_DIR = run6_moco_config.DATA_DIR

# run5's predicted heads (dx12_config.py: predict=True only on these two).
RUN5_PREDICT_HEADS = {'gene', 'aa_vaf_bin'}


def build_config(tokenized_data_path, clip_meta_w=0.3):
    """run6 config, with component `predict` flags reduced to run5's output set."""
    config = run6_moco_config.build_config(tokenized_data_path, clip_meta_w=clip_meta_w)

    cc = config['omics']['modalities']['dna']['architecture']['component_configs']
    for name, comp in cc.items():
        if name in RUN5_PREDICT_HEADS:
            comp['predict'] = True
            comp['loss_param'] = 1.0        # match run5's loss weighting
        else:
            comp['predict'] = False         # off (incl. run6 cross-preds protein/mutation/aa_vaf)
            comp.pop('loss_param', None)     # drop stale auxiliary loss weights

    return config


if __name__ == '__main__':
    c = build_config('/tmp/_run7_probe.pt')
    cc = c['omics']['modalities']['dna']['architecture']['component_configs']
    print('run7 component predict flags:')
    for k, v in cc.items():
        print(f'  {k:16s} encode={v.get("encode")!s:5s} predict={v.get("predict")!s:5s} '
              f'loss_param={v.get("loss_param", "-")}')
    pred = [k for k, v in cc.items() if v.get('predict') is True]
    print('\npredict=True heads:', pred)
    assert set(pred) == RUN5_PREDICT_HEADS, f'expected {RUN5_PREDICT_HEADS}, got {set(pred)}'
    print('clip_meta_w     :', c['training']['clip_meta_w'])
    print('recon_meta_w    :', c['training']['recon_meta_w'])
    print('baitset_filter  :', c['omics']['modalities']['dna']['data']['baitset_filter'])
    print('OK — output heads match run5 (gene + aa_vaf_bin).')
