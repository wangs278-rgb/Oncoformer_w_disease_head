#!/usr/bin/env python3
"""
One-time preprocessing for run8_disease: build the per-sample DiseaseTerm targets
(`metadata_disease_run8.pt`) and the disease vocab (`vocab_metadata_disease_run8.json`)
that the metadata prediction-head pathway of Oncoformer_moco consumes.

run8_disease = run6 (DNA-only, MLM + single-modality MoCo, DX1/DX2) + a supervised
DiseaseTerm classification head, trained jointly during (warm-started) pretraining via
OncoformerPost(pre_train_model=True). That pathway needs two artifacts that don't ship
with the converge data:

  1. vocab: metadata/vocab/fmi.yaml -> {"disease_term": {"levels": [...]}}. The label id
     of a term is its index in `levels`. OncoformerPost sizes the Classification head as
     len(levels) and derives ignore ids from `ignore_patterns` against these levels.
  2. targets: metadata/data/fmi.yaml data_path (a torch pickle). dataset.py loads it via
     load_tensor() when it already exists (skipping the adata.obsm path that our anndata
     lacks) and attaches metadata[sample_id] to every sample (dataset.py:832-833) — so it
     MUST cover every id in sample_metadata.index or the assembler KeyErrors.

Targets are built directly from sample_metadata.pkl['DiseaseTerm'] (511 classes, string
index of unique sample ids, no NaN — verified), so no anndata.obsm preprocessing is needed.
Per-sample target shape is (1,): collate stacks -> (B,1); OncoformerPost._common_step does
squeeze(-1) -> (B,). '(nos)'/'other' terms are kept as vocab levels but excluded from the
loss at train time (ignore_patterns -> -100), handled by OncoformerPost + Classification.
"""
import os, json
import torch
import pandas as pd

DATA_DIR   = '/cv/home/wangs278/scratch/fmi'
SM_PATH    = f'{DATA_DIR}/sample_metadata.pkl'
VOCAB_OUT  = f'{DATA_DIR}/vocab_metadata_disease_run8.json'
META_OUT   = f'{DATA_DIR}/metadata_disease_run8.pt'

TASK_NAME       = 'disease_term'     # must match run8_disease_config discrete key
SOURCE_COLUMN   = 'DiseaseTerm'      # column in sample_metadata.pkl
IGNORE_PATTERNS = ['other', 'nos']   # informational here; enforced at train time by OncoformerPost


def main():
    print(f'Loading {SM_PATH} ...')
    sm = pd.read_pickle(SM_PATH)
    assert SOURCE_COLUMN in sm.columns, f'{SOURCE_COLUMN} missing from sample_metadata columns {list(sm.columns)}'
    n = len(sm)
    assert sm.index.is_unique, 'sample_metadata index must be unique (used as target key)'
    assert sm[SOURCE_COLUMN].isna().sum() == 0, f'{SOURCE_COLUMN} has NaNs; add an explicit unknown level'

    # 1) Vocab: deterministic level ordering; label id = index in levels.
    levels = sorted(sm[SOURCE_COLUMN].astype(str).unique().tolist())
    term2id = {t: i for i, t in enumerate(levels)}
    vocab = {TASK_NAME: {'levels': levels}}
    with open(VOCAB_OUT, 'w') as f:
        json.dump(vocab, f)
    print(f'Wrote {VOCAB_OUT}: {len(levels)} classes')

    # 2) Targets keyed by the exact sample_metadata index values (str ids like "XRN:0003UY").
    ids  = sm.index.astype(str).tolist()
    vals = sm[SOURCE_COLUMN].astype(str).tolist()
    metadata = {}
    for sid, term in zip(ids, vals):
        tid = term2id[term]
        metadata[sid] = {('discrete', TASK_NAME, 'targets'): torch.tensor([tid], dtype=torch.long)}
    assert len(metadata) == n, f'coverage {len(metadata)} != {n} samples'
    torch.save(metadata, META_OUT)
    print(f'Wrote {META_OUT}: {len(metadata)} sample targets (full coverage)')

    # 3) Report how many levels/samples the ignore_patterns will drop from the disease loss.
    ignored_ids = [i for i, lvl in enumerate(levels)
                   if any(p in lvl.lower() for p in IGNORE_PATTERNS)]
    ignored_mask = sm[SOURCE_COLUMN].astype(str).str.lower().apply(
        lambda s: any(p in s for p in IGNORE_PATTERNS))
    print(f'ignore_patterns={IGNORE_PATTERNS}: {len(ignored_ids)} levels / '
          f'{int(ignored_mask.sum())} of {n} samples ({100*ignored_mask.mean():.1f}%) '
          f'-> target -100 (excluded from disease loss, still used for MLM+MoCo)')
    # Sanity: reload and spot-check one entry round-trips.
    _m = torch.load(META_OUT)
    _sid = ids[0]
    _t = _m[_sid][('discrete', TASK_NAME, 'targets')]
    assert _t.shape == (1,) and _t.dtype == torch.long and levels[int(_t)] == vals[0]
    print(f'OK — round-trip check passed (e.g. {_sid} -> id {int(_t)} = "{levels[int(_t)]}")')


if __name__ == '__main__':
    main()
