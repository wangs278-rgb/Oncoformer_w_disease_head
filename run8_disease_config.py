#!/usr/bin/env python3
"""
Shared config builder for run8_disease — run6 (DNA-only, MLM + single-modality MoCo,
DX1/DX2) PLUS a supervised DiseaseTerm classification head, trained jointly during a
warm-started continuation of pretraining.

How the disease head is added WITHOUT any model-code change: Oncoformer_moco's
`OncoformerPost` (models.py) wraps a pretraining backbone and, when
`training.pre_train_model=True`, runs the backbone's MLM+MoCo training_step AND a
supervised metadata-classification loss, summing them. Each `metadata.data.discrete`
task becomes a `Classification` head (CE, ignore_index=-100, label_smoothing=0.1) on the
backbone's pooled embedding via a per-task projection, weighted heteroscedastically
(exp(-log_var)*loss + log_var) * meta_weight.

This module reuses run6_moco_config.build_config() (inherits DX1/DX2 baitset filter, ESM /
tokenizer / vocab paths, clip_meta_w=0.3, recon_meta_w=1.0, wandb off) and then:
  - points `metadata` at the run8 DiseaseTerm targets + vocab built by
    build_disease_metadata_run8.py (511 classes),
  - sets a conservative disease meta_weight=0.3,
  - flips training.pre_train_model=True and shortens to 20 epochs (warm-start budget).

Tokenization is unaffected by metadata, so run8 REUSES run6's tokenized cache
(tokenized_dna_dx12_moco.pt) — no pre-tokenization job (same precedent as run7).
"""
import os
import run6_moco_config

# Re-export so train/dryrun scripts can call these off run8_disease_config directly.
use_moco_oncoformer = run6_moco_config.use_moco_oncoformer
DATA_DIR = run6_moco_config.DATA_DIR

# Artifacts produced by build_disease_metadata_run8.py
META_DATA_PATH  = f'{DATA_DIR}/metadata_disease_run8.pt'
META_VOCAB_PATH = f'{DATA_DIR}/vocab_metadata_disease_run8.json'

TASK_NAME        = 'disease_term'
DISEASE_WEIGHT   = 0.3          # conservative auxiliary weight (user choice)
IGNORE_PATTERNS  = ['other', 'nos']
NUM_EPOCHS       = 20           # warm-start continuation budget (user choice); backbone starts from run6 ep39

# Warm-start source: run6's final backbone checkpoint (used by train_run8_disease.py on cold start).
RUN6_CKPT = f'{DATA_DIR}/run6_moco_out/checkpoints/last.ckpt'


def build_config(tokenized_data_path, clip_meta_w=0.3):
    """run6 config + a DiseaseTerm metadata head, in joint-pretraining (pre_train_model) mode."""
    config = run6_moco_config.build_config(tokenized_data_path, clip_meta_w=clip_meta_w)

    # Metadata / disease-head task definition (consumed by OncoformerPost + dataset load_metadata).
    config['metadata'] = {
        'data': {
            'discrete': {
                TASK_NAME: {'meta_weight': DISEASE_WEIGHT, 'ignore_patterns': list(IGNORE_PATTERNS)},
            },
            'data_path': META_DATA_PATH,
        },
        'vocab': {'file': META_VOCAB_PATH},
        'architecture': {},          # no confounder / hypernetwork
    }

    # Joint pretraining: OncoformerPost trains backbone (MLM+MoCo) + disease head together.
    config['training']['pre_train_model'] = True
    config['training']['fine_tune_model'] = False
    config['training']['num_epochs'] = NUM_EPOCHS

    return config


if __name__ == '__main__':
    c = build_config('/tmp/_run8_probe.pt')
    disc = c['metadata']['data']['discrete']
    print('metadata.discrete   :', {k: v for k, v in disc.items()})
    print('metadata.data_path  :', c['metadata']['data']['data_path'])
    print('metadata.vocab.file :', c['metadata']['vocab']['file'])
    print('pre_train_model     :', c['training']['pre_train_model'])
    print('num_epochs          :', c['training']['num_epochs'])
    print('clip_meta_w         :', c['training']['clip_meta_w'])
    print('recon_meta_w        :', c['training']['recon_meta_w'])
    print('baitset_filter      :', c['omics']['modalities']['dna']['data']['baitset_filter'])
    print('run6 warm ckpt      :', RUN6_CKPT, '(exists)' if os.path.exists(RUN6_CKPT) else '(MISSING!)')

    assert c['training']['pre_train_model'] is True
    assert c['training']['fine_tune_model'] is False
    assert set(disc.keys()) == {TASK_NAME}, f'expected discrete task {TASK_NAME}, got {set(disc.keys())}'
    assert abs(c['training']['clip_meta_w'] - 0.3) < 1e-9
    assert c['omics']['modalities']['dna']['data']['baitset_filter'] == ['DX1', 'DX2']
    # Data artifacts must be built before training (build_disease_metadata_run8.py).
    for p in (META_DATA_PATH, META_VOCAB_PATH):
        print(f'artifact {os.path.basename(p):34s}:', 'present' if os.path.exists(p) else 'MISSING — run build_disease_metadata_run8.py')
    print('OK — config self-check passed.')
