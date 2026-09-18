#!/usr/bin/env python3
"""
Shared config builder for the DX1/DX2-only Oncoformer DNA-MLM variant.

Identical to train.py's run4 config EXCEPT:
  - config[...]['dna']['data']['baitset_filter'] = ['DX1', 'DX2']
  - tokenized_data_path is passed in by the caller

Side-effect free: importing this module builds nothing. Both pretokenize_dx12.py
and train_dx12.py import build_config() so their configs can never drift (a
mismatch would make the training job re-tokenize and race across DDP ranks).
"""

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
REPO     = '/cv/home/wangs278/scratch/fmi/Oncoformer'

PROTEIN_W  = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.protein.weight.pt'
PROTEIN_V  = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.protein.vocab.json'
MUTATION_W = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.mutation.weight.pt'
MUTATION_V = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.mutation.vocab.json'

# Baitsets to keep for this variant.
BAITSET_FILTER = ['DX1', 'DX2']


def build_config(tokenized_data_path):
    """Return the run4 config dict, restricted to DX1/DX2, using the given
    tokenized cache path."""
    config = {
        'anndata': f'{DATA_DIR}/pretraining_data_full.h5',
        'sample_metadata_path': f'{DATA_DIR}/sample_metadata.pkl',
        'data_file_prefix': '',
        'sample_index': {'join': 'union'},
        'omics': {
            'architecture': {
                'modalities': ['dna'],
                'embed_dim': 512,
                'num_heads': 8,
                'hidden_dim': 512,
                'num_layers': 4,
                'dropout': 0.3,
                'pad_idx': 0,
                'pool_embeddings': 'cls_token',
                'pool_attn': 'mean',
                'stitch': {'enabled': False},
            },
            'modalities': {
                'dna': {
                    'pad_idx': 0,
                    'architecture': {
                        'max_length': 128,
                        'component_configs': {
                            'gene': {
                                'encoder_type': 'LayerNormEncoder',
                                'encode': False, 'predict': True,
                                'params': {'num_embeddings': True, 'embedding_dim': 512, 'padding_idx': 0},
                                'combination': 'concat', 'loss_param': 1.0,
                            },
                            'alt_type': {
                                'encoder_type': 'LayerNormEncoder',
                                'encode': False, 'predict': False,
                                'params': {'num_embeddings': True, 'embedding_dim': 512, 'padding_idx': 0},
                                'combination': 'concat',
                            },
                            'pathogenicity': {
                                'encoder_type': 'LayerNormEncoder',
                                'encode': False, 'predict': False,
                                'params': {'num_embeddings': True, 'embedding_dim': 512, 'padding_idx': 0},
                                'combination': 'concat',
                            },
                            'zygosity': {
                                'encoder_type': 'LayerNormEncoder',
                                'encode': False, 'predict': False,
                                'params': {'num_embeddings': True, 'embedding_dim': 512, 'padding_idx': 0},
                                'combination': 'concat',
                            },
                            'aa_ref': {
                                'encoder_type': 'LayerNormEncoder',
                                'encode': False, 'predict': False,
                                'params': {'num_embeddings': True, 'embedding_dim': 256, 'padding_idx': 0},
                                'combination': 'concat',
                            },
                            'aa_mut': {
                                'encoder_type': 'LayerNormEncoder',
                                'encode': False, 'predict': False,
                                'params': {'num_embeddings': True, 'embedding_dim': 256, 'padding_idx': 0},
                                'combination': 'concat',
                            },
                            'protein': {
                                'encoder_type': 'PrecomputedEmbeddingsEncoder',
                                'encode': True, 'predict': False,
                                'params': {
                                    'weight_path': PROTEIN_W,
                                    'vocab_path':  PROTEIN_V,
                                    'trainable': False,
                                },
                                'combination': 'concat',
                            },
                            'mutation': {
                                'encoder_type': 'PrecomputedEmbeddingsAtomicEncoder',
                                'encode': True, 'predict': False,
                                'params': {
                                    'weight_path': MUTATION_W,
                                    'vocab_path':  MUTATION_V,
                                    'dtype': 'bf16',
                                    'trainable_base': False,
                                    'trainable_atomic': True,
                                },
                                'combination': 'concat',
                            },
                            'aa_vaf': {
                                'encoder_type': 'FourierFractionEncoder',
                                'encode': True, 'predict': False,
                                'params': {
                                    'embedding_dim': 512, 'fourier_dim': 32,
                                    'f_min': 1.0, 'f_max': 64.0, 'add_logit': True,
                                },
                                'combination': 'sum',
                            },
                            'aa_vaf_bin': {
                                'encoder_type': 'LayerNormEncoder',
                                'encode': False, 'predict': True,
                                'params': {'num_embeddings': True, 'embedding_dim': 128, 'padding_idx': 0},
                                'combination': 'concat', 'loss_param': 1.0,
                            },
                        },
                    },
                    'data': {
                        'source': 'anndata_uns',
                        'anndata_uns_key': 'unpaired',
                        'anndata_uns_patient_metadata_cols': ['BaitSet', 'DiseaseTerm', 'DiseaseGroup'],
                        'tokenizer_save_path': f'{DATA_DIR}/tokenizer_dna.pkl',
                        'tokenized_data_path': tokenized_data_path,
                        'baitset_filter': BAITSET_FILTER,
                        'rank_by_component': 'gene',
                        'rank_direction': 'asc',
                        'protein_weight_path': PROTEIN_W,
                        'protein_vocab_path':  PROTEIN_V,
                        'mutation_weight_path': MUTATION_W,
                        'mutation_vocab_path':  MUTATION_V,
                    },
                    'vocab': {'file': f'{DATA_DIR}/vocab_dna.json'},
                }
            },
        },
        'training': {
            'seed': 42,
            'optimizer': 'AdamW',
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'scheduler_curve': 'cosine',
            'num_epochs': 40,
            'train_test_split': 0.8,
            'batch_size': 512,
            'scheduler_warmup_fraction': 0.1,
            'mlm_targets': ['dna'],
            'dna_masking_rate': 0.25,
            'dna_val_mask_rate': 0.15,
            'w_dino': 0.0,
            'recon_meta_w': 1.0,
            'dino_prototypes': 1024,
            'dino_hidden': 2048,
            'dino_bottleneck': 512,
            'dino_head_lr_mult': 2.0,
            'dino_head_decay': 5e-4,
            'dino_student_temp': 0.13,
            'dino_teacher_temp_base': 0.10,
            'dino_teacher_temp_final': 0.06,
            'dino_teacher_m_base': 0.998,
            'dino_teacher_m_final': 0.9997,
            'dino_center_momentum': 0.99,
            'dna_contrast_mask_rate': 0.25,
            'moddrop_student_p': 0.0,
            'moddrop_teacher_p': 0.0,
            'context_moddrop_p': 0.0,
            'byol': {'enabled': False},
        },
        'sampler': {'enabled': True, 'cohorts': {'matched': 1.0}},
        'metadata': {},
        'visualization': {'save_self_attn': False},
    }
    return config
