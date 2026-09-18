#!/usr/bin/env python3
"""
Oncoformer DNA-only pretraining  —  MLM-only
Heads: gene + aa_vaf_bin only
DINO: disabled (w_dino = 0.0)
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
REPO     = '/cv/home/wangs278/scratch/fmi/Oncoformer'
OUT_DIR  = '/cv/home/wangs278/scratch/fmi/run4_out'

sys.path.insert(0, REPO)

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics

PROTEIN_W  = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.protein.weight.pt'
PROTEIN_V  = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.protein.vocab.json'
MUTATION_W = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.mutation.weight.pt'
MUTATION_V = f'{DATA_DIR}/esm_unpaired_layer_33_diff_embedding_full.mutation.vocab.json'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

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
                    'tokenized_data_path': (
                        f'/dev/shm/onco_tokenized_dna_{os.environ["SLURM_JOB_ID"]}.pt'
                        if os.environ.get('SLURM_JOB_ID') and
                           os.path.exists(f'/dev/shm/onco_tokenized_dna_{os.environ["SLURM_JOB_ID"]}.pt')
                        else f'{DATA_DIR}/tokenized_dna.pt'
                    ),
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

print('Building dataloaders...')
L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
print(f'  train={len(dl.train_dataloader)} steps/epoch  val={len(dl.val_dataloader)} steps')

print('Building model...')
model = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')

epoch_ckpt_cb = ModelCheckpoint(
    dirpath=f'{OUT_DIR}/checkpoints',
    filename='epoch={epoch}',
    every_n_epochs=5,
    save_top_k=-1,
    save_last=False,
    monitor=None,
)
step_ckpt_cb = ModelCheckpoint(
    dirpath=f'{OUT_DIR}/checkpoints',
    filename='step_ckpt',
    every_n_train_steps=100,
    save_top_k=1,
    save_last=True,
    enable_version_counter=False,
)
lr_cb = LearningRateMonitor(logging_interval='epoch')
csv_logger = pl_loggers.CSVLogger(save_dir=OUT_DIR, name='logs')

trainer = L.Trainer(
    accelerator='gpu',
    devices=torch.cuda.device_count(),
    strategy=DDPStrategy(find_unused_parameters=True, broadcast_buffers=False) if torch.cuda.device_count() > 1 else 'auto',
    max_epochs=config['training']['num_epochs'],
    precision='bf16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm',
    callbacks=[epoch_ckpt_cb, step_ckpt_cb, lr_cb],
    logger=[csv_logger],
    log_every_n_steps=50,
    num_sanity_val_steps=0,
    use_distributed_sampler=False,
)

last_ckpt = f'{OUT_DIR}/checkpoints/last.ckpt'
ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
if ckpt_path:
    print(f'Resuming from {ckpt_path}')

print(f'Training on {torch.cuda.device_count()} GPU(s)...')
trainer.fit(model, dl.train_dataloader, dl.val_dataloader, ckpt_path=ckpt_path)
print('Done.')
