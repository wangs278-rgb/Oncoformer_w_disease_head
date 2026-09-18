# Oncoformer Run 4 — Converge Handoff

## Goal
Continue pretraining the Oncoformer DNA-only MLM model (run 4) on converge.
Training was previously running on scoop and reached epoch 19 before being moved here for more GPUs.

---

## What is already on converge at `/cv/home/wangs278/scratch/fmi/` ✓ DONE

| File | Size | Notes |
|------|------|-------|
| `pretraining_data_full.h5` | 19G | Main anndata training dataset |
| `tokenized_dna.pt` | 7.7G | Pre-tokenized cache — saves ~50 min on startup |
| `sample_metadata.pkl` | 47M | Sample metadata |
| `tokenizer_dna.pkl` | 473K | Tokenizer |
| `esm_unpaired_layer_33_diff_embedding_full.mutation.weight.pt` | 3.3G | ESM mutation embeddings |
| `esm_unpaired_layer_33_diff_embedding_full.mutation.vocab.json` | 37M | ESM mutation vocab |
| `esm_unpaired_layer_33_diff_embedding_full.protein.weight.pt` | 2.9M | ESM protein embeddings |
| `esm_unpaired_layer_33_diff_embedding_full.protein.vocab.json` | 19K | ESM protein vocab |
| `vocab_dna.json` | 414K | DNA vocabulary |
| `train.py` | — | Training script (paths already updated for converge) |
| `submit.sh` | — | SLURM submit script (needs partition filled in, see below) |

---

## What still needs to be copied (run these ON converge) ← START HERE

### 1. Oncoformer code repo (~2MB of actual code)
```bash
mkdir -p /cv/home/wangs278/scratch/fmi/Oncoformer
scp -r sc1nc001is01.eth.rsiec.sc1.science.roche.com:/gpfs/scratchfs01/site/u/wangs278/Oncoformer/oncoformer \
    sc1nc001is01.eth.rsiec.sc1.science.roche.com:/gpfs/scratchfs01/site/u/wangs278/Oncoformer/config \
    sc1nc001is01.eth.rsiec.sc1.science.roche.com:/gpfs/scratchfs01/site/u/wangs278/Oncoformer/pyproject.toml \
    sc1nc001is01.eth.rsiec.sc1.science.roche.com:/gpfs/scratchfs01/site/u/wangs278/Oncoformer/requirements.txt \
    /cv/home/wangs278/scratch/fmi/Oncoformer/
```

### 2. Latest checkpoint (resume from epoch 19)
```bash
mkdir -p /cv/home/wangs278/scratch/fmi/run4_out/checkpoints
scp sc1nc001is01.eth.rsiec.sc1.science.roche.com:/home/wangs278/scratch/oncoformer_test/4_oncoformer_retrain_test1_mlm_only/checkpoints/last.ckpt \
    /cv/home/wangs278/scratch/fmi/run4_out/checkpoints/
```

---

## Directory layout expected by train.py

```
/cv/home/wangs278/scratch/fmi/
├── pretraining_data_full.h5
├── tokenized_dna.pt
├── sample_metadata.pkl
├── tokenizer_dna.pkl
├── vocab_dna.json
├── esm_unpaired_layer_33_diff_embedding_full.mutation.weight.pt
├── esm_unpaired_layer_33_diff_embedding_full.mutation.vocab.json
├── esm_unpaired_layer_33_diff_embedding_full.protein.weight.pt
├── esm_unpaired_layer_33_diff_embedding_full.protein.vocab.json
├── train.py
├── submit.sh
├── Oncoformer/              ← code repo (step 1 above)
│   ├── oncoformer/
│   ├── config/
│   ├── pyproject.toml
│   └── requirements.txt
└── run4_out/                ← created automatically by train.py
    ├── checkpoints/
    │   └── last.ckpt        ← copied in step 2 above (resumes from epoch 19)
    └── logs/
```

---

## Install the package

```bash
cd /cv/home/wangs278/scratch/fmi/Oncoformer
pip install -e .
```

---

## Submit the job

Edit `submit.sh` — fill in two placeholders:
- `FILL_IN_PARTITION` — GPU partition name on converge
- `FILL_IN_GPU_TYPE:4` — GPU type and count (e.g. `a100:4`, `h100:4`; or just `gpu:4` for any)

Then:
```bash
sbatch /cv/home/wangs278/scratch/fmi/submit.sh
```

---

## Model config summary

- Architecture: 4-layer transformer, embed_dim=512, 8 heads, max_seq_len=128
- Modality: DNA mutations only (MLM pretraining)
- Batch size: 512, 40 epochs total, AdamW lr=5e-5, cosine schedule
- Training resumes automatically from `run4_out/checkpoints/last.ckpt` if it exists
- Checkpoints saved every 5 epochs + every 100 steps (for preemption resilience)
- DINO disabled, bf16 mixed precision

---

## Source server reference
- Server: `sc1nc001is01.eth.rsiec.sc1.science.roche.com`
- Original run dir: `/home/wangs278/scratch/oncoformer_test/4_oncoformer_retrain_test1_mlm_only/`
- Original data: `/gstore/data/cancer_depmap/oncoformer/datasets/` and `/gpfs/scratchfs01/site/u/wangs278/Oncoformer/datasets/`
