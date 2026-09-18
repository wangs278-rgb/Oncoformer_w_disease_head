# Oncoformer with Disease Head

Code for **run8_disease**: the Oncoformer DNA-only pretraining model (MLM + single-modality
MoCo, DX1/DX2 baitsets) extended with a **supervised DiseaseTerm classification head**,
trained jointly during a warm-started continuation of pretraining. This repo also contains
the earlier run4–run7 lineage and the full downstream analysis pipeline (feature extraction,
in-silico perturbation, evaluation, plotting).

> **Code only.** All data, tokenized caches, model checkpoints and run outputs (~460 GB)
> are excluded via `.gitignore`. See [Data & artifacts](#data--artifacts-not-in-git) for
> how to regenerate them.

---

## The disease-head model (run8_disease)

The disease head is added **without any shared model-code change**. `OncoformerPost`
(`Oncoformer_moco/oncoformer/models.py`) wraps a pretraining backbone and, when
`training.pre_train_model=True`, runs the backbone's MLM+MoCo `training_step` **and** a
supervised metadata-classification loss, summing them:

```
backbone = OncoformerOmics(config)            # run6: MLM + MoCo (DNA-only, DX1/DX2)
model    = OncoformerPost(backbone, config)   # + DiseaseTerm head, pre_train_model=True
```

Each `metadata.data.discrete` task becomes a `Classification` head (cross-entropy,
`ignore_index=-100`, label smoothing 0.1) on the backbone's pooled embedding, weighted
heteroscedastically. run8 defines a single `disease_term` task (**511 classes**) with a
conservative `meta_weight=0.3`, trained for **20 epochs** warm-started from run6's final
backbone checkpoint (`run6_moco_out/checkpoints/last.ckpt`, `load_state_dict(strict=False)`).

**Warm-start / resume policy** (`train_run8_disease.py`):
- **Cold start** (no run8 checkpoint): init backbone from run6's `last.ckpt`, disease head
  random, fresh optimizer + cosine schedule over 20 epochs.
- **Requeue** (SLURM preemption): resume the *combined* model from run8's own newest valid
  checkpoint (corrupt/truncated checkpoints are quarantined and skipped).

---

## Model architecture

```
                 mutation tokens (per-variant, max 128 per sample)
                          │
   ┌──────────────────────┴───────────────────────┐
   │  component encoders  (concat/​sum → 512-d)     │   gene, alt_type, pathogenicity,
   │                                               │   zygosity, aa_ref, aa_mut,
   │                                               │   protein & mutation (ESM, frozen),
   │                                               │   aa_vaf (Fourier) + aa_vaf_bin
   └──────────────────────┬───────────────────────┘
                          │
             ┌────────────┴────────────┐
             │  PRETRAINING BACKBONE    │   OncoformerOmics
             │  4-layer Transformer     │   embed_dim=512, 8 heads, max_len=128, bf16
             │  encoder (per modality;  │
             │  DNA-only here)          │
             └────────────┬────────────┘
                          │ pooled / per-token hidden states
        ┌─────────────────┼──────────────────────────────┬─────────────────────┐
        ▼                 ▼                                ▼                     ▼
  MLM heads          MLM head: gene              MoCo/CLIP head          Disease head
  (per field)        (masked-token              (contrastive,           (SUPERVISED, run8)
  alt_type,           reconstruction)            student+teacher,        disease_term
  pathogenicity,     ← "the gene head"           queue 65536,            Classification
  zygosity,                                       out 256, τ=0.07)        511 classes
  aa_ref, aa_mut,                                                         weight 0.3
  aa_vaf_bin
  └────────────── self-supervised (from run6 backbone) ──────────────┘   └─ added on top ─┘
```

**Pretraining backbone — `OncoformerOmics`** (trained in run6, warm-started into run8):
- Per-modality Transformer encoder (DNA-only in this line of work): **4 layers, `embed_dim=512`,
  8 attention heads, `max_length=128`** variants/sample, bf16-mixed.
- Each variant is embedded from multiple **components** and combined (concat/sum → 512-d):
  `gene`, `alt_type`, `pathogenicity`, `zygosity`, `aa_ref`, `aa_mut`, `protein` &
  `mutation` (precomputed **ESM** embeddings, frozen), and `aa_vaf` (Fourier encoder) +
  `aa_vaf_bin`.
- **Self-supervised objectives:**
  1. **MLM** — a reconstruction head per masked field: `gene`, `alt_type`, `pathogenicity`,
     `zygosity`, `aa_ref`, `aa_mut`, `aa_vaf_bin` (`protein` is aliased to the gene target;
     `mutation` to `[alt_type, aa_ref, aa_mut, pathogenicity]`). **The "gene head" is one of
     these.**
  2. **MoCo / CLIP contrastive** — student/teacher projection heads (`CLIPMoCoHead`),
     `clip_out_dim=256`, `clip_hidden=2048`, momentum queue `65536`, temperature `0.07`
     (single-modality MoCo fallback since it's DNA-only).

**Supervised head — added by `OncoformerPost` in run8:**
- **Disease head** — `prediction_heads['disease_term']`, a `Classification` over **511
  DiseaseTerm classes** on the backbone's pooled embedding (CE, `ignore_index=-100`, label
  smoothing 0.1), auxiliary weight **0.3**.

**Total training loss (run8, `pre_train_model=True`):**
```
loss = MLM_loss (all reconstruction heads incl. gene)
     + MoCo_contrastive_loss
     + 0.3 · disease_term_CE
```

---

## Repository layout

```
.
├── run8_disease_config.py          # disease-head config (extends run6_moco_config)
├── train_run8_disease.py           # training entrypoint (warm-start + requeue logic)
├── build_disease_metadata_run8.py  # builds disease targets + vocab (one-time preprocessing)
├── dryrun_run8_disease.py          # fast config/data smoke test
├── eval_run8_disease_head.py       # supervised DiseaseTerm eval on DX1/DX2 val split
├── vocab_dna.json                  # DNA tokenizer vocab (small, kept in repo)
├── vocab_metadata_disease_run8.json# disease-term vocab (small, kept in repo)
│
├── run{4,5,6,7}*_config.py / train_run{4..7}*.py   # earlier run lineage
├── extract_*.py                    # embedding / feature / attention extraction
├── insilico_*.py                   # in-silico perturbation (KO, KI, dose, evo, shift, minsig)
├── eval_*.py                       # evaluation (VUS, gene-stratified, head competence, …)
├── plot_*.py                       # figures (umap, cup, dose, grammar, ki/ko, shift, …)
│
├── Oncoformer/                     # vendored backbone package (MLM baseline)
└── Oncoformer_moco/                # vendored package with the disease-head model code
    └── oncoformer/models.py        #   OncoformerOmics, OncoformerPost, Classification
```

---

## Setup

```bash
# Python env with torch + lightning (as used on the cluster)
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate   # or create your own venv

# Install the model package (editable). run8 uses the MoCo variant:
pip install -e Oncoformer_moco    # provides `oncoformer.models.OncoformerPost`
# (Oncoformer/ is the earlier MLM-only backbone package.)
```

`run8_disease_config.use_moco_oncoformer()` forces `import oncoformer` to resolve to the
`Oncoformer_moco` package; the training/eval scripts assert this at startup.

---

## Data & artifacts (not in git)

These live at `DATA_DIR = /cv/home/wangs278/scratch/fmi/` and are **not** committed:

| Artifact | Produced by | Notes |
|---|---|---|
| `sample_metadata.pkl` | (upstream) | Sample metadata incl. `DiseaseTerm`, `BaitSet` |
| `esm_unpaired_*` | (upstream) | ESM mutation/protein embeddings + vocabs |
| `tokenized_dna_dx12_moco.pt` | `pretokenize_run6.py` | DX1/DX2 MoCo tokenized cache (reused by run8) |
| `metadata_disease_run8.pt` | `build_disease_metadata_run8.py` | Per-sample disease targets (full coverage) |
| `run6_moco_out/checkpoints/last.ckpt` | run6 training | Warm-start source for run8 |
| `run8_disease_out/checkpoints/*.ckpt` | `train_run8_disease.py` | Trained disease-head model |

---

## End-to-end: reproduce run8_disease

```bash
# 0. Env
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
pip install -e Oncoformer_moco

# 1. Build disease targets + vocab (needs sample_metadata.pkl)
python build_disease_metadata_run8.py
#   -> metadata_disease_run8.pt (511 classes), vocab_metadata_disease_run8.json

# 2. (If missing) build the DX1/DX2 MoCo tokenized cache
python pretokenize_run6.py            # -> tokenized_dna_dx12_moco.pt

# 3. Sanity-check config + data wiring
python dryrun_run8_disease.py

# 4. Train (multi-GPU, warm-starts from run6 last.ckpt; auto-requeues on preemption)
python train_run8_disease.py          # -> run8_disease_out/checkpoints/

# 5. Evaluate the disease head on the DX1/DX2 non-degenerate val split
python eval_run8_disease_head.py
#   -> run8_disease_out/eval/run8_disease_head_eval.json
#      (acc@1/@5, macro/weighted F1, OvR AUROC/AUPRC vs. majority baseline)
```

> The SLURM launcher scripts (`submit_*.sh`) are cluster-specific and are **not** part of
> this repo; run the `python` entrypoints above directly, or wrap them in your own scheduler
> job. `train_run8_disease.py` picks up `torch.cuda.device_count()` GPUs via DDP.

Downstream analysis (feature extraction, in-silico KO/KI/dose/evo/shift, plotting) uses the
matching `extract_*.py` / `insilico_*.py` / `plot_*.py` scripts, all reading from
`run8_disease_out/`.

---

## Notes

- Training config: DNA-only, 4-layer transformer, `embed_dim=512`, 8 heads,
  `max_seq_len=128`, batch 512, AdamW + cosine, bf16-mixed. Checkpoints every 5 epochs and
  every 100 steps for preemption resilience.
- Terms matching `ignore_patterns=['other','nos']` are kept as vocab levels but excluded
  from the disease loss (target `-100`); they still contribute to MLM+MoCo.
- The logged `val_disease_term_loss` spans all baitsets and is not directly interpretable —
  `eval_run8_disease_head.py` recomputes clean metrics on DX1/DX2 non-degenerate val samples,
  scoring only over the supervised class set.
