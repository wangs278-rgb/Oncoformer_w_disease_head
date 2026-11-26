# CCF decoder (arm-loss targets)

A transformer decoder, initialised from run8 (the disease-head Oncoformer in this repo), that generates a patient's alterations one at a time.
It is conditioned on disease, a purity class and the sample site, and emits the alterations in **descending cancer-cell fraction (CCF)**.
The folder holds code only. Data, checkpoints and generated cohorts are not included (see [Not included](#not-included)).

## Model (`ccf_model.py`, parent class in `decoder_model.py`)

Sequence, teacher forcing:

```
input  : [DISEASE + purity4 + site]  timed_1 .. timed_a  |U  untimed_1 .. untimed_b
target :  timed_1 ..                 |U  untimed_1 ..        END
```

- **timed**: alterations with a CCF (classes 1-3), sorted by CCF bin from high to low. There are 21 bins of 0.05, and bin 20 means CCF >= 1. Ties within a bin are put in random order, redrawn every batch.
- **untimed**: copy-number events and short variants without a CCF, in the original gene frequency-rank order, with no value.
- **One step per alteration** chains three predictions: gene (or `|U` / `END`), then the variant token of that gene, then the CCF bin (timed alterations only).
- **Grammar**, enforced in both training and sampling:
  - timed block: any gene; the CCF bin must be <= the previous bin; `END` is not allowed.
  - untimed block: genes in non-decreasing rank; `|U` is not allowed.
- **Input embedding** = run8's frozen token embedding plus run8's Fourier encoder evaluated at the bin centre.
- **Prefix** = `disease_emb + purity4_emb + site_emb`.
- **Transformer**: 4 layers, d = 512, 8 heads, initialised from run8; 51.0 M trainable parameters.
- **Loss** = CE(gene) + CE(variant | gene) + CE(CCF bin | gene, variant).

## CCF targets

- `ccf_prep.py` (v1): per alteration CCF, by class:
  - class 1 (heterozygous, no arm gain): CCF = 2·VAF/p
  - class 2 (homozygous): CCF = VAF/p
  - class 3 (heterozygous on a gained arm): CCF from the MAP copy state of run9 `evolution/copy_states.py`

  Classes 4/5 and copy-number events are untimed.

  It also writes the prefix labels:
  - purity4: `<20% or unknown`, `20-40%`, `40-60%`, `>=60%`
  - site: FMI `A1_samples.local_met_status`, as given (5 values)
- `ccf_prep_armloss.py` (v2, used for this model): the arm-loss rule, decided 2026-11-20 and applied to all diseases. A class-2 short variant whose gene's own autosomal arm is lost (`B1_samples chr<arm>_loss == 'positive'`) gets CCF = VAF·(2−p)/p instead of VAF/p. All other targets are copied unchanged from v1.

## Files

| file | role |
|---|---|
| `ccf_model.py` | `CCFDecoder`, `build_ccf_decoder` |
| `decoder_model.py` | run9 `PatientDecoder` (parent class), unchanged copy |
| `ccf_data.py` | `CCFData`: loads run9 sequences plus CCF targets and prefix labels; builds batches in decoder order |
| `train_ccf.py` | training (AdamW, warmup + cosine, bf16 autocast). Resumes after preemption, with batches identical to an uninterrupted run |
| `generate_ccf.py` | one synthetic patient per real held-out patient of a disease, on that patient's prefix. Checks every patient against the grammar |
| `eval_ccf.py` | real vs synthetic metrics M1-M7 (counts, prevalence, co-occurrence, timed share, CCF bins, pair order, clusters) |
| `ccf_prep.py`, `ccf_prep_armloss.py`, `prep_common.py` | CCF targets and prefix labels |
| `ccf_selftest.py` | self-tests on toy data, CPU (collate, brute-force loss, causality, greedy vs teacher forcing, grammar, cap, prefix, train + exact resume) |

## Inputs (not in git)

| env var / option | content | made by |
|---|---|---|
| `RUN9_DATA` / `--data-dir` | `sequences.npz`, `vocab.npz`, `embeddings.pt`, `run8_init.pt` (prep also uses `purity3.npz`, `timing_targets_cs.npz`) | run9 `code/prep_data.py` from the run8 checkpoint; run9 timing prep |
| `CCF_DATA` / `--ccf-dir` (default `./data`) | `ccf_targets.npz`, `prefix_labels.npz` | `ccf_prep.py` → `ccf_prep_armloss.py` |
| `RUN9_ROOT` | run9 `code/` and `evolution/`; needed only by `ccf_prep.py` | run9 |
| `FMI_PARQUET` | `A1_samples.parquet`, `B1_samples.parquet` | FMI |
| `SAMPLES_PARQUET` | sample table with `xrn`, `computationalTumorPurity` | FMI |
| `REFSEQ` | UCSC `ncbiRefSeq.txt.gz` (gene → arm) | UCSC |
| `TOP50_DIR` / `--top50-dir` | `top50_states.npz`, `top50_profile_fits.npz`. Optional: M7 is skipped if unset | 50-gene latent-class fit (separate analysis) |

## Usage

```bash
pip install -r requirements.txt
python ccf_selftest.py                                   # toy data, CPU, ~minutes; prints "ALL 8 PASSED"

export RUN9_DATA=... FMI_PARQUET=... SAMPLES_PARQUET=... REFSEQ=... RUN9_ROOT=...
python -I ccf_prep.py                                    # -> data/v1/
python -I ccf_prep_armloss.py                            # -> data/ (targets used by the model)

python train_ccf.py --ckpt-dir checkpoints --keep-every 1                      # the arm-loss run: 8 epochs, bs 256, lr 3e-4
python generate_ccf.py --ckpt checkpoints/epoch_7.pt --out gen/crc_epoch7.npz --disease 'colon adenocarcinoma (crc)'
python -I eval_ccf.py gen/crc_epoch7.npz gen/eval_crc_epoch7.json
```

The original runs used 1 × A10G (training about 14 min per epoch).

The arm-loss model was trained on 318,058 train and 79,762 validation patients (run9 split). Validation loss was lowest at epoch 4 (`best.pt`). All downstream generation used the final checkpoint, `epoch_7.pt`.

## Not included

- patient-level data: FMI sequences, CCF targets, prefix labels
- checkpoints: about 200 MB each, over GitHub's 100 MB file limit
- generated cohorts and analysis outputs
