# In-silico gene KNOCK-IN (addition) — resume note

**Status as of 2026-09-11:** full run **job 21252359** RUNNING on the cluster (independent of any
Claude session — it will finish on its own). Audit already PASSED (job 21251471).

## What this is
Mirror of the knockout ([ko/](run8_disease_out/ko/)). For every panel gene G (same set as KO:
≥100 real carriers), in each DX1/DX2 patient that does NOT carry G, INSERT G's **modal
(most-frequent real) mutation token** into a free slot (mask→1) and measure
ΔP = P_added − P_baseline over the 426 supervised disease classes.
Positive ΔP = adding G raises that lineage; negative = lowers it.

## Files (all in /cv/home/wangs278/scratch/fmi/)
- `insilico_ki_disease.py`  — analysis (`--mode audit|full`); RECIP_CAP=10000, MIN_CARRIERS=100.
- `submit_insilico_ki.sh`   — 1×B200 preempt, mem=196G, time=05:00:00, `KI_MODE` env var.
- `plot_ki_violin.py`       — CPU plot (Arial) → `run8_disease_out/ki/figures/fig_ki_gene_violin.{pdf,png}`.
- Output (when done): `run8_disease_out/ki/ki_results.npz` + `ki_meta.json`.
- Log: `run8_disease_out/logs/ki_21252359.out`.

## Audit result (job 21251471) — ALL EXACT ZERO (fp32)
determinism=0; add-then-remove==baseline=0; add-effect=0.503 (TP53); free-slot position
invariance=0; mask-insert==physical-append=0. Insertion ≡ genuinely carrying the token.
Token has 11 scalar components: gene, alt_type, pathogenicity, zygosity, aa_ref, aa_mut,
aa_pos, aa_vaf_bin, aa_vaf(float), protein, mutation.

## IMPORTANT caveat (surfaced by audit)
Modal-mutation injection is faithful for **hotspot oncogenes** (KRAS/BRAF/EGFR: 1 dominant
mutation) but only a small slice for **dispersed tumor suppressors** (TP53 modal = 4.6% of its
14,995 distinct mutations). `modal_frac` is saved per gene; the figure flags genes with
modal_frac<0.20 with a dagger (†). The declined "marginalize over mutations" option would fix this.

## TO FINISH (after job 21252359 completes)
```bash
cd /cv/home/wangs278/scratch/fmi
sacct -j 21252359 --format=JobID,State,Elapsed -X -n        # confirm COMPLETED
tail -n 30 run8_disease_out/logs/ki_21252359.out           # top-12 genes summary is printed here
source /cv/scratch/u/wangs278/venv_oncoformer/bin/activate
python3 plot_ki_violin.py                                   # regenerate the violin figure
```
Then compare KI (top-RAISED lineage per gene) vs KO (top-lowered) — expect mirror image:
adding VHL→raises kidney clear cell, EGFR→lung adeno, KRAS→pancreas, etc.

## If the job was preempted / needs resubmit
```bash
cd /cv/home/wangs278/scratch/fmi
sbatch --export=ALL,KI_MODE=full submit_insilico_ki.sh
```
(Everything is stateless/reproducible — SEED=42, stable per-(sample,gene) hash subsample.)
