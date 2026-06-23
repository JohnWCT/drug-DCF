# Round 10 Conditional Adversarial Deconfounding Manual

## Motivation from Round 9

Round 9 showed global domain adaptation works, but **within the same cancer type**, source and target remain separable (macro conditional domain AUC ~0.84–0.88 for exp_048). Round 10 tests **cancer-type-conditioned** adversarial deconfounding:

`D_cond(z, cancer_type) → source / target`

## Branch design

| Branch | Purpose | Jobs |
|--------|---------|------|
| **10A** | Global ADV reproduction control from exp_048 | 3 |
| **10B** | Conditional ADV replacement (main) | 108 |
| **10C** | Conditional ADV + weak global guard (λ×0.25) | 12 |

Total: **123** pretrain jobs.

## Config generation

```bash
python tools/round10_config_builder.py \
  --settings config/round10_cond_adv_settings.json \
  --outdir result/optimization_runs/round10_cond_adv \
  --force
```

## Training behavior

- `conditional_adv_enabled=false`: identical to legacy pretrain (backward compatible).
- `conditional_replacement`: conditional critic only; global discriminator not updated.
- `conditional_plus_weak_global`: both critics; generator global loss scaled by `lambda_global_adv_multiplier`.

## Selection

Mode: `round10_cond_adv_qc` (Top-K=24). Scores conditional leakage improvement, cancer retention, global alignment safety — **not** downstream AUC at pretrain stage.

## Pipeline

```bash
bash tools/run_round10_cond_adv_pipeline.sh
```

## Results (2026-06-22 run)

**Status:** `no_conditional_improvement` — pipeline complete; downstream improved vs R9 reproduction but conditional leakage QC not verified.

| Metric | Value |
|--------|-------|
| Pretrain | 115/123 success (8 failed, all 10B) |
| Selected | 24 models (20×10B, 4×10C) |
| Finetune | 96/96 success |
| Best Avg TCGA | **0.5749** (`exp_111`, 10C, λ=0.001, dim=16) |
| vs R9 repro best | +0.0078 (0.5671) |
| vs R7 exp_048 | −0.0169 (0.5918) |
| Top-24 mean Avg TCGA | 0.5193 |

**Best model:** `exp_111` — `10C_conditional_plus_weak_global`, not pure 10B replacement.

**Limitation:** Round 9 conditional leakage diagnostics were not re-run; `mean_conditional_leakage_strength` is NaN in summaries. Conditional ADV training is confirmed via `gan_metrics.json`.

**Failed pretrain:** 7 jobs at λ=0.001, 2 at λ=0.0003 dim=16 (early exit code 1).

Full report: `docs/round10_final_report.md`

## Round 11 decision

**Completed (2026-06-22).** Round 11A re-ran Round 9-style conditional diagnostics on Round 10 Top-24; 11B/11C pretrain and downstream finetune finished (**195/195** pretrain, **120/120** finetune). Best downstream **exp_035** Avg TCGA **0.5828** (+0.0079 vs Round 10 exp_111). Round 11A: exp_111 leakage 0.400 vs exp_048 0.409 (improved).

**Next:** `go_prototype_alignment` — see `docs/round11_final_report.md` and `docs/pipeline_summary.md` §19.8.
