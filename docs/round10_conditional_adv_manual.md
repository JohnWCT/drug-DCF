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

## Round 11 decision

Go if 10B/10C reduce conditional leakage without biology collapse and downstream does not degrade materially.
