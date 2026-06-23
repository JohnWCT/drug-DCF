# Round 11 Optimization Manual

## Purpose

Round 11 addresses three gaps after Round 10:

1. **Missing conditional leakage QC** — Round 10 trained Conditional ADV but did not re-run Round 9-style diagnostics.
2. **10C stabilization** — Best model `exp_111` used weak global guard; sweep λ_cond, schedule, and λ_global multiplier.
3. **SmoothL1 reconstruction ablation** — Test whether Huber-style reconstruction improves latent stability vs MSE (AE/VAE only).

Round 11 does **not** add Prototype Alignment, SupCon, or importance weighting.

## Branch design

| Branch | Purpose | Jobs |
|--------|---------|------|
| **11A** | Round 10 Top-24 post-hoc conditional QC | diagnostics only |
| **11B** | 10C weak global guard stabilization | 120 + 12 (10B control) |
| **11C** | SmoothL1 / hybrid reconstruction ablation | 27 + 36 |
| **11D** | Best 10C + SmoothL1 combo (optional) | 5–15 |

Total pretrain (11B+11C): **195** jobs.

## Key files

| File | Role |
|------|------|
| `config/round11_settings.json` | Sweep dimensions, seeds, references |
| `tools/reconstruction_losses.py` | `mse` / `smooth_l1` / `hybrid_mse_smooth_l1` |
| `tools/round11_config_builder.py` | Generate configs + manifest |
| `tools/run_round11a_round10_qc.py` | Post-hoc Round 10 conditional QC |
| `tools/round11_selection.py` | `round11_stability_qc` selection |
| `tools/analyze_round11_qc.py` | Reports + Round 12 go/no-go |
| `tools/run_round11_pipeline.sh` | End-to-end pipeline |

## Config generation

```bash
python tools/round11_config_builder.py \
  --settings config/round11_settings.json \
  --outdir result/optimization_runs/round11_stability_recon \
  --force
```

Manifest columns include: `round11_branch`, `reconstruction_loss_type`, `smooth_l1_beta`, `global_adv_mode`, `lambda_cond_adv`.

## Full pipeline

```bash
bash tools/run_round11_pipeline.sh
```

Environment overrides: `PRETRAIN_PARALLEL`, `FINETUNE_PARALLEL`, `FINETUNE_EPOCHS`.

## Reconstruction loss params

Default (backward compatible):

```json
{
  "reconstruction_loss_type": "mse",
  "smooth_l1_beta": 1.0,
  "reconstruction_loss_reduction": "mean",
  "reconstruction_loss_scale": 1.0
}
```

## Selection

Mode: `round11_stability_qc`, `top_k=30`, force `exp_111`.

Groups: downstream proxy, low leakage, best 10C, SmoothL1, hybrid, MSE control, alignment, cancer retention, forced reference, fill.

## Success criteria

**Basic:** 11A completes; pretrain success ≥95%; selection retains MSE + SmoothL1 + 10C; finetune completes.

**Method:** 10C leakage < Round 10 `exp_111`; SmoothL1 no collapse; Avg TCGA ≥ 0.5749.

## Round 12 decision

Go to Prototype Alignment only if conditional leakage drops, kmeans_ari holds, and downstream ≥ Round 10 best.

See `docs/pipeline_summary.md` §19.

## Results (completed 2026-06-22)

**Run:** `result/optimization_runs/round11_stability_recon`  
**Status:** ALL_DONE — pretrain **195/195**, finetune **120/120**

| Stage | Result |
|-------|--------|
| Round 11A QC | 25 models; exp_111 leakage 0.400 vs exp_048 0.409 |
| Best downstream | **exp_035** Avg TCGA **0.5828** (vs Round 10 exp_111 **0.5749**) |
| Round 12 | **go_prototype_alignment** |

Full report: `docs/round11_final_report.md` · runtime CSVs: `result/optimization_runs/round11_stability_recon/final_report/`
