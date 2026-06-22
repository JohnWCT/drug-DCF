# Round 11 Final Report

**Run:** `result/optimization_runs/round11_stability_recon`  
**Completed:** 2026-06-22  
**Pipeline status:** ALL_DONE (pretrain 195/195, finetune 120/120)

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|----------------------|
| R7 original exp_048 | 0.5918 |
| Round 10 best exp_111 | 0.5749 |
| Round 9 reproduction | 0.5671 |
| **Round 11 best exp_035** | **0.5828** |

## Pipeline summary

| Stage | Jobs | Success |
|-------|------|---------|
| Round 11A post-hoc QC | 25 models | completed |
| Pretrain (11B+11C) | 195 | 195 |
| Selection | top-30 | 30 |
| Finetune | 120 (30×4) | 120 |

Branches: 11B_10C_stabilization (120), 11C_global_recon_control (27), 11C_10C_recon_ablation (36), 11B_10B_control (12).

## Round 11A conditional QC (Round 10 models)

- Models analyzed: 25
- **exp_111** mean conditional leakage: 0.400 (vs exp_048 0.409) → **improved**
- exp_111 leakage delta vs exp_048: −0.0089
- Per-cancer delta table: `round11_per_cancer_qc_delta.csv` (672 rows)

## Pretrain — reconstruction ablation (latent proxy)

| reconstruction_loss_type | n | mean kmeans_ari | mean wasserstein |
|--------------------------|---|-----------------|------------------|
| hybrid_mse_smooth_l1 | 12 | 0.770 | 0.635 |
| smooth_l1 | 36 | 0.705 | 0.944 |
| mse | 147 | 0.539 | 1.109 |

SmoothL1 configs logged in `run_summary.json` → `reconstruction_loss` block (36 smooth_l1 models verified).

## Downstream (finetune aggregate)

**Best model: exp_035** — 11B 10C stabilization, MSE reconstruction, `λ_cond_adv=0.0001`, `conditional_plus_weak_global`, λ_global_mult=0.25.

| Rank | Model_ID | Average_TCGA_AUC_mean | Global_TCGA_AUC_mean |
|------|----------|----------------------|---------------------|
| 1 | exp_035 | **0.5828** | 0.6026 |
| 2 | exp_142 | 0.5632 | 0.6246 |
| 3 | exp_097 | 0.5566 | 0.6261 |
| 4 | exp_134 | 0.5548 | 0.5894 |
| 5 | exp_153 | 0.5528 | 0.5926 |

Round 11 finetuned **exp_111** (forced reference): Average_TCGA_AUC_mean ≈ 0.5043.

## Round 12 decision

**Recommendation:** `go_prototype_alignment` (conditional)

Rationale:
1. Round 11A confirms exp_111 lowers conditional leakage vs exp_048 with cancer retention intact.
2. Round 11 best downstream **0.5828** exceeds Round 10 exp_111 (**0.5749**).
3. SmoothL1 / hybrid reconstruction improves latent stability vs MSE without collapse.

Caveat: still below R7 original exp_048 (0.5918). Prototype Alignment should build on **exp_035** or top-10C stabilized candidates, not smoke-era checkpoints.

## Artifacts

- Runtime: `result/optimization_runs/round11_stability_recon/final_report/`
- Selection: `selection/pretrain_top10.csv`
- Aggregate: `aggregate/aggregate_scores.csv`
- Logs: `logs/pipeline_resume.log`, `logs/pipeline_downstream.log`
