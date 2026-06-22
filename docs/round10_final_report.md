# Round 10 Conditional ADV — Final Report

**Run date:** 2026-06-22  
**Docker image:** `dapl:5.1`  
**Output dir:** `result/optimization_runs/round10_cond_adv/`  
**round10_success_status:** `no_conditional_improvement`

---

## 1. Pipeline execution summary

| Stage | Planned | Completed | Notes |
|-------|---------|-----------|-------|
| Pretrain | 123 | **115 success / 8 failed** (93.5%) | All failures are **10B**; 7× `λ=0.001`, 2× `λ=0.0003 dim=16` |
| Analyze + Select | — | **24 models** selected | 20×10B, 4×10C; **no 10A** in Top-24 |
| Finetune | 96 | **96/96 success** | 24 models × 4 finetune combos |
| Aggregate + report | — | Done | |

**Parallelism:** `PRETRAIN_PARALLEL=33`, `FINETUNE_PARALLEL=26`, `batch_size=128`.

---

## 2. Pretrain branch summary (successful runs)

| Branch | n (success) | mean wasserstein | mean kmeans_ari | mean FID |
|--------|-------------|------------------|-----------------|----------|
| 10A global ADV repro | 3 | 0.51 | 0.16 | 25.8 |
| 10B conditional replacement | 100 | 1.63 | 0.58 | 50.5 |
| 10C conditional + weak global | 12 | 1.30 | 0.61 | 48.5 |

Conditional ADV **did train**: `gan_metrics.json` records `conditional_adv_enabled=true`, `cond_critic_loss_mean`, etc.

**Caveat:** Round 9-style per-cancer conditional leakage diagnostics were **not re-run**. `mean_conditional_leakage_strength` is NaN in pretrain summaries.

---

## 3. Downstream finetune results

| Reference | Avg TCGA | Notes |
|-----------|----------|-------|
| **Round 10 best** (`exp_111`) | **0.5749** | 10C, `λ_cond=0.001`, dim=16, weak global ×0.25 |
| Round 9 exp_048 reproduction best | 0.5671 | exp_010 @ seed 303 |
| R7 original exp_048 checkpoint | 0.5918 | Project primary baseline |
| Round 10 Top-24 mean | 0.5193 | — |

### Top-5 by Average TCGA AUC

| Rank | Model | Avg TCGA | Global TCGA | Branch |
|------|-------|----------|-------------|--------|
| 1 | exp_111 | **0.5749** | 0.6318 | 10C |
| 2 | exp_034 | 0.5636 | 0.6321 | 10B |
| 3 | exp_052 | 0.5608 | 0.5828 | 10B |
| 4 | exp_090 | 0.5516 | 0.5898 | 10B |
| 5 | exp_029 | 0.5464 | 0.6055 | 10B |

---

## 4. Conclusions

1. Pipeline complete end-to-end (pretrain → select → finetune → aggregate).
2. Best model **exp_111** (10C weak global guard) beats Round 9 reproduction by **+0.0078** Avg TCGA.
3. QC status **`no_conditional_improvement`** — conditional leakage not measured; Round 9 diagnostics needed before Round 11.
4. Pretrain failures cluster at **λ=0.001** (7/8 failed jobs).

---

## 5. Artifact paths (runtime, not in git)

| Artifact | Path |
|----------|------|
| Aggregate | `result/optimization_runs/round10_cond_adv/aggregate/aggregate_scores.csv` |
| Selection | `result/optimization_runs/round10_cond_adv/selection/pretrain_top10.csv` |
| Pretrain manifest | `result/optimization_runs/round10_cond_adv/manifests/pretrain_sweep_manifest.csv` |
| Finetune manifest | `result/optimization_runs/round10_cond_adv/manifests/finetune_dispatch_manifest.csv` |
