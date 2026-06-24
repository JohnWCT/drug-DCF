# Round 14 Final Report

**Run:** `result/optimization_runs/round14_vicreg_stabilizer`  
**Pipeline started:** 2026-06-24T08:08:26Z  
**Status:** **IN PROGRESS** — pretrain complete; downstream (feature extract → finetune → aggregate) running

> 本文件為 pretrain 完成後的 interim 報告。Finetune 132/132 完成後請以 `final_report/round14_final_report.md` 為準並更新本檔 downstream 數字。

## Timeline

| Phase | When | Status |
|-------|------|--------|
| Config + pretrain | 2026-06-24 08:08Z–~17:30Z | **84/84** success (`PRETRAIN_PARALLEL=20`) |
| Analyze (1st) | — | Failed: missing `latent_active_dims` columns → **fixed** |
| Selection (1st) | — | Failed: empty Top-K (`latent_active_dims=0` collapse false-positive) → **fixed** |
| Selection (2nd) | 2026-06-24 ~17:48Z | **11/16** candidates (structure filter pool) |
| Feature extract + finetune | started ~17:50Z | **132 jobs** planned (11×3 modes×4 combos), `FINETUNE_PARALLEL=12` |

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|----------------------|
| Round 13 best `r13_exp_008_own_plus_summary` | **0.6112** |
| Round 13 exp_035 z-only | 0.6059 |
| Round 12 exp_037 | 0.5972 |
| Round 11 exp_035 | 0.5828 |
| R7 exp_048 | 0.5918 |

## Pretrain (84/84)

| Branch | Jobs | Route |
|--------|------|-------|
| 14B | 54 | `exp_008` proto-response (R12 12B) |
| 14C | 30 | `exp_035` strong z-only 10C (R11) |

**Pretrain QC (all 84):**

| Metric | Mean | Std |
|--------|------|-----|
| kmeans_ari | 0.569 | 0.072 |
| latent_cov_offdiag_mean (from g_loss) | 0.569 | — |
| tumor_vicreg_var_loss_mean (VICReg active) | ~5e-5 | — |
| tumor_vicreg_cov_loss_mean (VICReg active) | ~36.5 | — |

**Top pretrain kmeans_ari (exp_035 route, λ=3e-5):** exp_057 / exp_063 / exp_069 ≈ **0.715**

VICReg λ sweep 彙整見 `reports/round14_vicreg_sweep_summary.csv`。14C（exp_035）在 paired λ=3e-5 時 mean kmeans_ari **0.610** vs 14B **0.556**（同 λ 設定）。

## Selection (11 candidates)

Structure-first filter 後僅 **11/84** 進入 aggregated pool（非 16）。已選模型含 exp_024, exp_027, exp_045, exp_051, exp_056, exp_057, exp_062, exp_063, exp_068, exp_069, exp_078。

## Downstream

- **Finetune jobs:** 132（因 filter pool 11 而非原計畫 16）
- **Best model / Avg TCGA:** pending finetune completion
- **vs Round 13 0.6112:** pending
- **Stretch 0.6200:** pending

## Pipeline fixes applied mid-run

1. `analyze_round14_vicreg_stabilizer.py` — optional column agg + g_loss `latent_cov_offdiag_mean` fallback  
2. `round14_selection.py` — do not treat missing `latent_active_dims` as collapse  
3. `tools/run_round14_resume_from_select.sh` — resume helper

## Round 15 decision

**Hold** until downstream aggregate completes.

成功條件：Best Round 14 > **0.6112** 或 seed stability 改善且 AUC 不 regress → `go_importance_weighting`。
