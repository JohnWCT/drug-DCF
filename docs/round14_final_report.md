# Round 14 Final Report

**Run:** `result/optimization_runs/round14_vicreg_stabilizer`  
**Completed:** 2026-06-24（downstream 14:07Z）  
**Pipeline status:** ALL_DONE（pretrain **84/84**，feature extract **33/33**，finetune **132/132**）

## Timeline

| Phase | When | Notes |
|-------|------|-------|
| Config + pretrain | 2026-06-24 08:08Z–~17:30Z | **84/84** success；`PRETRAIN_PARALLEL=20` |
| Analyze / selection | ~17:48Z | 初次 analyze / selection bug 已修復；**11/16** 入選 |
| Feature extract + finetune | 09:50Z–14:07Z | **132/132**；`FINETUNE_PARALLEL=12` |
| Aggregate + final analysis | 14:07Z | Done |

Mid-run fixes：`analyze_round14` optional columns、`round14_selection` collapse false-positive、`run_round14_resume_from_select.sh`。

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|----------------------|
| **Round 13 best `r13_exp_008_own_plus_summary`** | **0.6112** |
| Round 13 exp_035 z-only | 0.6059 |
| Round 12 exp_037 | 0.5972 |
| Round 11 exp_035 | 0.5828 |
| R7 exp_048 | 0.5918 |

## Pipeline summary

| Stage | Planned | Completed | Notes |
|-------|---------|-----------|-------|
| Pretrain 14B+14C | 84 | **84/84** | VICReg λ sweep on exp_008 + exp_035 lineages |
| Selection | Top-16 | **11** | structure filter 後僅 11 模型；**全為 exp_035 route** |
| Feature extract | 33 | **33/33** | 11 models × 3 compact modes |
| Finetune | 192 → **132** | **132/132** | 11×3×4 combos |
| Aggregate + analyze | — | Done | |

**Parallelism:** `PRETRAIN_PARALLEL=20`，`FINETUNE_PARALLEL=12`；`batch_size=128`（pretrain），`12288/3072`（finetune）。

## Downstream

- **Best model:** `r14_exp_078_own_plus_summary`（pretrain `exp_078`，exp_035 route，VICReg λ=1e-4 paired）
- **Average_TCGA_AUC_mean:** **0.5908654300736795**
- **Global_TCGA_AUC_mean:** 0.6069798243795909
- **vs Round 13 best (0.6112):** **−0.0204**（未超越）
- **vs Round 12 exp_037 (0.5972):** **−0.0063**
- **vs R7 exp_048 (0.5918):** **−0.0009**（略低）
- **Stretch goal 0.6200:** **not met**

### Top-5 downstream（by Avg TCGA）

| Rank | Model_ID | feature_mode | Average_TCGA_AUC_mean | Global_TCGA_AUC_mean |
|------|----------|--------------|----------------------|---------------------|
| 1 | **r14_exp_078_own_plus_summary** | own_plus_summary | **0.5909** | 0.6070 |
| 2 | r14_exp_078_none | none (z-only) | 0.5747 | **0.6114** |
| 3 | r14_exp_027_none | none | 0.5440 | 0.5914 |
| 4 | r14_exp_062_own_plus_summary | own_plus_summary | 0.5408 | 0.6201 |
| 5 | r14_exp_078_own_cancer | own_cancer | 0.5326 | 0.6002 |

Note：`r14_exp_078_none` Global TCGA **0.6114** 接近 Round 13 best，但 Avg TCGA 仍低於 `own_plus_summary` peak。

## Feature mode ablation

| Mode | Best model | Best Avg TCGA | Mean Avg TCGA (n=11) |
|------|------------|---------------|----------------------|
| **own_plus_summary** | r14_exp_078_own_plus_summary | **0.5909** | 0.5038 |
| none (z-only) | r14_exp_078_none | 0.5747 | 0.5088 |
| own_cancer | r14_exp_078_own_cancer | 0.5326 | 0.5061 |

`own_plus_summary` 仍為 peak feature mode，但整體 mean 與 z-only 接近，未重現 Round 13 在 `exp_008` 上的大幅 proto gain。

## z-only vs prototype features（per selected pretrain）

| source | z-only | best proto | delta |
|--------|--------|------------|-------|
| exp_078 | 0.5747 | 0.5909 (`own_plus_summary`) | **+0.0161** |
| exp_051 | 0.4321 | 0.4765 (`own_cancer`) | **+0.0444** |
| exp_057 | 0.5059 | 0.5314 (`own_cancer`) | **+0.0255** |
| exp_056 | 0.5037 | 0.5191 (`own_plus_summary`) | **+0.0154** |
| exp_062 | 0.5190 | 0.5408 (`own_plus_summary`) | **+0.0219** |
| exp_024 | 0.4861 | 0.4983 (`own_cancer`) | +0.0122 |
| exp_045 | 0.4459 | 0.4534 (`own_plus_summary`) | +0.0075 |
| exp_063 | 0.5311 | 0.5316 (`own_cancer`) | +0.0005 |
| exp_068 | 0.5251 | 0.5210 (`own_cancer`) | −0.0042 |
| exp_069 | 0.5290 | 0.5050 (`own_cancer`) | −0.0239 |
| exp_027 | 0.5440 | 0.5226 (`own_plus_summary`) | −0.0214 |

Proto benefit：**8/11** 為正，但幅度遠小於 Round 13 `exp_008`（+0.0449）。

## Pretrain / VICReg findings

- **84/84** pretrain 成功；mean kmeans_ari **0.569**（std 0.072）
- 14C（exp_035）在 paired λ=3e-5 時 mean kmeans_ari **0.610** > 14B **0.556**（同 λ）
- VICReg var/cov loss mean 已正確記錄於 `run_summary.json` / `g_loss.csv`
- **Downstream 僅涵蓋 exp_035 route**（11 模型）；exp_008 14B 模型未通過 structure filter 進入 selection，**未測試 Round 13 最佳 proto-response stack + VICReg**

## Latent stability

- `latent_cov_offdiag_mean`（from g_loss）：pretrain mean **~0.569**
- `latent_active_dims` 未於 pretrain 直接記錄；selection 已修正 missing→collapse 誤判
- Seed proxy std：n/a（downstream 未做跨 seed 同一 checkpoint 比較）

## Round 15 decision

**Recommendation:** **`hold`** — 不進 importance-aware weighting（Round 15 原計畫）

理由：
1. Best Round 14 **0.5909** < Round 13 **0.6112**（−0.0204）
2. 未達 stretch **0.6200**
3. VICReg 未帶來可觀 AUC 提升；downstream 未覆蓋 exp_008 proto-response route
4. z-only Global TCGA 對 `exp_078` 達 0.6114，但 Avg TCGA 未整體超越 Round 13

### 建議 Round 14.1（若繼續 VICReg 路線）

1. 放寬 selection / 強制納入 **exp_008** route Top-K
2. 更低 VICReg λ 或更晚 start epoch
3. 僅 `none` / `own_plus_summary` downstream
4. 5-seed 驗證 Round 13 best 可重現性

## Artifacts

- Runtime final report: `result/optimization_runs/round14_vicreg_stabilizer/final_report/round14_final_report.md`
- Pretrain sweep: `reports/round14_vicreg_sweep_summary.csv`
- Feature mode: `final_report/round14_response_feature_summary.csv`
- z vs proto: `final_report/round14_z_vs_proto_delta.csv`
- Aggregate: `aggregate/aggregate_scores.csv`
- Logs: `logs/pipeline.log`, `logs/downstream.log`
