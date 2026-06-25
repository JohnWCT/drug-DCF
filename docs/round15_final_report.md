# Round 15 Final Report

**Run:** `result/optimization_runs/round15_repro_rescue`  
**Completed:** 2026-06-24 18:37Z  
**Pipeline status:** ALL_DONE（pretrain **24/24**；manifest finetune **144/144** marked success；有效 downstream **72/144**（僅 z-only））

## Timeline

| Phase | When | Notes |
|-------|------|-------|
| Config + initial manifest | 15:57Z | 15A/15B finetune manifest 80 jobs；15C pretrain 24 jobs |
| 15C pretrain | 15:57Z–~16:14Z | **24/24** success；`PRETRAIN_PARALLEL=12` |
| Analyze + selection | ~16:14Z | Top-8 selected；rebuild finetune manifest **144** jobs |
| Feature extract | ~16:14Z | 36 feature rows written |
| Finetune | ~16:14Z–18:37Z | manifest 144 success；**72** 有效 `run_summary`（全為 `none`） |
| Aggregate + final analysis | 18:37Z | Done |

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|----------------------|
| **Round 13 best `r13_exp_008_own_plus_summary`** | **0.6112** |
| Round 14 best `r14_exp_078_own_plus_summary` | 0.5909 |
| Round 13 exp_035 z-only | 0.6059 |
| Round 12 exp_037 | 0.5972 |
| R7 exp_048 | 0.5918 |

## Pipeline summary

| Stage | Planned | Completed | Notes |
|-------|---------|-----------|-------|
| 15C pretrain | 24 | **24/24** | ultra-low / late VICReg on exp_008 route |
| Selection | Top-12 → **8** | **8** | `round15_repro_rescue_qc`；route_id 欄位標註異常（見下） |
| Feature extract | 36 | **36** rows | 實際輸出路徑與 manifest 不一致 |
| Finetune | 144 | **144** manifest success | **72** 有效 AUC；**0** `own_plus_summary` |
| Aggregate | — | **18** models | 僅 z-only best-combo per model |

**Parallelism:** `PRETRAIN_PARALLEL=12`，`FINETUNE_PARALLEL=12`。

## Downstream

- **Best model:** `r15b_exp_035_none`（Round 11 exp_035 z-only reference）
- **Average_TCGA_AUC_mean:** **0.5876**
- **Global_TCGA_AUC_mean:** 0.6138
- **vs Round 13 best (0.6112):** **−0.0236**（未重現、未超越）
- **vs Round 14 best (0.5909):** **−0.0033**
- **Stretch goal 0.6200:** **not met**

### Top-10 downstream（by Avg TCGA，皆 z-only）

| Rank | Model_ID | branch | Avg TCGA | Global TCGA |
|------|----------|--------|----------|-------------|
| 1 | **r15b_exp_035_none** | 15B | **0.5876** | 0.6138 |
| 2 | r15a_exp_008_none_s505 | 15A | 0.5838 | 0.6078 |
| 3 | r15c_exp_012_none | 15C | 0.5745 | 0.5901 |
| 4 | r15a_exp_008_none_s303 | 15A | 0.5714 | 0.6094 |
| 5 | r15a_exp_008_none_s202 | 15A | 0.5653 | 0.5934 |
| 6 | r15c_exp_005_none | 15C | 0.5615 | 0.6053 |
| 7 | r15c_exp_024_none | 15C | 0.5594 | 0.5893 |
| 8 | r15a_exp_008_none_s404 | 15A | 0.5583 | 0.5903 |
| 9 | r15b_exp_008_none | 15B | 0.5520 | 0.5846 |
| 10 | r15a_exp_008_none_s101 | 15A | 0.5494 | 0.5713 |

**`own_plus_summary` downstream：0 筆有效結果**（Round 15 核心假設未驗證）。

## Q1. Round 13 best 5-seed reproducibility？

**15A `exp_008` z-only（5 seeds）：**

| seed | Avg TCGA | Global TCGA |
|------|----------|-------------|
| 101 | 0.5494 | 0.5713 |
| 202 | 0.5653 | 0.5934 |
| 303 | 0.5714 | 0.6094 |
| 404 | 0.5583 | 0.5903 |
| 505 | 0.5838 | 0.6078 |

- **mean ± std：** **0.5656 ± 0.0131**
- **vs Round 13 best 0.6112：** **−0.0456**
- **vs floor 0.6000：** **FAIL**
- **own_plus_summary：** 無有效 downstream（見 infra issue）

**結論：** Round 13 peak **不可重現**（至少在此 run 的 z-only 5-seed 與缺失的 proto downstream 下）。

## Q2–Q4. exp_008 route + Round 14 coverage gap

| 問題 | 結果 |
|------|------|
| exp_008 route 是否穩定受益於 own_plus_summary？ | **未測到**（proto finetune 全失敗） |
| Round 14 漏測 exp_008 是否影響結論？ | 15B/15C 有測 exp_008 lineage，但 best 仍為 exp_035 z-only |
| 15A best exp_008 z-only | s505 **0.5838** << R13 **0.6112** |

## Q5–Q7. Route / feature / stack

| 項目 | 結果 |
|------|------|
| exp_035 backup | **是** — `r15b_exp_035_none` **0.5876** 為 Round 15 best |
| own_plus_summary 應為 final Step 2 mode？ | **無法判定** — 無有效 proto downstream |
| Avg TCGA > 0.6000？ | **否** — best **0.5876** |

## 15C ultra-low / late VICReg rescue

- Pretrain **24/24** success；mean kmeans_ari **0.600**
- Best pretrain structure：exp_012 / exp_018 / exp_024（λ=6e-5, start=90, seed=303, ARI **0.722**）
- Best 15C downstream z-only：`r15c_exp_012_none` **0.5745** — 未超越 15A/15B best
- VICReg rescue **未帶來** Round 13 級別 downstream 提升

## Infra issues（影響結論可信度）

### 1. Feature path mismatch（`own_plus_summary` 全滅）

- `round15_config_builder` 寫入 manifest 路徑：`features/{A|B|C}/{model_id}/{mode}[/s{seed}]`
- `extract_round13_proto_features` 實際輸出：`features/{model_id}/{mode}/`
- Finetune `combined_latent_dir` 不存在 → `own_plus_summary` 72 jobs 產出空結果（`best_overall_auc: -Infinity`），manifest 卻標記 success

### 2. Selection route_id 標註

- 15C pretrain `exp_005` 等實際 `route_id=exp008_proto_response_route`
- `selection/pretrain_top10.csv` 中 `round15_route_id` 全為 `exp035_strong_zonly_route`（繼承 Round 14 annotate 誤判）

## Round 16 Go / No-Go

**Recommendation:** **`NO-GO`** — 不進 importance-aware weighting

理由：
1. Round 13 best **未重現**（15A z-only mean **0.5656**）
2. **`own_plus_summary` 無有效 downstream** — 核心 Round 15 問題未回答
3. Best Round 15 **0.5876** < Round 13 **0.6112**、Round 14 **0.5909**
4. 需先修復 feature path + 重跑 `own_plus_summary` finetune，再決定是否 Round 15.1

### 建議 Round 15.1

1. 對齊 `combined_latent_dir`（config builder 或 feature extract）
2. 僅重跑 **72** `own_plus_summary` finetune jobs
3. 修正 selection `route_id` 標註
4. 若 own_plus_summary 仍無法接近 **0.6112** → 固定 Round 13 best 做 final validation / instability analysis

## Artifacts

- Manifests: `result/optimization_runs/round15_repro_rescue/manifests/`
- Reports: `result/optimization_runs/round15_repro_rescue/final_report/`
- Pipeline log: `result/optimization_runs/round15_repro_rescue/pipeline.log`
- Selection: `result/optimization_runs/round15_repro_rescue/selection/pretrain_top10.csv`
