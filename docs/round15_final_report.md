# Round 15 Final Report

**Run:** `result/optimization_runs/round15_repro_rescue`  
**Initial pipeline:** 2026-06-24 15:57Z–18:37Z  
**Proto resume:** 2026-06-25 02:38Z–04:57Z（feature path fix + `own_plus_summary` finetune retry）  
**Final status:** ALL_DONE（pretrain **24/24**；finetune **144/144**；aggregate **36** models）

## Timeline

| Phase | When | Notes |
|-------|------|-------|
| Config + 15C pretrain | 06-24 15:57Z | 24 ultra-low / late VICReg jobs |
| Selection + finetune (1st pass) | 06-24 16:14Z–18:37Z | 144 manifest success；**72 z-only 有效**；`own_plus_summary` 因 path mismatch 全滅 |
| Path fix + resume | 06-25 02:38Z–04:57Z | `extract_round13_proto_features` 尊重 `combined_latent_dir`；重跑 72 proto finetune |
| Aggregate + analysis | 06-25 04:57Z | **36** models（18 none + 18 own_plus_summary） |

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
| 15C pretrain | 24 | **24/24** | mean kmeans_ari **0.600** |
| Selection | Top-12 → **8** | **8** | 15C rescue candidates |
| Feature extract | 36 | **36** | resume 後 path 與 manifest 一致 |
| Finetune | 144 | **144/144** | 72 none + 72 own_plus_summary |
| Aggregate | — | **36** | best combo per model×mode |

**Parallelism:** `PRETRAIN_PARALLEL=12`，`FINETUNE_PARALLEL=12`。

## Downstream（resume 後最終）

- **Best model:** `r15c_exp_005_own_plus_summary`（15C ultra-low VICReg rescue）
- **Average_TCGA_AUC_mean:** **0.6083**
- **Global_TCGA_AUC_mean:** 0.6138
- **vs Round 13 best (0.6112):** **−0.0029**（接近但未重現）
- **vs Round 14 best (0.5909):** **+0.0174**
- **Stretch goal 0.6200:** **not met**

### Top-10 downstream（by Avg TCGA）

| Rank | Model_ID | mode | Avg TCGA | Global TCGA |
|------|----------|------|----------|-------------|
| 1 | **r15c_exp_005_own_plus_summary** | own_plus_summary | **0.6083** | 0.6138 |
| 2 | r15c_exp_024_own_plus_summary | own_plus_summary | 0.6065 | 0.6264 |
| 3 | r15b_exp_035_own_plus_summary | own_plus_summary | 0.5990 | 0.6020 |
| 4 | r15c_exp_012_own_plus_summary | own_plus_summary | 0.5917 | 0.6075 |
| 5 | r15c_exp_018_own_plus_summary | own_plus_summary | 0.5912 | 0.5976 |
| 6 | r15b_exp_008_own_plus_summary | own_plus_summary | 0.5889 | 0.6140 |
| 7 | r15b_exp_035_none | none | 0.5876 | 0.6138 |
| 8 | r15a_exp_008_own_plus_summary_s505 | own_plus_summary | 0.5839 | 0.6122 |
| 9 | r15a_exp_008_none_s505 | none | 0.5838 | 0.6078 |
| 10 | r15a_exp_008_own_plus_summary_s202 | own_plus_summary | 0.5828 | 0.6242 |

## Q1. Round 13 best 5-seed reproducibility？

**15A `exp_008`（Round 12 source checkpoint）：**

| seed | z-only | own_plus_summary | Δ |
|------|--------|------------------|---|
| 101 | 0.5494 | 0.5724 | +0.0229 |
| 202 | 0.5653 | 0.5828 | +0.0176 |
| 303 | 0.5714 | 0.5699 | −0.0015 |
| 404 | 0.5583 | 0.5641 | +0.0058 |
| 505 | 0.5838 | 0.5839 | +0.0000 |

| 指標 | z-only | own_plus_summary |
|------|--------|------------------|
| mean ± std | **0.5656 ± 0.0131** | **0.5746 ± 0.0085** |
| vs R13 0.6112 | −0.0456 | −0.0366 |
| vs floor 0.6000 | FAIL | FAIL |

**結論：** Round 13 peak **不可 5-seed 重現**；proto 在 4/5 seeds 有正增益但幅度遠小於 Round 13 單次 +0.0449。

## Q2–Q4. exp_008 route + Round 14 gap

| 問題 | 結果 |
|------|------|
| exp_008 是否受益於 own_plus_summary？ | **是** — 15B `exp_008` +0.0369（0.5520→0.5889）；15A 4/5 seeds 為正 |
| Round 14 漏測 exp_008 是否影響結論？ | 15C VICReg rescue 補測後 best 達 **0.6083**，高於 Round 14 **0.5909** |
| Best exp_008 lineage downstream | `r15b_exp_008_own_plus_summary` **0.5889** |

## Feature mode ablation（aggregate, n=18 pairs）

| 指標 | 結果 |
|------|------|
| own_plus_summary > z-only | **13/18** |
| mean Δ (ops − none) | **+0.0142** |
| Best proto gain (15C) | exp_024 **+0.0471**；exp_005 **+0.0469** |

### 15B forced route（z vs own_plus_summary）

| source | z-only | own_plus_summary | Δ |
|--------|--------|------------------|---|
| exp_008 | 0.5520 | 0.5889 | **+0.0369** |
| exp_035 | 0.5876 | 0.5990 | +0.0114 |
| exp_078 | 0.5427 | 0.5702 | +0.0275 |
| exp_028 | 0.5292 | 0.5531 | +0.0239 |
| exp_001 | 0.5487 | 0.5380 | −0.0107 |

## 15C ultra-low / late VICReg rescue

- Pretrain **24/24**；best structure ARI **0.722**（exp_012/018/024, λ=6e-5, start=90）
- Best downstream：`r15c_exp_005_own_plus_summary` **0.6083**（Global **0.6138**）
- 15C own_plus_summary 整體優於第一次 pass 的 z-only only 結論

## Infra fix（resume）

**問題：** `extract_round13_proto_features` 忽略 manifest `combined_latent_dir`，寫入 `features/{model_id}/{mode}/` 而非 `features/{branch}/{model_id}/{mode}/[seed]`。

**修復：** commit `3ff2f52` — honor manifest path；`run_round15_resume_proto_finetune.sh` 重跑 72 proto jobs。

**驗證：** `response_input_dim=75`（64+11）；18/18 proto artifact paths 正確。

## Round 16 Go / No-Go

**Recommendation:** **`NO-GO`** — 不進 importance-aware weighting

理由：
1. Round 13 best **0.6112** 未重現（15A mean **0.5746**；best overall **0.6083** 仍差 **−0.0029**）
2. `own_plus_summary` 穩定優於 z-only（13/18），但增益不足以回到 Round 13 peak
3. 15A 5-seed 顯示結果仍具 seed variance，不適合作為單點 final stack 宣告

### 建議後續

1. **Final validation** — 固定 Round 13 best `r13_exp_008_own_plus_summary` 做嚴格 per-cancer / per-drug 分析
2. 調查 Round 13 single-run peak vs 15A 5-seed gap（finetune seed / combo selection）
3. 若需再優化 — 15C `exp_005` / `exp_024` stack 值得 paper appendix，但不取代 Round 13 champion

## Artifacts

- `result/optimization_runs/round15_repro_rescue/aggregate/aggregate_scores.csv`
- `result/optimization_runs/round15_repro_rescue/final_report/`
- `result/optimization_runs/round15_repro_rescue/resume_proto.log`
- `result/optimization_runs/round15_repro_rescue/pipeline.log`
