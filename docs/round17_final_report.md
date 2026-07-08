# Round 17 Final Report

**Run:** `result/optimization_runs/round17_direct_proto`  
**Pipeline:** Stage 17A → 17B → 17C（含 Telegram stage 通知）  
**Final status:** ALL_DONE（17A **1440/1440**；17B **30/30**；17C **50/50**；`failed=0`）

## Timeline

| Phase | When | Notes |
|-------|------|-------|
| Phase 0（5-target eval） | 2026-06 | `Integrated5_*`、AACDR extended SMILES |
| Stage 17A feature sweep | 2026-06-30 ~ 2026-07-07 | 1440 finetune jobs；`max-parallel=22` |
| Stage 17B head search | 2026-07-07 | 30 jobs（`concat_mlp`）；3 筆 `r13_exp_035_control` 初跑失敗後重跑成功 |
| Stage 17C 10-seed confirm | 2026-07-07 | 5 candidates × 10 seeds = 50 jobs |
| Code fix | `287dd73` | `r13_exp_035_control` model key 解析修正（避免 17B/17C `model_select` 路徑錯誤） |

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|----------------------|
| **Round 13 best `r13_exp_008_own_plus_summary`** | **0.6112** |
| Round 15 best `r15c_exp_005_own_plus_summary` | 0.6083 |
| Round 16 best | 0.6068 |

## Pipeline summary

| Stage | Planned | Completed | Notes |
|-------|---------|-----------|-------|
| 17A feature optimization | 1440 | **1440/1440** | 4 models × 多 feature modes × combos × 3 seeds |
| 17B prototype head | 30 | **30/30** | Top-10 from 17A × `concat_mlp` × 3 seeds |
| 17C confirmation | 50 | **50/50** | Top-5 from 17B × 10 seeds |

**GPU 參數（正式跑）：** `FINETUNE_PARALLEL=22`，`FINETUNE_BATCH_SIZE=24576`，`FINETUNE_MINI_BATCH_SIZE=6144`，`FINETUNE_EPOCHS=1500`

**報表路徑：**

- `reports_stage17a/round17_top_candidates.csv`
- `reports_stage17b/round17_top_candidates.csv`
- `reports_stage17c/round17_top_candidates.csv`

## Downstream（Stage 17C 最終）

### Best historical（13 seen drugs, `gdsc_intersect13`）

| Model | feature_mode | Avg TCGA mean ± std | Integrated5 mean ± std |
|-------|--------------|---------------------|------------------------|
| **r13_exp_008_own_proto_context_projected_16** | own_proto_context_projected_16 | **0.5892 ± 0.0336** | 0.5606 ± 0.0246 |
| r15c_exp_005_own_plus_summary | own_plus_summary | 0.5868 ± 0.0304 | **0.5617 ± 0.0282** |
| r13_exp_008_own_proto_delta_projected_8 | own_proto_delta_projected_8 | 0.5840 ± 0.0306 | 0.5574 ± 0.0273 |

- **vs Round 13 best (0.6112):** **−0.0220**（最佳 historical 仍低於 R13）
- **vs stretch goal 0.6200:** **not met**
- **own_plus_summary 10-seed mean（17C top-5 中）：** 0.5868（`r15c_exp_005`）

### 17A 階段最佳（feature sweep, single seed per combo）

17A top candidate（by Integrated5）：`r13_exp_008_control_own_plus_summary` — historical **0.5998**，Integrated5 **0.5782**

### 17B 階段最佳（head search）

17B best historical：**0.6083**（`r13_exp_008_own_proto_delta_projected_8`）；best Integrated5：**0.5762**（`r15c_exp_024_own_plus_summary`）

## 替代基準：各藥物 AUC macro mean（5 datasets）

若不以單一 headline target（`gdsc_intersect13` / `Average_TCGA_AUC`）排序，改以**每個 eval target 內各藥物 AUC 的 macro mean**（即 `Average_TCGA_AUC`，非 sample-pooled `Global_TCGA_AUC`），再對 5 個 target 取平均：

```text
mean_5target_drug_macro_auc =
  mean(
    gdsc_intersect13 Average_TCGA_AUC,
    tcga_only3 Average_TCGA_AUC,
    dapl Average_TCGA_AUC,
    aacdr_tcga_only Average_TCGA_AUC,
    aacdr_gdsc_intersect Average_TCGA_AUC
  )
```

資料來源：`stage17a/aggregate/aggregate_scores.csv`（60 models，feature sweep 全空間）。

### Overall Top-5（5-target drug-macro mean）

| Rank | Model | feature_mode | mean_5target | gdsc13 | tcga3 | dapl | aacdr_tcga | aacdr_gdsc |
|------|-------|--------------|-------------|--------|-------|------|------------|------------|
| 1 | `r13_exp_008_control` | `own_plus_summary` | **0.5782** | 0.5998 | 0.6255 | 0.5543 | 0.5999 | 0.5115 |
| 2 | `r15c_exp_024` | `own_plus_summary` | **0.5742** | 0.5942 | 0.6192 | 0.5551 | 0.5952 | 0.5074 |
| 3 | `r13_exp_008` | `own_plus_summary` | **0.5735** | 0.5896 | 0.6134 | 0.5562 | 0.5913 | 0.5173 |
| 4 | `r15c_exp_005` | `own_plus_summary` | **0.5732** | 0.5989 | 0.6022 | 0.5516 | 0.5926 | 0.5209 |
| 5 | `r13_exp_035_control` | `own_proto_delta_projected_16` | **0.5612** | 0.5873 | 0.5998 | 0.5761 | 0.5745 | 0.4685 |

**觀察：** Overall Top-5 中有 **4/5 為 `own_plus_summary`**；direct prototype 僅 `own_proto_delta_projected_16` 進入第 5 名。

### 各 Dataset Top-5（drug-macro AUC）

| Target | #1 | #2 | #3 | #4 | #5 |
|--------|----|----|----|----|-----|
| **gdsc_intersect13** (13 seen) | `r13_exp_035_control` / `none` (0.6049) | `r13_exp_008_control` / `own_plus_summary` (0.5998) | `r15c_exp_005` / `own_plus_summary` (0.5989) | `r15c_exp_024` / `own_plus_summary` (0.5942) | `r13_exp_008` / `own_plus_summary` (0.5896) |
| **tcga_only3** (3 unseen) | `r13_exp_035_control` / `minimal_source_only_min_margin` (0.6391) | `r15c_exp_024` / `own_proto_context_projected_32` (0.6335) | `r15c_exp_024` / `own_proto_delta_projected_64` (0.6268) | `r13_exp_008_control` / `own_plus_summary` (0.6255) | `r13_exp_008_control` / `own_proto_context_projected_32` (0.6243) |
| **dapl** (5 drugs) | `r13_exp_008` / `minimal_source_only_min_margin` (0.5955) | `r15c_exp_005` / `minimal_source_only_min_margin` (0.5940) | `r15c_exp_024` / `minimal_source_only_min_margin` (0.5906) | `r13_exp_035_control` / `own_plus_summary` (0.5839) | `r13_exp_035_control` / `own_plus_summary_plus_delta_projected_16` (0.5802) |
| **aacdr_tcga_only** | `r13_exp_008_control` / `own_plus_summary` (0.5999) | `r15c_exp_024` / `own_plus_summary` (0.5952) | `r15c_exp_005` / `own_plus_summary` (0.5926) | `r13_exp_008` / `own_plus_summary` (0.5913) | `r15c_exp_005` / `own_proto_context_projected_32` (0.5770) |
| **aacdr_gdsc_intersect** | `r15c_exp_005` / `own_plus_summary` (0.5209) | `r13_exp_008` / `own_plus_summary` (0.5173) | `r13_exp_008_control` / `own_plus_summary` (0.5115) | `r15c_exp_005` / `none` (0.5085) | `r15c_exp_024` / `own_plus_summary` (0.5074) |

### Stage 17C 確認（10-seed, 5 candidates）

| Rank | Model | feature_mode | mean_5target drug-macro |
|------|-------|--------------|-------------------------|
| 1 | `r15c_exp_005` | `own_plus_summary` | **0.5618** |
| 2 | `r13_exp_008` | `own_proto_context_projected_16` | 0.5606 |
| 3 | `r13_exp_008` | `own_proto_delta_projected_8` | 0.5574 |
| 4 | `r15c_exp_024` | `own_proto_context_projected_16` | 0.5542 |
| 5 | `r15c_exp_024` | `own_plus_summary` | 0.5537 |

> 對照欄位：`Integrated5_DrugMacro_TCGA_AUC_mean`（5 target 全部藥物 pooled macro）與上表 `mean_5target` 排序高度一致，但數值略低（因跨 target 藥物數不均）。

## 結論

| 問題 | 結果 |
|------|------|
| direct prototype 是否全面超越 `own_plus_summary`？ | **否**；以 headline `gdsc_intersect13` 或 **5-target drug-macro mean** 看，`own_plus_summary` 仍占多數 Top-5 |
| 最佳 projected 維度？ | **context_projected_16**（R13 exp_008）與 **delta_projected_8** 表現較佳 |
| AACDR 5-target 是否改變排序？ | `aacdr_*` 單獨看仍多為 `own_plus_summary` 居前；`tcga_only3` / `dapl` 則 minimal / projected 較強 |
| 是否重現 Round 13？ | **未達**（gap ≈ 0.022 on best 10-seed historical） |

## 已知問題與修復

| 問題 | 修復 |
|------|------|
| `r13_exp_035_control` 在 17B 被誤解析為 `r13_exp_035`，導致 `Missing model_select_path` | `287dd73`：config builder 改為最長匹配 + `model_select` 路徑驗證；3 筆 17B job 已重跑成功 |

## 待辦（未執行）

- **Stage 17F**：prototype-aware tSNE（`tools/run_round17_prototype_tsne_stage17f.sh`）
- 17B 進階 head（`two_tower_proto` / `proto_film`）待 `step1` classifier 接線

---

*Generated from `reports_stage17c/` aggregate on 2026-07-08; drug-macro rankings from `stage17a/aggregate/aggregate_scores.csv`.*
