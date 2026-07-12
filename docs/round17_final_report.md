# Round 17 Final Report

**Run:** `result/optimization_runs/round17_direct_proto`  
**Pipeline:** Stage 17A → 17B → 17C（含 Telegram stage 通知）  
**Pre-18class status:** ALL_DONE（17A **1440/1440**；17B **30/30**；17C **50/50**；17F **2/2** models；`failed=0`）  
**18-class-clean status:** 完整 17A 重跑已暫停；改走 **Round 17R**（見下方 §Round 17R）

## Timeline

| Phase | When | Notes |
|-------|------|-------|
| Phase 0（5-target eval） | 2026-06 | `Integrated5_*`、AACDR extended SMILES |
| Stage 17A feature sweep | 2026-06-30 ~ 2026-07-07 | 1440 finetune jobs；`max-parallel=22` |
| Stage 17B head search | 2026-07-07 | 30 jobs（`concat_mlp`）；3 筆 `r13_exp_035_control` 初跑失敗後重跑成功 |
| Stage 17C 10-seed confirm | 2026-07-07 | 5 candidates × 10 seeds = 50 jobs |
| Stage 17F prototype tSNE | 2026-07-08 | `r13_exp_008`、`r13_exp_035_control` 各產出 png/pdf/csv |
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
| 17F prototype tSNE | 2 models | **2/2** | coordinates + png + pdf per model |

**GPU 參數（正式跑）：** `FINETUNE_PARALLEL=22`，`FINETUNE_BATCH_SIZE=24576`，`FINETUNE_MINI_BATCH_SIZE=6144`，`FINETUNE_EPOCHS=1500`

**報表路徑：**

- `reports_stage17a/round17_top_candidates.csv`
- `reports_stage17b/round17_top_candidates.csv`
- `reports_stage17c/round17_top_candidates.csv`
- `visualizations/prototype_tsne/{model}/prototype_tsne_*`

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

### 各 Dataset Top-5 去重清單（16 models，含重現路徑）

**排名來源：** `result/optimization_runs/round17_direct_proto/stage17a/aggregate/aggregate_scores.csv`  
**指標：** 各 dataset 的 per-drug macro AUC（`Average_TCGA_AUC_*` 欄位）

**共用設定（重現 17A finetune / eval）：**

| 用途 | 路徑 |
|------|------|
| Round 17 設定 | `config/round17_direct_proto_settings.json` |
| Finetune 超參 | `config/params_finetune_round17_direct_proto.json` |
| Drug SMILES | `data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv` |
| 17A manifest | `result/optimization_runs/round17_direct_proto/manifests/stage17a_finetune_dispatch_manifest.csv` |
| Aggregate 來源 | `result/optimization_runs/round17_direct_proto/stage17a/aggregate/aggregate_scores.csv` |

**TCGA eval 五資料集：**

| eval key | 路徑 |
|----------|------|
| `gdsc_intersect13` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` |
| `tcga_only3` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` |
| `dapl` | `data/TCGA/TCGA_drug_response_from_DAPL.csv` |
| `aacdr_tcga_only` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv` |
| `aacdr_gdsc_intersect` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv` |

**去重模型清單（依出現 dataset 數排序）：**

| # | model_id | feature_mode | 出現於 Top-5（dataset:rank） | pretrain checkpoint | feature_dir |
|---|----------|--------------|------------------------------|---------------------|-------------|
| 1 | `r13_exp_008_control_own_plus_summary` | own_plus_summary | gdsc13:#2, tcga3:#4, aacdr_tcga:#1, aacdr_gdsc:#3 | `result/optimization_runs/round12_proto_alignment/pretrain/exp_008` | `.../features/r13_exp_008_control/own_plus_summary` |
| 2 | `r15c_exp_005_own_plus_summary` | own_plus_summary | gdsc13:#3, aacdr_tcga:#3, aacdr_gdsc:#1 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_005` | `.../features/r15c_exp_005/own_plus_summary` |
| 3 | `r13_exp_008_own_plus_summary` | own_plus_summary | gdsc13:#5, aacdr_tcga:#4, aacdr_gdsc:#2 | `result/optimization_runs/round12_proto_alignment/pretrain/exp_008` | `.../features/r13_exp_008/own_plus_summary` |
| 4 | `r15c_exp_024_own_plus_summary` | own_plus_summary | gdsc13:#4, aacdr_tcga:#2, aacdr_gdsc:#5 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_024` | `.../features/r15c_exp_024/own_plus_summary` |
| 5 | `r13_exp_035_control_minimal_source_only_min_margin` | minimal_source_only_min_margin | tcga3:#1 | `result/optimization_runs/round11_stability_recon/pretrain/exp_035` | `.../features/r13_exp_035_control/minimal_source_only_min_margin` |
| 6 | `r15c_exp_024_own_proto_context_projected_32` | own_proto_context_projected_32 | tcga3:#2 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_024` | `.../features/r15c_exp_024/own_proto_context_projected_32` |
| 7 | `r15c_exp_024_own_proto_delta_projected_64` | own_proto_delta_projected_64 | tcga3:#3 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_024` | `.../features/r15c_exp_024/own_proto_delta_projected_64` |
| 8 | `r13_exp_008_control_own_proto_context_projected_32` | own_proto_context_projected_32 | tcga3:#5 | `result/optimization_runs/round12_proto_alignment/pretrain/exp_008` | `.../features/r13_exp_008_control/own_proto_context_projected_32` |
| 9 | `r13_exp_035_control_none` | none | gdsc13:#1 | `result/optimization_runs/round11_stability_recon/pretrain/exp_035` | `.../features/r13_exp_035_control/none` |
| 10 | `r13_exp_008_minimal_source_only_min_margin` | minimal_source_only_min_margin | dapl:#1 | `result/optimization_runs/round12_proto_alignment/pretrain/exp_008` | `.../features/r13_exp_008/minimal_source_only_min_margin` |
| 11 | `r15c_exp_005_minimal_source_only_min_margin` | minimal_source_only_min_margin | dapl:#2 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_005` | `.../features/r15c_exp_005/minimal_source_only_min_margin` |
| 12 | `r15c_exp_024_minimal_source_only_min_margin` | minimal_source_only_min_margin | dapl:#3 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_024` | `.../features/r15c_exp_024/minimal_source_only_min_margin` |
| 13 | `r13_exp_035_control_own_plus_summary` | own_plus_summary | dapl:#4 | `result/optimization_runs/round11_stability_recon/pretrain/exp_035` | `.../features/r13_exp_035_control/own_plus_summary` |
| 14 | `r13_exp_035_control_own_plus_summary_plus_delta_projected_16` | own_plus_summary_plus_delta_projected_16 | dapl:#5 | `result/optimization_runs/round11_stability_recon/pretrain/exp_035` | `.../features/r13_exp_035_control/own_plus_summary_plus_delta_projected_16` |
| 15 | `r15c_exp_005_own_proto_context_projected_32` | own_proto_context_projected_32 | aacdr_tcga:#5 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_005` | `.../features/r15c_exp_005/own_proto_context_projected_32` |
| 16 | `r15c_exp_005_none` | none | aacdr_gdsc:#4 | `result/optimization_runs/round15_repro_rescue/pretrain/exp_005` | `.../features/r15c_exp_005/none` |

上表 `...` = `result/optimization_runs/round17_direct_proto`。

**每個 model 另需路徑（重現 finetune 結果）：**

| 類型 | 路徑樣式 |
|------|----------|
| model_select | `result/optimization_runs/round17_direct_proto/manifests/model_selects/<model_id>.csv` |
| 17A finetune 輸出 | `result/optimization_runs/round17_direct_proto/stage17a/finetune/<model_id>/combo_*/seed_*/` |
| 整合 eval 摘要 | `.../combo_*/seed_*/eval_metrics_integrated_summary.csv` |
| 單 target eval | `.../combo_*/seed_*/<model_id>/param_001/target_eval_<eval_key>/target_metrics_summary.csv` |

**純 model_id 去重 list（16）：**

```text
r13_exp_008_control_own_plus_summary
r15c_exp_005_own_plus_summary
r13_exp_008_own_plus_summary
r15c_exp_024_own_plus_summary
r13_exp_035_control_minimal_source_only_min_margin
r15c_exp_024_own_proto_context_projected_32
r15c_exp_024_own_proto_delta_projected_64
r13_exp_008_control_own_proto_context_projected_32
r13_exp_035_control_none
r13_exp_008_minimal_source_only_min_margin
r15c_exp_005_minimal_source_only_min_margin
r15c_exp_024_minimal_source_only_min_margin
r13_exp_035_control_own_plus_summary
r13_exp_035_control_own_plus_summary_plus_delta_projected_16
r15c_exp_005_own_proto_context_projected_32
r15c_exp_005_none
```

### Stage 17C 確認（10-seed, 5 candidates）

| Rank | Model | feature_mode | mean_5target drug-macro |
|------|-------|--------------|-------------------------|
| 1 | `r15c_exp_005` | `own_plus_summary` | **0.5618** |
| 2 | `r13_exp_008` | `own_proto_context_projected_16` | 0.5606 |
| 3 | `r13_exp_008` | `own_proto_delta_projected_8` | 0.5574 |
| 4 | `r15c_exp_024` | `own_proto_context_projected_16` | 0.5542 |
| 5 | `r15c_exp_024` | `own_plus_summary` | 0.5537 |

> 對照欄位：`Integrated5_DrugMacro_TCGA_AUC_mean`（5 target 全部藥物 pooled macro）與上表 `mean_5target` 排序高度一致，但數值略低（因跨 target 藥物數不均）。

## Average_TCGA_AUC 參數調整方向（重點）

分析基礎：

- 17A 全量 job-level 結果（`stage17a/finetune/**/parameter_comparison_tcga_focus.csv`，共 1440 jobs）
- 17A manifest（`manifests/stage17a_finetune_dispatch_manifest.csv`，含 combo/seed）
- 17C 10-seed top-5（`reports_stage17c_pre18class_fix_20260708T035033Z/round17_top_candidates.csv`）

### 1) 有貢獻的 feature 方向（看 `Average_TCGA_AUC`）

| feature_mode | 17A 平均 | 17A 單點最高 | Top-100 job 出現次數 |
|--------------|----------|--------------|----------------------|
| `own_plus_summary` | **0.5925** | 0.6485 | **28** |
| `none` | 0.5855 | **0.6557** | 21 |
| `own_proto_delta_projected_8` | 0.5804 | 0.6296 | 11 |
| `own_proto_delta_projected_16` | 0.5788 | 0.6359 | 13 |
| `own_proto_context_projected_16` | 0.5723 | 0.6487 | 5 |

判讀：

- **最穩定主線：** `own_plus_summary`（平均最好、Top 命中率最高）。
- **高峰但不穩：** `none` 可跑出最高單點，但均值與跨 seed 穩定度不如 `own_plus_summary`。
- **direct prototype 可延伸方向：** `delta_projected_8/16`（均值次佳）與 `context_projected_16`（17C 最佳 historical）。

### 2) 有貢獻的 finetune combo 方向（看 `Average_TCGA_AUC`）

| combo | 平均 | 單點最高 | 主要設定 |
|------|------|----------|----------|
| **4** | **0.5805** | 0.6309 | `lr=1e-4`, `wd=1e-4`, `dropout=0.2`, default head, `12288/3072`, patience=100 |
| 3 | 0.5764 | 0.6485 | `lr=3e-4`, `wd=3e-5`, `dropout=0.2`, default head, `12288/3072`, patience=100 |
| 2 | 0.5753 | 0.6468 | `lr=5e-4`, `wd=1e-5`, `dropout=0.1`, default head, `12288/3072`, patience=50 |
| 7 | 0.5715 | 0.6407 | `lr=5e-4`, `wd=1e-5`, `dropout=0.1`, default head, `8192/2048`, patience=50 |
| 6 | 0.5563 | 0.6487 | `lr=3e-4`, `wd=1e-5`, `dropout=0.1`, **`head=512,256`** |

判讀（可持續優化）：

- **正向訊號：** 較強 regularization（`dropout=0.2` + 較高 `weight_decay`）在均值上更有利（combo 3/4）。
- **head 維度：** custom head（`256,128` / `512,256`）均值落後 default，短期先以 default 為主。
- **batch：** `12288/3072` 整體優於 `8192/2048`（combo 7 相對下滑）。
- **學習率：** `1e-4`（combo 4）在均值最穩，`3e-4` 可保留做次選；`5e-4` 較偏高風險。

### 3) 後續優化優先順序（以 `Average_TCGA_AUC`）

1. **主幹保留：** `own_plus_summary` + default head + `dropout=0.2` 設定帶（combo 4/3）。
2. **prototype 強化線：** `own_proto_delta_projected_8/16`、`own_proto_context_projected_16` 分支加密搜尋。
3. **超參縮圈：** 先集中 `lr ∈ {1e-4, 2e-4, 3e-4}`、`wd ∈ {3e-5, 1e-4, 3e-4}`、`dropout ∈ {0.15, 0.2, 0.25}`。
4. **減少低效組合：** 先暫停 custom head 大維度（尤其 `512,256`）與小 batch 線。
5. **穩健評估：** 對新候選至少做 5-seed，再進 10-seed confirm，避免單點高分誤判。

## 結論

| 問題 | 結果 |
|------|------|
| direct prototype 是否全面超越 `own_plus_summary`？ | **否**；以 headline `gdsc_intersect13` 或 **5-target drug-macro mean** 看，`own_plus_summary` 仍占多數 Top-5 |
| 最佳 projected 維度？ | **context_projected_16**（R13 exp_008）與 **delta_projected_8** 表現較佳 |
| AACDR 5-target 是否改變排序？ | `aacdr_*` 單獨看仍多為 `own_plus_summary` 居前；`tcga_only3` / `dapl` 則 minimal / projected 較強 |
| 是否重現 Round 13？ | **未達**（gap ≈ 0.022 on best 10-seed historical） |

## Stage 17F（prototype tSNE）

僅使用 **pretrain 訓練的 18 類癌症**（與 `metadata/cancer_type_mapping.json`、`kmeans_k=18` 一致；CCLE≥10 ∩ TCGA，並排除 `config/pretrain_cancer_type_exclude.json` 所列類型）。tSNE 樣本與 prototype 皆過濾至這 18 類，不再出現 Engineered / Fibroblast 等非訓練類型。

| Model | n_points | source prototypes | target prototypes plotted | trainable cancer types |
|-------|----------|-------------------|---------------------------|------------------------|
| `r13_exp_008` | 3973 | 18 | 18 | 18 |
| `r13_exp_035_control` | 3973 | 18 | 18 | 18 |

輸出目錄：`result/optimization_runs/round17_direct_proto/visualizations/prototype_tsne/<model>/`

實作：`extract_round12_prototypes.py` 讀取 checkpoint `metadata/cancer_type_mapping.json`；`visualize_round17_prototype_tsne.py` 依同一清單過濾 latent 樣本。

## 已知問題與修復

| 問題 | 修復 |
|------|------|
| `r13_exp_035_control` 在 17B 被誤解析為 `r13_exp_035`，導致 `Missing model_select_path` | `287dd73`：config builder 改為最長匹配 + `model_select` 路徑驗證；3 筆 17B job 已重跑成功 |
| sklearn 1.0 `TSNE` 不支援 `max_iter` | `visualize_round17_prototype_tsne.py` 自動 fallback 至 `n_iter` |
| tSNE / prototype cache 使用 metadata 全量癌症（含 Engineered 等 28 類） | 改為讀取 checkpoint `metadata/cancer_type_mapping.json`（18 類）；17F 已重跑 |

## 18 類 prototype 修正後重跑（已改走 Round 17R）

因 prototype feature 曾基於 28 類 cache，原規劃重跑 Stage **17A → 17B → 17C**（1440 jobs）。完整重跑已暫停（manifest：`1404 pending` + `36 failed`）；改以 **Round 17R** focused rerun 取代。

本報告 §Downstream 結論仍基於 `reports_stage17*_pre18class_fix_*`；18-class-clean 最新數字見 Round 17R 報告。

## Round 17R（18-class-clean focused）— ALL_DONE

**Run:** `result/optimization_runs/round17r_18class`  
**完整報告：** `docs/round17r_18class_final_report.md`（含各資料集 Top-5 + 策略）

| Stage | 狀態 | 說明 |
|-------|------|------|
| 17R-A feature smoke | ✅ **20/20** | 18-class QC 全通過 |
| 17R-B focused finetune | ✅ **126/126** | 7 candidates × 6 combos × 3 seeds |
| 17R-C refine | ✅ **180/180** | top-6 × 6 combos × 5 seeds |
| 17R-D 10-seed confirm | ✅ **50/50** | top-5 × 10 seeds |
| 17R-F tSNE | ✅ | `r13_exp_008`（18/18）；`r13_exp_035_control` skip |

### 最終（17R-D）vs Round 13 / Pre-18class

| 指標 | Pre-18class 17C | 17R-B peak | **17R-D 10-seed** |
|------|-----------------|------------|-------------------|
| Best historical AUC | 0.5892（`context_16`） | 0.6074（`r15c_exp_024` summary） | **0.5915 ± 0.036**（`r13_exp_008` summary） |
| vs Round 13（0.6112） | −0.0220 | −0.0039 | **−0.0197** |
| Primary strategy | mixed | `own_plus_summary` | **`own_plus_summary`** |

### 17R-D 各資料集冠軍（strategy）

| Dataset | #1 Model | Strategy |
|---------|----------|----------|
| `gdsc_intersect13` | `r13_exp_008` | distance-to-proto summary |
| `tcga_only3` | `r15c_exp_024` | distance-to-proto summary |
| `dapl` | `r13_exp_008` | **direct proto context-16** |
| `aacdr_tcga_only` | `r13_exp_008` | distance-to-proto summary |
| `aacdr_gdsc_intersect` | `r13_exp_008` | distance-to-proto summary |

樣本覆蓋統計：`docs/round17r_18class_dataset_sample_usage.md`

## 未來可選

- 17B 進階 head（`two_tower_proto` / `proto_film`）待 `step1` classifier 接線

---

*Generated from `reports_stage17c_pre18class_fix_20260708T035033Z/` aggregate + Stage 17F visualizations; Round 17R ALL_DONE updated 2026-07-12.*
