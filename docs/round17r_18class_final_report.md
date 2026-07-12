# Round 17R Final Report（18-class-clean）

**Run:** `result/optimization_runs/round17r_18class`  
**Pipeline:** Stage 17R-A → 17R-B → 17R-C → 17R-D → 17R-F  
**Final status:** **ALL_DONE**（2026-07-12）

## 整體完成度

| Stage | 計畫 | 完成 | 說明 |
|-------|------|------|------|
| 17R-A feature smoke | 20 features | ✅ **20/20** | 18-class QC 全通過 |
| 17R-B focused finetune | 126 jobs | ✅ **126/126** | 7 candidates × 6 combos × 3 seeds |
| 17R-C hyperparameter refine | 180 jobs | ✅ **180/180** | top-6 × 6 combos × 5 seeds |
| 17R-D 10-seed confirm | 50 jobs | ✅ **50/50** | top-5 × 1 combo × 10 seeds |
| 17R-F prototype tSNE | 2 models | ✅ **1/2** | `r13_exp_008` OK；`r13_exp_035_control` 不在 17R feature manifest（skip） |

**GPU 參數：** `FINETUNE_PARALLEL=20`，`batch=24576`，`mini_batch=6144`，`epochs=1500`

---

## 基準對照

| Benchmark | Average_TCGA_AUC |
|-----------|------------------|
| Round 13 best `r13_exp_008_own_plus_summary` | **0.6112** |
| Round 17C pre-18class best（10-seed） | 0.5892（`context_16`） |
| Round 17C pre-18class best `own_plus_summary` | 0.5868（`r15c_exp_005`） |

---

## 階段結果摘要（historical `gdsc_intersect13`）

| Stage | Best model | Strategy | AUC mean ± std | vs R13 |
|-------|------------|----------|----------------|--------|
| 17R-B（3-seed×6 combo） | `r15c_exp_024` | `own_plus_summary` | **0.6074 ± 0.0191** | −0.0039 |
| 17R-C（5-seed×6 combo） | `r13_exp_008` | `own_plus_summary` | **0.5979 ± 0.0285** | −0.0134 |
| **17R-D（10-seed confirm）** | **`r13_exp_008`** | **`own_plus_summary`** | **0.5915 ± 0.0360** | **−0.0197** |

> 單點峰值（17R-B）接近 R13，但 10-seed confirm 後均值回落到 ~0.59，仍優於 pre-18class 17C 的 `own_plus_summary`（0.5868），未達 R13（0.6112）。

---

## 17R-D 最終確認（10-seed）

資料來源：`stage17r_d/aggregate/aggregate_scores.csv`（每候選 10 runs）

| Rank | Model | Strategy（feature_mode） | Historical AUC | Integrated5 TargetMacro | Integrated5 DrugMacro |
|------|-------|--------------------------|----------------|-------------------------|----------------------|
| 1 | `r13_exp_008` | **distance-to-proto summary**（`own_plus_summary`） | **0.5915 ± 0.0360** | 0.5537 | **0.5632** |
| 2 | `r13_exp_008_control` | distance-to-proto summary（`own_plus_summary`） | 0.5899 ± 0.0313 | 0.5505 | 0.5595 |
| 3 | `r15c_exp_005` | distance-to-proto summary（`own_plus_summary`） | 0.5889 ± 0.0392 | 0.5504 | 0.5593 |
| 4 | `r15c_exp_024` | distance-to-proto summary（`own_plus_summary`） | 0.5886 ± 0.0274 | 0.5509 | 0.5590 |
| 5 | `r13_exp_008` | **direct proto context-16**（`own_proto_context_projected_16`） | 0.5813 ± 0.0187 | 0.5513 | 0.5591 |

### vs Pre-18class Round 17C

| Model | Strategy | 17R-D | Pre-18 17C | Δ |
|-------|----------|-------|------------|---|
| `r15c_exp_005` | `own_plus_summary` | 0.5889 | 0.5868 | **+0.0020** |
| `r15c_exp_024` | `own_plus_summary` | 0.5886 | 0.5821 | **+0.0065** |
| `r13_exp_008` | `own_proto_context_projected_16` | 0.5813 | 0.5892 | −0.0079 |

---

## 各資料集 Top-5 與策略（17R-D 最終）

指標：各 eval target 的 `Average_TCGA_AUC`（per-drug macro mean）± std。  
策略說明：

| feature_mode | 策略含義 |
|--------------|----------|
| `own_plus_summary` | distance-to-prototype summary（主幹） |
| `own_proto_context_projected_16` | direct prototype context projected-16 |
| `own_proto_delta_projected_8` | direct prototype delta projected-8 |
| `minimal_source_only_min_margin` | minimal source geometry（min-margin） |

### 1) `gdsc_intersect13`（historical primary，13 seen drugs）

| Rank | Model | Strategy | AUC |
|------|-------|----------|-----|
| 1 | `r13_exp_008` | distance-to-proto summary | **0.5915 ± 0.0360** |
| 2 | `r13_exp_008_control` | distance-to-proto summary | 0.5899 ± 0.0313 |
| 3 | `r15c_exp_005` | distance-to-proto summary | 0.5889 ± 0.0392 |
| 4 | `r15c_exp_024` | distance-to-proto summary | 0.5886 ± 0.0274 |
| 5 | `r13_exp_008` | direct proto context-16 | 0.5813 ± 0.0187 |

**判讀：** Top-4 全為 `own_plus_summary`；context-16 略低但 std 最小（最穩）。

### 2) `tcga_only3`（3 unseen drugs）

| Rank | Model | Strategy | AUC |
|------|-------|----------|-----|
| 1 | `r15c_exp_024` | distance-to-proto summary | **0.5286 ± 0.0347** |
| 2 | `r13_exp_008_control` | distance-to-proto summary | 0.5266 ± 0.0442 |
| 3 | `r13_exp_008` | distance-to-proto summary | 0.5255 ± 0.0392 |
| 4 | `r15c_exp_005` | distance-to-proto summary | 0.5238 ± 0.0410 |
| 5 | `r13_exp_008` | direct proto context-16 | 0.5154 ± 0.0468 |

**判讀：** unseen drugs 上仍以 `own_plus_summary` 為主；整體 AUC 低於 historical。

### 3) `dapl`（DAPL 5-drug response）

| Rank | Model | Strategy | AUC |
|------|-------|----------|-----|
| 1 | `r13_exp_008` | **direct proto context-16** | **0.5442 ± 0.0231** |
| 2 | `r15c_exp_024` | distance-to-proto summary | 0.5230 ± 0.0336 |
| 3 | `r13_exp_008` | distance-to-proto summary | 0.5223 ± 0.0295 |
| 4 | `r15c_exp_005` | distance-to-proto summary | 0.5221 ± 0.0316 |
| 5 | `r13_exp_008_control` | distance-to-proto summary | 0.5210 ± 0.0302 |

**判讀：** **唯一由 direct prototype 奪冠的 target**；context-16 明顯優於 summary（+0.021）。

### 4) `aacdr_tcga_only`（AACDR TCGA-only）

| Rank | Model | Strategy | AUC |
|------|-------|----------|-----|
| 1 | `r13_exp_008` | distance-to-proto summary | **0.5752 ± 0.0464** |
| 2 | `r15c_exp_005` | distance-to-proto summary | 0.5685 ± 0.0397 |
| 3 | `r15c_exp_024` | distance-to-proto summary | 0.5671 ± 0.0458 |
| 4 | `r13_exp_008` | direct proto context-16 | 0.5650 ± 0.0517 |
| 5 | `r13_exp_008_control` | distance-to-proto summary | 0.5610 ± 0.0615 |

**判讀：** `own_plus_summary` 主導；context-16 可進 Top-4，但波動較大。

### 5) `aacdr_gdsc_intersect`（AACDR GDSC-intersect）

| Rank | Model | Strategy | AUC |
|------|-------|----------|-----|
| 1 | `r13_exp_008` | distance-to-proto summary | **0.5541 ± 0.0304** |
| 2 | `r13_exp_008_control` | distance-to-proto summary | 0.5540 ± 0.0410 |
| 3 | `r13_exp_008` | direct proto context-16 | 0.5505 ± 0.0291 |
| 4 | `r15c_exp_005` | distance-to-proto summary | 0.5487 ± 0.0363 |
| 5 | `r15c_exp_024` | distance-to-proto summary | 0.5472 ± 0.0291 |

**判讀：** summary 與 context-16 接近；整體水準低於 historical / aacdr_tcga_only。

### 5-target 整體（Integrated5）

| Rank | Model | Strategy | TargetMacro | DrugMacro |
|------|-------|----------|-------------|-----------|
| 1 | `r13_exp_008` | distance-to-proto summary | **0.5537** | **0.5632** |
| 2 | `r13_exp_008` | direct proto context-16 | 0.5513 | 0.5591 |
| 3 | `r15c_exp_024` | distance-to-proto summary | 0.5509 | 0.5590 |
| 4 | `r13_exp_008_control` | distance-to-proto summary | 0.5505 | 0.5595 |
| 5 | `r15c_exp_005` | distance-to-proto summary | 0.5504 | 0.5593 |

---

## 策略對照：17R-B 廣域候選（含未進 17R-D 者）

17R-D 僅保留 top-5，未含 `minimal_source` / `delta_8`。以下取自 **17R-B aggregate**（7 候選，各 18 runs），用於看 target-specific 策略：

### 各 Dataset 冠軍策略（17R-B）

| Dataset | #1 Model | Strategy | AUC |
|---------|----------|----------|-----|
| `gdsc_intersect13` | `r15c_exp_024` | distance-to-proto summary | 0.6074 |
| `tcga_only3` | `r13_exp_008` | distance-to-proto summary | 0.5481 |
| **`dapl`** | `r13_exp_008` | **minimal source min-margin** | **0.5641** |
| `aacdr_tcga_only` | `r13_exp_008` | **direct proto delta-8** | 0.5714 |
| `aacdr_gdsc_intersect` | `r15c_exp_024` | distance-to-proto summary | 0.5820 |

### 17R-C 補充（refine，各 30 runs）

| Dataset | #1 Model | Strategy | AUC |
|---------|----------|----------|-----|
| `gdsc_intersect13` | `r13_exp_008` | distance-to-proto summary | 0.5979 |
| `tcga_only3` | `r15c_exp_024` | distance-to-proto summary | 0.5403 |
| **`dapl`** | `r13_exp_008` | **minimal source min-margin** | **0.5529** |
| `aacdr_tcga_only` | `r15c_exp_005` | distance-to-proto summary | 0.5861 |
| `aacdr_gdsc_intersect` | `r15c_exp_024` | distance-to-proto summary | 0.5917 |

> **`dapl` 上 minimal_source / context-16 持續優於 summary**；但因 historical headline 落後，未進入 17R-D confirm。可作 ablation / target-specific insight，不作 primary。

---

## 去重模型清單（跨 5 datasets Top 出現）

以 **17R-D** 為最終確認集合：

| # | model_id | Strategy | 出現於 Top（dataset） |
|---|----------|----------|----------------------|
| 1 | `r13_exp_008_own_plus_summary` | distance-to-proto summary | gdsc13:#1, tcga3:#3, dapl:#3, aacdr_tcga:#1, aacdr_gdsc:#1, Integrated5:#1 |
| 2 | `r13_exp_008_control_own_plus_summary` | distance-to-proto summary | gdsc13:#2, tcga3:#2, dapl:#5, aacdr_tcga:#5, aacdr_gdsc:#2 |
| 3 | `r15c_exp_005_own_plus_summary` | distance-to-proto summary | gdsc13:#3, tcga3:#4, dapl:#4, aacdr_tcga:#2, aacdr_gdsc:#4 |
| 4 | `r15c_exp_024_own_plus_summary` | distance-to-proto summary | gdsc13:#4, tcga3:#1, dapl:#2, aacdr_tcga:#3, aacdr_gdsc:#5 |
| 5 | `r13_exp_008_own_proto_context_projected_16` | direct proto context-16 | gdsc13:#5, tcga3:#5, **dapl:#1**, aacdr_tcga:#4, aacdr_gdsc:#3 |

另（僅見於 17R-B/C，未進 17R-D）：

| model_id | Strategy | 強項 dataset |
|----------|----------|--------------|
| `r13_exp_008_minimal_source_only_min_margin` | minimal source min-margin | **dapl**（17R-B/C #1） |
| `r13_exp_008_own_proto_delta_projected_8` | direct proto delta-8 | aacdr_tcga_only（17R-B #1） |

---

## 核心問題答覆

| 問題 | 結果 |
|------|------|
| 18-class-clean 後 ranking 是否改變？ | **是**；`own_plus_summary` 回升為明確主幹 |
| `own_plus_summary` 是否仍最穩定主線？ | **是**；17R-D Top-4 全為 summary |
| direct prototype 是否接近 summary？ | **否**（historical gap ≈ 0.010）；但在 **dapl** 上 context-16 可奪冠 |
| minimal_source 優勢是否保留？ | **僅 target-specific**（dapl）；未進 17R-D |
| 是否重現 Round 13（0.6112）？ | **否**；10-seed best = 0.5915（gap −0.0197） |
| 是否進入 final validation？ | **否**（未達 0.6112）；建議以 `own_plus_summary` 作 primary |

---

## Stage 17R-F（prototype tSNE）

| Model | 狀態 | source / target prototypes |
|-------|------|-----------------------------|
| `r13_exp_008` | ✅ | 18 / 18（無 Engineered / Fibroblast） |
| `r13_exp_035_control` | ⏭ skip | 不在 17R-A feature manifest |

輸出：`result/optimization_runs/round17r_18class/visualizations/prototype_tsne/r13_exp_008/`

---

## 結論與建議

1. **Primary strategy：** `own_plus_summary`（distance-to-prototype summary）+ Round 13/15 強 checkpoint（`r13_exp_008` / `r15c_exp_005` / `r15c_exp_024`）。
2. **Secondary / target-specific：** `own_proto_context_projected_16`（尤其 **dapl**）；`minimal_source_only_min_margin` 作 ablation。
3. **不建議** 以 direct prototype 全面取代 summary；也不建議僅因 17R-B 單點接近 R13 而宣稱重現。
4. **下一方法輪（Round 18）** 可另開；Round 17R 作為 18-class-clean confirmation 已結束。

---

## 產出路徑

| 類型 | 路徑 |
|------|------|
| 17R-B/C/D reports | `result/optimization_runs/round17r_18class/reports_stage17r_{b,c,d}/` |
| Aggregates | `.../stage17r_{b,c,d}/aggregate/aggregate_scores.csv` |
| Manifests | `.../manifests/stage17r_*_finetune_dispatch_manifest.csv` |
| 樣本統計 | `docs/round17r_18class_dataset_sample_usage.md` |
| Round 17 總報告 | `docs/round17_final_report.md` |

---

*Generated from 17R-B/C/D aggregates + reports on 2026-07-12.*
