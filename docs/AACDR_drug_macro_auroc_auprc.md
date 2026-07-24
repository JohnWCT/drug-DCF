# DrugMacro AUROC / AUPRC（eval3 × 3 + target_infer × 2）

僅整理本專案模型結果：

- **eval3**（`outputs_aacdr_eval3`）：訓練 + 評估，DAPL 標註，3 個 TCGA 評估集
- **target_infer**（`outputs_aacdr_eval_aacdr_target_infer`）：以 eval3 checkpoint 推論，AACDR 標註，2 個 TCGA 評估集

兩者皆為 5-fold、seed=0、omics 為 DAPL pretrain（1426 features）。不含 `main_ckpt`。

## 實驗設定（評估集對照）

| 評估角色 | eval3（DAPL 標註） | target_infer（AACDR 標註） |
|----------|--------------------|----------------------------|
| Primary | `gdsc_intersect13` | `aacdr_gdsc_intersect` |
| Target-only | `tcga_only3` | `aacdr_tcga_only` |
| Auxiliary | `TCGA_drug_response_from_DAPL` | —（未評估） |

| 評估集 | 資料路徑 |
|--------|----------|
| `gdsc_intersect13` | `PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` |
| `tcga_only3` | `PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` |
| `TCGA_drug_response_from_DAPL` | `TCGA_drug_response_from_DAPL.csv` |
| `aacdr_gdsc_intersect` | `TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv` |
| `aacdr_tcga_only` | `TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv` |

---

## Round 24 現行超越標準（無 10% test set / `stest0`）

**協議：** 訓練不含另留 10% internal test（與 Round 24 NoHoldout 資料用量對齊）。  
**實驗標籤：** `eval3_stest0` / `target_infer_stest0`。  
**判定：** fold-mean DrugMacro **嚴格大於**下表 mean。

| 實驗 | 評估集 | DrugMacro AUROC | DrugMacro AUPRC |
|------|--------|----------------:|----------------:|
| eval3_stest0 | `gdsc_intersect13` | **0.5197 ± 0.0269** | 0.5981 ± 0.0222 |
| eval3_stest0 | `tcga_only3` | **0.5536 ± 0.0449** | 0.6960 ± 0.0286 |
| eval3_stest0 | `TCGA_drug_response_from_DAPL` | **0.5304 ± 0.0061** | 0.5570 ± 0.0117 |
| target_infer_stest0 | `aacdr_gdsc_intersect` | **0.5279 ± 0.0312** | 0.5710 ± 0.0122 |
| target_infer_stest0 | `aacdr_tcga_only` | **0.4804 ± 0.0414** | 0.6300 ± 0.0419 |

**Round 24 硬閘 PASS（僅兩組）：**  
`aacdr_gdsc_intersect` > **0.5279** ∧ `aacdr_tcga_only` > **0.4804**。  
其餘三組必報、不擋 lock。候選 vs 標準：[`reports/round24/vs_aacdr_standard.md`](../reports/round24/vs_aacdr_standard.md)。

---

## 歷史對照（含 holdout / 原 eval3；已非 Round 24 硬閘）

| 實驗 | 評估集 | 標註來源 | 藥物數 | Pair 數 | DrugMacro AUROC | DrugMacro AUPRC |
|------|--------|----------|--------|---------|-----------------|-----------------|
| eval3 | `gdsc_intersect13` | DAPL | 12 | 906 | 0.5184 ± 0.0437 | 0.6011 ± 0.0252 |
| eval3 | `tcga_only3` | DAPL | 3 | 129 | 0.5586 ± 0.0442 | 0.7130 ± 0.0272 |
| eval3 | `TCGA_drug_response_from_DAPL` | DAPL | 5 | 178 | 0.5356 ± 0.0570 | 0.5591 ± 0.0318 |
| target_infer | `aacdr_gdsc_intersect` | AACDR | 11 | 425 | 0.5582 ± 0.0618 | 0.6017 ± 0.0487 |
| target_infer | `aacdr_tcga_only` | AACDR | 8 | 97 | 0.4394 ± 0.0372 | 0.5942 ± 0.0206 |

## 備註

- **DrugMacro** = `macro_auroc` / `macro_auprc`（各藥物指標的 macro 平均）。
- `stest0` 數值由使用者確認（無 10% testset）；四捨五入至小數點後 4 位。
- 歷史表數值來自 `target_*_metrics_summary_fold_mean_std.csv`。
- 詳細 fold 明細見 `doc/drug_macro_auroc_auprc_gdsc_intersect_tcga_only.md`。
- `configs/round24/eval3.yaml` 的 `gate_*` 已同步為 **stest0** 標準。
