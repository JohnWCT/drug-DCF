# drug-DCF Round 9 Deconfounding QC 操作手冊

## 0. 核心定位

Round 9 = **Deconfounding Quality Control**（不是新方法）。

檢查既有 pretrain domain adaptation 是否：

- 僅 global source/target 對齊，或
- 在每個 **cancer type** 內 source/target 也對齊，
- 同時保留 cancer biology 與 response-relevant heterogeneity。

**不新增** Conditional ADV / Prototype / SupCon 等 training loss。

## 1. 新增檔案

| 類別 | 路徑 |
|------|------|
| Config | `config/round9_baselines.json` |
| Config | `config/pretrain_sweeps/vaewc_round9_reproduction.json` |
| Config | `config/pretrain_sweeps/vaewc_round10_cond_adv_template.json`（template only） |
| Tools | `tools/round9_baseline_resolver.py` |
| Tools | `tools/build_round9_reproduction_manifest.py` |
| Tools | `tools/analyze_deconfounding_qc.py` |
| Tools | `tools/analyze_conditional_domain_leakage.py` |
| Tools | `tools/analyze_cancer_prototypes.py` |
| Tools | `tools/analyze_latent_stability.py` |
| Tools | `tools/analyze_round9_diagnostics.py` |
| Tools | `tools/build_round9_finetune_select.py` |
| Tools | `tools/run_round9_diagnostics_pipeline.sh` |
| Shared | `tools/round9_diagnostics_common.py` |

## 2. Baselines

| exp_id | role | required |
|--------|------|----------|
| exp_048 | primary_best_vicreg | **是** |
| exp_021 | secondary_vicreg | 否 |
| exp_188 | round8_best_control_wide_encoder | 否 |
| exp_010 | round6_control_like_winner | 否 |
| exp_012 | round6_active_vicreg_integrated_reference | 否 |
| exp_746 | historical_baseline | 否 |

```bash
python tools/round9_baseline_resolver.py \
  --baseline-config config/round9_baselines.json \
  --search-root result \
  --outdir result/optimization_runs/round9_diagnostics/baselines
```

`exp_048` 找不到 → exit code 2。

## 3. 3-seed Reproduction

```bash
python tools/build_round9_reproduction_manifest.py \
  --resolved-baselines result/optimization_runs/round9_diagnostics/baselines/resolved_baselines.csv \
  --baseline-config config/round9_baselines.json \
  --outdir result/optimization_runs/round9_reproduction \
  --force
```

- Seeds：**101 / 202 / 303**
- Hyperparameters 從 `params.json` 還原
- 預期：**6 baselines × 3 = 18** pretrain jobs

## 4. 一鍵 Pipeline

```bash
bash tools/run_round9_diagnostics_pipeline.sh
# OOM 調整
PRETRAIN_PARALLEL=10 bash tools/run_round9_diagnostics_pipeline.sh
FINETUNE_PARALLEL=20 bash tools/run_round9_diagnostics_pipeline.sh
```

流程：resolve baselines → reproduction manifest → pretrain → QC / leakage / prototype / stability → finetune select → finetune → aggregate → final report。

## 5. 診斷輸出

| 報告 | 路徑 |
|------|------|
| Deconfounding QC | `round9_diagnostics/reports/deconfounding_qc_*` |
| Conditional leakage | `round9_diagnostics/reports/conditional_domain_*` |
| Prototypes | `round9_diagnostics/reports/prototype_*` |
| Latent stability | `round9_diagnostics/reports/latent_stability_*` |
| Final | `round9_diagnostics/final_report/round9_final_report.md` |

### QC 狀態

- `good_conditional_deconfounding`
- `global_only_alignment`
- `biology_collapse_risk`
- `insufficient_evidence`

## 6. 測試

```bash
python -m compileall .

pytest tests/test_round9_baseline_resolver.py \
  tests/test_round9_reproduction_config.py \
  tests/test_deconfounding_qc.py \
  tests/test_conditional_domain_leakage.py \
  tests/test_cancer_prototype_diagnostics.py \
  tests/test_latent_stability_diagnostics.py \
  tests/test_build_round9_finetune_select.py -q
```

## 7. 成功標準

**必須：** exp_048 解析成功、diagnostics CSV 輸出、mini finetune aggregate 完成、`round9_final_report.md` 產出。

**理想：** 6 baselines × 3 seeds 全完成；Spearman correlation 可計算；Round 10 優先癌別清單可產出。

## 8. 執行結果摘要（2026-06-22 完成）

| 階段 | 結果 |
|------|------|
| Baseline 解析 | **6/6** |
| Pretrain reproduction | **18/18** |
| Finetune | **72/72** |
| GPU | pretrain **33** / finetune **26** parallel |
| 執行時間 | ~**2.9 h** |
| QC | `global_only_alignment` **8**、`insufficient_evidence` **10** |
| exp_048 最佳 reproduction | **0.5671** Avg TCGA（seed 303） |
| vs R7 exp_048（0.5918） | **未復現** |
| Round 10 高優先癌別 | Brain、Esophageal、Liver、Lung、Ovarian |

**備註：** 初跑 final report 因 prototype 欄位與 TCGA patient-key 對應問題失敗；已修復 `round9_diagnostics_common.py` / `analyze_round9_diagnostics.py` 後重跑 diagnostics。

## 9. Round 10

- `vaewc_round10_cond_adv_template.json` 僅 template，**Round 9 不執行**
- Go 條件見 `docs/pipeline_summary.md` §17.10

詳見 `docs/pipeline_summary.md` §17。
