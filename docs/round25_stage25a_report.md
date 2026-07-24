# Round 25 — Stage 25A Report

**容器：** `DAPL` · `/workspace/DAPL`  
**標準：** Stage2 alignment screen（S0/S2/S1，條件 S3/S2b）  
**Downstream XA：** 固定，不搜尋

## 決策

| 項目 | 結果 |
|------|------|
| Status | `PROMOTE_S1` |
| Selected | **S1**（AADA AE 取代 global WGAN + always-on prototype） |
| S2 | FAIL（`prototype_hinge_active_fraction=0` → margin 過寬，loss 失效） |
| S3 | 已執行（因 S1 PASS）；同樣 hinge=0 → 不晉升 |
| S2b | 未觸發（無過度重疊證據） |

## 設計摘要

1. **共用 AE：** 先訓練一次 shared AE，所有 variant 以 `skip_ae_train` + `ae_init_dir` 載入，保證 encoder 初值一致。  
2. **S0：** dual WGAN + always-on prototype（現行基準）。  
3. **S2：** 僅把 prototype 改成 margin-gated；`prototype_upper_margin` 由 source radius P90 估計後 freeze。  
4. **S1：** `global_adv_mode=conditional_replacement`，加入 Latent AE + zero-init target residual adapter。  
5. **選模：** 不用 TCGA；要求 source/target geometry 與 C32 readiness 相對 S0 不崩壞。

## 產物

- `reports/round25_stage25a_metrics.csv`（12 jobs = S0/S1/S2/S3 × 3 seeds）
- `reports/round25_stage25a_geometry.json`
- `reports/round25_stage25a_context_readiness.json`
- `reports/round25_stage25a_decision.json`
- checkpoints：`result/optimization_runs/round25_stage25a/`
