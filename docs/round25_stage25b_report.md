# Round 25 — Stage 25B / 25C Report

**容器：** `DAPL` · `/workspace/DAPL`  
**Downstream：** 固定 fresh no-pooling BioCDA-XA（不搜尋拓撲）  
**選模：** 未使用 TCGA

## Stage 25B 配對結果

| Arm | Stage2 特徵 | mean DrugMacro AUC | mean AUPRC |
|-----|-------------|--------------------|------------|
| B0 | S0（基準 dual WGAN + always-on proto） | **0.6303** | 0.4151 |
| B1 | S1（AADA，25A 晉升） | 0.6241 | 0.4258 |
| B2 | S1 的 Z64-only（C32=0 pad） | 0.6327 | — |

**決策：`KEEP_S0`（不晉升 Stage2）**

| Gate | 結果 |
|------|------|
| mean AUC Δ (B1−B0) | −0.0063 &lt; 0 |
| noninferior seeds | 2/3 &lt; 3 |
| worst-seed Δ | −0.0599 ≤ −0.010 |

### 設計邏輯

1. **固定 XA：** 三臂共用 splits / seeds / budget / fresh GIN→XA；只換 cell 特徵來源。  
2. **B0 vs B1：** 檢驗 25A 晉升的 S1 是否真能改善下游；失敗則 lock 回 S0。  
3. **B2：** 同 S1 checkpoint 的 Z64，C32 置零，供 25C 判斷 context 是否具預測貢獻。  
4. **預算：** smoke 平行 3 workers（對齊 NoHoldout screen 節奏）；殘留 formal orphan 已終止，避免覆寫完成產物。

## Stage 25C（C32 ablation）

| 項目 | 結果 |
|------|------|
| B1−B2 AUC Δ | −0.0086 |
| predictive effect | weak |
| attention effect | weak |
| **final claim** | `do_not_emphasize_C32` |

B2 ≥ B1：在本輪固定 XA 下，C32 未帶來可重現的預測增益，不應強調「生物 context 導引」。

## 產物

- `reports/round25_stage25b_paired_performance.csv`
- `reports/round25_selection_decision.json`
- `reports/round25_c32_xa_effect.json`
- `outputs/round25_stage25b/{B0,B1,B2}/`
