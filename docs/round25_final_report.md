# Round 25 — Stage2 Margin / AADA → No-Pooling XA Closure

**狀態：** `LOCKED_KEEP_S0`  
**容器：** `DAPL` · `/workspace/DAPL`  
**架構：** biocda-xa-v2（fresh no-pooling；拓撲不搜尋）

## 結論

Stage25A 幾何／對齊 screen 晉升 **S1（AADA）**，但 Stage25B 固定下游 XA 配對後 **B1 未勝過 B0**，因此 **不晉升 Stage2**，lock 維持 **S0**。Stage25C 顯示 C32 預測貢獻弱（`do_not_emphasize_C32`）。

| Stage | 決策 |
|-------|------|
| 25A | `PROMOTE_S1`（S2/S3 hinge inactive；S2b 未觸發） |
| 25B | `KEEP_S0`（mean AUC Δ≈−0.006；noninf 2/3；worst Δ≈−0.060） |
| 25C | `do_not_emphasize_C32`（B1−B2 AUC Δ≈−0.009） |
| Lock | `LOCKED_KEEP_S0` · `promoted_stage2_variant=S0` |

## 硬性約束（均已遵守）

- Round23 GDSC XA lock 維持 `REJECTED`（未覆寫 `reports/biocda_xa_model_lock.json`）
- Downstream XA 固定 fresh no-pooling；未搜尋拓撲
- TCGA 未參與選模
- `reconstruction_margin` / `prototype_upper_margin` / `prototype_lower_margin` 欄位分離
- Telegram 僅整輪 closure 完成後通知一次

## 設計思路摘要

1. **25A：** 共用 AE 一次 → S0/S2/S1（條件 S3）平行 screen；只搜尋 Stage2 alignment，不碰 XA 拓撲。
2. **25B：** 以晉升 Stage2 重產特徵，對 B0(S0) / B1(S1) / B2(Z-only) 跑固定 fresh XA；晉升門檻看配對 AUC／AUPRC／seed 穩定性。
3. **25C：** B1 vs B2 量化 C32；弱效應則不強調 context-guided 敘事。
4. **Lock：** 寫入 `reports/biocda_xa_stage2_lock.json`（獨立於 Round23 XA lock）。

## 關鍵數字（validation DrugMacro）

| Arm | mean AUC | mean AUPRC |
|-----|----------|------------|
| B0 (S0) | 0.6303 | 0.4151 |
| B1 (S1) | 0.6241 | 0.4258 |
| B2 (Z-only) | 0.6327 | — |

## 產物

```text
reports/round25_stage25a_decision.json
reports/round25_selection_decision.json
reports/round25_c32_xa_effect.json
reports/biocda_xa_stage2_lock.json
docs/round25_stage25a_report.md
docs/round25_stage25b_report.md
docs/round25_final_report.md
```
