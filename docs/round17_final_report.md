# Round 17 — Direct Prototype Features

**狀態：** ALL_DONE（17A–17C、17F）；18-class 重跑改走 Round 17R

見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md#round-17--direct-prototype-features)。

## 基準

| 基準 | Avg TCGA |
|------|----------|
| R13 best | **0.6112** |
| R15 best | 0.6083 |

## 17C 10-seed 最佳

| Model | feature_mode | Avg TCGA mean ± std |
|-------|--------------|---------------------|
| r13_exp_008 | own_proto_context_projected_16 | **0.5892 ± 0.034** |
| r15c_exp_005 | own_plus_summary | 0.5868 ± 0.030 |

vs R13：**−0.022**

## 5-target drug-macro mean（17A 全空間 Top-1）

| Model | feature_mode | mean_5target |
|-------|--------------|--------------|
| r13_exp_008_control | own_plus_summary | **0.5782** |

## 結論

- direct prototype **未全面超越** `own_plus_summary`。
- 最佳 projected：`context_projected_16`、`delta_projected_8`。
- **未重現 R13**（gap ≈ 0.022）。
- 18-class 修正後結果見 [`round17r_18class_final_report.md`](round17r_18class_final_report.md)。
