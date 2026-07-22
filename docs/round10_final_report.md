# Round 10 — Conditional Adversarial Deconfounding

**狀態：** DONE · `no_conditional_improvement`

## 分數

| 參考 | Avg TCGA |
|------|----------|
| **R10 最佳 exp_111**（10C weak global） | **0.5749** |
| R9 exp_048 reproduction 最佳 | 0.5671 |
| R7 原始 exp_048 | 0.5918 |

## 結論

Conditional ADV 訓練完成，下游略優 R9 reproduction（+0.0078），但仍低於 R7 原始 0.5918。最佳為 10C（weak global guard），非純 10B replacement。Conditional leakage 改善未經 R9 式 diagnostics 重跑確認。
