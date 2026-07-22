# Round 11 — Stabilized Conditional ADV + SmoothL1

**狀態：** DONE

## 分數

| 參考 | Avg TCGA |
|------|----------|
| **R11 最佳 exp_035** | **0.5828** |
| R10 exp_111 | 0.5749 |
| R7 exp_048 | 0.5918 |

Round 11A QC：exp_111 leakage 0.400 vs exp_048 0.409（改善）。

## 結論

10C 穩定化 + conditional ADV 超越 Round 10。SmoothL1 改善 latent stability；最佳下游來自 stabilized 10C（MSE recon），非 pure SmoothL1。**→ 進入 Round 12 prototype alignment。**
