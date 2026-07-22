# Round 15 — Reproducibility + exp_008 Route Rescue

**狀態：** ALL_DONE

## 分數

| 分支 | 最佳 | Avg TCGA |
|------|------|----------|
| 15A 5-seed exp_008 repro | ops mean | 0.5746 ± 0.009 |
| 15B forced exp_008 route | own_plus_summary | 0.5889 |
| **15C ultra-low VICReg** | r15c_exp_005_own_plus_summary | **0.6083** |
| R13 best | — | 0.6112 |

vs R13：**−0.0029** · vs R14：**+0.0174**

## 結論

未重現 R13 5-seed 水準；15C 最接近 R13。Proto feature path 修復後 own_plus_summary 多數優於 z-only（mean Δ +0.0142）。**→ NO-GO Round 16 bruteforce。**
