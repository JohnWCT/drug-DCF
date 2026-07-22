# Round 18 — Architecture Screening

**狀態：** 18A–18E DONE；18F（可解釋性）未完成

見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md#round-18--architecture-screening)。

## Formal 5CV DrugMacro AUC

| 排名 | 架構 | mean AUC |
|------|------|----------|
| 1 | X3 pure × context16 | **0.6181** |
| 2 | X3 pooled_residual × context16 | 0.6176 |
| 3 | P1 compact64 × context16 | 0.6169 |
| 4 | P3 deeper128 × context16 | 0.6105 |
| 5 | MLP × own_plus_summary | 0.6078 |

Screening 峰值：X3 × pooled_residual × context16 **0.6230**。

## Internal held-out（ensemble）

| 架構 | DrugMacro AUC |
|------|---------------|
| P3 × context16 | **0.6131** |
| X3 pure × context16 | 0.6056 |
| MLP × own_plus_summary | 0.5358 |

X3 pure vs MLP：+0.070（bootstrap P(Δ>0)≈0.9995）。

## TCGA external

- X3 pure vs MLP：**2/5** non-worse → `cross_attention_external_success = false`
- Integrated5 最高：**MLP 0.5288**（X3 pure 0.4748）

## 結論

1. Cross-attention + **context16** 在 CV / internal 優於 MLP；**context16 效應（~+0.018）遠大於 residual（~+0.002）**。
2. none omics 下 cross-attn 優勢消失（~−0.015～−0.020 vs context16）。
3. Formal 中 pure ≈ residual；GIN shortcut 非主要增益來源。
4. TCGA 外推**未通過**預定門檻；formal/internal 增益未能穩定外推。
5. 18F attention export / masking 待完成。
