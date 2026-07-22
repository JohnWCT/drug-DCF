# Round 21 — Cross-Attention Validation v1（BioCDA-XA-Candidate）

**狀態：** COMPLETE · **REJECTED**

完整彙總見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md#round-21--cross-attention-v1biocda-xa-candidate)。

## 分數（development unseen-drug，seeds 17/29/43）

| 模型 | mean DrugMacro AUC | Δ vs M0 |
|------|-------------------|---------|
| M0 pooled_baseline | **0.746** | — |
| M1 biocda_xa_z | 0.714 | −0.032 |
| M2 biocda_xa_zc | 0.709 | **−0.037** |

Guardrail：mean ΔAUC(M2−M0) ≥ −0.005 → **未達**。

## 結論

- **根因：** performance_failure（attention health / context utilization 正常）。
- **保留：** M0 / BioCDA-Predictive 作為唯一正式預測模型。
- **C32：** 在 Predictive 上已證實優於 C16（Round 20）；在 XA 上 context 改變 attention 但未改善預測。
- **TCGA / 可解釋性：** 未執行；須待 XA 性能追平後再開。
