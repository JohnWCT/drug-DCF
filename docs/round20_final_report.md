# Round 20 — Unseen-Drug Closure（BioCDA-Predictive）

**狀態：** COMPLETE · LOCKED_RELEASE

完整彙總見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md#round-20--unseen-drug-closurebiocda-predictive)。

## 分數

### C16 vs C32（Stage 20A）

| | C16 | C32 | Δ |
|--|-----|-----|---|
| mean DrugMacro AUC | 0.7434 | **0.7509** | **+0.0074** |

### Pooled E3 vs Gated（Stage 20B，C32 上）

| | B_E3 | B_GATED | Δ |
|--|------|---------|---|
| mean DrugMacro AUC | baseline | −0.0020 | gated 未過 guardrails |

### TCGA post-lock（Stage 20D）

| Target | DrugMacro AUC | Global AUC |
|--------|---------------|------------|
| aacdr_gdsc_intersect | **0.6173** | 0.6020 |
| aacdr_tcga_only | 0.5391 | 0.4182 |
| gdsc_intersect13 | 0.4714 | 0.5506 |
| tcga_only3 | 0.4591 | 0.3826 |
| dapl | 0.4284 | 0.4632 |

## 鎖定結果

- **Context：** C32（96-d O2）
- **Model：** B_E3 / AdapterMLPFusion + ResponseHead
- **原因：** C32 stable_improvement；gated_failed_guardrails

## 結論

C32 在固定 E3 下穩定提升 repeated drug-held-out AUC；gated fusion 未過預定門檻，保留 parsimonious pooled E3。此組合即 **BioCDA-Predictive**。TCGA 僅作選模後描述性評估。
