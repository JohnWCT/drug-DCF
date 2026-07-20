# Round 20 Final Report — Unseen-Drug Closure

## Status

**COMPLETE** — completion audit: `PASS`.

## Executive summary

Round 20 compared prototype-context dimension (C16 vs C32) and predictor architecture
(pooled E3 vs gated fusion) under repeated drug-held-out validation. The locked model
uses **C32** (96-d O2), **D0 GIN32**, and **AdapterMLPFusion+ResponseHead** (`B_E3`).

## Stage 20A — Context dimension

| Metric | C16 | C32 | Δ (C32−C16) |
|--------|-----|-----|-------------|
| mean DrugMacro AUC | 0.7434 | 0.7509 | 0.0074 |

**Locked context:** C32 (`stable_improvement`)

## Stage 20B — Predictor

| Guardrail | Pass |
|-----------|------|
| G1 mean AUC | False |
| G2 seed majority | False |
| G3 AUPRC | True |
| G4 no major fail | True |
| G5 complete | True |

Mean AUC Δ (gated−E3): **-0.0020** — `all_pass=False`

## Stage 20C — Model lock

- Context: C32 (96-d)
- Model: B_E3 / AdapterMLPFusion+ResponseHead
- Reason: gated_failed_guardrails
- Forbidden metrics used: False

## Stage 20D — TCGA (post-lock only)

| Target | DrugMacro AUC | Global AUC |
|--------|---------------|------------|
| aacdr_gdsc_intersect | 0.6173 | 0.6020 |
| aacdr_tcga_only | 0.5391 | 0.4182 |
| dapl | 0.4284 | 0.4632 |
| gdsc_intersect13 | 0.4714 | 0.5506 |
| tcga_only3 | 0.4591 | 0.3826 |

## Stage 20E — Release

- Release status: LOCKED
- Artifacts hashed: 53

## Final architecture

```text
Raw omics [G] → frozen encoder → Z [64]
Raw prototype context → PCA (n=32) → context [32]
Z + context → O2 [96]
SMILES → D0 GIN → graph embedding [32]
O2 + graph → AdapterMLPFusion+ResponseHead → probability [1]
```

## Limitations

- Unseen cancer-type optimization: out of scope.
- Encoder unfreezing: not formally evaluated in Round 20.
- TCGA results must not be used to revisit model selection.

## Official conclusion

Increasing prototype-context dimension from 16 to 32 produced stable repeated
drug-held-out improvement under the locked E3 contract. Gated fusion did not pass
predefined guardrails; the parsimonious pooled E3 predictor was retained on **C32**.
