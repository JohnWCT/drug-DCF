# Round 20 Stage 20B — Pooled E3 vs Gated Fusion (C32)

## Status

**COMPLETE** — gated failed guardrails; baseline retained.

| Metric | B_E3 (C32) | B_GATED | Δ (gated−E3) |
|--------|------------|---------|--------------|
| mean DrugMacro AUC | baseline | −0.0020 | fails G1 |
| mean DrugMacro AUPRC | — | −0.0049 | passes G3 (≥ −0.01) |
| seed Δ AUC 52/62/72 | — | −0.0023 / −0.0014 / −0.0023 | 0/3 non-worse |

Guardrails: G1 fail, G2 fail, G3 pass, G4 pass, G5 pass → `all_pass=false`.

## Jobs

- 15 E3 jobs reused from Stage 20A `A_C32_E3`
- 15 gated jobs trained fresh under identical splits / D0 / budget
- 15/15 gated COMPLETE, 0 failed

## Decision implication (Stage 20C)

Retain **B_E3** (`gated_failed_guardrails`) on locked **C32**.
