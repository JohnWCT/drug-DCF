# Round 20 Stage 20D — Locked TCGA Inference

## Status

**COMPLETE** — post-selection evaluation only (no re-selection).

| Field | Value |
|-------|-------|
| Context | C32 (96-d O2) |
| Model | B_E3 (pooled AdapterMLPFusion + ResponseHead) |
| Ensemble | 15-fold probability mean |
| Checkpoint load | `strict=True` |
| Preflight | PASS |

## Aggregate metrics (DrugMacro AUC)

| Target | DrugMacro AUC | Global AUC |
|--------|---------------|------------|
| gdsc_intersect13 | 0.471 | 0.551 |
| tcga_only3 | 0.459 | 0.383 |
| dapl | 0.428 | 0.463 |
| aacdr_tcga_only | 0.539 | 0.418 |
| aacdr_gdsc_intersect | 0.617 | 0.602 |

Full per-drug tables: `result/.../stage20d_tcga/tcga_metrics.json`.

TCGA results are locked post-selection evaluation and must not be used to revisit C16/C32 or E3/gated.
