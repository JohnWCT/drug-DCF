# Round 20 Final Report — Unseen-Drug Closure

## Status

**COMPLETE / ALL_DONE** — scenario-focused unseen-drug closure finished.

### Acceptance checklist

| Stage | Criterion | Result |
|-------|-----------|--------|
| 20-0 | GO + resolved E3 + comparable C16/C32 | PASS |
| 20A | 30/30 jobs + dimension lock | **C32** LOCKED |
| 20B | gated vs E3 + guardrails | gated fail → keep E3 |
| 20C | immutable model lock (no TCGA) | **C32 + B_E3** |
| 20D | TCGA strict ensemble (5 targets) | COMPLETE |
| 20E | release audit | **PASS** |
| Tests | `tests/round20` + prior Round 20 unit tests | 22 passed |

## Locked configuration

| Component | Choice | Reason |
|-----------|--------|--------|
| Context | **C32** (Z64+context32 = 96-d) | Stage 20A stable improvement (ΔAUC +0.0075; 3/3 seeds non-worse) |
| Drug encoder | D0 GIN32 | Fixed from Round 19 E3 contract |
| Predictor | **pooled E3** (`AdapterMLPFusion+ResponseHead`) | Stage 20B gated failed G1/G2 (ΔAUC −0.002) |
| Checkpoint policy | 15-fold probability-mean ensemble | Inherit Round 18/19 TCGA protocol |

## Stage outcomes

1. **20-0 GO** — E3 uniquely resolved; C16/C32 comparable after rebuild.
2. **20A** — 30/30 jobs; lock **C32**.
3. **20B** — 15 gated + 15 reused E3; gated fails guardrails.
4. **20C** — immutable lock `C32 + B_E3` (`gated_failed_guardrails`); no TCGA in selection.
5. **20D** — strict TCGA inference on five existing targets; ensemble complete.
6. **20E** — release archive + `ROUND20_RELEASE_AUDIT=PASS`.

## Official conclusion

Increasing the prototype-context projection from 16 to 32 dimensions produced a
stable repeated drug-held-out improvement, while gated fusion did not provide
sufficient additional benefit.

The final model therefore used **C32 O2**, **D0 GIN32**, and **pooled E3**.

## Limitations

- Not validated for unseen cancer type.
- Omics encoder remained frozen; raw-omics path is capability-only.
- TCGA was not used for model selection.

## Artifacts

```text
result/optimization_runs/round20_unseen_drug_closure/
  stage20_0/
  stage20a_dimension/stage20a_dimension_decision.json
  stage20b_predictor/stage20b_guardrail_report.json
  stage20c_lock/final_model_lock.json
  stage20d_tcga/
  stage20e_release/   # RELEASE_MANIFEST.json, MODEL_CARD.md, hashes/
```
