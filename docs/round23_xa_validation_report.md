# BioCDA Round 23 — No-Pooling XA Performance Closure

Round code name: **Round 23 — No-Pooling XA Performance Closure and Architecture Lock**

Canonical names:

| Role | Name | Status |
|------|------|--------|
| Predictive reference | **BioCDA-Predictive** (pooled E3) | `LOCKED_REFERENCE` |
| XA student | **BioCDA-XA-Candidate** (`biocda-xa-v2`) | **`REJECTED`** |

Baselines (immutable science): `reports/round20_round21_architecture_diff.json`, `docs/round20_round21_scientific_audit.md`.

## Final outcome

| Item | Result |
|------|--------|
| Pipeline status | **COMPLETE** |
| Model lock status | **REJECTED** (not LOCKED) |
| Root cause | **performance_failure only** |
| Retained predictive model | **BioCDA-Predictive** |
| Best XA by mean DrugMacro AUC | `biocda_xa_fresh` (still fails seed non-worse gate) |
| TCGA used for selection | **false** |
| XA attention may explain Predictive? | **No** |

Mean validation DrugMacro AUC (seeds 17/29/43):

| Model | Mean AUC | ΔAUC vs P0 | Mean AUPRC | ΔAUPRC vs P0 |
|-------|----------|------------|------------|--------------|
| P0 BioCDA-Predictive | **0.744** | — | **0.512** | — |
| X0 fresh XA | 0.740 | **−0.0043** | 0.506 | −0.0059 |
| X2 transfer+KD | 0.720 | −0.0247 | 0.490 | −0.0214 |
| X1 transfer | 0.699 | −0.0455 | 0.477 | −0.0342 |

Gate details for closest candidate (`biocda_xa_fresh`):

- mean ΔAUC −0.0043 ≥ −0.005 → pass mean AUROC
- mean ΔAUPRC −0.0059 ≥ −0.010 → pass mean AUPRC
- non-worse seeds (≥ −0.005): **1/3** (need 2/3) → **fail**
- floor (no seed < −0.020): pass

Transfer and KD fail mean and floor rules. Therefore XA stays **REJECTED**.

Artifacts: `reports/round23_paired_performance.csv`, `reports/round23_selection_decision.json`, `reports/biocda_xa_model_lock.json`.

## Scope completed

| Stage | Status |
|-------|--------|
| Architecture plan + `biocda_xa_v2_architecture_spec.json` | DONE |
| No-pooling XA modules (`biocda/models/xa/*`) | DONE |
| Predictive reference loader (`biocda/models/predictive/*`) | DONE |
| E3 GIN transfer + KD + freeze schedule | DONE |
| Strict no-pooling audit | PASS |
| Contract tests (`test_biocda_xa_v2_contracts.py`) | 15 passed |
| Paired GDSC training 12 runs (P0/X0/X1/X2 × 3 seeds) | DONE |
| Attention health / query–drug utilization | PASS |
| Selection gate + lock manifest | DONE (`REJECTED`) |
| X3 Z64-only C32 ablation | Deferred (performance already rejected) |

**Out of scope:** TCGA inference, atom masking faithfulness tables, 2D molecule rendering. Do not use rejected XA attention to explain BioCDA-Predictive predictions.

## Architecture (BioCDA-XA v2)

```text
Z64 + C32 → LN → concat 96-d → Linear+LN → Q0 [B,1,128]
GIN 5×32 jk=last BN dropout0.1 → atom nodes only (no pooling)
atom tokens 32→128 → 2-layer cross-attn (d=128, H=4, FFN=256)
response head(Qfinal[:,0,:]) → logit
```

Forbidden on XA path: graph pooling, pooled/raw concat into head, teacher in student checkpoint, absolute atom position / modality embeddings.

## Training matrix

| ID | Model | GIN | C32 | KD |
|----|-------|-----|-----|----|
| P0 | BioCDA-Predictive | warm-start Round20 + train on same split | ✓ | — |
| X0 | XA fresh | random | ✓ | ✗ |
| X1 | XA transfer | Round20 E3 GIN | ✓ | ✗ |
| X2 | XA KD | Round20 E3 GIN | ✓ | ✓ |

P0 is **retrained** on seeds `{17,29,43}` (paired). Round20 `seed52_fold0.pt` is used only for GIN transfer / frozen KD teacher.

Phases for XA: attention warm-up (GIN frozen + BN eval) → last GIN block FT → joint stabilize.

## Diagnostics

- No-pooling audit: PASS (`reports/round23_no_pooling_architecture_audit.json`)
- E3 GIN transfer: 50 `convs.*`/`bns.*` keys loaded; `fc1_xd`/`out` ignored
- Attention health (trained fresh seed17): mean normalized entropy ≈ 0.84
- Query/drug utilization: PASS

## Decision policy applied

```text
BioCDA-Predictive = LOCKED_REFERENCE
BioCDA-XA-Candidate = REJECTED (performance_failure)
```

Closest recovery signal: fresh no-pooling XA nearly matches P0 on mean ΔAUC, but fails the 2/3-seed non-worse rule. Transfer/KD did **not** close the gap in this schedule.

## Next steps (only if revisiting XA)

1. Keep Predictive as the only formal prediction model.
2. If another recovery round is opened, start from **fresh XA** (best mean), not transfer/KD as currently configured.
3. Run X3 C32 ablation only after a candidate passes the paired performance gate.
4. Do not open TCGA atom-level interpretability on rejected XA.

## Reproduce

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/audit_xa_no_pooling.py --strict --transfer-smoke'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 -m pytest test_biocda_xa_v2_contracts.py -q'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/train_xa_performance_closure.py --config configs/biocda/xa_v2_closure.yaml'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/evaluate_xa_candidates.py && python3 scripts/diagnose_xa_utilization.py --checkpoint outputs/xa_v2_closure/biocda_xa_fresh_seed17/best.pt && python3 scripts/lock_biocda_xa.py'
```
