# Round 23 — No-Pooling XA Performance Closure and Architecture Lock

Canonical names (immutable):

| Role | Name | Status |
|------|------|--------|
| Round 20 pooled E3 | **BioCDA-Predictive** | `LOCKED_REFERENCE` |
| This round XA student | **BioCDA-XA-Candidate** (`biocda-xa-v2`) | `CANDIDATE` until gates pass |

Baselines (do not rewrite science):

- `reports/round20_round21_architecture_diff.json`
- `docs/round20_round21_scientific_audit.md`

## Goal

Close the Round 21 performance gap with a **strict no-pooling** sample-to-atom cross-attention student:

```text
Z64 + C32 → sample query Q0 [B,1,128]
GIN → atom nodes H [N,32] → atom tokens [B,N,128]
2-layer cross-attention → Qfinal [B,1,128]
response head(Qfinal[:,0,:]) → logits [B]
```

No graph pooling, no pooled concat into the head, no teacher in the student checkpoint.

## Candidates (paired matrix)

| ID | Model | GIN | C32 | KD |
|----|-------|-----|-----|----|
| P0 | BioCDA-Predictive | Round20 | ✓ | — |
| X0 | XA-A fresh | random | ✓ | ✗ |
| X1 | XA-B transfer | E3 GIN | ✓ | ✗ |
| X2 | XA-C transfer+KD | E3 GIN | ✓ | ✓ |
| X3 | XA C32 ablation | best strategy | ✗ (Z64 only) | same as best |

## Performance gate (vs P0)

- mean Δ DrugMacro AUROC ≥ −0.005
- mean Δ DrugMacro AUPRC ≥ −0.010
- ≥ 2/3 seeds non-worse
- no seed ΔAUC < −0.020
- plus attention health / query–drug sensitivity / C32 contract / reproduction

Pass → upgrade to **BioCDA-XA** (or **BioCDA-XA-KD** if only X2 passes).  
Fail → Predictive stays LOCKED; XA remains REJECTED; no XA attention explaining Predictive.

## P0 pairing rule

`biocda_predictive` is **retrained** on the same unseen-drug seeds `{17,29,43}` as XA
(architecture = Round20 pooled E3). The Round20 `seed52_fold0.pt` checkpoint is used
only for:

- GIN weight transfer into X1/X2
- frozen teacher logits for X2 KD

Do **not** score the Round20 checkpoint directly on Round21 splits (unpaired / leakage risk).

## Docker

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/audit_xa_no_pooling.py --strict --transfer-smoke'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 -m pytest test_biocda_xa_v2_contracts.py -q'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/train_xa_performance_closure.py --config configs/biocda/xa_v2_closure.yaml --smoke'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/train_xa_performance_closure.py --config configs/biocda/xa_v2_closure.yaml'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/evaluate_xa_candidates.py && python3 scripts/diagnose_xa_utilization.py && python3 scripts/lock_biocda_xa.py'
```

Machine-readable architecture: `reports/biocda_xa_v2_architecture_spec.json`.


## Round 23 closure status (executed)

| Item | Status |
|------|--------|
| Training matrix P0/X0/X1/X2 × seeds {17,29,43} | COMPLETE (12/12) |
| No-pooling audit + contract tests | PASS |
| Attention / utilization diagnostics | PASS |
| Paired performance gate | **FAIL** (all XA candidates) |
| Lock manifest | `reports/biocda_xa_model_lock.json` → **REJECTED** |
| Validation report | `docs/round23_xa_validation_report.md` |
| X3 C32 ablation | Deferred after performance rejection |

Closest candidate: `biocda_xa_fresh` (mean ΔAUC ≈ −0.0043) failed the ≥2/3-seed non-worse rule.
