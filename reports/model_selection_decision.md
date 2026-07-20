# BioCDA Round 21 Model Selection Decision

- Status: **TRAINING_IN_PROGRESS**
- Selected model: pending (awaiting GDSC repeated unseen-drug runs)
- Protocol: repeated unseen-drug on GDSC development rows (seeds 17, 29, 43)
- Primary metric: drug_macro_auc
- TCGA not used for selection

## Pipeline status

| Step | Status |
|------|--------|
| Repository audit | PASS |
| Runtime architecture audit | PASS |
| Unit tests (37) | PASS |
| Smoke training | PASS |
| GDSC train M0/M1/M2 × 3 seeds | **IN PROGRESS** (see `outputs/xa_validation/full_run.log`) |
| Attention diagnostics | pending trained M2 checkpoints |
| Paired comparison | pending |
| Final lock | pending |

## Interim gate preview (untrained / synthetic smoke)

Previous smoke evaluation on uninitialized weights:

- functional_correctness: PASS
- performance_guardrails: PASS (placeholder until real metrics)
- context_utilization: PASS
- attention_health: FAIL (expected before training)

## Next action

When Docker training completes:

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml diagnose-attention'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml compare'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml lock'
```

If all gates pass → `reports/biocda_final_model_lock.json` status becomes **LOCKED**.
