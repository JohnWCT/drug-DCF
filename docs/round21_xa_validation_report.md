# BioCDA Round 21 — Cross-Attention Validation and Model Lock

Round code name: **Round 21 — Cross-Attention Validation and Model Lock**

Paper working title: **BioCDA: An Interpretable Biological-Context-Guided Framework for Cell-Line-to-Patient Drug Response Prediction**

## Scope completed in this round

| Stage | Status |
|-------|--------|
| Repository sync audit | `scripts/audit_repository_state.py` |
| Runtime architecture audit + forward trace | `scripts/audit_biocda_architecture.py` |
| Cross-attention contract fixes (raw/dropout, masked softmax, atom_ptr) | `biocda/models/` |
| M0 / M1 / M2 factory (`pooled_baseline`, `biocda_xa_z`, `biocda_xa_zc`) | `biocda/models/model_factory.py` |
| Unseen-drug split manifest | `reports/splits/unseen_drug_split_manifest.json` |
| GDSC repeated validation training CLI | `scripts/run_xa_validation.py` |
| Attention / context / query diagnostics | `biocda/diagnostics/` |
| Selection gates + model lock manifest | `biocda/validation/` |
| Unit tests (37+) | `tests/test_*biocda*` |
| CI workflow | `.github/workflows/biocda-ci.yml` |

**Out of scope (deferred to Round 22):** TCGA inference, attention heatmaps, RDKit 2D rendering, gene-level IG/DeepLIFT.

## Architecture (BioCDA-XA v1)

```text
omics → Z64 (frozen O2)
context → C32 (frozen prototype context)
LayerNorm(Z), LayerNorm(C) → concat → sample projection → S
GIN atom nodes → K, V
S → Q; cross-attention → attended drug representation
[S ; attended drug] → response head → logit [B]
```

Candidates:

- **M0** `pooled_baseline` — D0 max-pool + adapter fusion (performance baseline)
- **M1** `biocda_xa_z` — query from Z only
- **M2** `biocda_xa_zc` — query from [Z; C] (primary BioCDA candidate)

## Docker execution (DAPL)

All commands run inside container `/workspace/DAPL`:

```bash
docker start DAPL

# Unit tests + architecture smoke
docker exec DAPL bash -lc '/workspace/DAPL/scripts/biocda/run_architecture_finalization.sh'

# Full Round 21 pipeline (splits → audit → smoke → GDSC train → diagnose → compare → lock)
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml all'

# Verify lock manifest
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/verify_model_lock.py --strict'
```

Repository audit on host (requires clean git tree):

```bash
python3 scripts/audit_repository_state.py --strict
```

## Training configuration

- Data: Round 19 `development_rows.csv` + Round 20 C32 features (`z_plus_context32`)
- Split: drug-level 70/15/15 train/val/test per seed `{17, 29, 43}` — no drug leakage
- Freeze policy (Phase A): omics encoder, context path, GIN frozen; train sample projection, cross-attention, response head
- Optimizer: AdamW lr=5e-4, ReduceLROnPlateau, AMP, batch=512, grad clip=5
- Early stopping: validation DrugMacro AUC, patience=20
- **TCGA not used for model selection**

## Key reports

```text
reports/repository_state_audit.json
reports/biocda_architecture_runtime_audit.json
reports/biocda_forward_trace.json
reports/model_parameter_comparison.csv
reports/model_comparison_summary.csv
reports/paired_model_deltas.csv
reports/attention_health_summary.csv
reports/query_sensitivity_summary.csv
reports/context_sensitivity_summary.csv
reports/modality_scale_summary.json
reports/model_selection_decision.md
reports/biocda_final_model_lock.json
```

## Model lock policy

`reports/biocda_final_model_lock.json` status is **`LOCKED`** only when all selection gates pass:

1. Functional correctness (attention sums, no bypass, checkpoint load)
2. Performance guardrails vs M0
3. Context utilization (M2 uses context in query path)
4. Attention health (non-uniform, non-collapsed, head diversity)

If any gate fails → `NEEDS_REVISION` or `REJECTED` with reasons in `model_selection_decision.md`.

## Design notes

### Attention dropout contract

- `attention_probabilities` = pre-dropout raw softmax (exported for interpretability)
- `attention_probabilities_used` = dropout applied in train mode only (debug/full trace)

### Masked softmax

`biocda/utils/masked_softmax.py` renormalizes over valid atoms; padding=0; empty graphs raise `ValueError`.

### Atom batch metadata

`atom_ptr`, `atom_batch_index`, and RDKit/model atom indices preserved for Round 22 long-table attention export.
