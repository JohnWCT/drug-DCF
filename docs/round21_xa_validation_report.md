# BioCDA Round 21 — Cross-Attention Validation and Model Lock

Round code name: **Round 21 — Cross-Attention Validation and Model Lock**

Paper working title: **BioCDA: An Interpretable Biological-Context-Guided Framework for Cell-Line-to-Patient Drug Response Prediction**

## Final outcome

| Item | Result |
|------|--------|
| Pipeline status | **COMPLETE** |
| Model lock status | **REJECTED** (not LOCKED) |
| Retained predictive baseline | **M0 `pooled_baseline`** |
| M2 BioCDA-XA-ZC | Failed performance guardrails vs M0 |
| TCGA used for selection | **false** |

Mean validation DrugMacro AUC: M0 **0.746**, M1 **0.714**, M2 **0.709**.  
Mean ΔAUC(M2−M0) ≈ **−0.037** (guardrail requires ≥ −0.005).

See [model selection decision](../reports/model_selection_decision.md) and `reports/biocda_final_model_lock.json`.

## Scope completed in this round

| Stage | Status |
|-------|--------|
| Repository sync audit | PASS |
| Runtime architecture audit + forward trace | PASS |
| Cross-attention contract (raw/dropout, masked softmax, atom_ptr) | DONE |
| M0 / M1 / M2 factory | DONE |
| Unseen-drug split manifest (seeds 17/29/43) | DONE |
| GDSC repeated training (9 runs) | DONE |
| Parallel GPU dispatch (`max_parallel=3`) | DONE |
| Attention / context / query diagnostics | DONE |
| Selection gates + model lock manifest | DONE (`REJECTED`) |
| Unit tests | PASS |
| CI workflow | `.github/workflows/biocda-ci.yml` |

**Out of scope (deferred):** TCGA inference, attention heatmaps, RDKit 2D rendering, gene-level IG/DeepLIFT. Per Outcome 3, do **not** proceed to TCGA interpretability until XA performance recovers.

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

- **M0** `pooled_baseline` — D0 max-pool + adapter fusion (retained baseline)
- **M1** `biocda_xa_z` — query from Z only
- **M2** `biocda_xa_zc` — query from `[Z;C]` (primary BioCDA candidate; rejected for lock)

## Docker execution (DAPL)

```bash
docker start DAPL

# Unit tests + architecture smoke
docker exec DAPL bash -lc '/workspace/DAPL/scripts/biocda/run_architecture_finalization.sh'

# Parallel GDSC train (resume-safe)
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml train'

# Diagnose / compare / lock
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml diagnose-attention'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml compare'
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/run_xa_validation.py --config configs/biocda/xa_validation.yaml lock'
```

## Training configuration

- Data: Round 19 `development_rows.csv` + Round 20 C32 (`z_plus_context32`)
- Split: drug-level 70/15/15 train/val/test per seed `{17, 29, 43}`
- Freeze Phase A: omics / context / GIN frozen
- Optimizer: AdamW lr=5e-4, ReduceLROnPlateau, AMP
- Throughput: `max_parallel=3`, `micro_batch_size=768`, `accumulation_steps=2`, shared graph cache
- DataLoader workers: `0` (Docker `/dev/shm` constraint)
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

## Next round recommendations (Outcome 3)

1. Gated residual / attention distillation toward pooled baseline
2. Limited last-block GIN fine-tuning (Phase B) after functional + attention gates only
3. Temperature / QK scale sweeps if attention still weak
4. Do **not** run full TCGA interpretability until predictive parity with M0
